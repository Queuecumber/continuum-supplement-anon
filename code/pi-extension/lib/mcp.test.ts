import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  ContinuumClient,
  adaptToolSchema,
  describeProgress,
  extractImage,
  extractText,
  formatError,
  prepareCallArgs,
} from "./mcp.ts";

describe("tool schema adaptation", () => {
  test("media params become workspace refs and $defs are dropped", () => {
    const fn = adaptToolSchema({
      name: "edit_image",
      description: "Edit an image.",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string" },
          image: { $ref: "#/$defs/ImageContent" },
          references: { type: "array", items: { $ref: "#/$defs/ImageContent" } },
          steps: { type: "integer" },
        },
        $defs: { ImageContent: { type: "object" } },
      },
    });
    const props = fn.function.parameters.properties;
    assert.equal(fn.function.name, "edit_image");
    assert.equal(props.image.type, "string");
    assert.equal(props.references.type, "array");
    assert.deepEqual(props.references.items, { type: "string" });
    // Non-media params are untouched.
    assert.equal(props.prompt.type, "string");
    assert.equal(props.steps.type, "integer");
    assert.equal(fn.function.parameters.$defs, undefined);
  });

});

describe("call argument preparation", () => {
  const bytes = Buffer.from("PNGDATA");
  const resolve = (ref: string) => (ref === "@light" || ref === "@image-2" ? bytes : undefined);

  test("refs become ImageContent and non-media args pass through", () => {
    const call = prepareCallArgs({ prompt: "x", image: "@light", steps: 30 }, resolve);
    assert.deepEqual(call.image, {
      type: "image",
      data: bytes.toString("base64"),
      mimeType: "image/png",
    });
    assert.equal(call.prompt, "x");
    assert.equal(call.steps, 30);
  });

  test("reference lists resolve element-wise", () => {
    const call = prepareCallArgs({ references: ["@light", "@image-2"] }, resolve);
    assert.equal(call.references.length, 2);
    assert.equal(call.references[0].type, "image");
  });

  test("an unknown ref raises instead of silently dropping conditioning", () => {
    // Dropping it would still generate — just not the image the user asked for.
    assert.throws(() => prepareCallArgs({ image: "@nope" }, resolve), /unknown workspace ref @nope/);
    assert.throws(() => prepareCallArgs({ references: ["@nope"] }, resolve), /unknown workspace ref/);
  });

  test("pinSize and stripSeed are enforced over the model's choices", () => {
    const call = prepareCallArgs(
      { prompt: "x", width: 512, height: 512, seed: 42 },
      resolve,
      { pinSize: [1024, 768], stripSeed: true },
    );
    assert.equal(call.width, 1024);
    assert.equal(call.height, 768);
    assert.equal(call.seed, undefined);
  });

  test("the caller's args object is not mutated", () => {
    const args = { image: "@light" };
    prepareCallArgs(args, resolve);
    assert.equal(args.image, "@light");
  });
});

describe("result extraction", () => {
  test("image bytes come off ImageContent", () => {
    const result = {
      content: [
        { type: "text", text: "done" },
        { type: "image", data: Buffer.from("IMG").toString("base64") },
      ],
    };
    assert.equal(extractImage(result)!.toString(), "IMG");
    assert.equal(extractText(result), "done");
  });



  test("absent media yields undefined rather than throwing", () => {
    const textOnly = { content: [{ type: "text", text: "hi" }] };
    assert.equal(extractImage(textOnly), undefined);
  });
});

describe("error formatting", () => {
  test("aggregate errors flatten to leaf causes", () => {
    // A transport wrapper reads as a useless message without this.
    const inner = new Error("429 Too Many Requests");
    const agg = new AggregateError([inner], "unhandled errors");
    assert.equal(formatError(agg), "Error: 429 Too Many Requests");
  });

  test("nested aggregates flatten all the way down", () => {
    const nested = new AggregateError([new AggregateError([new TypeError("bad")], "inner")], "outer");
    assert.equal(formatError(nested), "TypeError: bad");
  });

  test("plain errors and non-errors still render", () => {
    assert.equal(formatError(new Error("boom")), "Error: boom");
    assert.equal(formatError("plain string"), "plain string");
  });
});

describe("schema $defs", () => {
  test("keeps $defs when a $ref still points into it", () => {
    // continuum's tools $ref LoraSpec from `loras`. Dropping $defs leaves that
    // reference dangling, and some gateways answer with a bare
    // "invalid_argument" 400 that names neither tool nor parameter — which is
    // exactly how this presented, as a model configuration problem.
    const adapted = adaptToolSchema({
      name: "generate_image",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string" },
          image: { $ref: "#/$defs/ImageContent" },
          loras: { type: "array", items: { $ref: "#/$defs/LoraSpec" } },
        },
        $defs: { ImageContent: { type: "object" }, LoraSpec: { type: "object" } },
      },
    });
    const params = adapted.function.parameters;
    assert.ok(params.$defs, "$defs must survive while loras references it");
    assert.ok(params.$defs.LoraSpec, "the referenced definition specifically");
    // The media param was rewritten, so its definition is no longer needed —
    // but removing only that is not worth the risk of a stale reference.
    assert.equal(params.properties.image.type, "string");
  });

  test("drops $defs once nothing references it", () => {
    const adapted = adaptToolSchema({
      name: "edit_image",
      inputSchema: {
        type: "object",
        properties: { prompt: { type: "string" }, image: { $ref: "#/$defs/ImageContent" } },
        $defs: { ImageContent: { type: "object" } },
      },
    });
    assert.equal(adapted.function.parameters.$defs, undefined);
  });
});

describe("long-running tool calls", () => {
  test("asks for progress so the timeout can reset on it", async () => {
    // Two coupled requirements in the SDK: resetTimeoutOnProgress defaults to
    // false, and the progressToken is attached only when an onprogress handler
    // is supplied. Setting the flag without the handler means the server sends
    // no notifications and nothing ever resets — which is how a model load
    // still times out despite a generous limit.
    const seen: any[] = [];
    const client = new ContinuumClient("http://unused");
    (client as any).client = {
      callTool: async (_params: any, _schema: any, options: any) => {
        seen.push(options);
        return { content: [] };
      },
    };

    await client.call("generate_image", { prompt: "x" });
    const options = seen[0];
    assert.equal(options.resetTimeoutOnProgress, true, "must opt in to resetting");
    assert.equal(typeof options.onprogress, "function", "handler is what requests the token");
    assert.ok(options.timeout >= 60_000, "must exceed a model load");
    assert.ok(options.maxTotalTimeout > options.timeout, "backstop above the per-idle limit");
  });
})

describe("progress lines", () => {
  test("reads well for every shape continuum sends", () => {
    // A phase name during a model load, a step count during denoising, or both.
    assert.equal(describeProgress({ message: "downloading model" }), "downloading model");
    assert.equal(describeProgress({ progress: 3, total: 20 }), "3/20");
    assert.equal(describeProgress({ message: "denoising", progress: 3, total: 20 }), "denoising 3/20");
  });

  test("never shows an empty line or a meaningless total", () => {
    // total=1 is the server saying "done", not a count worth rendering.
    assert.equal(describeProgress({ progress: 1, total: 1 }), "working");
    assert.equal(describeProgress({}), "working");
    assert.equal(describeProgress(undefined), "working");
  });
})
