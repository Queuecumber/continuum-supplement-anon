/** Continuum image tools for Pi. */

import { readFileSync, readdirSync } from "node:fs";
import { extname } from "node:path";

import {
  ContinuumClient,
  adaptToolSchema,
  extractImage,
  extractText,
  formatError,
  prepareCallArgs,
} from "../lib/mcp.ts";

import { resolveRef, writeResult } from "../lib/store.ts";
import {
  describeResult,
  isGenerative,
  nameFromPrompt,
  unknownRefMessage,
} from "../lib/tools.ts";

/** Where the continuum server is, unless overridden. */
const DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp";

interface State {
  mcp: ContinuumClient | null;
  cwd: string | null;
  /** continuum's own operating manual, from the MCP handshake. */
  instructions: string | null;
}

const state: State = { mcp: null, cwd: null, instructions: null };

function cwd(): string {
  return state.cwd ?? process.cwd();
}

function mcpUrl(): string {
  return process.env.CONTINUUM_MCP_URL || DEFAULT_MCP_URL;
}

/** Media files in the working directory, to name in an unknown-ref message. */
const MEDIA_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".gif"];

function mediaFiles(): string[] {
  try {
    return readdirSync(cwd())
      .filter((f) => MEDIA_EXTENSIONS.includes(extname(f).toLowerCase()))
      .sort();
  } catch {
    return [];
  }
}

export default function (pi: any) {
  pi.on?.("session_start", async (_event: any, ctx: any) => {
    state.cwd = ctx.cwd ?? process.cwd();
    try {
      await connectMcp(pi, ctx);
    } catch (err) {
      // Continuum being down is normal — the GPU server is not always up — and
      // saying so once beats failing the session or retrying in the background.
      ctx.ui.notify(`continuum unavailable (${mcpUrl()}): ${formatError(err)}`, "warning");
    }
  });

  // continuum's instructions, verbatim, every turn.
  //
  // The tool descriptions cover individual calls; these cover how to use them
  // together, and they are the half of continuum that is not code. They are
  // maintained on the server, so they are passed through rather than
  // paraphrased here — a copy would drift.
  pi.on?.("before_agent_start", async (event: any) => {
    if (!state.instructions) return undefined;
    return { systemPrompt: `${event.systemPrompt}\n\n${state.instructions}` };
  });

  pi.on?.("session_shutdown", async () => {
    await state.mcp?.close().catch(() => {});
    state.mcp = null;
  });
}

/**
 * Register continuum's tools with Pi, adapted to file refs.
 *
 * The whole surface is exposed rather than a curated subset: continuum's tools
 * and its instructions are written to be used together, and picking a few would
 * quietly drop capabilities the model is told it has.
 */
async function connectMcp(pi: any, ctx: any): Promise<void> {
  if (state.mcp) return;
  const client = new ContinuumClient(mcpUrl());
  const { instructions, tools } = await client.connect();
  state.mcp = client;
  state.instructions = instructions ?? null;

  let first = true;
  for (const tool of tools) {
    const adapted = adaptToolSchema(tool);
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description ?? "",
      parameters: adapted.function.parameters,
      promptSnippet: `${tool.name} — continuum: ${(tool.description ?? "").split("\n")[0].slice(0, 90)}`,
      // Once, on the first tool: repeating this per tool would bloat the
      // system prompt with a dozen copies of the same paragraph.
      promptGuidelines: first
        ? [
            "Continuum's image parameters take a filename like @lighthouse.png, " +
              "not bytes. Pass the ref straight through; the extension loads the file. " +
              "Ignore any instruction in a continuum tool description to pass an " +
              "ImageContent — those describe the raw protocol.",
            "A continuum tool returns a new filename. Pass that to the next call rather " +
              "than the original, so edits compose.",
            "The first call loads a model and can take minutes. That is normal; do not " +
              "retry with a different model because a call is slow.",
          ]
        : undefined,
      execute: async (
        _id: string,
        params: Record<string, any>,
        _signal: AbortSignal | undefined,
        onUpdate: ((partial: any) => void) | undefined,
      ) => runTool(ctx, tool.name, params, onUpdate),
    });
    first = false;
  }
}

const text = (body: string) => ({ content: [{ type: "text", text: body }], details: {} });

async function runTool(
  ctx: any,
  name: string,
  params: Record<string, any>,
  onUpdate: ((partial: any) => void) | undefined,
): Promise<any> {
  const report = (message: string) => {
    onUpdate?.({ content: [{ type: "text", text: `${name}: ${message}` }], details: {} });
  };

  let args: Record<string, any>;
  try {
    args = prepareCallArgs(params, (ref) => bytesForRef(ref));
  } catch {
    const missing = [params.image, params.mask, ...(params.references ?? [])].find(
      (r: any) => typeof r === "string" && !bytesForRef(r),
    );
    return text(unknownRefMessage(String(missing ?? "that reference"), mediaFiles()));
  }

  let result: any;
  try {
    result = await state.mcp!.call(name, args, report);
  } catch (err) {
    return text(`${name} failed: ${formatError(err)}`);
  }

  if (!isGenerative(name)) return text(extractText(result) || "done");

  const image = extractImage(result);
  if (!image) return text(extractText(result) || "no image in result");

  const base = nameFromPrompt(params.prompt, name.replace(/_image$/, ""));
  const { name: file, path } = writeResult(cwd(), base, ".png", image);
  ctx.ui.setStatus?.("continuum", `wrote ${file}`);

  // The ImageContent goes back so Pi renders it in the transcript — that is
  // the whole display layer, and it is Pi's.
  return {
    content: [
      { type: "text", text: describeResult({ ref: `@${file}`, model: params.model, note: path }) },
      { type: "image", data: image.toString("base64"), mimeType: "image/png" },
    ],
    details: { ref: `@${file}`, path },
  };
}

/** Bytes behind a ref, for the MCP boundary. */
function bytesForRef(ref: string): Buffer | undefined {
  const path = resolveRef(cwd(), ref);
  return path ? readFileSync(path) : undefined;
}



