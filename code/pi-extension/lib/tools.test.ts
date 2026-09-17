import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  describeResult,
  isGenerative,
  nameFromPrompt,
  unknownRefMessage,
} from "./tools.ts";


describe("which results are kept", () => {
  test("only image-producing tools are written to the store", () => {
    for (const t of ["generate_image", "edit_image", "inpaint_image", "upscale_image"]) {
      assert.equal(isGenerative(t), true, t);
    }
    // Viewing tools return images too; writing them would fill the directory
    // with crops of things it already has.
    for (const t of ["inspect_region", "segment", "list_models"]) {
      assert.equal(isGenerative(t), false, t);
    }
  });
});

describe("naming", () => {
  test("names come from the prompt so they can be typed back", () => {
    assert.equal(nameFromPrompt("A lighthouse at dawn on a cliff", "kf"), "lighthouse-dawn-cliff");
  });

  test("filler words are dropped", () => {
    assert.equal(nameFromPrompt("make a picture of the harbour", "kf"), "picture-harbour");
  });

  test("an unusable prompt falls back rather than producing an empty name", () => {
    assert.equal(nameFromPrompt("", "image-3"), "image-3");
    assert.equal(nameFromPrompt("!!! ???", "image-3"), "image-3");
    assert.equal(nameFromPrompt("the a of", "image-3"), "image-3");
  });

  test("punctuation cannot leak into a name that has to round-trip as a ref", () => {
    assert.match(nameFromPrompt("a cat's \"hat\", stylised!", "kf"), /^[a-z0-9-]+$/);
  });
});

describe("what the model is told", () => {
  test("leads with the ref, since that is what it needs to continue", () => {
    const text = describeResult({ ref: "@harbour", width: 1024, height: 1024, model: "flux2-klein" });
    assert.ok(text.startsWith("Saved as @harbour"));
    assert.ok(text.includes("1024x1024"));
    assert.ok(text.includes("flux2-klein"));
  });

  test("an unknown ref is reported with what does exist", () => {
    // Returned rather than thrown: a stale ref should cost a turn, not the run.
    const text = unknownRefMessage("@nope", ["@a", "@b"]);
    assert.ok(text.includes("@nope"));
    assert.ok(text.includes("@a, @b"));
  });

  test("with nothing tracked yet it says so instead of listing nothing", () => {
    assert.ok(unknownRefMessage("@nope", []).includes("none yet"));
  });
});

