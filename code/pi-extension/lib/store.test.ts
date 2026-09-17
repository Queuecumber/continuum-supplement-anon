import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, test } from "node:test";

import { freeName, resolveRef, writeResult } from "./store.ts";

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "continuum-store-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe("naming a result", () => {
  test("an unused name is taken as-is", () => {
    assert.equal(freeName(dir, "harbour", ".png"), "harbour.png");
  });

  test("an existing file is never overwritten", () => {
    // Editing produces a new file rather than replacing the old one: the
    // earlier image is the only 'go back' left now the version graph is gone.
    writeFileSync(join(dir, "harbour.png"), "first");
    assert.equal(freeName(dir, "harbour", ".png"), "harbour-2.png");
    writeFileSync(join(dir, "harbour-2.png"), "second");
    assert.equal(freeName(dir, "harbour", ".png"), "harbour-3.png");
  });

  test("writing returns the name that was actually used", () => {
    writeFileSync(join(dir, "shot.png"), "first");
    const { name, path } = writeResult(dir, "shot", ".png", Buffer.from("second"));
    assert.equal(name, "shot-2.png");
    assert.equal(readFileSync(path, "utf8"), "second");
    assert.equal(readFileSync(join(dir, "shot.png"), "utf8"), "first", "the original survives");
  });
});

describe("resolving a ref", () => {
  beforeEach(() => {
    writeFileSync(join(dir, "lighthouse.png"), "png bytes");
  });

  test("the leading @ is the extension's own sigil, so it comes off", () => {
    assert.equal(resolveRef(dir, "@lighthouse.png"), join(dir, "lighthouse.png"));
    assert.equal(resolveRef(dir, "lighthouse.png"), join(dir, "lighthouse.png"));
  });


  test("everything else is taken literally — a ref is a filename", () => {
    // No extension guessing: @lighthouse is not lighthouse.png. The model is
    // told the filename by the tool that wrote it, and completion inserts the
    // whole name, so inferring one would only paper over a wrong ref.
    assert.equal(resolveRef(dir, "@lighthouse"), undefined);
    assert.equal(resolveRef(dir, "@nope.png"), undefined);
    assert.equal(resolveRef(dir, "@"), undefined);
    assert.equal(resolveRef(dir, ""), undefined);
  });
});
