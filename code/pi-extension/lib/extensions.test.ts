import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, test } from "node:test";

/**
 * Pi loads *every* file in extensions/ and rejects any that does not default-export
 * a factory. A test or helper parked there fails the whole extension at startup,
 * which is why nothing but entry points lives in that directory.
 */
describe("extension entry points", () => {
  const dir = join(import.meta.dirname, "..", "extensions");
  const files = readdirSync(dir).filter((f) => f.endsWith(".ts"));

  test("the directory holds only loadable extensions", () => {
    const strays = files.filter((f) => f.includes(".test.") || f.includes(".spec."));
    assert.deepEqual(strays, [], "tests and helpers belong in lib/, not extensions/");
  });

  for (const file of files) {
    test(`${file} default-exports a factory`, async () => {
      const mod = await import(join(dir, file));
      assert.equal(typeof mod.default, "function", `${file} must export default function(pi)`);
    });
  }
});
