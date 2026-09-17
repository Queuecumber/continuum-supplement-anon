/**
 * Where generated media lands: the working directory.
 *
 * Files, in a directory. There is no manifest, no version graph and no
 * database: continuum writes what it makes, and the conversation is the
 * history — Pi already keeps that, including the prompt that produced each
 * file. A store that also tracked provenance would be a second, worse copy of
 * something the transcript already holds.
 *
 * The consequence worth noticing is that a reference is now just a filename.
 * Nothing is opaque, so a model that goes looking on disk finds exactly what it
 * was promised, and a user can open the directory in anything.
 */

import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * A filename that does not overwrite an existing one.
 *
 * Editing an image produces a new file rather than replacing the old: the
 * earlier one is what "go back" means, and it is the only versioning left now
 * that the version graph is gone.
 */
export function freeName(dir: string, base: string, ext: string): string {
  let candidate = `${base}${ext}`;
  let n = 2;
  while (existsSync(join(dir, candidate))) {
    candidate = `${base}-${n}${ext}`;
    n += 1;
  }
  return candidate;
}

/** Write bytes into the working directory, returning the filename chosen. */
export function writeResult(dir: string, base: string, ext: string, data: Buffer): { name: string; path: string } {
  const name = freeName(dir, base, ext);
  const path = join(dir, name);
  writeFileSync(path, data);
  return { name, path };
}

/**
 * The file a `@ref` names, or undefined.
 *
 * Only the leading `@` is removed, because that sigil is the extension's own —
 * completion inserts it and the tool schemas ask for it. Everything else is
 * taken literally: a ref is a filename.
 */
export function resolveRef(dir: string, ref: string): string | undefined {
  const name = ref.replace(/^@/, "");
  if (!name) return undefined;
  const path = join(dir, name);
  return existsSync(path) ? path : undefined;
}
