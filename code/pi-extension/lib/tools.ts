/**
 * Turning continuum's MCP tools into Pi tools.
 *
 * The model works in filenames, never bytes. Media parameters are rewritten to
 * refs in the tool schema (see mcp.ts), the extension swaps them for image
 * content at the MCP boundary, and the result is written to the store and
 * named back. That keeps whole images out of the transcript except where Pi
 * renders one for the model to look at.
 *
 * The pure decisions live here so they can be tested without a server: what a
 * result is called, and what the model is told about it.
 */

/** Continuum tools whose output is a new image worth keeping. */
export const GENERATIVE_TOOLS = new Set([
  "generate_image",
  "edit_image",
  "inpaint_image",
  "outpaint_image",
  "upscale_image",
]);

export function isGenerative(tool: string): boolean {
  return GENERATIVE_TOOLS.has(tool);
}

/**
 * A short filename from a prompt.
 *
 * Names are for the user to type back, so they are drawn from the prompt's
 * first few words rather than being opaque ids.
 */
export function nameFromPrompt(prompt: string | undefined, fallback: string): string {
  const words = String(prompt ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .filter((w) => !STOP_WORDS.has(w))
    .slice(0, 3);
  return words.length ? words.join("-") : fallback;
}

const STOP_WORDS = new Set([
  "a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "for",
  "is", "are", "it", "its", "this", "that", "make", "create", "generate", "add",
]);

/**
 * What the model is told after a call succeeds.
 *
 * It leads with the filename because that is what it passes to the next call,
 * and the dimensions because they are the thing it most often needs to reason
 * about and cannot read precisely off a rendered image.
 */
export function describeResult(opts: {
  ref: string;
  width?: number;
  height?: number;
  model?: string;
  note?: string;
}): string {
  const parts = [`Saved as ${opts.ref}`];
  if (opts.width && opts.height) parts.push(`${opts.width}x${opts.height}`);
  if (opts.model) parts.push(`model ${opts.model}`);
  const head = parts.join("  ");
  return opts.note ? `${head}\n${opts.note}` : head;
}

/**
 * Message for a ref that does not resolve.
 *
 * Returned to the model as a result rather than thrown, so it can correct the
 * reference and carry on — a hallucinated or stale ref should cost one turn,
 * not the whole run.
 */
export function unknownRefMessage(ref: string, known: string[]): string {
  const list = known.length ? known.slice(0, 12).join(", ") : "none yet";
  return `No file is called ${ref}. Available: ${list}.`;
}
