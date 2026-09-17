/**
 * Continuum MCP client — standard mode only.
 *
 * The extension owns the loop and a client-side ref<->bytes mapping against a
 * STATELESS continuum: the model emits workspace refs (`@name`, `@name:3`,
 * `@image-2`) and this layer swaps them for ImageContent bytes at the MCP
 * boundary. No `continuum://` URIs, no server-side resource store — that is the
 * legacy compat path and it is deliberately unused.
 *
 * Ported from continuum/harnesses/agent_loop.py, which is the reference
 * implementation and already paid for several of these seams in production.
 *
 * The pure functions here are unit-tested; `ContinuumClient` needs a live
 * server and is exercised only in a real session.
 */

export const SINGLE_MEDIA_PARAMS = new Set(["image", "mask"]);
export const LIST_MEDIA_PARAMS = new Set(["references"]);

export interface ToolDef {
  name: string;
  description?: string;
  inputSchema: Record<string, any>;
}

/** Resolves a workspace ref to bytes; the workspace supplies this. */
export type MediaResolver = (ref: string) => Buffer | undefined;

/**
 * Rewrite a continuum tool schema so media parameters are workspace refs
 * instead of inline content. The model should never be asked to emit bytes,
 * and `$defs` for ImageContent become unreferenced once they are gone.
 */
export function adaptToolSchema(tool: ToolDef): Record<string, any> {
  const schema = JSON.parse(JSON.stringify(tool.inputSchema ?? {}));
  const props = schema.properties ?? {};
  for (const key of Object.keys(props)) {
    if (SINGLE_MEDIA_PARAMS.has(key)) {
      props[key] = {
        type: "string",
        description: "workspace ref, e.g. '@lighthouse' or '@lighthouse:2'; never paste bytes",
      };
    } else if (LIST_MEDIA_PARAMS.has(key)) {
      props[key] = {
        type: "array",
        items: { type: "string" },
        description: "workspace refs, e.g. ['@lighthouse', '@image-2']; never paste bytes",
      };
    }
  }
  // Drop $defs only once nothing references it. Rewriting the media params
  // removes the ImageContent definitions, but `loras` still $refs LoraSpec, and
  // a dangling $ref makes some gateways reject the whole request with a bare
  // "invalid_argument" 400 that names no tool and no parameter. The Python
  // harness lost every run to this before the same fix.
  const withoutDefs = JSON.stringify(
    Object.fromEntries(Object.entries(schema).filter(([key]) => key !== "$defs")),
  );
  if (schema.$defs && !withoutDefs.includes('"$ref"')) {
    delete schema.$defs;
  }

  return {
    type: "function",
    function: { name: tool.name, description: tool.description ?? "", parameters: schema },
  };
}

function toImageContent(data: Buffer): Record<string, any> {
  return { type: "image", data: data.toString("base64"), mimeType: "image/png" };
}

/**
 * Swap workspace refs for ImageContent, and enforce harness-side parameters.
 *
 * pinSize is applied regardless of what the model asked for, so a run stays
 * dimensionally consistent; an unresolvable ref raises rather than silently
 * dropping the conditioning image, which would produce a plausible-looking but
 * wrong generation.
 */
export function prepareCallArgs(
  args: Record<string, any>,
  resolve: MediaResolver,
  opts: { pinSize?: [number, number]; stripSeed?: boolean } = {},
): Record<string, any> {
  const call: Record<string, any> = { ...args };

  for (const key of SINGLE_MEDIA_PARAMS) {
    const value = call[key];
    if (typeof value === "string") {
      const bytes = resolve(value);
      if (!bytes) throw new Error(`unknown workspace ref ${value} for parameter '${key}'`);
      call[key] = toImageContent(bytes);
    }
  }
  for (const key of LIST_MEDIA_PARAMS) {
    const value = call[key];
    if (Array.isArray(value)) {
      call[key] = value.map((v) => {
        if (typeof v !== "string") return v;
        const bytes = resolve(v);
        if (!bytes) throw new Error(`unknown workspace ref ${v} in '${key}'`);
        return toImageContent(bytes);
      });
    }
  }

  if (opts.stripSeed) delete call.seed;
  if (opts.pinSize) {
    call.width = opts.pinSize[0];
    call.height = opts.pinSize[1];
  }
  return call;
}

/** First ImageContent in a CallToolResult, or undefined. */
export function extractImage(result: any): Buffer | undefined {
  for (const item of result?.content ?? []) {
    if (item?.type === "image" && typeof item.data === "string") {
      return Buffer.from(item.data, "base64");
    }
  }
  return undefined;
}

export function extractText(result: any): string {
  return (result?.content ?? [])
    .filter((c: any) => typeof c?.text === "string")
    .map((c: any) => c.text)
    .join("\n");
}

/**
 * Flatten nested aggregate errors to their leaf causes.
 *
 * The Python harness had to do this because anyio buries the real failure in a
 * task-group ExceptionGroup; the JS transports do the same with AggregateError,
 * and an unflattened message reads as a useless wrapper.
 */
export function formatError(err: unknown): string {
  if (err && typeof err === "object" && "errors" in err && Array.isArray((err as any).errors)) {
    return (err as any).errors.map(formatError).join("; ");
  }
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  return String(err);
}

/**
 * A progress notification as a line worth showing.
 *
 * continuum sends a message for the phases that take the time ("downloading
 * model", "setting up quantization") and a bare count for denoising steps, so
 * both shapes have to read well: a message alone, a count alone, or both.
 */
export function describeProgress(notification: any): string {
  const message = typeof notification?.message === "string" ? notification.message : "";
  const done = notification?.progress;
  const total = notification?.total;
  const counted =
    typeof done === "number" && typeof total === "number" && total > 1 ? `${done}/${total}` : "";
  if (message && counted) return `${message} ${counted}`;
  return message || counted || "working";
}

export class ContinuumClient {
  private client: any = null;
  private tools: ToolDef[] = [];
  private url: string;
  private timeoutMs: number;
  private maxTotalTimeoutMs: number;

  // Explicit field + assignment rather than a parameter property: node's
  // strip-only TypeScript mode rejects parameter properties (they emit code),
  // and the tests run under plain `node --test`.
  constructor(url: string, opts: { timeoutMs?: number; maxTotalTimeoutMs?: number } = {}) {
    this.url = url;
    // Generous, because it only bites when nothing is happening at all: a
    // loading model reports progress and keeps resetting this.
    this.timeoutMs = opts.timeoutMs ?? 10 * 60_000;
    this.maxTotalTimeoutMs = opts.maxTotalTimeoutMs ?? 60 * 60_000;
  }

  async connect(): Promise<{ instructions?: string; tools: ToolDef[] }> {
    const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
    const { StreamableHTTPClientTransport } = await import(
      "@modelcontextprotocol/sdk/client/streamableHttp.js"
    );
    const client = new Client({ name: "continuum-pi", version: "0.0.0" }, { capabilities: {} });
    await client.connect(new StreamableHTTPClientTransport(new URL(this.url)));
    this.client = client;
    const listed = await client.listTools();
    this.tools = (listed.tools ?? []) as ToolDef[];
    // Server instructions are continuum's own operating manual — the agent
    // should see them, not just the tool docstrings.
    return { instructions: client.getInstructions?.(), tools: this.tools };
  }

  toolSchemas(allow?: Set<string>): Record<string, any>[] {
    return this.tools.filter((t) => !allow || allow.has(t.name)).map(adaptToolSchema);
  }

  /**
   * Call a tool, allowing for model loads.
   *
   * The SDK times requests out after 60 seconds by default, which every first
   * call exceeds: continuum loads the model before it can generate, and that
   * alone runs into minutes. The failure arrives as a bare "Request timed out"
   * with the work still running server-side, so the model retries and pays the
   * cost again.
   *
   * continuum reports progress throughout, so the timeout resets on each
   * notification: a call that is doing something is never cut off, while one
   * that has genuinely stalled still ends. maxTotalTimeout is the backstop for
   * a job that reports progress forever.
   */
  async call(name: string, args: Record<string, any>, onProgress?: (message: string) => void): Promise<any> {
    if (!this.client) throw new Error("not connected");
    return this.client.callTool({ name, arguments: args }, undefined, {
      timeout: this.timeoutMs,
      resetTimeoutOnProgress: true,
      maxTotalTimeout: this.maxTotalTimeoutMs,
      // Always supplied, even with nothing to report it to: the SDK attaches
      // the progressToken only when this is present, and without that token the
      // server sends no progress notifications — leaving resetTimeoutOnProgress
      // with nothing to reset on.
      onprogress: (p: any) => onProgress?.(describeProgress(p)),
    });
  }

  async close(): Promise<void> {
    await this.client?.close?.();
    this.client = null;
  }
}
