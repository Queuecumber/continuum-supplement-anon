"""Agent loop: drive an OpenAI-compatible LLM through Continuum's MCP tools.

The harness owns the loop and a client-side id<->bytes registry against a
STATELESS standard-mode Continuum server (ImageContent bytes in and out — no
server-side URI store). The model only ever emits image *ids*; the harness
swaps ids -> bytes when calling the server, registers returned bytes under a
fresh id, and shows the model each result as a downscaled image_content block.
Full-res bytes stay in the registry so edits run on pristine pixels; only the
model's *view* is downscaled.

openai / mcp / PIL are imported lazily so `import continuum.harnesses` stays
cheap and the LLM client is an optional extra (continuum[agent]).
"""

import asyncio
import base64
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

# Continuum tool parameters that carry image bytes (ImageContent). Single-image
# params become an id string for the model; list params become a list of ids.
_SINGLE_IMAGE_PARAMS = {"image", "mask"}
_LIST_IMAGE_PARAMS = {"references"}

# Tools whose result is a candidate final output. Others still return images —
# inspect_region crops, segment masks — and are registered so the model can
# reference them by id, but they must never become the run's output: agents
# routinely finish by zooming in to check their work, which would otherwise
# save a thumbnail of a detail instead of the image.
# Generation and editing only. upscale is excluded because it changes the
# output dimensions, which would break the pinned size a benchmark run compares
# against. inpaint/outpaint count as editing: they produce
# a corrected full-size image, and omitting them would save the *pre*-fix
# version when a run ends on a surgical correction — the same class of bug.
MARK_BEST_TOOL_NAME = "mark_best"

# The alias the model may pass wherever an image id is accepted.
BEST_ALIAS = "best"

# Harness-local, never sent to the server: image ids are the harness's own
# registry and continuum has never heard of them.
#
# Without this the run's result is whichever image happened to be produced last,
# which is wrong precisely when it matters. Measured over this benchmark, 12% of
# generative calls source an *older* image than the newest one — the model
# discarding an edit that made things worse. Ending on one of those, or being
# cut off just after one, returns the version the model had already rejected.
MARK_BEST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MARK_BEST_TOOL_NAME,
        "description": (
            "Mark which image is your best result so far. Call this as soon as you have an "
            "image you would be willing to submit, and call it again whenever a later image "
            "is better. Only one image is marked at a time — the most recent call wins — and "
            "you may mark any image, including an earlier one, if a later edit made things "
            "worse. Marking does not end the run: mark, then carry on improving. Afterwards "
            "you can pass 'best' anywhere an image id is accepted, so edit_image(image='best') "
            "continues from your marked image without tracking its id. If the run is cut "
            "short by a step limit or an error, the marked image is what is returned — work "
            "you never marked can be lost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image id to mark, e.g. 'img_2'."},
                "reason": {
                    "type": "string",
                    "description": "Briefly, why this is your best result so far.",
                },
            },
            "required": ["image"],
        },
    },
}


_OUTPUT_TOOLS = {
    "generate_image",
    "edit_image",
    "inpaint_image",
    "outpaint_image",
}


# A tool call the server failed to parse -----------------------------------
#
# Qwen emits tool calls as XML. A server started with a reasoning parser the
# model does not match returns the whole response as reasoning_content with
# tool_calls empty -- the call is well formed and sitting right there, the
# deployment just never extracted it. Seen on qwen3.5-0.8b, which emits no
# <think> markers, so `--reasoning-parser qwen3` treats everything it says as
# reasoning.
#
# Salvaging it here is free when the server did its job: it only runs when
# tool_calls is empty AND the text holds a <tool_call> block.
_XML_TOOL_CALL = re.compile(
    r"<\s*tool_call\s*>(.*?)(?:<\s*/\s*tool_call\s*>|$)", re.DOTALL)
_XML_FUNCTION = re.compile(
    r"<\s*function\s*=\s*([^>]*?)\s*>(.*?)(?:<\s*/\s*function\s*>|$)", re.DOTALL)
_XML_PARAM = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=)|$)", re.DOTALL)


def _trim_one_newline(value: str) -> str:
    """Strip one leading and one trailing newline -- the template's markup."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _coerce_arg(value: str, schema: dict[str, Any]) -> Any:
    """Cast one parameter to what the tool schema asks for.

    A value that will not cast passes through as the string it was, so the tool
    answers with a validation error the model can read rather than a silent None.
    """
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    text = value.strip()
    try:
        if kind == "integer":
            return int(text)
        if kind == "number":
            return float(text)
        if kind == "boolean":
            if text.lower() in ("true", "yes", "1"):
                return True
            if text.lower() in ("false", "no", "0"):
                return False
            return value
        if kind in ("object", "array"):
            return json.loads(text)
        if kind is None and (
            text[:1] in "[{"
            or text in ("true", "false", "null")
            or re.fullmatch(r"-?\d+(\.\d+)?", text)
        ):
            return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def salvage_xml_tool_calls(text: str, tools: list[dict[str, Any]] | None) -> list[Any]:
    """Parse Qwen's XML tool-call syntax out of free text.

    Returns objects shaped like the OpenAI SDK's tool calls (.id, .type,
    .function.name/.arguments) so callers cannot tell them from ones the server
    parsed itself.
    """
    if not text or "<tool_call" not in text:
        return []
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if fn.get("name"):
            schemas[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}

    calls: list[Any] = []
    for block in _XML_TOOL_CALL.finditer(text):
        for fn_match in _XML_FUNCTION.finditer(block.group(1)):
            name = fn_match.group(1).strip()
            if not name:
                continue
            args: dict[str, Any] = {}
            for param in _XML_PARAM.finditer(fn_match.group(2)):
                key = param.group(1).strip()
                if key:
                    args[key] = _coerce_arg(
                        _trim_one_newline(param.group(2)),
                        schemas.get(name, {}).get(key, {}),
                    )
            calls.append(
                SimpleNamespace(
                    id=f"call_{uuid.uuid4().hex[:20]}",
                    type="function",
                    function=SimpleNamespace(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    ),
                )
            )
    return calls


@dataclass
class ImageRegistry:
    """Client-side id -> full-resolution image bytes."""

    _by_id: dict[str, bytes] = field(default_factory=dict)
    _counter: int = 0

    def register(self, data: bytes) -> str:
        img_id = f"img_{self._counter}"
        self._counter += 1
        self._by_id[img_id] = data
        return img_id

    def get(self, img_id: str) -> bytes:
        if img_id not in self._by_id:
            raise KeyError(f"unknown image id {img_id!r}")
        return self._by_id[img_id]

    def __contains__(self, img_id: str) -> bool:
        return img_id in self._by_id

    def as_dict(self) -> dict[str, bytes]:
        """Snapshot of every id -> bytes, for persisting the full image chain."""
        return dict(self._by_id)


def adapt_tool_schema(name: str, description: str | None, input_schema: dict[str, Any]) -> dict[str, Any]:
    """Turn a Continuum MCP tool into an OpenAI function def.

    Image-typed params are rewritten to id strings (or arrays of id strings)
    so the model references prior images by id instead of re-emitting bytes.
    """
    schema = json.loads(json.dumps(input_schema))  # deep copy
    props = schema.get("properties", {})
    for param in list(props):
        if param in _SINGLE_IMAGE_PARAMS:
            props[param] = {
                "type": "string",
                "description": "id of an image in this session (e.g. 'img_0'); do not paste bytes",
            }
        elif param in _LIST_IMAGE_PARAMS:
            props[param] = {
                "type": "array",
                "items": {"type": "string"},
                "description": "ids of images in this session (e.g. ['img_0']); do not paste bytes",
            }
    # Drop $defs only once nothing references them. Media params become plain
    # strings so their defs go unused, but others (loras -> LoraSpec) keep a
    # $ref, and removing its target leaves a dangling reference — an invalid
    # schema that strict endpoints reject with 400 invalid_argument while
    # lenient ones silently accept it.
    if "$defs" in schema and "$ref" not in json.dumps({k: v for k, v in schema.items() if k != "$defs"}):
        schema.pop("$defs")

    return {
        "type": "function",
        "function": {"name": name, "description": description or "", "parameters": schema},
    }


def _select_tool_names(
    all_names: list[str],
    allow: set[str] | None,
    block: set[str] | None,
) -> set[str]:
    """Which tools the agent may use: intersect with `allow` (if given), then
    subtract `block`. block wins over allow.
    """
    names = set(all_names)
    if allow:
        names &= allow
    if block:
        names -= block
    return names


class UnknownImageId(Exception):
    """The model referenced an image id that was never produced.

    Its own mistake, and a recoverable one: it should cost a turn, not the run.
    Weaker models do this on the very first call, naming img_0 before anything
    has been generated -- which used to raise out of the loop and fail the item
    at step 1.
    """

    def __init__(self, img_id: str, known: list[str]) -> None:
        self.img_id, self.known = img_id, known
        super().__init__(img_id)

    def message(self) -> str:
        have = ", ".join(self.known[-6:]) if self.known else "none yet"
        return (f"No image {self.img_id!r} exists. Available ids: {have}. "
                "Generate one first, or use an id that was returned to you.")


def _id_to_image_content(img_id: str, registry: ImageRegistry) -> dict[str, Any]:
    if img_id not in registry:
        raise UnknownImageId(img_id, list(registry.as_dict()))
    data = registry.get(img_id)
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": "image/png"}


# The only tools that take a model. Forcing `model` onto anything else would
# be rejected by the server's schema.
_MODEL_TOOLS = frozenset({"generate_image", "edit_image"})


def prepare_call_args(
    args: dict[str, Any],
    registry: ImageRegistry,
    *,
    pin_size: tuple[int, int] | None = None,
    strip_seed: bool = True,
    best_id: str | None = None,
    pin_model: str | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Translate a model tool-call's args into a real MCP call.

    Image ids -> ImageContent bytes from the registry; width/height clamped to
    pin_size; seed stripped so samples vary; `model` forced to pin_model on the
    tools that accept one (all enforced regardless of what the model emitted).

    `best_id` resolves the "best" alias, so the model can continue from the
    image it marked without carrying its id around — which is the case where
    remembering an id is hardest, because marking an earlier image means the
    newest id is the wrong one.
    """

    def resolve(value: str) -> str:
        return best_id if value == BEST_ALIAS and best_id else value

    call = dict(args)
    for param in _SINGLE_IMAGE_PARAMS:
        if isinstance(call.get(param), str):
            call[param] = _id_to_image_content(resolve(call[param]), registry)
    for param in _LIST_IMAGE_PARAMS:
        value = call.get(param)
        if isinstance(value, list):
            call[param] = [
                _id_to_image_content(resolve(v), registry) if isinstance(v, str) else v for v in value
            ]
    if strip_seed:
        call.pop("seed", None)
    if pin_size is not None:
        call["width"], call["height"] = int(pin_size[0]), int(pin_size[1])
    if pin_model is not None and (tool_name is None or tool_name in _MODEL_TOOLS):
        # Omitting `model` does not use the model the policy names -- it uses the
        # server's defaults, and the call succeeds, so a benchmark arm can run to
        # completion on the wrong generator with nothing reporting a swap.
        # Measured at 2.4% of items for one model and ~10% for another, which is
        # why this cannot be left to instruction-following.
        call["model"] = pin_model
    return call


# Quality for the model's view of an image. JPEG rather than PNG: these are
# photographic frames, and lossless encoding of an already-downscaled image is
# incoherent — the resize has thrown away far more than the codec will. At 768px
# this is ~5x smaller than PNG (925KB -> 158KB mean, measured over this
# benchmark's outputs), which is what keeps a long conversation under the
# upstream proxy's request-size limit without dropping images from context.
#
# 90 verified against the case that would notice first: an inspect_region crop
# of rendered text reads identically to PNG. Raise it if fine detail ever looks
# marginal; the working bytes are unaffected either way, since the registry
# keeps full-resolution PNG and tool calls are served from there.
MODEL_VIEW_JPEG_QUALITY = 90


def image_content_part(
    data: bytes, max_side: int = 768, quality: int = MODEL_VIEW_JPEG_QUALITY
) -> dict[str, Any]:
    """An OpenAI image_url content part, downscaled to bound vision tokens."""
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(data))
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    buf = BytesIO()
    # convert after resizing: RGBA -> RGB is required for JPEG, and doing it
    # once here also covers images that never needed downscaling.
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def extract_image_bytes(call_result: Any) -> bytes | None:
    """First ImageContent's bytes from an MCP CallToolResult, or None."""
    for content in getattr(call_result, "content", []) or []:
        blob = getattr(content, "data", None)
        if blob is not None and getattr(content, "type", None) == "image":
            return base64.b64decode(blob)
    return None


def extract_result_text(call_result: Any) -> str:
    parts = []
    for content in getattr(call_result, "content", []) or []:
        text = getattr(content, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


@dataclass
class LoopResult:
    status: str
    output_image: bytes | None
    output_image_id: str | None
    steps: int
    transcript: list[dict[str, Any]]  # chat history with image bytes replaced by ids (see redact_transcript)
    error: str | None = None
    images: dict[str, bytes] = field(default_factory=dict)  # id -> full-res bytes for every image shown
    # How the loop ended: completed (model returned control — no tool call — accepting
    # its output), max_steps (harness cut it off mid-work), or error.
    stop_reason: str = "completed"
    # Provenance: image id -> the model asked to produce it, so a downstream
    # training-data export can filter by licence. "unspecified" means the caller
    # omitted `model` and the server's default applied (not recoverable here —
    # continuum's tools return bytes, not the model they used).
    image_models: dict[str, str] = field(default_factory=dict)
    # The image the model explicitly kept, if it used the keep tool, and why.
    # None means it never kept anything and the output is whatever was produced
    # last — which is the case the tool exists to remove.
    kept_id: str | None = None
    kept_reason: str = ""


def _assistant_message(message: Any) -> dict[str, Any]:
    """Serialize an OpenAI assistant message (with any tool_calls) for history.

    `reasoning_content` is carried through when the model returns one. Reasoning
    models put their chain of thought there rather than in `content` — so a turn
    that only makes a tool call has an empty `content` — and the gateway accepts
    it back on prior assistant turns, which lets the model continue a line of
    reasoning across a tool call instead of re-deriving it each step. It also
    means the saved transcript shows what the model was thinking, which it
    previously did not.
    """
    out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and str(reasoning).strip():
        out["reasoning_content"] = str(reasoning)
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return out


def _system_message(server_instructions: str | None, system_prompt: str | None) -> str | None:
    """Compose the system message: the MCP server's instructions (continuum's
    operating manual / editing discipline) first, then the run's system prompt
    (AGENTS.md + policy) on top.
    """
    parts = [p.strip() for p in (server_instructions, system_prompt) if p and p.strip()]
    return "\n\n".join(parts) or None


def _truncate(text: str, limit: int) -> str:
    """One-line, length-bounded rendering for progress output."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_tool_args(args: dict[str, Any], limit: int = 60) -> str:
    """Compact one-line view of a tool call's args for progress output.

    Image params are still ids here (bytes are swapped in downstream), so this
    stays short — long strings truncate, lists collapse to a count.
    """
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            parts.append(f"{key}={_truncate(value, limit)!r}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)} ids]")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def cap_inline_images(messages: list[dict[str, Any]], keep: int) -> int:
    """Replace all but the last `keep` inline images with a text placeholder.

    Off by default (`keep < 0`): the model reasons better with every image it
    has produced still in view, and for this work that is worth the payload.
    Kept as a lever because it was added for a real failure — every tool result
    appends an image and each request re-sends the whole history, so payload
    grows with step count until the endpoint refuses it, observed as HTTP 413 at
    step 15. If that returns, cap rather than shortening the run.

    Only the model's *view* shrinks: full-resolution bytes stay in the registry,
    so edits still run on pristine pixels and the agent can inspect_region any
    earlier image by id. Returns how many were elided.
    """
    if keep < 0:
        return 0
    positions: list[tuple[int, int]] = []
    for mi, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                positions.append((mi, ci))

    elided = 0
    for mi, ci in positions[: max(0, len(positions) - keep)]:
        part = messages[mi]["content"][ci]
        if part.get("_elided"):
            continue
        messages[mi]["content"][ci] = {
            "type": "text",
            "text": "(earlier image elided; inspect it by id if needed)",
            "_elided": True,
        }
        elided += 1
    return elided


def redact_transcript(messages: list[dict[str, Any]], image_ids: list[str]) -> list[dict[str, Any]]:
    """Copy of `messages` with base64 image data-urls replaced by image ids.

    Every image shown to the model is appended to `image_ids` in the same order
    it appears, so the k-th image_url part maps to image_ids[k]. Keeps the saved
    transcript small and legible (no megabyte data-urls) while preserving the
    chat shape — the bytes live in LoopResult.images, persisted alongside. A
    caller that stores those images can relink each id to its file path.
    """
    clean = [dict(message) for message in messages]
    k = 0
    for message in clean:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                ref = image_ids[k] if k < len(image_ids) else "image"
                k += 1
                new_content.append({"type": "image_url", "image_url": {"url": ref}})
            else:
                new_content.append(part)
        message["content"] = new_content
    return clean


def _format_exc(exc: BaseException) -> str:
    """Render an exception for the run log, unwrapping ExceptionGroups.

    anyio/MCP run the transport in a task group, so a plain failure (a 429, a
    bad tool schema, a vision error) surfaces as `ExceptionGroup: unhandled
    errors in a TaskGroup` with the real cause buried inside `.exceptions`.
    Recurse to the leaf causes so the log shows what actually went wrong.
    """
    inner = getattr(exc, "exceptions", None)
    if inner:
        return "; ".join(_format_exc(e) for e in inner)
    return f"{type(exc).__name__}: {exc}"


async def run_direct_tool_call(
    *,
    task_prompt: str,
    mcp_url: str,
    input_images: list[bytes] | None = None,
    tool_name: str = "generate_image",
    image_model: str | None = None,
    pin_size: tuple[int, int] | None = None,
    strip_seed: bool = True,
    on_event: Callable[[str], None] | None = None,
) -> LoopResult:
    """One blind Continuum tool call with no LLM in the loop.

    The benchmark control: the prompt goes to the generator verbatim and
    whatever comes back is accepted. Deliberately skips the agent entirely —
    routing the baseline through the loop (even restricted to generate_image)
    would let the model rewrite the prompt, which is itself part of the uplift
    being measured. Needs no LLM endpoint and no `openai` install.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    emit = on_event or (lambda _message: None)
    registry = ImageRegistry()
    transcript: list[dict[str, Any]] = []
    image_models: dict[str, str] = {}
    output_image_id: str | None = None
    kept_id: str | None = None
    kept_reason: str = ""
    error: str | None = None
    try:
        async with streamable_http_client(mcp_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                args: dict[str, Any] = {"prompt": task_prompt}
                if image_model:
                    args["model"] = image_model
                for data in input_images or []:
                    args["image"] = registry.register(data)  # edit-style baselines
                    break
                emit(f"direct {tool_name}({_summarize_tool_args(args)})")
                transcript.append({"role": "harness", "tool": tool_name, "arguments": dict(args)})
                call_args = prepare_call_args(args, registry, pin_size=pin_size, strip_seed=strip_seed)
                result = await session.call_tool(tool_name, call_args)
                image_bytes = extract_image_bytes(result)
                if image_bytes is not None:
                    output_image_id = registry.register(image_bytes)
                    image_models[output_image_id] = image_model or "unspecified"
                    emit(f"  ↳ produced {output_image_id}")
                    transcript.append(
                        {"role": "tool", "name": tool_name, "content": f"Produced image {output_image_id}."}
                    )
                else:
                    text = extract_result_text(result) or "no image returned"
                    emit(f"  ↳ {_truncate(text, 160)}")
                    transcript.append({"role": "tool", "name": tool_name, "content": text})
    except Exception as exc:
        error = _format_exc(exc)
        emit(f"error: {error}")

    output_image = registry.get(output_image_id) if output_image_id else None
    if error is not None:
        status, stop_reason = "failed", "error"
    else:
        status = "ok" if output_image is not None else "no-output"
        stop_reason = "direct"
    return LoopResult(
        status=status,
        output_image=output_image,
        output_image_id=output_image_id,
        steps=1,
        transcript=transcript,
        error=error,
        images=registry.as_dict(),
        stop_reason=stop_reason,
        image_models=image_models,
        kept_id=kept_id,
        kept_reason=kept_reason,
    )


async def run_agent_loop(
    *,
    task_prompt: str,
    system_prompt: str | None,
    input_images: list[bytes],
    mcp_url: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int = 20,
    max_image_side: int = 768,
    pin_size: tuple[int, int] | None = None,
    strip_seed: bool = True,
    include_server_instructions: bool = True,
    on_event: Callable[[str], None] | None = None,
    max_retries: int = 6,
    request_delay: float = 0.0,
    allow_tools: set[str] | None = None,
    block_tools: set[str] | None = None,
    fail_on_truncation: bool = False,
    max_inline_images: int = -1,
    reasoning_effort: str | None = None,
    image_model: str | None = None,
) -> LoopResult:
    """Run one agent episode: LLM <-> Continuum MCP tools, harness-owned ids.

    on_event, if given, receives one-line progress strings (step timing +
    per-call token usage, each tool call and its result) for streaming to a log.
    max_retries is the SDK's per-call retry budget (absorbs transient 5xx/429
    mid-loop); request_delay is a min gap before each LLM call (eases a shared
    or rate-limited endpoint).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from openai import AsyncOpenAI

    emit = on_event or (lambda _message: None)
    registry = ImageRegistry()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)

    output_image_id: str | None = None
    kept_id: str | None = None
    kept_reason: str = ""
    steps = 0
    total_prompt_tokens = 0
    messages: list[dict[str, Any]] = []
    shown_ids: list[str] = []  # image ids in the order they appear in messages (for redact_transcript)
    image_models: dict[str, str] = {}
    error: str | None = None
    returned_control = False  # True once the model ends its turn without a tool call
    try:
        async with streamable_http_client(mcp_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                init = await session.initialize()
                server_instructions = (
                    getattr(init, "instructions", None) if include_server_instructions else None
                )
                listed = (await session.list_tools()).tools
                allowed_names = _select_tool_names([t.name for t in listed], allow_tools, block_tools)
                if allow_tools:
                    missing = allow_tools - {t.name for t in listed}
                    if missing:
                        emit(f"warning: --tools not offered by the server: {sorted(missing)}")
                tools = [
                    adapt_tool_schema(t.name, t.description, t.inputSchema)
                    for t in listed
                    if t.name in allowed_names
                ]
                # Always offered, and not subject to --tools/--exclude-tools:
                # it is the loop's own bookkeeping, not a capability of the
                # server that a run might legitimately withhold.
                tools.append(MARK_BEST_TOOL)
                allowed_names = allowed_names | {MARK_BEST_TOOL_NAME}
                emit(
                    f"connected — {len(tools)}/{len(listed)} MCP tools enabled, "
                    f"{len(input_images)} input image(s)"
                )

                # Server instructions (operating manual) first, run policy on top.
                system = _system_message(server_instructions, system_prompt)
                if system:
                    messages.append({"role": "system", "content": system})
                user_content: list[dict[str, Any]] = [{"type": "text", "text": task_prompt}]
                for data in input_images:
                    img_id = registry.register(data)
                    user_content.append({"type": "text", "text": f"Input image {img_id}:"})
                    user_content.append(image_content_part(data, max_image_side))
                    shown_ids.append(img_id)
                messages.append({"role": "user", "content": user_content})
                for steps in range(1, max_steps + 1):
                    if max_inline_images >= 0:
                        dropped = cap_inline_images(messages, max_inline_images)
                        if dropped:
                            emit(f"  (elided {dropped} older image(s) to bound request size)")
                    if request_delay > 0:
                        await asyncio.sleep(request_delay)
                    t0 = time.monotonic()
                    # reasoning_effort is only sent when asked for: it is not
                    # universal, and an endpoint that does not know it may
                    # reject the request rather than ignore the field.
                    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
                    response = await client.chat.completions.create(
                        model=model, messages=messages, tools=tools, tool_choice="auto", **extra
                    )
                    dt = time.monotonic() - t0
                    usage = getattr(response, "usage", None)
                    prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
                    completion_tok = getattr(usage, "completion_tokens", None) if usage else None
                    if prompt_tok:
                        total_prompt_tokens += prompt_tok
                    tok = (
                        f"  {prompt_tok}→{completion_tok or 0} tok  (Σ prompt {total_prompt_tokens})"
                        if prompt_tok
                        else ""
                    )
                    emit(f"step {steps}/{max_steps}  {dt:.1f}s{tok}")
                    message = response.choices[0].message
                    if fail_on_truncation and getattr(response.choices[0], "finish_reason", None) == "length":
                        messages.append(_assistant_message(message))
                        raise RuntimeError("agent response reached its token limit; refusing to count it as completion")
                    if not message.tool_calls:
                        # The server may have handed us a tool call it did not
                        # parse -- see salvage_xml_tool_calls. Reasoning text is
                        # where it lands, and the SDK does not model that field.
                        raw = getattr(message, "reasoning_content", None) or ""
                        recovered = salvage_xml_tool_calls(
                            raw or (message.content or ""), tools)
                        if recovered:
                            emit(f"  recovered {len(recovered)} unparsed tool call(s) "
                                 f"from the model's text")
                            message.tool_calls = recovered
                            # Drop the XML now it is a real tool call. History
                            # carries reasoning_content *and* tool_calls, and
                            # Qwen's template renders both -- leaving it would
                            # show the model its own call twice every turn.
                            if raw:
                                try:
                                    message.reasoning_content = _XML_TOOL_CALL.sub("", raw).strip()
                                except (AttributeError, ValueError):
                                    pass
                    messages.append(_assistant_message(message))
                    if message.content and message.content.strip():
                        emit(f"  says: {_truncate(message.content, 200)}")
                    if not message.tool_calls:
                        emit("  finished (no tool call)")
                        returned_control = True
                        break
                    for call in message.tool_calls:
                        if call.function.name not in allowed_names:
                            emit(f"  ✗ blocked disallowed tool {call.function.name}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": f"Tool {call.function.name!r} is not available.",
                                }
                            )
                            continue
                        args = json.loads(call.function.arguments or "{}")
                        emit(f"  → {call.function.name}({_summarize_tool_args(args)})")
                        if call.function.name == MARK_BEST_TOOL_NAME:
                            wanted = str(args.get("image") or "")
                            if wanted == BEST_ALIAS and kept_id:
                                wanted = kept_id
                            if wanted not in registry:
                                text = f"No image {wanted!r} to mark. Mark an id that exists, e.g. the last one produced."
                            else:
                                output_image_id = wanted
                                kept_id = wanted
                                kept_reason = str(args.get("reason") or "")
                                text = f"Marked {wanted} as best. Pass 'best' to any image parameter to continue from it."
                            emit(f"    ↳ {text}")
                            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
                            continue
                        try:
                            call_args = prepare_call_args(
                                args,
                                registry,
                                pin_size=pin_size,
                                strip_seed=strip_seed,
                                best_id=kept_id,
                                pin_model=image_model,
                                tool_name=call.function.name,
                            )
                        except UnknownImageId as unknown:
                            # The model's mistake, and recoverable: tell it what
                            # exists and let it try again on the next turn.
                            emit(f"    ↳ {unknown.message()}")
                            messages.append({"role": "tool", "tool_call_id": call.id,
                                             "content": unknown.message()})
                            continue
                        result = await session.call_tool(call.function.name, call_args)
                        image_bytes = extract_image_bytes(result)
                        if image_bytes is not None:
                            new_id = registry.register(image_bytes)
                            if call.function.name in _OUTPUT_TOOLS:
                                # Only track the latest automatically while the
                                # model has not spoken. Once it has kept
                                # something, that choice stands until it keeps
                                # again — otherwise the next edit silently
                                # overrides the judgement we asked it for.
                                if kept_id is None:
                                    output_image_id = new_id
                                # call_args, not args: what was sent, not what the
                                # model asked for. They differ whenever it omitted
                                # the model and the pin supplied one.
                                image_models[new_id] = str(call_args.get("model") or "unspecified")
                            else:
                                # Derived view (crop/mask): inherit provenance from
                                # its source rather than claiming a model made it.
                                source = args.get("image")
                                image_models[new_id] = image_models.get(str(source), "derived")
                            emit(f"    ↳ produced {new_id}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": f"Produced image {new_id}.",
                                }
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Result {new_id}:"},
                                        image_content_part(image_bytes, max_image_side),
                                    ],
                                }
                            )
                            shown_ids.append(new_id)
                        else:
                            text = extract_result_text(result) or "ok"
                            emit(f"    ↳ {_truncate(text, 160)}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": text,
                                }
                            )
    except Exception as exc:
        # Capture the real cause + partial transcript instead of raising, so a
        # failed run still leaves a transcript.json and a legible error.
        error = _format_exc(exc)
        emit(f"error: {error}")
    finally:
        # Close the client's httpx pool inside this event loop. Otherwise its
        # connections are finalized after asyncio.run() closes the loop, spewing
        # harmless "RuntimeError: Event loop is closed" tracebacks — one per run
        # across a batch.
        await client.close()

    output_image = registry.get(output_image_id) if output_image_id else None
    if error is not None:
        status, stop_reason = "failed", "error"
    else:
        # The model returned control (accepting its output) or the harness cut it off.
        stop_reason = "completed" if returned_control else "max_steps"
        status = "ok" if output_image is not None else "no-output"
    return LoopResult(
        status=status,
        output_image=output_image,
        output_image_id=output_image_id,
        steps=steps,
        transcript=redact_transcript(messages, shown_ids),
        error=error,
        images=registry.as_dict(),
        stop_reason=stop_reason,
        image_models=image_models,
        kept_id=kept_id,
        kept_reason=kept_reason,
    )
