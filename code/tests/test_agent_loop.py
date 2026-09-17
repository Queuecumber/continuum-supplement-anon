import base64
import json
import types
from io import BytesIO

import pytest
from PIL import Image

from continuum.harnesses import agent_loop


def _png_bytes(size=(32, 32), color="red") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_image_registry_assigns_incrementing_ids() -> None:
    reg = agent_loop.ImageRegistry()
    a = reg.register(b"aaa")
    b = reg.register(b"bbb")
    assert (a, b) == ("img_0", "img_1")
    assert reg.get(a) == b"aaa"
    assert "img_1" in reg and "img_9" not in reg


def test_adapt_tool_schema_rewrites_image_params_to_ids() -> None:
    fn = agent_loop.adapt_tool_schema(
        "edit_image",
        "Edit an image.",
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "image": {"$ref": "#/$defs/ImageContent"},
                "references": {"type": "array", "items": {"$ref": "#/$defs/ImageContent"}},
                "steps": {"type": "integer"},
            },
            "$defs": {"ImageContent": {"type": "object"}},
        },
    )
    props = fn["function"]["parameters"]["properties"]
    assert fn["function"]["name"] == "edit_image"
    assert props["image"] == {"type": "string", "description": props["image"]["description"]}
    assert props["image"]["type"] == "string"
    assert props["references"]["type"] == "array" and props["references"]["items"] == {"type": "string"}
    assert props["prompt"]["type"] == "string" and props["steps"]["type"] == "integer"
    assert "$defs" not in fn["function"]["parameters"]  # image defs no longer referenced


def test_adapt_tool_schema_keeps_defs_that_are_still_referenced() -> None:
    # Media params become plain strings so their defs go unused, but `loras`
    # keeps a $ref to LoraSpec. Dropping $defs anyway leaves a dangling
    # reference: an invalid schema that lenient endpoints accept silently and
    # strict ones reject with 400 invalid_argument.
    fn = agent_loop.adapt_tool_schema(
        "generate_image",
        "Generate.",
        {
            "type": "object",
            "properties": {
                "image": {"$ref": "#/$defs/ImageContent"},
                "loras": {
                    "anyOf": [{"items": {"$ref": "#/$defs/LoraSpec"}, "type": "array"}, {"type": "null"}]
                },
            },
            "$defs": {"ImageContent": {"type": "object"}, "LoraSpec": {"type": "object"}},
        },
    )
    params = fn["function"]["parameters"]
    assert "$defs" in params, "LoraSpec is still referenced; its target must survive"
    assert "LoraSpec" in params["$defs"]
    assert params["properties"]["image"]["type"] == "string"


def test_adapt_tool_schema_drops_defs_when_nothing_references_them() -> None:
    fn = agent_loop.adapt_tool_schema(
        "edit_image",
        "Edit.",
        {
            "type": "object",
            "properties": {"image": {"$ref": "#/$defs/ImageContent"}, "prompt": {"type": "string"}},
            "$defs": {"ImageContent": {"type": "object"}},
        },
    )
    assert "$defs" not in fn["function"]["parameters"], "unreferenced defs are noise for the model"


def test_prepare_call_args_swaps_ids_clamps_size_strips_seed() -> None:
    reg = agent_loop.ImageRegistry()
    img_id = reg.register(b"PNGBYTES")
    ref_id = reg.register(b"REFBYTES")

    call = agent_loop.prepare_call_args(
        {"prompt": "x", "image": img_id, "references": [ref_id], "seed": 7, "width": 1000, "height": 1000},
        reg,
        pin_size=(64, 48),
        strip_seed=True,
    )

    assert call["image"] == {
        "type": "image",
        "data": base64.b64encode(b"PNGBYTES").decode(),
        "mimeType": "image/png",
    }
    assert call["references"][0]["data"] == base64.b64encode(b"REFBYTES").decode()
    assert "seed" not in call
    assert (call["width"], call["height"]) == (64, 48)
    assert call["prompt"] == "x"


def test_image_content_part_downscales_large_image() -> None:
    part = agent_loop.image_content_part(_png_bytes(size=(2048, 1024)), max_side=256)
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")  # the model's view is JPEG
    decoded = Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(decoded.size) == 256  # long edge clamped


def test_system_message_composes_server_instructions_then_policy() -> None:
    # Server instructions (operating manual) first, run policy on top.
    assert agent_loop._system_message("MANUAL", "POLICY") == "MANUAL\n\nPOLICY"
    assert agent_loop._system_message("MANUAL", None) == "MANUAL"
    assert agent_loop._system_message(None, "POLICY") == "POLICY"
    assert agent_loop._system_message(None, None) is None
    assert agent_loop._system_message("  ", "") is None


def test_select_tool_names_allow_then_block() -> None:
    names = ["generate_image", "edit_image", "inspect_region", "upscale_image"]
    assert agent_loop._select_tool_names(names, None, None) == set(names)  # default: all
    assert agent_loop._select_tool_names(names, {"generate_image", "edit_image"}, None) == {
        "generate_image",
        "edit_image",
    }
    assert agent_loop._select_tool_names(names, None, {"inspect_region", "upscale_image"}) == {
        "generate_image",
        "edit_image",
    }
    # block wins over allow
    assert agent_loop._select_tool_names(names, {"generate_image", "upscale_image"}, {"upscale_image"}) == {
        "generate_image"
    }


def test_summarize_tool_args_is_compact_and_truncates() -> None:
    summary = agent_loop._summarize_tool_args(
        {"prompt": "x" * 200, "image": "img_0", "references": ["img_1", "img_2"], "steps": 28}
    )
    assert "image='img_0'" in summary  # image ids stay readable
    assert "references=[2 ids]" in summary  # lists collapse to a count
    assert "steps=28" in summary
    assert "…" in summary and len(summary) < 160  # long prompt truncated


def test_only_generative_tools_are_candidate_outputs() -> None:
    # Agents finish by zooming in to check their work, so the last image a run
    # produces is usually an inspect_region crop. Treating that as the output
    # saved a thumbnail of a detail instead of the generated image — 81% of one
    # benchmark arm came out as crops.
    assert "generate_image" in agent_loop._OUTPUT_TOOLS
    assert "edit_image" in agent_loop._OUTPUT_TOOLS
    assert "inpaint_image" in agent_loop._OUTPUT_TOOLS
    for viewer in ("inspect_region", "segment", "segment_points", "invert_mask"):
        assert viewer not in agent_loop._OUTPUT_TOOLS, f"{viewer} must not become the output"
    # upscale changes dimensions, which breaks the pinned size a run is compared at.
    assert "upscale_image" not in agent_loop._OUTPUT_TOOLS


def _msgs_with_images(n: int) -> list:
    out = []
    for i in range(n):
        out.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Result img_{i}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{i}"}},
                ],
            }
        )
    return out


def _inline_count(messages) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    )


def test_cap_inline_images_keeps_only_the_most_recent() -> None:
    # Each request re-sends the history, so payload grows with step count until
    # the endpoint refuses it (seen as HTTP 413 at step 15).
    messages = _msgs_with_images(10)
    dropped = agent_loop.cap_inline_images(messages, keep=3)
    assert dropped == 7
    assert _inline_count(messages) == 3
    # The surviving ones are the newest, which is what the agent is reasoning about.
    kept = [
        p["image_url"]["url"]
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    assert kept == [f"data:image/png;base64,IMG{i}" for i in (7, 8, 9)]


def test_cap_inline_images_leaves_a_placeholder_not_a_hole() -> None:
    messages = _msgs_with_images(3)
    agent_loop.cap_inline_images(messages, keep=1)
    elided = [p for m in messages for p in m["content"] if p.get("_elided")]
    assert len(elided) == 2
    assert all(p["type"] == "text" and "elided" in p["text"] for p in elided)


def test_cap_inline_images_is_idempotent_and_bounded() -> None:
    # Called before every step, so re-capping must not keep re-eliding.
    messages = _msgs_with_images(6)
    first = agent_loop.cap_inline_images(messages, keep=2)
    second = agent_loop.cap_inline_images(messages, keep=2)
    assert first == 4 and second == 0
    assert _inline_count(messages) == 2


def test_cap_inline_images_disabled_and_under_limit() -> None:
    messages = _msgs_with_images(5)
    assert agent_loop.cap_inline_images(messages, keep=-1) == 0
    assert _inline_count(messages) == 5
    assert agent_loop.cap_inline_images(messages, keep=10) == 0
    assert _inline_count(messages) == 5


def test_redact_transcript_replaces_data_urls_with_ids_in_order() -> None:
    messages = [
        {"role": "system", "content": "policy"},  # string content untouched
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Input image img_0:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Result img_1:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ],
        },
    ]
    clean = agent_loop.redact_transcript(messages, ["img_0", "img_1"])

    assert clean[0]["content"] == "policy"
    assert clean[1]["content"][1] == {"type": "image_url", "image_url": {"url": "img_0"}}
    assert clean[2]["content"][1] == {"type": "image_url", "image_url": {"url": "img_1"}}
    # the big data-urls are gone from the copy...
    assert "data:image" not in json.dumps(clean)
    # ...and the original is not mutated
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image")


class _FakeGroup(Exception):
    """Mimics an ExceptionGroup's `.exceptions` without needing the 3.11 builtin."""

    def __init__(self, message, exceptions):
        super().__init__(message)
        self.exceptions = exceptions


def test_format_exc_unwraps_exception_groups() -> None:
    assert agent_loop._format_exc(ValueError("boom")) == "ValueError: boom"
    # anyio/MCP bury the real cause in a task-group ExceptionGroup — surface the leaf.
    group = _FakeGroup("unhandled errors in a TaskGroup", [RuntimeError("429 Too Many Requests")])
    assert agent_loop._format_exc(group) == "RuntimeError: 429 Too Many Requests"
    nested = _FakeGroup("outer", [_FakeGroup("inner", [KeyError("x")])])
    assert agent_loop._format_exc(nested) == "KeyError: 'x'"


def test_prepare_call_args_image_is_a_valid_mcp_imagecontent() -> None:
    # The dict prepare_call_args sends for an image param must be a valid
    # ImageContent — that is what standard-mode edit_image(image: ImageContent)
    # receives via call_tool. Pins the serialization contract vs the real type.
    from mcp.types import ImageContent

    reg = agent_loop.ImageRegistry()
    img_id = reg.register(b"PNGDATA")
    call = agent_loop.prepare_call_args({"image": img_id, "prompt": "x"}, reg)
    ic = ImageContent(**call["image"])
    assert ic.type == "image"
    assert base64.b64decode(ic.data) == b"PNGDATA"


def test_extract_image_bytes_reads_a_real_mcpimage_return() -> None:
    # continuum returns MCPImage(data=png, format="png"); FastMCP turns it into
    # an ImageContent in the call_tool result. Confirm extract_image_bytes reads
    # that exact shape (pins the return contract vs the real FastMCP type).
    from mcp.server.fastmcp import Image as MCPImage

    png = _png_bytes()
    content = MCPImage(data=png, format="png").to_image_content()
    result = types.SimpleNamespace(content=[content])
    assert agent_loop.extract_image_bytes(result) == png


def test_extract_image_bytes_finds_image_content() -> None:
    result = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(type="text", text="done", data=None),
            types.SimpleNamespace(type="image", text=None, data=base64.b64encode(b"IMG").decode()),
        ]
    )
    assert agent_loop.extract_image_bytes(result) == b"IMG"
    assert agent_loop.extract_result_text(result) == "done"

    text_only = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="hi", data=None)])
    assert agent_loop.extract_image_bytes(text_only) is None


def test_reasoning_is_carried_on_assistant_turns() -> None:
    """A reasoning model's chain of thought survives into the next request.

    Reasoning models return it in `reasoning_content`, not `content` — so a turn
    that only makes a tool call otherwise looks empty. The gateway accepts it
    back on prior assistant turns, which lets the model continue a line of
    reasoning across a tool call rather than re-deriving it, and it is what
    makes the saved transcript readable.
    """

    class Msg:
        content = ""
        reasoning_content = "The banner text is garbled; I should zoom in."
        tool_calls = None

    out = agent_loop._assistant_message(Msg())
    assert out["reasoning_content"] == "The banner text is garbled; I should zoom in."
    assert out["content"] == ""


def test_absent_or_blank_reasoning_adds_no_key() -> None:
    """Models that do not reason must not gain an empty field.

    Some endpoints reject unknown keys, and an always-present empty string would
    put one on every assistant turn for every non-reasoning model.
    """

    class Plain:
        content = "done"
        tool_calls = None

    class Blank(Plain):
        reasoning_content = "   "

    assert "reasoning_content" not in agent_loop._assistant_message(Plain())
    assert "reasoning_content" not in agent_loop._assistant_message(Blank())


def test_images_are_kept_by_default() -> None:
    """Nothing is elided unless a cap is asked for.

    The agent reasons about images it produced several steps ago — whether an
    edit actually landed, whether an earlier version was better — and it does
    that far better seeing them than holding an id. The cap exists for the
    endpoint's sake (HTTP 413 once), not the model's, so it is opt-in.
    """
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "b"}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "c"}}]},
    ]
    assert agent_loop.cap_inline_images(messages, -1) == 0
    assert all(m["content"][0]["type"] == "image_url" for m in messages)


def test_mark_best_is_always_offered() -> None:
    """mark_best is the loop's own bookkeeping, not a server capability.

    It is exempt from --tools/--exclude-tools: a run may legitimately withhold
    outpaint_image, but withholding the model's only way to say which result it
    wants is never what the caller meant.
    """
    assert agent_loop.MARK_BEST_TOOL["function"]["name"] == "mark_best"
    params = agent_loop.MARK_BEST_TOOL["function"]["parameters"]
    assert params["required"] == ["image"]
    assert "reason" in params["properties"]


def test_mark_best_description_tells_the_model_what_it_is_for() -> None:
    """The description is the only place the model learns the consequence.

    It has to say four things: marking does not end the run, only one image is
    marked at a time, an earlier image may be marked if a later edit was worse,
    and unmarked work is lost if the run is cut short. Without the last one
    there is no reason to call it early.
    """
    text = agent_loop.MARK_BEST_TOOL["function"]["description"]
    assert "does not end the run" in text
    assert "most recent call wins" in text
    assert "earlier" in text
    assert "cut short" in text


def test_best_alias_resolves_to_the_marked_image() -> None:
    """'best' stands in for the marked id wherever an image id is accepted.

    This is the point of the alias: after marking an *earlier* image, the newest
    id is the wrong one to carry around, and that is exactly when the model
    would otherwise have to remember which id it chose.
    """
    registry = agent_loop.ImageRegistry()
    first = registry.register(b"first")
    registry.register(b"second")

    call = agent_loop.prepare_call_args({"image": "best"}, registry, strip_seed=False, best_id=first)
    assert call["image"]["data"], "the alias resolved to real bytes"

    import base64

    assert base64.b64decode(call["image"]["data"]) == b"first"


def test_best_alias_without_a_mark_is_left_alone() -> None:
    """Nothing marked yet: 'best' is not silently pointed at some other image."""
    registry = agent_loop.ImageRegistry()
    registry.register(b"only")
    with pytest.raises(agent_loop.UnknownImageId) as caught:
        agent_loop.prepare_call_args({"image": "best"}, registry, strip_seed=False, best_id=None)
    assert caught.value.img_id == "best"
    assert "img_0" in caught.value.message()


def test_model_view_is_jpeg_not_png() -> None:
    """The model's view is JPEG; the working bytes stay PNG.

    A conversation re-sends every image on every step, and PNG-encoding frames
    that have already been downscaled to 768px is what pushed requests past the
    upstream proxy's size limit (HTTP 413). The registry still holds full-
    resolution PNG, so nothing an editing tool receives is affected.
    """
    img = Image.new("RGB", (1024, 1024), "red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    part = agent_loop.image_content_part(buf.getvalue())
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_model_view_is_much_smaller_than_png() -> None:
    """The saving is the point, so it is worth asserting rather than assuming."""
    img = Image.new("RGB", (1024, 1024))
    # structured noise: a flat fill compresses to nothing in either format and
    # would make this test pass for the wrong reason.
    px = img.load()
    for y in range(0, 1024, 2):
        for x in range(0, 1024, 2):
            px[x, y] = ((x * 7) % 256, (y * 13) % 256, ((x + y) * 3) % 256)
    buf = BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    jpeg = agent_loop.image_content_part(raw)["image_url"]["url"]
    png_part = agent_loop.image_content_part(raw, quality=100)
    assert len(jpeg) < len(png_part["image_url"]["url"]), "q90 must beat q100"


def test_rgba_images_survive_the_jpeg_conversion() -> None:
    """JPEG has no alpha; a mask or a crop with transparency must not raise."""
    img = Image.new("RGBA", (256, 256), (255, 0, 0, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    part = agent_loop.image_content_part(buf.getvalue())
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
