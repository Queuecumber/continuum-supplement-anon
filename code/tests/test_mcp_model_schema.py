from PIL import Image

from continuum import mcp_server


def _desc(tool) -> str:
    """Tool description with docstring line-wrapping collapsed to single spaces."""
    return " ".join((tool.description or "").split())


def test_image_model_parameters_expose_enums_and_descriptions() -> None:
    tools = mcp_server._build_server(False)._tool_manager._tools

    generate_model = tools["generate_image"].parameters["properties"]["model"]
    assert generate_model["enum"] == [
        "flux2-dev",
        "flux2-klein",
        "flux1-dev",
        "qwen-image",
    ]
    # The default is surfaced on the parameter so it shows on schema inspection.
    assert generate_model["default"] == "flux2-dev"
    # Prose is the docstring's job (single source): it lives in the tool description.
    assert "flux2-dev" in _desc(tools["generate_image"])
    assert "cheap with high prompt adherence" in _desc(tools["generate_image"])

    edit_model = tools["edit_image"].parameters["properties"]["model"]
    edit_enum = edit_model["anyOf"][0]["enum"]
    assert edit_enum == [
        "firered",
        "flux2-dev",
        "flux2-klein",
        "qwen-image",
    ]
    assert edit_model["default"] == "firered"
    assert "default firered" in _desc(tools["edit_image"])
    assert "does NOT support LoRAs" in _desc(tools["edit_image"])


def test_loras_are_a_structured_loraspec_type() -> None:
    tools = mcp_server._build_server(False)._tool_manager._tools
    params = tools["generate_image"].parameters

    # loras is list[LoraSpec] — the object shape is self-documenting in the schema.
    lora_spec = params["$defs"]["LoraSpec"]
    assert set(lora_spec["properties"]) == {"source", "scale", "weight_name"}
    assert lora_spec["properties"]["scale"]["default"] == 1.0
    assert lora_spec["required"] == ["source"]

    # edit_image no longer supports masked edits — masks live on inpaint_image.
    assert "mask" not in tools["edit_image"].parameters["properties"]
    assert "mask_method" not in tools["edit_image"].parameters["properties"]


def test_inpaint_image_is_opinionated() -> None:
    tools = mcp_server._build_server(False)._tool_manager._tools

    inpaint = tools["inpaint_image"].parameters["properties"]
    # Minimal surface: mask + prompt + generation basics + dilate.
    assert "mask" in inpaint
    assert "dilate" in inpaint
    assert set(inpaint) == {
        "image",
        "mask",
        "prompt",
        "steps",
        "guidance",
        "seed",
        "dilate",
        "loras",
    }
    # loras' FLUX.1-dev family is documented in the tool description.
    assert "FLUX.1-dev" in _desc(tools["inpaint_image"])


def test_outpaint_image_shares_the_inpaint_surface() -> None:
    tools = mcp_server._build_server(False)._tool_manager._tools

    outpaint = tools["outpaint_image"].parameters["properties"]
    for side in ("left", "right", "top", "bottom"):
        assert side in outpaint
    assert outpaint["overlap"]["default"] == 20
    # Minimal surface: padding + prompt + generation basics + overlap.
    assert set(outpaint) == {
        "image",
        "prompt",
        "left",
        "right",
        "top",
        "bottom",
        "overlap",
        "steps",
        "guidance",
        "seed",
        "loras",
    }
    assert "FLUX.1-dev" in _desc(tools["outpaint_image"])


def test_edit_dimensions_only_default_for_flux_family(monkeypatch) -> None:
    from continuum.core import ops

    img = Image.new("RGB", (1600, 900), "white")

    monkeypatch.setattr(ops, "_space_edit_dimensions", lambda _img: (1024, 576))

    assert ops.edit_dimensions_for_model("firered", img, None, None) == (None, None)
    assert ops.edit_dimensions_for_model("qwen-image-edit", img, None, None) == (None, None)
    assert ops.edit_dimensions_for_model("flux2", img, None, None) == (1024, 576)
    assert ops.edit_dimensions_for_model("flux2-klein", img, None, None) == (1024, 576)
