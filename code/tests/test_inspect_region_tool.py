import asyncio
import base64
import io

import pytest
from PIL import Image

from continuum import mcp_server


def _image_content_from_pil(img: Image.Image) -> dict[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mimeType": "image/png",
    }


def _pil_from_tool_image(out) -> Image.Image:
    content = out.to_image_content()
    return Image.open(io.BytesIO(base64.b64decode(content.data))).convert("RGB")


def test_inspect_region_schema_exposes_roi_union() -> None:
    tools = mcp_server._build_server(False)._tool_manager._tools
    inspect_tool = tools["inspect_region"]

    assert inspect_tool.parameters["properties"]["image"]["$ref"] == "#/$defs/ImageContent"
    roi_schema = inspect_tool.parameters["properties"]["roi"]
    assert len(roi_schema["anyOf"]) == 3
    # roi prose lives in the tool description (docstring Args), not a Field.
    desc = " ".join((inspect_tool.description or "").split())
    assert "the logo" in desc


def test_inspect_region_compat_schema_accepts_uri_image() -> None:
    tools = mcp_server._build_server(True)._tool_manager._tools
    inspect_tool = tools["inspect_region"]

    assert inspect_tool.parameters["properties"]["image"]["type"] == "string"


def test_coerce_roi_box_accepts_supported_rectangle_shapes() -> None:
    assert mcp_server._coerce_roi_box([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)
    assert mcp_server._coerce_roi_box("1, 2, 3, 4") == (1.0, 2.0, 3.0, 4.0)
    assert mcp_server._coerce_roi_box({"x": 1, "y": 2, "width": 3, "height": 4}) == (
        1.0,
        2.0,
        4.0,
        6.0,
    )
    assert mcp_server._coerce_roi_box("the left eye") is None


def test_crop_image_roi_clips_expands_and_scales_normalized_box() -> None:
    image = Image.new("RGB", (100, 50), "white")

    crop = mcp_server._crop_image_roi(
        image,
        (0.25, 0.2, 0.75, 0.6),
        normalized=True,
        padding=0.1,
    )

    assert crop.size == (60, 24)


def test_bbox_from_mask_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="did not match"):
        mcp_server._bbox_from_mask(Image.new("L", (4, 4), 0))


def test_inspect_region_crops_inline_image_content_rectangle() -> None:
    server = mcp_server._build_server(False)
    tool = server._tool_manager._tools["inspect_region"]
    img = Image.new("RGB", (4, 4), "black")
    img.putpixel((1, 1), (255, 0, 0))
    img.putpixel((2, 2), (0, 255, 0))

    async def run_tool():
        return await tool.run(
            {
                "image": _image_content_from_pil(img),
                "roi": [1, 1, 3, 3],
            }
        )

    crop = _pil_from_tool_image(asyncio.run(run_tool()))

    assert crop.size == (2, 2)
    assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert crop.getpixel((1, 1)) == (0, 255, 0)
