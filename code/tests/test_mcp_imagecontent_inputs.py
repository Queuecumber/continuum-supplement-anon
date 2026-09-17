import asyncio
import base64
import io

import pytest
from PIL import Image

from continuum.mcp_server import _build_server


def _image_content_from_pil(img: Image.Image) -> dict[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mimeType": "image/png",
    }


def test_non_compat_invert_mask_accepts_image_content() -> None:
    server = _build_server(compat_mode=False)
    tool = server._tool_manager._tools["invert_mask"]

    schema = tool.parameters
    assert schema["properties"]["mask"]["$ref"] == "#/$defs/ImageContent"

    async def run_tool():
        return await tool.run({"mask": _image_content_from_pil(Image.new("L", (1, 1), 255))})

    out = asyncio.run(run_tool())
    content = out.to_image_content()
    raw = base64.b64decode(content.data)
    pixel = Image.open(io.BytesIO(raw)).convert("L").getpixel((0, 0))

    assert content.mimeType == "image/png"
    assert pixel == 0


def test_tool_errors_are_logged_server_side(capsys) -> None:
    server = _build_server(compat_mode=False)
    tool = server._tool_manager._tools["invert_mask"]

    async def run_tool():
        return await tool.run(
            {
                "mask": {
                    "type": "image",
                    "data": base64.b64encode(b"not an image").decode("ascii"),
                    "mimeType": "image/png",
                }
            }
        )

    with pytest.raises(Exception):
        asyncio.run(run_tool())

    out = capsys.readouterr().out
    assert "[mcp] invert_mask: ERROR" in out
    assert "Traceback" in out
