"""Byte-level media helpers: decode images."""

import io

from PIL.Image import Image as PILImage
from PIL.Image import open as _pil_open


def bytes_to_pil(data: bytes, mode: str = "RGB") -> PILImage:
    try:
        return _pil_open(io.BytesIO(data)).convert(mode)
    except Exception as e:
        head = data[:32]
        ascii_preview = head.decode("ascii", errors="replace")
        raise ValueError(
            f"PIL couldn't identify image bytes: {e}. "
            f"len={len(data)}, head_hex={head.hex()}, head_ascii={ascii_preview!r}"
        ) from e
