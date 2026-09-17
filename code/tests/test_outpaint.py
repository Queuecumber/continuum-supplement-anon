import types

import pytest
from PIL import Image

from continuum.backends.base import EditParams
from continuum.backends.flux_fill import FluxFillInpainter
from continuum.imaging import prepare_outpaint_canvas


def test_canvas_geometry_and_mask_layout() -> None:
    src = Image.new("RGB", (100, 80), "red")

    canvas, mask = prepare_outpaint_canvas(
        src,
        left=10,
        right=30,
        top=20,
        bottom=40,
        overlap=0,
    )

    assert canvas.size == (140, 140)
    assert mask.size == (140, 140)
    # Original pasted at (left, top).
    assert canvas.getpixel((10, 20)) == (255, 0, 0)
    assert canvas.getpixel((109, 99)) == (255, 0, 0)
    # Border is the neutral fill color.
    assert canvas.getpixel((0, 0)) == (127, 127, 127)
    assert canvas.getpixel((139, 139)) == (127, 127, 127)
    # Mask: black (keep) over the original, white (fill) over the border.
    assert mask.getpixel((10, 20)) == 0
    assert mask.getpixel((109, 99)) == 0
    assert mask.getpixel((0, 0)) == 255
    assert mask.getpixel((139, 139)) == 255
    assert mask.getpixel((9, 20)) == 255  # one pixel left of the paste box
    assert mask.getpixel((110, 99)) == 255  # one pixel right of the paste box


def test_single_side_padding() -> None:
    src = Image.new("RGB", (64, 64), "blue")

    canvas, mask = prepare_outpaint_canvas(
        src,
        left=0,
        right=32,
        top=0,
        bottom=0,
        overlap=0,
    )

    assert canvas.size == (96, 64)
    assert canvas.getpixel((0, 0)) == (0, 0, 255)
    assert mask.getpixel((0, 0)) == 0
    assert mask.getpixel((95, 0)) == 255


def test_overlap_extends_fill_into_padded_sides_only() -> None:
    src = Image.new("RGB", (100, 80), "red")

    _canvas, mask = prepare_outpaint_canvas(
        src,
        left=0,
        right=30,
        top=0,
        bottom=0,
        overlap=20,
    )

    # Right side padded: the fill region reaches 20px into the image.
    assert mask.getpixel((99, 40)) == 255  # inside the image, overlap strip
    assert mask.getpixel((80, 40)) == 255  # overlap boundary (first fill px)
    assert mask.getpixel((79, 40)) == 0  # keep region starts here
    # Unpadded sides keep their full edge.
    assert mask.getpixel((0, 40)) == 0
    assert mask.getpixel((40, 0)) == 0
    assert mask.getpixel((40, 79)) == 0


def test_overlap_is_clamped_for_small_images() -> None:
    src = Image.new("RGB", (16, 16), "red")

    _canvas, mask = prepare_outpaint_canvas(
        src,
        left=8,
        right=8,
        top=8,
        bottom=8,
        overlap=100,
    )

    # Overlap is clamped so a keep region survives at the centre.
    assert mask.getpixel((16, 16)) == 0  # canvas centre = image centre


def test_rejects_negative_and_zero_padding() -> None:
    src = Image.new("RGB", (64, 64))

    with pytest.raises(ValueError, match="non-negative"):
        prepare_outpaint_canvas(src, left=-1, right=0, top=0, bottom=0)
    with pytest.raises(ValueError, match="at least one side"):
        prepare_outpaint_canvas(src, left=0, right=0, top=0, bottom=0)


class _DummyFillPipe:
    def __call__(self, **kwargs):
        out = Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")
        return types.SimpleNamespace(images=[out])


def test_fill_outpaint_preserves_original_region() -> None:
    editor = FluxFillInpainter()
    editor._pipe = _DummyFillPipe()

    src = Image.new("RGB", (256, 256), "black")
    canvas, mask = prepare_outpaint_canvas(
        src,
        left=64,
        right=64,
        top=0,
        bottom=0,
        overlap=0,
    )

    result, meta = editor.inpaint(
        canvas,
        EditParams(prompt="a scene", seed=None),
        mask,
        feather=0,
    )

    # Output covers the full expanded canvas.
    assert result.size == canvas.size
    # Original region untouched (black), border filled from the model (white).
    assert result.getpixel((64 + 128, 128)) == (0, 0, 0)
    assert result.getpixel((10, 128)) == (255, 255, 255)
    assert result.getpixel((canvas.width - 10, 128)) == (255, 255, 255)
