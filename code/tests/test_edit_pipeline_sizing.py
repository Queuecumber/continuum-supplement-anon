import types

import pytest
from PIL import Image

from continuum.backends.base import EditParams
from continuum.backends.qwen import FireRedEditor
from continuum.imaging import fit_image, resize_if_aspect_matches


def test_fit_image_preserves_portrait_aspect_inside_canvas() -> None:
    img = Image.new("RGB", (600, 1000), "white")

    fitted = fit_image(img, 616, 1024)

    assert fitted.size == (608, 992)


def test_resize_if_aspect_matches_rejects_square_to_portrait_stretch() -> None:
    img = Image.new("RGB", (1024, 1024), "white")

    with pytest.raises(RuntimeError, match="refusing to stretch"):
        resize_if_aspect_matches(
            img,
            (608, 992),
            context="test pipeline",
        )


def test_resize_if_aspect_matches_allows_same_aspect_resize() -> None:
    img = Image.new("RGB", (304, 496), "white")

    resized = resize_if_aspect_matches(
        img,
        (608, 992),
        context="test pipeline",
    )

    assert resized.size == (608, 992)


def test_firered_fit_inputs_preserves_portrait_call_size() -> None:
    editor = FireRedEditor.__new__(FireRedEditor)
    imgs, width, height = editor._fit_inputs(
        [Image.new("RGB", (600, 1000), "white")],
        EditParams(prompt="edit", width=616, height=1024),
    )

    assert (width, height) == (616, 1024)
    assert imgs[0].size == (608, 992)


def test_firered_edit_calls_pipeline_with_fitted_portrait_size() -> None:
    class DummyPipe:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            out = Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")
            return types.SimpleNamespace(images=[out])

    pipe = DummyPipe()
    editor = FireRedEditor()
    editor._pipe = pipe

    out, meta = editor.edit(
        Image.new("RGB", (600, 1000), "white"),
        EditParams(prompt="edit", width=616, height=1024),
    )

    assert pipe.kwargs["width"] == 608
    assert pipe.kwargs["height"] == 992
    assert out.size == (608, 992)
    assert meta["requested_size"] == [616, 1024]
    assert meta["call_size"] == [608, 992]


def test_firered_edit_passes_inferred_size_when_size_not_explicit() -> None:
    class DummyPipe:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            out = Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")
            return types.SimpleNamespace(images=[out])

    pipe = DummyPipe()
    editor = FireRedEditor()
    editor._pipe = pipe

    out, meta = editor.edit(
        Image.new("RGB", (600, 1000), "white"),
        EditParams(prompt="edit"),
    )

    assert pipe.kwargs["width"] == 608
    assert pipe.kwargs["height"] == 992
    assert out.size == (608, 992)
    assert meta["requested_size"] is None
    assert meta["call_size"] == [608, 992]


def test_firered_edit_rejects_square_output_instead_of_stretching() -> None:
    class DummyPipe:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[Image.new("RGB", (1024, 1024), "white")])

    editor = FireRedEditor()
    editor._pipe = DummyPipe()

    with pytest.raises(RuntimeError, match="FireRed edit pipeline.*refusing to stretch"):
        editor.edit(
            Image.new("RGB", (600, 1000), "white"),
            EditParams(prompt="edit"),
        )
