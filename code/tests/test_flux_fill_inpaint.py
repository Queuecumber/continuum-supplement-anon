import types

import pytest
from PIL import Image

from continuum.backends.base import EditParams
from continuum.backends.flux_fill import FluxFillInpainter
from continuum.imaging import dilate_mask, match_color_statistics


class _DummyFillPipe:
    """Stand-in for FluxFillPipeline: echoes a blank image at the requested size."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        out = Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")
        return types.SimpleNamespace(images=[out])


def test_flux_fill_snaps_canvas_and_preserves_native_size() -> None:
    editor = FluxFillInpainter()
    pipe = _DummyFillPipe()
    editor._pipe = pipe

    original = Image.new("RGB", (600, 1000), "black")
    mask = Image.new("L", (600, 1000), 0)
    # Paint a white (fill) block in the middle.
    for x in range(200, 400):
        for y in range(300, 700):
            mask.putpixel((x, y), 255)

    result, meta = editor.inpaint(
        original,
        EditParams(prompt="a red silk shirt", guidance_scale=30.0, seed=None),
        mask,
        on_step=None,
    )

    # Generation canvas is snapped to multiples of 16 for the VAE.
    assert pipe.kwargs["width"] % 16 == 0
    assert pipe.kwargs["height"] % 16 == 0
    # A mask image is actually passed to the fill pipeline.
    assert pipe.kwargs["mask_image"] is not None
    # The editor passes the requested guidance straight through.
    assert pipe.kwargs["guidance_scale"] == 30.0
    # Output is composited back to the native image size.
    assert result.size == original.size
    assert meta["mode"] == "flux1-fill"


def test_flux_fill_uses_native_resolution_when_within_budget() -> None:
    editor = FluxFillInpainter()
    pipe = _DummyFillPipe()
    editor._pipe = pipe

    original = Image.new("RGB", (600, 512), "black")
    mask = Image.new("L", (600, 512), 0)
    mask.paste(255, (100, 100, 200, 200))

    editor.inpaint(original, EditParams(prompt="x", seed=None), mask)

    # ≤1024 long edge: generate at snapped-native size, not upscaled to 1024.
    assert pipe.kwargs["width"] == 608
    assert pipe.kwargs["height"] == 512


def test_match_color_statistics_removes_global_drift() -> None:
    import numpy as np

    _match = match_color_statistics

    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, size=(64, 64, 3)).astype(np.uint8)
    reference = Image.fromarray(base)
    # Generator output = same content with a +18 global tone drift.
    edited = Image.fromarray(np.clip(base.astype(np.int16) + 18, 0, 255).astype(np.uint8))
    fill_mask = Image.new("L", (64, 64), 0)
    fill_mask.paste(255, (24, 24, 40, 40))

    corrected = np.asarray(_match(edited, reference, fill_mask), dtype=np.float32)

    # The drift is removed everywhere, including inside the fill region.
    diff = corrected - base.astype(np.float32)
    assert abs(float(diff.mean())) < 2.0


def test_dilate_mask_grows_fill_region() -> None:

    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (30, 30, 34, 34))

    grown = dilate_mask(mask, 5)

    assert grown.getpixel((32, 32)) == 255  # original fill survives
    assert grown.getpixel((26, 32)) == 255  # grown ~5px outward
    assert grown.getpixel((20, 32)) == 0  # but not unbounded
    # Zero/negative are no-ops.
    assert dilate_mask(mask, 0) is mask


def test_match_color_statistics_skips_degenerate_estimates() -> None:
    _match = match_color_statistics

    # Uniform keep region → no reliable estimate → identity.
    reference = Image.new("RGB", (64, 64), (0, 0, 0))
    edited = Image.new("RGB", (64, 64), (255, 255, 255))
    fill_mask = Image.new("L", (64, 64), 0)

    corrected = _match(edited, reference, fill_mask)
    assert corrected.getpixel((32, 32)) == (255, 255, 255)


class _DummyLoraFillPipe(_DummyFillPipe):
    def __init__(self):
        super().__init__()
        self.lora_loads = []
        self.adapters = None

    def load_lora_weights(self, repo, **kw):
        self.lora_loads.append((repo, kw.get("adapter_name")))

    def unload_lora_weights(self):
        pass

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights or []))


def test_flux_fill_applies_loras() -> None:
    editor = FluxFillInpainter()
    pipe = _DummyLoraFillPipe()
    editor._pipe = pipe

    editor.inpaint(
        Image.new("RGB", (256, 256), "black"),
        EditParams(prompt="x", seed=None),
        Image.new("L", (256, 256), 0),
        loras=[{"source": "someuser/style-lora", "scale": 0.7}],
    )

    assert pipe.lora_loads == [("someuser/style-lora", "lora_0")]
    assert pipe.adapters == (["lora_0"], [0.7])


def test_flux_fill_reports_lora_phases(monkeypatch) -> None:
    import continuum.backends.lora as lora_mod

    # Keep the test offline: the pre-download only exists to split the phases.
    monkeypatch.setattr(lora_mod, "_prefetch_lora", lambda *a, **k: None)

    editor = FluxFillInpainter()
    editor._pipe = _DummyLoraFillPipe()

    events: list[str] = []
    editor.inpaint(
        Image.new("RGB", (256, 256), "black"),
        EditParams(prompt="x", seed=None),
        Image.new("L", (256, 256), 0),
        loras=[{"source": "someuser/style-lora", "scale": 0.7}],
        load_progress=lambda message, current, total: events.append(message),
    )

    # The client sees a distinct download phase then an apply phase.
    assert any(m.startswith("downloading LoRA") and "someuser/style-lora" in m for m in events)
    assert any(m.startswith("applying LoRA") for m in events)
    assert events.index(next(m for m in events if m.startswith("downloading LoRA"))) < events.index(
        next(m for m in events if m.startswith("applying LoRA"))
    )


def test_flux_fill_rejects_mismatched_pipeline_output() -> None:
    editor = FluxFillInpainter()

    class _WrongSizePipe:
        def __call__(self, **kwargs):
            # Model ignores the request and returns a square — the old
            # silent-stretch/zoom bug class. Must raise, not resize.
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), "white")])

    editor._pipe = _WrongSizePipe()

    with pytest.raises(RuntimeError, match="FLUX.1 Fill pipeline.*refusing to stretch"):
        editor.inpaint(
            Image.new("RGB", (600, 1000), "black"),
            EditParams(prompt="x", seed=None),
            Image.new("L", (600, 1000), 0),
        )


def test_flux_fill_preserves_pixels_outside_mask() -> None:
    editor = FluxFillInpainter()
    # Fill pipe returns solid white; original is solid black.
    editor._pipe = _DummyFillPipe()

    original = Image.new("RGB", (256, 256), "black")
    mask = Image.new("L", (256, 256), 0)  # all-black mask => nothing to fill

    result, _meta = editor.inpaint(
        original,
        EditParams(prompt="unused", seed=None),
        mask,
        on_step=None,
        feather=0,
    )

    # With an empty mask and no feather, the outside (whole image) must be the
    # untouched original, not the white fill output.
    assert result.getpixel((10, 10)) == (0, 0, 0)
    assert result.getpixel((128, 128)) == (0, 0, 0)
