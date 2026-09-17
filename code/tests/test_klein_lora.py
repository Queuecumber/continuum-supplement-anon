"""Klein text_to_image LoRA plumbing (shared lora helpers are covered in
test_lora_helpers.py)."""

import types

from PIL import Image

from continuum.backends.base import EditParams
from continuum.backends.klein import Flux2KleinEditor


class _DummyKleinPipe:
    def __init__(self):
        self.lora_loads = []
        self.adapters = None
        self.kwargs = None

    def load_lora_weights(self, repo, **kw):
        self.lora_loads.append((repo, kw.get("adapter_name")))

    def unload_lora_weights(self):
        pass

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights or []))

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        out = Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")
        return types.SimpleNamespace(images=[out])


def _make_editor():
    editor = Flux2KleinEditor.__new__(Flux2KleinEditor)
    editor._pipe = _DummyKleinPipe()
    editor.model_id = "klein-test"
    editor.device = "cpu"
    editor.dtype = "bfloat16"
    editor.hf_token = None
    return editor


def test_klein_text_to_image_applies_loras() -> None:
    editor = _make_editor()
    _out, meta = editor.text_to_image(
        EditParams(prompt="x", seed=None, width=512, height=512),
        loras=[{"source": "someuser/klein-lora", "scale": 0.6}],
    )

    assert editor._pipe.lora_loads == [("someuser/klein-lora", "lora_0")]
    assert editor._pipe.adapters == (["lora_0"], [0.6])
    assert meta["loras"] == ["someuser/klein-lora"]


def test_klein_text_to_image_without_loras_touches_no_adapters() -> None:
    editor = _make_editor()
    _out, meta = editor.text_to_image(
        EditParams(prompt="x", seed=None, width=512, height=512),
    )

    assert editor._pipe.lora_loads == []
    assert editor._pipe.adapters is None
    assert meta["loras"] is None
