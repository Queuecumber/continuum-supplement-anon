"""LoRA support is declared in the model registry and enforced in ops."""

import asyncio
import types

import pytest
from PIL import Image

from continuum.backends.base import EditParams
from continuum.core import ops


class _DummyLoraPipe:
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
        out = Image.new(
            "RGB",
            (kwargs.get("width", 512), kwargs.get("height", 512)),
            "white",
        )
        return types.SimpleNamespace(images=[out])


def _editor(cls):
    editor = cls.__new__(cls)
    editor._pipe = _DummyLoraPipe()
    editor.model_id = "test"
    editor.device = "cpu"
    editor.dtype = "bfloat16"
    editor.hf_token = None
    return editor


# ---- registry declarations --------------------------------------------------


def test_registry_declares_lora_bases() -> None:
    from continuum.models import IMAGE_BACKENDS

    bases = {k: v.lora_base for k, v in IMAGE_BACKENDS.items()}
    assert bases["flux2"] is None  # FP8-quantized; no LoRA
    assert bases["flux2-klein"] == "black-forest-labs/FLUX.2-klein-9B"
    assert bases["flux1-fill"] == "black-forest-labs/FLUX.1-dev"
    assert bases["flux1-dev"] == "black-forest-labs/FLUX.1-dev"
    assert bases["firered"] == "Qwen/Qwen-Image-Edit-2511"
    assert bases["qwen-image-edit"] == "Qwen/Qwen-Image-Edit-2511"
    assert bases["qwen-image-generate"] == "Qwen/Qwen-Image"


# ---- ops gate ---------------------------------------------------------------


def test_generate_rejects_loras_for_flux2_dev() -> None:
    with pytest.raises(ValueError, match="does not support LoRAs"):
        asyncio.run(
            ops.generate_image(
                "x",
                model="flux2-dev",
                loras=[{"source": "a/b"}],
            )
        )


def test_edit_rejects_loras_for_flux2_dev() -> None:
    with pytest.raises(ValueError, match="LoRA-capable models"):
        asyncio.run(
            ops.edit_image(
                Image.new("RGB", (64, 64)),
                "x",
                model="flux2-dev",
                loras=[{"source": "a/b"}],
            )
        )


def test_gate_error_names_capable_models() -> None:
    with pytest.raises(ValueError) as exc:
        ops.require_lora_support("flux2", [{"source": "a/b"}])
    message = str(exc.value)
    for capable in (
        "flux1-dev",
        "flux1-fill",
        "flux2-klein",
        "firered",
        "qwen-image-edit",
        "qwen-image-generate",
    ):
        assert capable in message


def test_gate_is_noop_without_loras() -> None:
    ops.require_lora_support("flux2", [])


# ---- newly wired backends ---------------------------------------------------


def test_klein_edit_applies_loras() -> None:
    from continuum.backends.klein import Flux2KleinEditor

    editor = _editor(Flux2KleinEditor)
    _out, meta = editor.edit(
        Image.new("RGB", (512, 512), "black"),
        EditParams(prompt="x", seed=None, width=512, height=512),
        loras=[{"source": "someuser/klein-lora", "scale": 0.5}],
    )
    assert editor._pipe.lora_loads == [("someuser/klein-lora", "lora_0")]
    assert meta["loras"] == ["someuser/klein-lora"]


def test_firered_edit_applies_loras() -> None:
    from continuum.backends.qwen import FireRedEditor

    editor = _editor(FireRedEditor)
    _out, meta = editor.edit(
        Image.new("RGB", (512, 512), "black"),
        EditParams(prompt="x", seed=None, width=512, height=512),
        loras=[{"source": "someuser/qwen-edit-lora"}],
    )
    assert editor._pipe.lora_loads == [("someuser/qwen-edit-lora", "lora_0")]
    assert meta["loras"] == ["someuser/qwen-edit-lora"]


def test_qwen_generate_applies_loras() -> None:
    from continuum.backends.qwen import QwenImageGenerator

    editor = _editor(QwenImageGenerator)
    _out, meta = editor.text_to_image(
        EditParams(prompt="x", seed=None, width=512, height=512),
        loras=[{"source": "someuser/qwen-lora"}],
    )
    assert editor._pipe.lora_loads == [("someuser/qwen-lora", "lora_0")]
    assert meta["loras"] == ["someuser/qwen-lora"]
