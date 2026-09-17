"""FLUX.1-dev text-to-image generator (LoRA-capable).

Loaded in bf16 (NOT quantized) on purpose: that is what lets it take LoRAs,
and FLUX.1-dev has the largest, most battle-tested LoRA ecosystem
(base_model:adapter:black-forest-labs/FLUX.1-dev). FLUX.1-dev is text-to-image
only — it is not an instruction editor; use the Qwen/klein edit models for
that. The same FLUX.1-dev LoRAs also apply to the FLUX.1 Fill inpaint/outpaint
tools.
"""

import os
from collections.abc import Callable
from typing import Any

import torch
from diffusers import FluxPipeline
from PIL import Image

from continuum.backends.base import (
    DeviceName,
    DTypeName,
    EditParams,
    accepts_kw,
    model_log,
    pick_device,
    pick_dtype,
)
from continuum.backends.lora import apply_loras, normalize_lora_specs
from continuum.progress import ProgressCallback, emit_progress, patch_tqdm_progress


class Flux1Generator:
    """Text-to-image generator using FLUX.1-dev, bf16 so LoRAs apply."""

    GEN_LONG_EDGE = 1024
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE = 3.5

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: DeviceName = "auto",
        dtype: DTypeName = "auto",
        hf_token: str | None = None,
    ):
        self.model_id = model_id
        self.device: str = pick_device(device)
        self.dtype_name: DTypeName = dtype
        self.dtype = pick_dtype(dtype, self.device)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.split_devices = False
        self.text_encoder_device = self.device
        self._pipe = None

    def runtime_info(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "load_mode": "bf16" if str(self.dtype).endswith("bfloat16") else str(self.dtype),
            "device": self.device,
            "text_encoder_device": self.text_encoder_device,
            "split_devices": False,
            "dtype": str(self.dtype).replace("torch.", ""),
            "quantization": "none",
        }

    def load(self, load_progress: ProgressCallback | None = None):
        return self._load_pipeline(load_progress=load_progress)

    def _load_pipeline(self, load_progress: ProgressCallback | None = None):
        if self._pipe is not None:
            return self._pipe

        emit_progress(load_progress, "setting up model", 0, None)
        info = self.runtime_info()
        model_log(f"[flux1-dev] loading dtype={info['dtype']} device={info['device']} model={info['model']}")
        with patch_tqdm_progress(load_progress, "loading model"):
            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    @staticmethod
    def _snap(x: int) -> int:
        return max(16, round(int(x) / 16) * 16)

    def edit(
        self,
        image: Image.Image | list[Image.Image],
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
        **_ignored: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        raise RuntimeError(
            "FLUX.1-dev is text-to-image only; it is not an instruction editor. "
            "Use edit_image(model='qwen-image' or 'flux2-klein'), or inpaint_image/"
            "outpaint_image (which take FLUX.1-dev LoRAs)."
        )

    def text_to_image(
        self,
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
        loras: Any = None,
        load_progress: ProgressCallback | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Text-to-image generation with FLUX.1-dev.

        Optional loras are FLUX.1-dev-compatible adapters (the large public
        FLUX.1-dev ecosystem).
        """
        pipe = self._load_pipeline()
        lora_specs = normalize_lora_specs(loras)
        apply_loras(self, pipe, lora_specs, hf_token=self.hf_token, load_progress=load_progress)

        width = self._snap(params.width or self.GEN_LONG_EDGE)
        height = self._snap(params.height or self.GEN_LONG_EDGE)

        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(params.seed))

        call_kwargs = dict(
            prompt=params.prompt,
            height=height,
            width=width,
            num_inference_steps=int(params.num_inference_steps),
            guidance_scale=float(params.guidance_scale),
            generator=generator,
        )
        if accepts_kw(pipe.__call__, "max_sequence_length"):
            call_kwargs["max_sequence_length"] = 512

        if on_step is not None and accepts_kw(pipe.__call__, "callback_on_step_end"):
            total = int(params.num_inference_steps)

            def _cb(pipe_obj, step_index, timestep, cb_kwargs):
                on_step(step_index + 1, total)
                return cb_kwargs

            call_kwargs["callback_on_step_end"] = _cb

        with torch.no_grad():
            out = pipe(**call_kwargs).images[0]

        meta = {
            "model": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "mode": "flux1-dev-generate",
            "loras": [spec["source"] for spec in lora_specs] or None,
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "output_size": [int(out.width), int(out.height)],
        }
        return out, meta
