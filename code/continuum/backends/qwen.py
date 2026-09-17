"""Qwen-Image text-to-image and FireRed / Qwen-Edit instruction editors."""

import os
from collections.abc import Callable
from typing import Any

import torch
from diffusers import DiffusionPipeline, QwenImagePipeline
from PIL import Image

from continuum import imaging
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


class QwenImageGenerator:
    """
    Text-to-image generator using Qwen/Qwen-Image.

    Qwen's diffusers pipeline uses true_cfg_scale for classifier-free
    guidance. The guidance_scale argument exists for future distilled
    variants and is ignored by the standard checkpoint.
    """

    GEN_LONG_EDGE = 1024
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 4.0

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image",
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
        model_log(
            f"[qwen-image] loading dtype={info['dtype']} device={info['device']} model={info['model']}",
        )
        with patch_tqdm_progress(load_progress, "loading model"):
            pipe = QwenImagePipeline.from_pretrained(
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
    def _snap_qwen(x: int) -> int:
        return max(32, round(int(x) / 32) * 32)

    def edit(
        self,
        image: Image.Image | list[Image.Image],
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
        **_ignored: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        raise RuntimeError(
            "Qwen/Qwen-Image is text-to-image only in this adapter; "
            "use edit_image(model='qwen-image'), which routes to "
            "Qwen/Qwen-Image-Edit-2511."
        )

    def text_to_image(
        self,
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
        loras: Any = None,
        load_progress: ProgressCallback | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Text-to-image generation with Qwen/Qwen-Image.

        Optional loras are Qwen-Image-compatible adapters.
        """
        pipe = self._load_pipeline()
        lora_specs = normalize_lora_specs(loras)
        apply_loras(self, pipe, lora_specs, hf_token=self.hf_token, load_progress=load_progress)

        width = self._snap_qwen(params.width or self.GEN_LONG_EDGE)
        height = self._snap_qwen(params.height or self.GEN_LONG_EDGE)

        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(params.seed))

        call_kwargs = dict(
            prompt=params.prompt,
            height=height,
            width=width,
            num_inference_steps=int(params.num_inference_steps),
            generator=generator,
        )

        if accepts_kw(pipe.__call__, "true_cfg_scale"):
            call_kwargs["true_cfg_scale"] = float(params.guidance_scale)
            if accepts_kw(pipe.__call__, "negative_prompt"):
                call_kwargs["negative_prompt"] = " "
        elif accepts_kw(pipe.__call__, "guidance_scale"):
            call_kwargs["guidance_scale"] = float(params.guidance_scale)

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
            "mode": "qwen-image-generate",
            "loras": [spec["source"] for spec in lora_specs] or None,
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "output_size": [int(out.width), int(out.height)],
        }
        return out, meta


class FireRedEditor:
    """
    Image editor using FireRed-Image-Edit-1.1 / Qwen Image Edit Plus.

    Whole-image instruction edits only. For masked inpainting use
    FluxFillInpainter or Flux2Editor.inpaint().
    """

    GEN_LONG_EDGE = 1024
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 4.0

    def __init__(
        self,
        model_id: str = "FireRedTeam/FireRed-Image-Edit-1.1",
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
        model_log(
            f"[firered] loading dtype={info['dtype']} device={info['device']} model={info['model']}",
        )

        with patch_tqdm_progress(load_progress, "loading model"):
            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _fit_inputs(
        self,
        imgs: list[Image.Image],
        params: EditParams,
    ) -> tuple[list[Image.Image], int, int]:
        if params.width and params.height:
            width = int(params.width)
            height = int(params.height)
        else:
            width, height = imaging.space_edit_dimensions(imgs[0], self.GEN_LONG_EDGE)
        return [imaging.fit_image(img, width, height) for img in imgs], width, height

    def edit(
        self,
        image: Image.Image | list[Image.Image],
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
        loras: Any = None,
        load_progress: ProgressCallback | None = None,
        **_ignored: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Edit image with FireRed (whole-image, no mask).

        Optional loras are Qwen-Image-Edit-compatible adapters.
        """
        pipe = self._load_pipeline()
        lora_specs = normalize_lora_specs(loras)
        apply_loras(self, pipe, lora_specs, hf_token=self.hf_token, load_progress=load_progress)

        if isinstance(image, list):
            imgs = [img.convert("RGB") for img in image]
        else:
            imgs = [image.convert("RGB")]
        multi = len(imgs) > 1
        explicit_size = params.width is not None and params.height is not None
        imgs, width, height = self._fit_inputs(imgs, params)
        call_width, call_height = imgs[0].size

        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(params.seed))

        call_kwargs = {
            "image": imgs if multi else imgs[0],
            "prompt": params.prompt,
            "num_inference_steps": int(params.num_inference_steps),
        }
        if accepts_kw(pipe.__call__, "generator"):
            call_kwargs["generator"] = generator
        if accepts_kw(pipe.__call__, "width"):
            call_kwargs["width"] = call_width
        if accepts_kw(pipe.__call__, "height"):
            call_kwargs["height"] = call_height

        # Qwen Image Edit Plus exposes true_cfg_scale; guidance_scale is
        # ineffective for some Qwen edit variants.
        if accepts_kw(pipe.__call__, "true_cfg_scale"):
            call_kwargs["true_cfg_scale"] = float(params.guidance_scale)
            if accepts_kw(pipe.__call__, "guidance_scale"):
                call_kwargs["guidance_scale"] = 1.0
            if accepts_kw(pipe.__call__, "negative_prompt"):
                call_kwargs["negative_prompt"] = " "
        elif accepts_kw(pipe.__call__, "guidance_scale"):
            call_kwargs["guidance_scale"] = float(params.guidance_scale)

        if on_step is not None and accepts_kw(pipe.__call__, "callback_on_step_end"):
            total = int(params.num_inference_steps)

            def _cb(pipe_obj, step_index, timestep, cb_kwargs):
                on_step(step_index + 1, total)
                return cb_kwargs

            call_kwargs["callback_on_step_end"] = _cb

        with torch.no_grad():
            out = pipe(**call_kwargs).images[0]

        out = imaging.resize_if_aspect_matches(
            out,
            (call_width, call_height),
            context="FireRed edit pipeline",
        )

        meta = {
            "model": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "mode": "firered",
            "loras": [spec["source"] for spec in lora_specs] or None,
            "multi_image": multi,
            "num_images": len(imgs),
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "requested_size": [width, height] if explicit_size else None,
            "call_size": [call_width, call_height],
            "conditioning_sizes": [[int(img.width), int(img.height)] for img in imgs],
            "crop_region": None,
            "output_size": [int(out.width), int(out.height)],
        }
        return out, meta

    def text_to_image(
        self,
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        raise RuntimeError("FireRed-Image-Edit-1.1 is image-to-image only; use flux2 for generate_image.")
