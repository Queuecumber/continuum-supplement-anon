"""FLUX.2 Klein editor: cheap edits and text-to-image (LoRA-capable)."""

import os
from collections.abc import Callable
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
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
from continuum.backends.offload import image_offload_mode, place_image_pipeline
from continuum.progress import ProgressCallback, emit_progress, patch_tqdm_progress


class Flux2KleinEditor:
    """
    Image generator/editor using FLUX.2 [klein] 9B.

    Klein uses diffusers' Flux2KleinPipeline rather than Flux2Pipeline. The
    9B checkpoint is the distilled klein model (low-step, low-guidance) and
    is the base the public FLUX.2 LoRA ecosystem targets.
    """

    GEN_LONG_EDGE = 1024
    DEFAULT_STEPS = 4
    DEFAULT_GUIDANCE = 1.0

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-9B",
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
        self.cpu_offload = image_offload_mode()
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
            "cpu_offload": self.cpu_offload,
        }

    def load(self, load_progress: ProgressCallback | None = None):
        return self._load_pipeline(load_progress=load_progress)

    def _load_pipeline(self, load_progress: ProgressCallback | None = None):
        if self._pipe is not None:
            return self._pipe

        emit_progress(load_progress, "setting up model", 0, None)
        info = self.runtime_info()
        model_log(
            f"[flux2-klein] loading dtype={info['dtype']} device={info['device']} model={info['model']}",
        )
        with patch_tqdm_progress(load_progress, "loading model"):
            pipe = Flux2KleinPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        place_image_pipeline(pipe, self.device, self.cpu_offload)
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
        """Edit image with FLUX.2 [klein] (whole-image, no mask).

        Optional loras are FLUX.2-klein-compatible adapters.
        """
        pipe = self._load_pipeline()
        lora_specs = normalize_lora_specs(loras)
        apply_loras(self, pipe, lora_specs, hf_token=self.hf_token, load_progress=load_progress)

        if isinstance(image, list):
            imgs = [img.convert("RGB") for img in image]
        else:
            imgs = [image.convert("RGB")]
        multi = len(imgs) > 1
        imgs, width, height = self._fit_inputs(imgs, params)
        call_width, call_height = imgs[0].size

        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(params.seed))

        call_kwargs = {
            "image": imgs if multi else imgs[0],
            "prompt": params.prompt,
            "num_inference_steps": int(params.num_inference_steps),
            "guidance_scale": float(params.guidance_scale),
            "generator": generator,
        }
        if accepts_kw(pipe.__call__, "width"):
            call_kwargs["width"] = call_width
        if accepts_kw(pipe.__call__, "height"):
            call_kwargs["height"] = call_height

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
            context="FLUX.2 Klein edit pipeline",
        )

        meta = {
            "model": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "mode": "flux2-klein",
            "loras": [spec["source"] for spec in lora_specs] or None,
            "multi_image": multi,
            "num_images": len(imgs),
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "requested_size": [width, height],
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
        loras: Any = None,
        load_progress: ProgressCallback | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Text-to-image generation with FLUX.2 [klein].

        Optional loras are FLUX.2-klein-compatible adapters.
        """
        pipe = self._load_pipeline()
        lora_specs = normalize_lora_specs(loras)
        apply_loras(self, pipe, lora_specs, hf_token=self.hf_token, load_progress=load_progress)

        width = imaging.snap(params.width or self.GEN_LONG_EDGE)
        height = imaging.snap(params.height or self.GEN_LONG_EDGE)

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
            "mode": "flux2-klein-generate",
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "loras": [spec["source"] for spec in lora_specs] or None,
            "output_size": [int(out.width), int(out.height)],
        }
        return out, meta
