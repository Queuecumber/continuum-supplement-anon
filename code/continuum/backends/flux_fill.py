"""FLUX.1 Fill inpainting backend (masked inpaint/outpaint, LoRA-capable)."""

import os
from collections.abc import Callable
from typing import Any

import torch
from diffusers import FluxFillPipeline
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


class FluxFillInpainter:
    """Native inpainting with FLUX.1 Fill [dev].

    Unlike the instruction-edit backends, FLUX.1 Fill is trained for masked
    inpainting: it is conditioned on the *masked* image (the fill region
    erased), so it must synthesise the hole rather than copy the source. The
    unmasked region is preserved pixel-exact with a feathered composite against
    the original, which is safe here because Fill produces tonally consistent
    fills (this is the failure mode that broke LanPaint-on-edit-models).

    Mask convention: white = fill, black = keep.
    """

    GEN_LONG_EDGE = 1024
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 30.0

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
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
        self._active_lora_sig = None

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
            f"[flux-fill] loading dtype={info['dtype']} device={info['device']} model={info['model']}",
        )
        with patch_tqdm_progress(load_progress, "loading model"):
            pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        place_image_pipeline(pipe, self.device, self.cpu_offload)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def edit(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "FLUX.1 Fill is inpaint-only; call inpaint() (edit_image does not "
            "route here). Use edit_image for instruction edits."
        )

    def text_to_image(self, *args: Any, **kwargs: Any):
        raise RuntimeError("FLUX.1 Fill is inpaint-only; use generate_image for text-to-image.")

    def inpaint(
        self,
        image: Image.Image | list[Image.Image],
        params: EditParams,
        mask_image: Image.Image,
        on_step: Callable[[int, int], None] | None = None,
        feather: int = 3,
        loras: Any = None,
        load_progress: ProgressCallback | None = None,
        **_ignored: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Inpaint the white region of mask_image to match the prompt.

        The prompt should describe the desired *contents* of the region (e.g.
        "a red silk shirt"), not an edit instruction. The region outside the
        mask is preserved pixel-exact via a feathered composite at native
        resolution. Optional loras are FLUX.1-dev-compatible adapters.
        """
        # Absolute import so the module also works when loaded standalone.
        pipe = self._load_pipeline()
        lora_specs = normalize_lora_specs(loras)
        apply_loras(self, pipe, lora_specs, hf_token=self.hf_token, load_progress=load_progress)

        original = (image[0] if isinstance(image, list) else image).convert("RGB")
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")

        # Generation canvas: snap to the pipeline's ×16 requirement. When the
        # image already fits the generation budget, work at (snapped) native
        # size — upscaling to 1024 and back gives the generated region a
        # different resampling character than the untouched original, which
        # reads as a seam at the composite boundary.
        if params.width and params.height:
            gen_w, gen_h = imaging.snap(int(params.width)), imaging.snap(int(params.height))
        elif max(original.size) <= self.GEN_LONG_EDGE:
            gen_w, gen_h = imaging.snap(original.width), imaging.snap(original.height)
        else:
            gen_w, gen_h = imaging.space_edit_dimensions(original, self.GEN_LONG_EDGE)
            gen_w, gen_h = imaging.snap(gen_w), imaging.snap(gen_h)

        img_in = original.resize((gen_w, gen_h), Image.LANCZOS)
        mask_in = mask_image.resize((gen_w, gen_h), Image.NEAREST)

        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(params.seed))

        call_kwargs: dict[str, Any] = dict(
            image=img_in,
            mask_image=mask_in,
            height=gen_h,
            width=gen_w,
            num_inference_steps=int(params.num_inference_steps),
            guidance_scale=float(params.guidance_scale),
            generator=generator,
        )
        call_kwargs["prompt"] = params.prompt
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

        # Align the generated image to the original at native resolution and
        # correct the generator's global tone drift (estimated on the keep
        # region) so the composite boundary carries no tonal step. The aspect
        # guard turns a pipeline returning unexpected dimensions into a loud
        # error instead of a silently stretched/zoomed result.
        out = imaging.resize_if_aspect_matches(
            out,
            original.size,
            context="FLUX.1 Fill pipeline",
        )
        mask_native = mask_image
        if mask_native.size != original.size:
            mask_native = mask_native.resize(original.size, Image.NEAREST)
        out = imaging.match_color_statistics(out, original, mask_native)

        # Preserve everything outside the mask pixel-exact at native resolution.
        result = imaging.composite_mask(
            original,
            out,
            mask_native,
            feather_radius=max(0, int(feather)),
        )

        meta = {
            "model": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "mode": "flux1-fill",
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "gen_size": [gen_w, gen_h],
            "feather": max(0, int(feather)),
            "color_corrected": True,
            "loras": [s["source"] for s in lora_specs] or None,
            "output_size": [int(result.width), int(result.height)],
        }
        return result, meta
