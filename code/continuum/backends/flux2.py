"""FLUX.2-dev editor: whole-image instruction edits."""

import os
from collections.abc import Callable
from typing import Any

import torch
from diffusers import Flux2Pipeline, PipelineQuantizationConfig
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from PIL import Image
from torchao.quantization import Float8WeightOnlyConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig

from continuum import imaging
from continuum.backends.base import (
    DeviceName,
    DTypeName,
    EditParams,
    model_log,
    pick_device,
    pick_dtype,
)
from continuum.progress import ProgressCallback, emit_progress, patch_tqdm_progress


def _patch_split_pipeline(pipe, *, transformer_device: str) -> None:
    """Make a Flux2Pipeline work with the text encoder on a different device
    from the transformer/VAE.

    Two things need to change vs. the single-device path:

    1. ``_execution_device`` defaults to walking ``self.components.values()``
       and returning the first nn.Module's device — which lands on the text
       encoder when it's on cuda:1, but the diffusion loop wants the
       transformer's device. Override it.
    2. ``encode_prompt`` runs the text encoder forward and uses ``device``
       for its input tensors. If we pass ``transformer_device`` it crashes
       because the encoder's weights live on a different card. Wrap it so
       it encodes on the encoder's own device, then transfers embeddings
       back to the requested target.
    """
    base_class = pipe.__class__
    split_class = type(
        f"{base_class.__name__}_Split",
        (base_class,),
        {"_execution_device": property(lambda self: self.transformer.device)},
    )
    pipe.__class__ = split_class

    original_encode_prompt = pipe.encode_prompt

    def split_encode_prompt(prompt, device=None, **kwargs):
        te_device = next(pipe.text_encoder.parameters()).device
        target = device if device is not None else transformer_device
        embeds, text_ids = original_encode_prompt(prompt, device=te_device, **kwargs)
        return embeds.to(target), text_ids.to(target)

    pipe.encode_prompt = split_encode_prompt


class Flux2Editor:
    """
    Image editor using FLUX.2-dev (32B) for whole-image instruction edits
    (masked inpainting lives in FluxFillInpainter).

    Edits use Flux2Pipeline with Mistral-3 text encoder and guidance
    embedding.

    Two deployment shapes:

    - **Single GPU** (text_encoder_device == device): FP8 weight-only
      quantization on transformer + text encoder so everything fits on
      80 GB. Slight quality cost vs BF16.
    - **Dual GPU** (text_encoder_device != device): pure BF16 with no
      quantization. Transformer + VAE on one card, text encoder on
      another.

    Multi-image: pass a list of PIL images directly via image= parameter.
    """

    GEN_LONG_EDGE = 1024
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 4.0

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-dev",
        device: DeviceName = "auto",
        dtype: DTypeName = "auto",
        hf_token: str | None = None,
        text_encoder_device: DeviceName | None = None,
    ):
        self.model_id = model_id
        self.device: str = pick_device(device)
        self.text_encoder_device: str = (
            pick_device(text_encoder_device) if text_encoder_device else self.device
        )
        self.split_devices: bool = self.text_encoder_device != self.device
        self.dtype_name: DTypeName = dtype
        self.dtype = pick_dtype(dtype, self.device)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self._pipe = None

    def runtime_info(self) -> dict[str, Any]:
        load_mode = "bf16-split" if self.split_devices else "fp8-weight-only"
        return {
            "model": self.model_id,
            "load_mode": load_mode,
            "device": self.device,
            "text_encoder_device": self.text_encoder_device,
            "split_devices": self.split_devices,
            "dtype": str(self.dtype).replace("torch.", ""),
            "quantization": "none" if self.split_devices else "torchao-fp8-weight-only",
        }

    def load(self, load_progress: ProgressCallback | None = None):
        return self._load_pipeline(load_progress=load_progress)

    def _load_pipeline(self, load_progress: ProgressCallback | None = None):
        if self._pipe is not None:
            return self._pipe

        emit_progress(load_progress, "setting up model", 0, None)
        info = self.runtime_info()
        model_log(
            "[flux2] loading "
            f"mode={info['load_mode']} dtype={info['dtype']} "
            f"device={info['device']} text_encoder_device={info['text_encoder_device']} "
            f"quantization={info['quantization']}",
        )

        if self.split_devices:
            # Dual-GPU: pure BF16, manual component placement, no quant.
            with patch_tqdm_progress(load_progress, "loading model"):
                pipe = Flux2Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    token=self.hf_token,
                )
            emit_progress(load_progress, "setting up model", 0, None)
            pipe.text_encoder.to(self.text_encoder_device)
            pipe.transformer.to(self.device)
            pipe.vae.to(self.device)
            _patch_split_pipeline(pipe, transformer_device=self.device)
        else:
            # Single-GPU: FP8 weight-only on transformer + text encoder.
            quantization_config = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusersTorchAoConfig(quant_type=Float8WeightOnlyConfig()),
                    "text_encoder": TransformersTorchAoConfig(quant_type=Float8WeightOnlyConfig()),
                }
            )

            with patch_tqdm_progress(load_progress, "loading model"):
                pipe = Flux2Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    quantization_config=quantization_config,
                    token=self.hf_token,
                )
            emit_progress(load_progress, "setting up model", 0, None)
            pipe.to(self.device)

        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def edit(
        self,
        image: Image.Image | list[Image.Image],
        params: EditParams,
        on_step: Callable[[int, int], None] | None = None,
        **_ignored: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Edit image with instruction prompt via FLUX.2 (whole-image, no mask).

        Accepts a single PIL image or a list of images. FLUX.2 handles
        multi-reference natively via the image= parameter. For masked
        inpainting use FluxFillInpainter.
        """
        pipe = self._load_pipeline()

        multi = isinstance(image, list) and len(image) > 1

        if multi:
            imgs = [img.convert("RGB") for img in image]
        else:
            img = (image[0] if isinstance(image, list) else image).convert("RGB")
            imgs = [img]

        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(params.seed))

        # Standard FLUX.2 editing path — mirror the official Space: pass
        # width/height and let diffusers preprocess each conditioning image
        # independently.
        call_kwargs = dict(
            image=imgs,
            prompt=params.prompt,
            num_inference_steps=int(params.num_inference_steps),
            guidance_scale=float(params.guidance_scale),
            generator=generator,
        )
        if params.width and params.height:
            call_kwargs["width"] = int(params.width)
            call_kwargs["height"] = int(params.height)

        if on_step is not None:
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
            "mode": "flux2",
            "multi_image": multi,
            "num_images": len(image) if isinstance(image, list) else 1,
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "requested_size": (
                [int(params.width), int(params.height)] if params.width and params.height else None
            ),
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
        """Text-to-image generation with FLUX.2-dev.

        No reference images — pure text-to-image. Uses params.width/height
        for output size (defaults to GEN_LONG_EDGE square).
        """
        pipe = self._load_pipeline()

        width = params.width or self.GEN_LONG_EDGE
        height = params.height or self.GEN_LONG_EDGE
        width = imaging.snap(width)
        height = imaging.snap(height)

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

        if on_step is not None:
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
            "mode": "flux2-generate",
            "prompt": params.prompt,
            "guidance_scale": float(params.guidance_scale),
            "num_inference_steps": int(params.num_inference_steps),
            "seed": params.seed,
            "output_size": [int(out.width), int(out.height)],
        }
        return out, meta
