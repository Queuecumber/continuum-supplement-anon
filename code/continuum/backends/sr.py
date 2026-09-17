"""FaithDiff super-resolution (SDXL community pipeline).

Quarantined: this backend wrangles a community pipeline (cv2 stubbing,
manual tiling, FP8 UNet) and is intentionally isolated from the image
editors.
"""

import gc
import os
import sys
import time
import types
from collections.abc import Callable
from importlib.machinery import ModuleSpec

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from huggingface_hub import hf_hub_download
from PIL import Image

from continuum.progress import ProgressCallback, emit_progress, patch_tqdm_progress


def _clear_vram():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# Module-level SR pipeline cache for batch operations (successive image calls).
# When keep_loaded=True, the pipeline persists across calls to avoid
# reloading ~10GB of weights per image.
_sr_pipe = None
_sr_pipe_device: str | None = None


def release_sr_pipeline() -> None:
    """Free cached SR pipeline from memory."""
    global _sr_pipe, _sr_pipe_device
    if _sr_pipe is not None:
        del _sr_pipe
        _sr_pipe = None
        _sr_pipe_device = None
        _clear_vram()


def upscale_faithdiff(
    img: Image.Image,
    target_w: int,
    target_h: int,
    on_step: Callable[[int, int], None] | None = None,
    load_progress: ProgressCallback | None = None,
    device: str = "cuda:0",
    keep_loaded: bool = False,
    prompt: str = "high-quality, detailed, sharp",
    negative_prompt: str = "blurry, pixelated, noisy, low resolution, artifacts",
    steps: int = 20,
    guidance: float = 5.0,
    target_tile_size: int = 1024,
    overlap: float = 0.5,
) -> Image.Image:
    """Upscale img using FaithDiff (CVPR 2025) SDXL-based super-resolution.

    Loads FaithDiff pipeline, runs SR with FP8 UNet. When keep_loaded=False
    (default), unloads after each call. When keep_loaded=True, caches the
    pipeline for subsequent calls (use release_sr_pipeline() to free).
    """
    global _sr_pipe, _sr_pipe_device

    # Stub cv2 before importing the community pipeline (it imports cv2 at
    # module level for check_image_size, which we replace with numpy padding)
    # Function-local: continuum.models imports this module (cycle).
    from continuum.models import RAMDISK_CACHE

    if "cv2" not in sys.modules:
        _cv2 = types.ModuleType("cv2")
        _cv2.BORDER_REPLICATE = 1
        _cv2.copyMakeBorder = lambda *a, **kw: None
        _cv2.__spec__ = ModuleSpec("cv2", None)
        sys.modules["cv2"] = _cv2
    elif sys.modules["cv2"].__spec__ is None:
        _cv2 = sys.modules["cv2"]
        _cv2.__spec__ = ModuleSpec("cv2", None, origin=getattr(_cv2, "__file__", None))

    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    if scale <= 1.0:
        return img

    # Reuse cached pipeline if available
    if _sr_pipe is not None and _sr_pipe_device == device:
        pipe = _sr_pipe
    else:
        emit_progress(load_progress, "setting up model", 0, None)
        # Release old pipeline if switching devices
        if _sr_pipe is not None:
            release_sr_pipeline()

        # Use ramdisk-cached paths if available, otherwise fall back to HF hub
        faithdiff_local = os.path.join(RAMDISK_CACHE, "jychen9811--FaithDiff")
        realvis_local = os.path.join(RAMDISK_CACHE, "SG161222--RealVisXL_V4.0")
        vae_local = os.path.join(RAMDISK_CACHE, "madebyollin--sdxl-vae-fp16-fix")

        faithdiff_path = faithdiff_local if os.path.exists(faithdiff_local) else "jychen9811/FaithDiff"
        realvis_path = realvis_local if os.path.exists(realvis_local) else "SG161222/RealVisXL_V4.0"
        vae_path = vae_local if os.path.exists(vae_local) else "madebyollin/sdxl-vae-fp16-fix"

        weight_path = os.path.join(faithdiff_path, "FaithDiff.bin")
        if not os.path.exists(weight_path):
            weight_path = hf_hub_download("jychen9811/FaithDiff", filename="FaithDiff.bin")

        start = time.time()

        with patch_tqdm_progress(load_progress, "loading VAE"):
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=torch.float16,
            )
        with patch_tqdm_progress(load_progress, "loading SR pipeline"):
            pipe = DiffusionPipeline.from_pretrained(
                realvis_path,
                torch_dtype=torch.float16,
                vae=vae,
                custom_pipeline="pipeline_faithdiff_stable_diffusion_xl",
                use_safetensors=True,
                variant="fp16",
            )
        with patch_tqdm_progress(load_progress, "loading SR UNet"):
            pipe.unet = pipe.unet_model.from_pretrained(
                realvis_path,
                subfolder="unet",
                variant="fp16",
                use_safetensors=True,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        pipe.unet.load_additional_layers(
            weight_path=weight_path,
            dtype=torch.float16,
        )
        pipe.unet.to(torch.float8_e4m3fn)
        pipe.set_encoder_tile_settings()
        # diffusers 0.40 made the pipeline-level shim an error rather than a
        # warning; tiling is configured on the VAE itself now. Guarded so the
        # backend still loads against a version that has only the old spelling.
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        elif hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        pipe.set_progress_bar_config(disable=True)

        load_time = time.time() - start
        emit_progress(load_progress, f"SR model loaded in {load_time:.1f}s", 0, None)

    upscaled_input = img.resize((target_w, target_h), Image.LANCZOS)

    # Pad to multiple of 8 (replaces pipe.check_image_size which uses cv2)

    padder = 8
    w_init, h_init = upscaled_input.size
    mod_pad_w = (padder - w_init % padder) % padder
    mod_pad_h = (padder - h_init % padder) % padder
    if mod_pad_w or mod_pad_h:
        arr = np.array(upscaled_input)
        arr = np.pad(arr, ((0, mod_pad_h), (0, mod_pad_w), (0, 0)), mode="edge")
        input_image = Image.fromarray(arr)
    else:
        input_image = upscaled_input
    w_now = w_init + mod_pad_w
    h_now = h_init + mod_pad_h

    sr_steps = int(steps)

    if on_step is not None:
        # FaithDiff tiles the image: callback fires 0→(sr_steps-1) per tile.
        # Report step within each tile (resets per tile) so the progress bar
        # shows 1/20..20/20 cleanly instead of counting to hundreds.
        def _cb(pipe_obj, step_index, timestep, cb_kwargs):
            on_step(step_index + 1, sr_steps)
            return cb_kwargs
    else:
        _cb = None

    call_kwargs = dict(
        lr_img=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=sr_steps,
        guidance_scale=float(guidance),
        start_point="lr",
        height=h_now,
        width=w_now,
        target_size=(int(target_tile_size), int(target_tile_size)),
        overlap=float(overlap),
    )
    if _cb is not None:
        call_kwargs["callback_on_step_end"] = _cb

    with torch.no_grad():
        result = pipe(**call_kwargs).images[0]

    if result.size != (target_w, target_h):
        result = result.crop((0, 0, target_w, target_h))

    if keep_loaded:
        _sr_pipe = pipe
        _sr_pipe_device = device
    else:
        del pipe
        _sr_pipe = None
        _sr_pipe_device = None
        _clear_vram()

    return result
