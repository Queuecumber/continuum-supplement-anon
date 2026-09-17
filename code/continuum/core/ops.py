"""Transport-free operations — the model layer behind the MCP views.

Every operation takes domain inputs (PIL images, plain params, raw bytes)
and returns domain outputs. Progress and cancellation flow through a Hooks
bundle supplied by the caller; nothing here knows MCP exists.
"""

import json
import math
import random
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from continuum.backends.base import EditParams
from continuum.backends.lora import normalize_lora_specs
from continuum.core import runtime
from continuum.imaging import (
    dilate_mask,
    prepare_outpaint_canvas,
    space_edit_dimensions,
)
from continuum.models import (
    DEFAULT_EDIT_MODEL,
    DEFAULT_GENERATE_MODEL,
    IMAGE_BACKENDS,
    canonical_image_model_name,
    is_flux1_model,
    is_flux2_klein_model,
    is_flux2_model,
    is_qwen_image_edit_model,
    is_qwen_image_model,
    normalize_image_model_name,
)

# ---------------------------------------------------------------------------
# Progress hooks
# ---------------------------------------------------------------------------


@dataclass
class Hooks:
    """Progress/cancellation callbacks supplied by a view.

    on_step(step, total): per-iteration generation progress (sync,
        worker-thread safe).
    on_load(message, progress=0, total=None): model download/load progress.
    cancel: set by the view when the host cancels; long loops should poll
        it (the bridges raise KeyboardInterrupt on the worker's behalf).
    notify(message): awaitable status notice ("queued", "decoding", ...).
    """

    on_step: Callable[[int, int], None] | None = None
    on_load: Callable[..., None] | None = None
    cancel: threading.Event | None = None
    notify: Callable[[str], Awaitable[None]] | None = None


async def _notify(hooks: Hooks | None, message: str) -> None:
    if hooks is not None and hooks.notify is not None:
        await hooks.notify(message)


def _hooks(hooks: Hooks | None) -> Hooks:
    return hooks if hooks is not None else Hooks()


# ---------------------------------------------------------------------------
# Defaults and model-name resolution
# ---------------------------------------------------------------------------


def resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else random.randint(0, 2**31 - 1)


def resolve_edit_defaults(
    model: str | None,
    steps: int | None,
    guidance: float | None,
) -> tuple[int, float]:
    model_key = normalize_image_model_name(model)
    if is_flux2_klein_model(model_key):
        default_steps = 4
        default_guidance = 1.0
    elif is_qwen_image_edit_model(model_key):
        default_steps = 40
        default_guidance = 4.0
    else:
        default_steps = 30
        default_guidance = 4.0
    return (
        int(steps) if steps is not None else default_steps,
        float(guidance) if guidance is not None else default_guidance,
    )


def resolve_generate_defaults(
    model: str | None,
    steps: int | None,
    guidance: float | None,
) -> tuple[int, float]:
    if is_flux2_klein_model(model):
        default_steps = 4
        default_guidance = 1.0
    elif is_flux1_model(model):
        default_steps = 28
        default_guidance = 3.5
    elif is_qwen_image_model(model):
        default_steps = 50
        default_guidance = 4.0
    else:
        default_steps = 30
        default_guidance = 4.0
    return (
        int(steps) if steps is not None else default_steps,
        float(guidance) if guidance is not None else default_guidance,
    )


def canonical_generate_model_name(model: str | None) -> str:
    key = normalize_image_model_name(model)
    if is_flux2_model(key):
        return "flux2"
    if is_flux2_klein_model(key):
        return "flux2-klein"
    if is_flux1_model(key):
        return "flux1-dev"
    if is_qwen_image_model(key):
        return "qwen-image-generate"
    return key


def canonical_edit_model_name(model: str | None) -> str:
    return canonical_image_model_name(model or DEFAULT_EDIT_MODEL)


def require_lora_support(model_key: str, lora_specs: list) -> None:
    """Raise if LoRAs were requested for a backend that cannot take them.

    Support is declared per backend in the model registry (lora_base names
    the base model adapters must be trained for; None = unsupported, e.g.
    flux2-dev whose FP8-quantized weights make LoRA application
    unreliable).

    Raises:
        ValueError: naming the LoRA-capable models.
    """
    if not lora_specs:
        return
    spec = IMAGE_BACKENDS.get(model_key)
    if spec is not None and spec.lora_base is not None:
        return
    capable = sorted(k for k, v in IMAGE_BACKENDS.items() if v.lora_base)
    raise ValueError(f"model {model_key!r} does not support LoRAs; LoRA-capable models: {', '.join(capable)}")


def _space_edit_dimensions(img: PILImage) -> tuple[int, int]:
    return space_edit_dimensions(img)


def edit_dimensions_for_model(
    model_key: str,
    base_img: PILImage,
    width: int | None,
    height: int | None,
) -> tuple[int | None, int | None]:
    """Return generation dimensions without forcing non-FLUX editors to crop."""
    if width is not None or height is not None:
        default_width, default_height = _space_edit_dimensions(base_img)
        return width or default_width, height or default_height
    if model_key in {"flux2", "flux2-klein"}:
        return _space_edit_dimensions(base_img)
    return None, None


# ---------------------------------------------------------------------------
# ROI / mask helpers
# ---------------------------------------------------------------------------


def coerce_roi_box(roi: Any) -> tuple[float, float, float, float] | None:
    """Return a rectangle if roi is rectangular, otherwise None for text roi."""
    if isinstance(roi, str):
        text = roi.strip()
        if not text:
            raise ValueError("roi text must not be empty")
        parsed: Any | None = None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
        else:
            parts = [part for part in text.replace(",", " ").split() if part]
            if len(parts) != 4:
                return None
            try:
                parsed = [float(part) for part in parts]
            except ValueError:
                return None
        return coerce_roi_box(parsed)

    if isinstance(roi, (list, tuple)):
        if len(roi) != 4:
            raise ValueError("rectangle roi must be [x1, y1, x2, y2]")
        try:
            x1, y1, x2, y2 = [float(value) for value in roi]
        except (TypeError, ValueError) as e:
            raise ValueError("rectangle roi values must be numeric") from e
        return x1, y1, x2, y2

    if isinstance(roi, dict):
        try:
            if {"x1", "y1", "x2", "y2"}.issubset(roi):
                return (
                    float(roi["x1"]),
                    float(roi["y1"]),
                    float(roi["x2"]),
                    float(roi["y2"]),
                )
            if {"x", "y", "width", "height"}.issubset(roi):
                x = float(roi["x"])
                y = float(roi["y"])
                return x, y, x + float(roi["width"]), y + float(roi["height"])
        except (TypeError, ValueError) as e:
            raise ValueError("rectangle roi values must be numeric") from e
        raise ValueError("object roi must contain x/y/width/height or x1/y1/x2/y2")

    return None


def _scale_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    normalized: bool,
) -> tuple[float, float, float, float]:
    if not normalized:
        return box
    width, height = image_size
    return (
        box[0] * width,
        box[1] * height,
        box[2] * width,
        box[3] * height,
    )


def _expand_and_clip_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    padding: float = 0.0,
    min_size: int = 1,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 == x1 or y2 == y1:
        raise ValueError("rectangle roi must have non-zero width and height")

    crop_w = x2 - x1
    crop_h = y2 - y1
    if 0.0 < padding <= 1.0:
        pad_x = crop_w * padding
        pad_y = crop_h * padding
    else:
        pad_x = max(0.0, padding)
        pad_y = max(0.0, padding)
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    min_size = max(1, int(min_size))
    if x2 - x1 < min_size:
        delta = (min_size - (x2 - x1)) / 2.0
        x1 -= delta
        x2 += delta
    if y2 - y1 < min_size:
        delta = (min_size - (y2 - y1)) / 2.0
        y1 -= delta
        y2 += delta

    left = max(0, min(width - 1, math.floor(x1)))
    top = max(0, min(height - 1, math.floor(y1)))
    right = max(left + 1, min(width, math.ceil(x2)))
    bottom = max(top + 1, min(height, math.ceil(y2)))
    return left, top, right, bottom


def bbox_from_mask(mask: PILImage) -> tuple[float, float, float, float]:
    bbox = mask.convert("L").point(lambda pixel: 255 if pixel > 0 else 0).getbbox()
    if bbox is None:
        raise ValueError("text roi did not match any visible region")
    left, top, right, bottom = bbox
    return float(left), float(top), float(right), float(bottom)


def crop_image_roi(
    image: PILImage,
    box: tuple[float, float, float, float],
    *,
    normalized: bool = False,
    padding: float = 0.0,
    min_size: int = 1,
) -> PILImage:
    scaled = _scale_box(box, image.size, normalized=normalized)
    crop_box = _expand_and_clip_box(
        scaled,
        image.size,
        padding=padding,
        min_size=min_size,
    )
    return image.crop(crop_box)


def subtract_masks(base: PILImage, negatives: list[PILImage]) -> PILImage:
    """Subtract white pixels in negative masks from a base L-mode mask."""
    base_arr = np.array(base.convert("L"), dtype=np.uint8)
    for neg in negatives:
        neg_img = neg.convert("L")
        if neg_img.size != base.size:
            neg_img = neg_img.resize(base.size, Image.NEAREST)
        base_arr[np.array(neg_img, dtype=np.uint8) > 0] = 0
    return Image.fromarray(base_arr, mode="L")


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------


async def generate_image(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    model: str = DEFAULT_GENERATE_MODEL,
    loras: Any = None,
    hooks: Hooks | None = None,
) -> PILImage:
    """Text-to-image generation.

    Returns:
        The generated image at the requested size.

    Raises:
        ValueError: if `model` is not a recognized generation model, or
            loras are requested for a model other than flux2-klein.
    """
    hooks = _hooks(hooks)
    image_model = canonical_generate_model_name(model)
    if image_model not in {"flux2", "flux2-klein", "flux1-dev", "qwen-image-generate"}:
        raise ValueError(
            "generate_image model must be 'flux2-dev', 'flux2-klein', 'flux1-dev', or 'qwen-image'"
        )
    # Validate LoRA specs early (raises on a bad source) and check the
    # backend declares LoRA support in the registry.
    lora_specs = normalize_lora_specs(loras)
    require_lora_support(image_model, lora_specs)
    gen_steps, gen_guidance = resolve_generate_defaults(image_model, steps, guidance)
    params = EditParams(
        prompt=prompt,
        guidance_scale=gen_guidance,
        num_inference_steps=gen_steps,
        seed=resolve_seed(seed),
        width=width,
        height=height,
    )
    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        mgr.enter_image_mode(image_model, load_progress=hooks.on_load)
        t2i_kwargs: dict[str, Any] = {"on_step": hooks.on_step}
        if lora_specs:
            t2i_kwargs["loras"] = loras
            t2i_kwargs["load_progress"] = hooks.on_load
        img, _meta = mgr.image_editor.text_to_image(params, **t2i_kwargs)
        return img

    return await runtime.run_locked(
        blocking,
        cancel=hooks.cancel,
        tool_name="generate_image",
    )


async def edit_image(
    image: PILImage,
    prompt: str,
    *,
    references: list[PILImage] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    model: str | None = None,
    loras: Any = None,
    hooks: Hooks | None = None,
) -> PILImage:
    """Whole-image instruction edit (no mask; see inpaint_image).

    Returns:
        The edited image.

    Raises:
        ValueError: if loras are requested for a model without LoRA
            support (flux2-dev), or a LoRA spec has a malformed source.
    """
    hooks = _hooks(hooks)
    images = [image, *(references or [])]

    image_model_key = canonical_edit_model_name(model)
    lora_specs = normalize_lora_specs(loras)
    require_lora_support(image_model_key, lora_specs)
    edit_steps, edit_guidance = resolve_edit_defaults(image_model_key, steps, guidance)
    edit_width, edit_height = edit_dimensions_for_model(
        image_model_key,
        image,
        width,
        height,
    )
    params = EditParams(
        prompt=prompt,
        guidance_scale=edit_guidance,
        num_inference_steps=edit_steps,
        seed=resolve_seed(seed),
        width=edit_width,
        height=edit_height,
    )
    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        mgr.enter_image_mode(image_model_key, load_progress=hooks.on_load)
        edit_kwargs: dict[str, Any] = {"on_step": hooks.on_step}
        if lora_specs:
            edit_kwargs["loras"] = loras
            edit_kwargs["load_progress"] = hooks.on_load
        out, _meta = mgr.image_editor.edit(images, params, **edit_kwargs)
        return out

    return await runtime.run_locked(
        blocking,
        cancel=hooks.cancel,
        tool_name="edit_image",
    )


async def _flux_fill(
    *,
    tool_name: str,
    base_img: PILImage,
    mask_img: PILImage,
    prompt: str,
    steps: int | None,
    guidance: float | None,
    seed: int | None,
    loras: Any,
    hooks: Hooks,
) -> PILImage:
    """Shared FLUX.1 Fill dispatch for inpaint_image and outpaint_image."""
    # Validate LoRA specs at the boundary (raises on a bad source).
    normalize_lora_specs(loras)

    params = EditParams(
        prompt=prompt,
        guidance_scale=float(guidance) if guidance is not None else 30.0,
        num_inference_steps=int(steps) if steps is not None else 50,
        seed=resolve_seed(seed),
    )
    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        mgr.enter_image_mode("flux1-fill", load_progress=hooks.on_load)
        out, _meta = mgr.image_editor.inpaint(
            base_img,
            params,
            mask_img,
            on_step=hooks.on_step,
            loras=loras,
            load_progress=hooks.on_load,
        )
        return out

    return await runtime.run_locked(
        blocking,
        cancel=hooks.cancel,
        tool_name=tool_name,
    )


async def inpaint_image(
    image: PILImage,
    mask: PILImage,
    prompt: str,
    *,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    dilate: int = 0,
    loras: Any = None,
    hooks: Hooks | None = None,
) -> PILImage:
    """Masked inpainting with FLUX.1 Fill (white = regenerate).

    Returns:
        The image with the masked region regenerated; pixels outside the
        (optionally dilated) mask are preserved exactly.

    Raises:
        ValueError: if a LoRA spec has a malformed source.
    """
    if dilate:
        mask = dilate_mask(mask, dilate)
    return await _flux_fill(
        tool_name="inpaint_image",
        base_img=image,
        mask_img=mask,
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
        loras=loras,
        hooks=_hooks(hooks),
    )


async def outpaint_image(
    image: PILImage,
    prompt: str,
    *,
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    overlap: int = 20,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    loras: Any = None,
    hooks: Hooks | None = None,
) -> PILImage:
    """Outpainting: pad the canvas per side and fill the border.

    Returns:
        The extended image at the padded canvas size.

    Raises:
        ValueError: if no side has padding > 0, padding is negative, or a
            LoRA spec has a malformed source.
    """
    canvas, mask_img = prepare_outpaint_canvas(
        image,
        left,
        right,
        top,
        bottom,
        overlap=overlap,
    )
    return await _flux_fill(
        tool_name="outpaint_image",
        base_img=canvas,
        mask_img=mask_img,
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
        loras=loras,
        hooks=_hooks(hooks),
    )


async def inspect_region(
    image: PILImage,
    roi: Any,
    *,
    normalized: bool = False,
    padding: float = 0.0,
    min_size: int = 1,
    threshold: float = 0.3,
    mask_threshold: float = 0.5,
    hooks: Hooks | None = None,
) -> PILImage:
    """Crop an ROI (rectangle, or text description segmented by SAM3).

    Returns:
        The crop at source resolution.

    Raises:
        ValueError: if `roi` is neither a valid rectangle nor a string, or
            a text roi matches no visible region.
    """
    hooks = _hooks(hooks)
    box = coerce_roi_box(roi)
    if box is not None:
        return crop_image_roi(
            image,
            box,
            normalized=normalized,
            padding=padding,
            min_size=min_size,
        )

    if not isinstance(roi, str):
        raise ValueError("roi must be a rectangle or text description")

    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        segmenter = mgr.get_segmenter(load_progress=hooks.on_load)
        mask = segmenter.segment(
            image,
            roi,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )
        return crop_image_roi(
            image,
            bbox_from_mask(mask),
            padding=padding,
            min_size=min_size,
        )

    return await runtime.run_locked(blocking, tool_name="inspect_region")


async def segment(
    image: PILImage,
    description: str,
    *,
    threshold: float = 0.3,
    mask_threshold: float = 0.5,
    negative_descriptions: list[str] | None = None,
    hooks: Hooks | None = None,
) -> PILImage:
    """Text-driven concept segmentation (SAM3 PCS).

    Returns:
        L-mode mask; white = matching region, with any
        negative_descriptions subtracted.
    """
    hooks = _hooks(hooks)
    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        segmenter = mgr.get_segmenter(load_progress=hooks.on_load)
        mask = segmenter.segment(
            image,
            description,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )
        negatives = []
        for negative in negative_descriptions or []:
            negatives.append(
                segmenter.segment(
                    image,
                    negative,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                )
            )
        return subtract_masks(mask, negatives) if negatives else mask

    return await runtime.run_locked(blocking, tool_name="segment")


async def segment_points(
    image: PILImage,
    points: list[list[float]],
    *,
    negative_points: list[list[float]] | None = None,
    labels: list[int] | None = None,
    normalized: bool = False,
    multimask_output: bool = True,
    mask_index: int | None = None,
    mask_threshold: float = 0.0,
    max_hole_area: float = 0.0,
    max_sprinkle_area: float = 0.0,
    apply_non_overlapping_constraints: bool = False,
    hooks: Hooks | None = None,
) -> PILImage:
    """Point-prompt segmentation (SAM3 Tracker / PVS).

    Returns:
        L-mode mask; white = selected region.
    """
    hooks = _hooks(hooks)
    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        return mgr.get_tracker_segmenter(load_progress=hooks.on_load).segment_points(
            image,
            points,
            labels=labels,
            negative_points=negative_points,
            normalized=normalized,
            multimask_output=multimask_output,
            mask_index=mask_index,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
            apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        )

    return await runtime.run_locked(blocking, tool_name="segment_points")


async def segment_boxes(
    image: PILImage,
    boxes: list[list[float]],
    *,
    normalized: bool = False,
    multimask_output: bool = False,
    mask_index: int | None = None,
    mask_threshold: float = 0.0,
    max_hole_area: float = 0.0,
    max_sprinkle_area: float = 0.0,
    apply_non_overlapping_constraints: bool = False,
    hooks: Hooks | None = None,
) -> PILImage:
    """Box-prompt segmentation (SAM3 Tracker / PVS); box masks are merged.

    Returns:
        L-mode mask; white = selected region.
    """
    hooks = _hooks(hooks)
    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        return mgr.get_tracker_segmenter(load_progress=hooks.on_load).segment_boxes(
            image,
            boxes,
            normalized=normalized,
            multimask_output=multimask_output,
            mask_index=mask_index,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
            apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        )

    return await runtime.run_locked(blocking, tool_name="segment_boxes")


def invert_mask(mask: PILImage) -> PILImage:
    """Invert an L-mode mask (255 <-> 0). Pure pixel op; no models."""
    return Image.fromarray(255 - np.array(mask.convert("L")), mode="L")


async def upscale_image(
    image: PILImage,
    *,
    scale: float | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    steps: int = 20,
    guidance: float = 5.0,
    target_tile_size: int = 1024,
    overlap: float = 0.5,
    hooks: Hooks | None = None,
) -> PILImage:
    """Tile-based super-resolution upscale (FaithDiff).

    Returns:
        The upscaled image at the target size (explicit dims win over
        `scale`).

    Raises:
        ValueError: if neither `scale` nor both target dimensions are
            provided.
    """
    hooks = _hooks(hooks)
    if target_width is not None and target_height is not None:
        tw, th = int(target_width), int(target_height)
    elif scale is not None:
        tw = int(round(image.width * scale))
        th = int(round(image.height * scale))
    else:
        raise ValueError("Provide either `scale` or both target_width and target_height")

    await _notify(hooks, "queued")

    def blocking() -> PILImage:
        mgr = runtime.get_models()
        return mgr.upscale(
            image,
            tw,
            th,
            on_step=hooks.on_step,
            load_progress=hooks.on_load,
            steps=steps,
            guidance=guidance,
            target_tile_size=target_tile_size,
            overlap=overlap,
        )

    return await runtime.run_locked(
        blocking,
        cancel=hooks.cancel,
        tool_name="upscale_image",
    )
