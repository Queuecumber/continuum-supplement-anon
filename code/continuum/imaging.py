"""Pixel-policy helpers shared by the image backends.

Pure PIL — no torch, no diffusers. This is the layer that owns geometry
(snapping, fitting, aspect guards) and pixel fidelity (compositing, color
correction, mask/canvas construction). Most historical GPU-visible bugs
(seams, silhouette anchoring, silent stretch/zoom) were fixed here.
"""

import numpy as np
from PIL import Image, ImageFilter

GEN_LONG_EDGE = 1024
SNAP = 16  # Pipelines snap to vae_scale_factor * 2 = 16
MAX_SAFE_ASPECT_RESIZE_ERROR = 0.03


def space_edit_dimensions(img: Image.Image, max_size: int = GEN_LONG_EDGE) -> tuple[int, int]:
    """Match the public FLUX.2 Space image-edit canvas selection."""
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height
    if aspect_ratio >= 1:
        width = max_size
        height = int(max_size / aspect_ratio)
    else:
        height = max_size
        width = int(max_size * aspect_ratio)

    width = round(width / 8) * 8
    height = round(height / 8) * 8
    width = max(256, min(max_size, width))
    height = max(256, min(max_size, height))
    return width, height


def snap(x: int) -> int:
    return max(SNAP, round(x / SNAP) * SNAP)


def fit_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale image to fit within target_w × target_h, preserving aspect ratio.

    Both dimensions are snapped to multiples of 16 to match the pipeline's
    internal snapping (vae_scale_factor * 2). Mismatched dimensions cause
    the pipeline to crop the input, creating a conditioning/output resolution
    mismatch that produces a global color shift.
    """
    src_w, src_h = img.size
    ar = src_w / src_h

    if ar >= target_w / target_h:
        new_w = min(src_w, target_w)
        new_h = round(new_w / ar)
    else:
        new_h = min(src_h, target_h)
        new_w = round(new_h * ar)

    new_w = snap(new_w)
    new_h = snap(new_h)

    if new_w != src_w or new_h != src_h:
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def resize_if_aspect_matches(
    img: Image.Image,
    target_size: tuple[int, int],
    *,
    context: str,
) -> Image.Image:
    """Resize only when the model output already matches the target aspect."""
    if img.size == target_size:
        return img

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0 or img.width <= 0 or img.height <= 0:
        raise ValueError("image dimensions must be positive")

    src_ratio = img.width / img.height
    target_ratio = target_w / target_h
    ratio_error = abs(src_ratio - target_ratio) / target_ratio
    if ratio_error > MAX_SAFE_ASPECT_RESIZE_ERROR:
        raise RuntimeError(
            f"{context} returned {img.width}x{img.height}; refusing to stretch "
            f"to {target_w}x{target_h} across mismatched aspect ratios"
        )

    return img.resize(target_size, Image.LANCZOS)


def match_color_statistics(
    edited: Image.Image,
    reference: Image.Image,
    fill_mask: Image.Image,
) -> Image.Image:
    """Correct edited's global color drift against reference before compositing.

    The keep region (black in fill_mask) exists in both images: reference has
    the ground-truth pixels, edited has the generator's reproduction of them.
    The per-channel affine map (mean/std) that aligns edited→reference over
    the keep region estimates the generator's global tone drift; applying it
    to the whole edited image aligns the fill region too, so the composite
    boundary doesn't show a tonal seam.
    """
    if edited.size != reference.size or fill_mask.size != reference.size:
        return edited

    e = np.asarray(edited.convert("RGB"), dtype=np.float32)
    r = np.asarray(reference.convert("RGB"), dtype=np.float32)
    keep = 1.0 - np.asarray(fill_mask.convert("L"), dtype=np.float32) / 255.0
    total = keep.sum()
    if total < 1024.0:  # not enough known pixels for a stable estimate
        return edited

    w = keep[..., None]

    def _stats(a):
        mean = (a * w).sum(axis=(0, 1)) / total
        var = (((a - mean) ** 2) * w).sum(axis=(0, 1)) / total
        return mean, np.sqrt(np.maximum(var, 1e-6))

    e_mean, e_std = _stats(e)
    r_mean, r_std = _stats(r)
    # A near-uniform keep region gives no reliable drift estimate.
    if float(e_std.mean()) < 1.0 or float(r_std.mean()) < 1.0:
        return edited
    gain = np.clip(r_std / e_std, 0.5, 2.0)
    out = (e - e_mean) * gain + r_mean
    return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8))


def dilate_mask(mask: Image.Image, px: int) -> Image.Image:
    """Grow the white (fill) region of an L-mode mask by ~px pixels.

    Used for shape-changing/removal inpaints: a mask that hugs an object's
    silhouette anchors the model to regenerate the same shape; growing it
    gives the fill room to replace the shape with something else.
    """
    px = int(px)
    if px <= 0:
        return mask
    if mask.mode != "L":
        mask = mask.convert("L")
    size = 2 * min(px, 64) + 1
    return mask.filter(ImageFilter.MaxFilter(size))


def prepare_outpaint_canvas(
    image: Image.Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
    overlap: int = 20,
    fill_color: tuple[int, int, int] = (127, 127, 127),
) -> tuple[Image.Image, Image.Image]:
    """Paste image on an expanded canvas and build the outpaint mask.

    Returns (canvas, mask): the canvas has the source pasted at
    (left, top) over a neutral background, and the L-mode mask is white
    over the new border area (fill) and black over the original image
    region (keep) — ready to feed to the inpaint backends.

    On each padded side the fill region extends *overlap* pixels into the
    original image (the upstream outpaint workflows do this via
    ImagePadForOutpaint feathering + threshold). The model regenerates
    that strip so tone and structures blend across the transition, and
    the downstream feathered composite lands on content-vs-content
    instead of mixing generated pixels with the neutral padding — which
    otherwise shows as a straight seam along the image edge.
    """
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    if min(left, right, top, bottom) < 0:
        raise ValueError("outpaint padding must be non-negative")
    if (left + right + top + bottom) == 0:
        raise ValueError("outpaint requires at least one side with padding > 0")

    src = image.convert("RGB")
    w, h = src.size
    # Overlap only applies on padded sides and must leave a keep region.
    overlap = max(0, int(overlap))
    max_overlap_x = max(0, (w - 8) // 2)
    max_overlap_y = max(0, (h - 8) // 2)
    ov_left = min(overlap, max_overlap_x) if left > 0 else 0
    ov_right = min(overlap, max_overlap_x) if right > 0 else 0
    ov_top = min(overlap, max_overlap_y) if top > 0 else 0
    ov_bottom = min(overlap, max_overlap_y) if bottom > 0 else 0

    canvas = Image.new("RGB", (w + left + right, h + top + bottom), fill_color)
    canvas.paste(src, (left, top))
    mask = Image.new("L", canvas.size, 255)
    mask.paste(
        0,
        (
            left + ov_left,
            top + ov_top,
            left + w - ov_right,
            top + h - ov_bottom,
        ),
    )
    return canvas, mask


def composite_mask(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    *,
    feather_radius: int = 0,
) -> Image.Image:
    """Composite edited pixels over original using mask white=edit."""
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)
    mask = mask.convert("L")

    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=int(feather_radius)))

    return Image.composite(edited, original, mask)
