"""Compat view: URI-only I/O for hosts that can't put bytes in the LLM
conversation.

Image inputs are URI strings; outputs are ResourceLinks under
continuum://images/{name}, served from
/tmp/continuum-mcp/. POST /upload bootstraps bytes into the image URI
namespace via curl.

Tools here are thin: resolve URIs to domain inputs, bundle progress hooks,
call the core op, persist and register the output. All defaults,
validation, and model routing live in continuum.core.ops. Tool docstrings
double as the MCP descriptions; the standard view carries the same prose
with inline-content I/O wording — keep the two in step when editing.
Parameters annotated with Field descriptions in schema.py (model, roi,
loras) are documented there, not in Args sections.
"""

import re
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ResourceLink
from PIL.Image import Image as PILImage
from starlette.responses import JSONResponse

from continuum.core import media, ops
from continuum.models import DEFAULT_EDIT_MODEL, DEFAULT_GENERATE_MODEL
from continuum.server import instructions, resources
from continuum.server.bridge import log_tool_errors, make_hooks
from continuum.server.schema import (
    EditImageModel,
    GenerateImageModel,
    InspectRegionRoi,
    LoraList,
)


def _to_pil(uri: str, mode: str = "RGB") -> PILImage:
    return media.bytes_to_pil(resources.fetch_uri(uri), mode)


def _to_pil_list(uris: list[str] | None, mode: str = "RGB"):
    return [_to_pil(uri, mode) for uri in uris] if uris else None


mcp = FastMCP("continuum", instructions=instructions.compat_instructions())


def set_upload_url(upload_url: str) -> None:
    """Rewrite the server instructions with the real /upload URL.

    The module-level server is built at import with a placeholder URL;
    the CLI calls this once the bind address is known.
    """
    mcp._mcp_server.instructions = instructions.compat_instructions(upload_url)


def _save_image(img: PILImage, name_hint: str = "out") -> ResourceLink:
    link = resources.save_image_as_resource(img, name_hint)
    resources.register_file_resource(
        mcp,
        resources.IMAGES_DIR / link.name,
        str(link.uri),
        link.mimeType,
    )
    return link


@mcp.tool()
@log_tool_errors
async def generate_image(
    prompt: str,
    ctx: Context,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    model: GenerateImageModel = DEFAULT_GENERATE_MODEL,
    loras: LoraList = None,
) -> ResourceLink:
    """Generate a new image from a text prompt.

    Args:
        prompt: text description of the image to generate.
        width: output width in pixels (default 1024).
        height: output height in pixels (default 1024).
        steps: diffusion steps; omit for the model default.
        guidance: guidance scale; omit for the model default.
        seed: random seed; randomized if omitted.
        model: generation model (default flux2-dev — medium cost, more
            creative). flux2-klein and qwen-image are cheap with high
            prompt adherence. flux1-dev is a weaker/older base model but
            has by far the largest LoRA ecosystem — reach for it when you
            need a specific adapter, not for base quality.
            flux2-klein/qwen-image/flux1-dev take LoRAs; flux2-dev does
            not.
        loras: LoRA adapters to stack. flux2-klein takes FLUX.2-klein-9B
            adapters and qwen-image takes Qwen-Image adapters; flux2-dev
            does NOT support LoRAs (raises ValueError). Put any trigger
            words in the prompt.

    Returns:
        ResourceLink with a continuum://images/<name> URI; pass it
        straight into the next tool that takes an image.
    """
    img = await ops.generate_image(
        prompt,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        model=model,
        loras=loras,
        hooks=make_hooks(ctx, "generate_image"),
    )
    return _save_image(img, "generate")


@mcp.tool()
@log_tool_errors
async def edit_image(
    image: str,
    prompt: str,
    ctx: Context,
    references: list[str] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    model: EditImageModel = DEFAULT_EDIT_MODEL,
    loras: LoraList = None,
) -> ResourceLink:
    """Edit an image using a text instruction (whole-image, no mask —
    for masked/region edits use `inpaint_image`).

    Describe only the CHANGE you want, in concrete visual language.
    Omit `steps`, `guidance`, `width`, and `height` unless you need to
    override the model defaults.

    Args:
        image: URI of the image to edit (continuum://images/<name>
            from a previous tool output or /upload, or http(s)://).
        prompt: the edit instruction.
        references: URIs of additional images whose visual details the
            edit should preserve or borrow.
        steps: diffusion steps; omit for the model default.
        guidance: guidance scale; omit for the model default.
        seed: random seed; randomized if omitted.
        width: output width; omit for the model default.
        height: output height; omit for the model default.
        model: editing model (default firered — cheap, good for most
            edits). qwen-image and flux2-klein are also cheap/high
            adherence; flux2-dev is medium cost, more creative.
            firered/qwen-image/flux2-klein take LoRAs; flux2-dev does
            not.
        loras: LoRA adapters to stack. qwen-image takes
            Qwen-Image-Edit-2511 adapters (the true base — prefer it when
            LoRA fidelity matters); firered loads the same adapters but,
            as a finetune, may weaken their effect; flux2-klein takes
            FLUX.2-klein-9B adapters; flux2-dev does NOT support LoRAs
            (raises ValueError). Put any trigger words in the prompt.

    Returns:
        ResourceLink with a continuum://images/<name> URI; pass it
        straight into the next tool that takes an image.
    """
    out = await ops.edit_image(
        _to_pil(image),
        prompt,
        references=_to_pil_list(references),
        steps=steps,
        guidance=guidance,
        seed=seed,
        width=width,
        height=height,
        model=model,
        loras=loras,
        hooks=make_hooks(ctx, "edit_image"),
    )
    return _save_image(out, "edit")


@mcp.tool()
@log_tool_errors
async def inpaint_image(
    image: str,
    mask: str,
    prompt: str,
    ctx: Context,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    dilate: int = 0,
    loras: LoraList = None,
) -> ResourceLink:
    """Inpaint the masked region of an image with FLUX.1 Fill.

    White mask pixels are regenerated; black pixels are preserved
    pixel-exact. Write the prompt as a description of the desired
    CONTENTS of the region (e.g. 'a red silk shirt'), not an edit
    instruction ('make it red').

    Prefer edit_image first: a targeted instruction naming what to
    replace and what it should become usually localizes well and gives
    better quality than inpainting. Use this tool when pixel-exact
    preservation outside the region is required, when edit_image keeps
    leaking changes outside the target, or when the region is only
    expressible as a precise mask.

    Limitations: the fill is driven by TEXT ONLY — it cannot copy a
    specific object, person, or pattern from another image (use `loras`
    for trained subjects, or edit_image with references when the whole
    image may change). Structural changes much larger than the mask are
    unreliable even with `dilate`, and small regions come back with
    limited fine detail. If 2-3 seeds fail, rework the mask or switch
    approach rather than iterating.

    Args:
        image: URI of the image to inpaint.
        mask: URI of the grayscale mask; white = regenerate,
            black = preserve.
        prompt: description of the desired contents of the masked
            region.
        steps: diffusion steps (default 50).
        guidance: guidance scale (default 30.0).
        seed: random seed; randomized if omitted.
        dilate: grow the fill region by ~N px before inpainting — use
            10-30 for shape-changing/removal edits so the mask
            silhouette doesn't anchor the old shape, and describe what
            replaces the removed parts (e.g. 'bare arms').
        loras: FLUX.1-dev LoRA adapters (FLUX.1 Fill takes FLUX.1-dev
            adapters). Put any trigger words in the prompt.

    Returns:
        ResourceLink with a continuum://images/<name> URI; pass it
        straight into the next tool that takes an image.
    """
    out = await ops.inpaint_image(
        _to_pil(image),
        _to_pil(mask, "L"),
        prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
        dilate=dilate,
        loras=loras,
        hooks=make_hooks(ctx, "inpaint_image"),
    )
    return _save_image(out, "inpaint")


@mcp.tool()
@log_tool_errors
async def outpaint_image(
    image: str,
    prompt: str,
    ctx: Context,
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    overlap: int = 20,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    loras: LoraList = None,
) -> ResourceLink:
    """Extend an image beyond its borders (outpainting) with FLUX.1 Fill.

    Pads the canvas and generates the new border area to match the
    prompt. Write the prompt as a description of the full scene being
    revealed (e.g. 'a wide sandy beach under a sunset sky'), not an
    edit instruction.

    Args:
        image: URI of the image to extend.
        prompt: description of the full scene being revealed.
        left/right/top/bottom: pixels to add on each side; at least
            one side must be > 0.
        overlap: pixels of the original border regenerated on padded
            sides so tone and structures blend across the transition
            (0 disables; risks a visible seam along the image edge).
        steps: diffusion steps (default 50).
        guidance: guidance scale (default 30.0).
        seed: random seed; randomized if omitted.
        loras: FLUX.1-dev LoRA adapters. Put any trigger words in the
            prompt.

    Returns:
        ResourceLink with a continuum://images/<name> URI at the
        expanded canvas size.

    Raises:
        ValueError: if no side has padding > 0 or padding is negative.
    """
    out = await ops.outpaint_image(
        _to_pil(image),
        prompt,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        overlap=overlap,
        steps=steps,
        guidance=guidance,
        seed=seed,
        loras=loras,
        hooks=make_hooks(ctx, "outpaint_image"),
    )
    return _save_image(out, "outpaint")


@mcp.tool()
@log_tool_errors
async def inspect_region(
    image: str,
    roi: InspectRegionRoi,
    ctx: Context,
    normalized: bool = False,
    padding: float = 0.0,
    min_size: int = 1,
    threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> ResourceLink:
    """Crop a region out of an image for full-resolution visual
    inspection.

    Use this when a VLM may have downsampled the full image and needs a
    small cutout to inspect details (logos, small text, hands, edges).

    Args:
        image: URI of the image to crop from.
        roi: region to crop. Pass a text description such as 'the logo'
            to segment with SAM, a rectangle as [x1, y1, x2, y2], or an
            object with x/y/width/height or x1/y1/x2/y2. Rectangle
            coordinates are pixels unless normalized=true.
        normalized: treat rectangle coordinates as 0-1 fractions of
            image width/height instead of absolute pixels.
        padding: extra context around the crop — pixels, or a fraction
            of each crop dimension when between 0 and 1.
        min_size: minimum crop side length in pixels.
        threshold: SAM3 instance filter (text roi only).
        mask_threshold: SAM3 mask binarization (text roi only).

    Returns:
        ResourceLink with a continuum://images/<name> URI of the crop
        at source resolution.
    """
    hooks = make_hooks(ctx, "inspect_region") if ctx is not None else None
    crop = await ops.inspect_region(
        _to_pil(image),
        roi,
        normalized=normalized,
        padding=padding,
        min_size=min_size,
        threshold=threshold,
        mask_threshold=mask_threshold,
        hooks=hooks,
    )
    if ctx is not None:
        await ctx.report_progress(1, 1, "done")
    return _save_image(crop, "inspect")


@mcp.tool()
@log_tool_errors
async def segment(
    image: str,
    description: str,
    ctx: Context,
    threshold: float = 0.3,
    mask_threshold: float = 0.5,
    negative_descriptions: list[str] | None = None,
) -> ResourceLink:
    """Run text-driven concept segmentation (SAM3 PCS) on an image.

    Returns a mask you can pass to inpaint_image's `mask` parameter
    for constrained inpainting.

    Args:
        image: URI of the image to segment.
        description: text description of the region to segment, e.g.
            "all clothing" or "the sky".
        threshold: filters detected instances.
        mask_threshold: binarizes masks.
        negative_descriptions: segmented separately and subtracted
            from the positive mask.

    Returns:
        ResourceLink with a continuum://images/<name> URI of the mask
        (grayscale PNG, white = matching region).
    """
    mask = await ops.segment(
        _to_pil(image),
        description,
        threshold=threshold,
        mask_threshold=mask_threshold,
        negative_descriptions=negative_descriptions,
        hooks=make_hooks(ctx, "segment"),
    )
    await ctx.report_progress(1, 1, "done")
    return _save_image(mask, "mask")


@mcp.tool()
@log_tool_errors
async def segment_points(
    image: str,
    points: list[list[float]],
    ctx: Context,
    negative_points: list[list[float]] | None = None,
    labels: list[int] | None = None,
    normalized: bool = False,
    multimask_output: bool = True,
    mask_index: int | None = None,
    mask_threshold: float = 0.0,
    max_hole_area: float = 0.0,
    max_sprinkle_area: float = 0.0,
    apply_non_overlapping_constraints: bool = False,
) -> ResourceLink:
    """Run point-prompt segmentation (SAM3 Tracker / PVS) on an image.

    Args:
        image: URI of the image to segment.
        points: positive [x, y] clicks for one target.
        negative_points: negative clicks for refinement.
        labels: overrides point labels directly (1=positive,
            0=negative).
        normalized: treat coordinates as 0-1 fractions of image
            width/height instead of absolute pixels.
        multimask_output: request candidate masks; by default the
            highest-IoU candidate is selected.
        mask_index: force a specific candidate mask.
        mask_threshold: mask decoding threshold.
        max_hole_area: fill holes up to this area in post-processing.
        max_sprinkle_area: remove specks up to this area.
        apply_non_overlapping_constraints: forwarded to SAM3 tracker
            post-processing.

    Returns:
        ResourceLink with a continuum://images/<name> URI of the mask
        (grayscale PNG, white = selected region).
    """
    mask = await ops.segment_points(
        _to_pil(image),
        points,
        negative_points=negative_points,
        labels=labels,
        normalized=normalized,
        multimask_output=multimask_output,
        mask_index=mask_index,
        mask_threshold=mask_threshold,
        max_hole_area=max_hole_area,
        max_sprinkle_area=max_sprinkle_area,
        apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        hooks=make_hooks(ctx, "segment_points"),
    )
    await ctx.report_progress(1, 1, "done")
    return _save_image(mask, "mask_points")


@mcp.tool()
@log_tool_errors
async def segment_boxes(
    image: str,
    boxes: list[list[float]],
    ctx: Context,
    normalized: bool = False,
    multimask_output: bool = False,
    mask_index: int | None = None,
    mask_threshold: float = 0.0,
    max_hole_area: float = 0.0,
    max_sprinkle_area: float = 0.0,
    apply_non_overlapping_constraints: bool = False,
) -> ResourceLink:
    """Run box-prompt segmentation (SAM3 Tracker / PVS) on an image.

    Args:
        image: URI of the image to segment.
        boxes: [x1, y1, x2, y2] boxes; all resulting object masks are
            merged.
        normalized: treat coordinates as 0-1 fractions of image
            width/height instead of absolute pixels.
        multimask_output: request candidate masks per box.
        mask_index: force a specific candidate mask.
        mask_threshold: mask decoding threshold.
        max_hole_area: fill holes up to this area in post-processing.
        max_sprinkle_area: remove specks up to this area.
        apply_non_overlapping_constraints: forwarded to SAM3 tracker
            post-processing.

    Returns:
        ResourceLink with a continuum://images/<name> URI of the mask
        (grayscale PNG, white = selected region).
    """
    mask = await ops.segment_boxes(
        _to_pil(image),
        boxes,
        normalized=normalized,
        multimask_output=multimask_output,
        mask_index=mask_index,
        mask_threshold=mask_threshold,
        max_hole_area=max_hole_area,
        max_sprinkle_area=max_sprinkle_area,
        apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        hooks=make_hooks(ctx, "segment_boxes"),
    )
    await ctx.report_progress(1, 1, "done")
    return _save_image(mask, "mask_boxes")


@mcp.tool()
@log_tool_errors
async def invert_mask(mask: str) -> ResourceLink:
    """Invert a mask (255 ↔ 0). Pure pixel operation — no model loaded.

    Args:
        mask: URI of the grayscale mask to invert.

    Returns:
        ResourceLink with a continuum://images/<name> URI of the
        inverted mask.
    """
    return _save_image(ops.invert_mask(_to_pil(mask, "L")), "inverted")


@mcp.tool()
@log_tool_errors
async def upscale_image(
    image: str,
    ctx: Context,
    scale: float | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    steps: int = 20,
    guidance: float = 5.0,
    target_tile_size: int = 1024,
    overlap: float = 0.5,
) -> ResourceLink:
    """Super-resolution upscale of an image (FaithDiff).


    Specify either `scale` or both `target_width` and `target_height`;
    if both are given, target dimensions win. Tile-based; reports
    progress per-tile.

    Args:
        image: URI of the image to upscale.
        scale: size multiplier like 2 or 4.
        target_width: explicit output width in pixels.
        target_height: explicit output height in pixels.
        steps: diffusion steps per tile.
        guidance: guidance scale.
        target_tile_size: SR tile size in pixels.
        overlap: tile overlap fraction.

    Returns:
        ResourceLink with a continuum://images/<name> URI at the
        target size.

    Raises:
        ValueError: if neither `scale` nor both target dimensions are
            provided.
    """
    out = await ops.upscale_image(
        _to_pil(image),
        scale=scale,
        target_width=target_width,
        target_height=target_height,
        steps=steps,
        guidance=guidance,
        target_tile_size=target_tile_size,
        overlap=overlap,
        hooks=make_hooks(ctx, "upscale_image"),
    )
    return _save_image(out, "upscale_image")


# ---- Resource handlers + HTTP /upload endpoint -----------------------


@mcp.resource("continuum://images/{name}", mime_type="image/png")
def _read_image(name: str) -> bytes:
    """Serve images saved under /tmp/continuum-mcp/images/."""
    path = resources.IMAGES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"image {name!r} not found")
    return path.read_bytes()


@mcp.custom_route("/upload", methods=["POST"])
async def _upload(request):
    """Out-of-band image upload that bypasses MCP entirely.

    POST a single 'file' multipart field; receive JSON containing
    a continuum://images/<name> URI. The way to bootstrap bytes
    into the URI namespace without inlining them in the LLM
    conversation::

        curl -F file=@input.jpg http://gpu-host:8742/upload
    """
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "missing 'file' field"}, status_code=400)

    data = await upload.read()
    try:
        img = media.bytes_to_pil(data, "RGB")
    except Exception as e:
        return JSONResponse({"error": f"not a valid image: {e}"}, status_code=400)

    # Use the uploaded filename's stem (stripped of path + extension,
    # sanitized) so the resulting URI is something the user can
    # recognize. Hash suffix preserved to avoid collisions when the
    # same name is uploaded twice.
    raw_name = Path(getattr(upload, "filename", "") or "").name
    stem = Path(raw_name).stem
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:50] or "upload"
    link = _save_image(img, stem)
    return JSONResponse(
        {
            "uri": str(link.uri),
            "name": link.name,
            "size": link.size,
            "mimeType": link.mimeType,
        }
    )


resources.scan_existing_resources(mcp)


if __name__ == "__main__":
    mcp.run()
