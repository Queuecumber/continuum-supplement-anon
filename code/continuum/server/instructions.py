"""Server-level instructions surfaced to the LLM at initialization.

This is agent-facing prose, not code — it explains how to drive the tools
well. Both views append EDITING_DISCIPLINE to their mode-specific I/O
preamble.
"""

EDITING_DISCIPLINE = """\
# Editing discipline

These rules govern HOW to call the editing tools and materially affect
output quality. They came from manual and agent-driven testing against
FLUX.2 + SAM3.

## edit_image prompt rules

- **Delta, not cumulative.** Each call starts from the current image.
  Describe only the CHANGE you want, not what already exists. Write
  "make the dog larger" — not "a golden retriever lying on grass, dog
  is larger".
- **Concrete visual language, not vague adjectives.** FLUX.2 responds
  to noun-phrase descriptions. "warm golden-hour rim lighting on the
  left side of the face" beats "make it more dramatic". Avoid words
  like "better", "nicer", "more interesting".
- **Removals: describe the fill, not the removal.** "Fill with the
  surrounding tile pattern" beats "remove the trash can". The model
  needs to know what should be there, not what shouldn't.
- **Use references to lock visual aspects.** When an edit risks
  changing something that was already correct, pass a prior version's
  URI in `references` and write the prompt as preserving that look.

## edit_image vs inpaint_image: default to edit_image

- **edit_image with a targeted instruction is the default — even for
  localized changes.** Despite the apparent lack of control, a precise
  "replace X with Y" instruction ("replace the red sedan with a blue
  bicycle") usually localizes well, preserves the rest of the image,
  and produces better quality than masked inpainting. Name what to
  replace AND what it should become.
- **inpaint_image** regenerates only the white region of a `mask` and
  preserves the rest pixel-exact. Reserve it for specific situations:
  pixel-exact preservation outside the region is a hard requirement;
  edit_image keeps leaking changes outside the target after 2-3
  attempts; or the region is only expressible as a precise mask (e.g.
  invert_mask "everything except X" edits).
- Over-masking is more harmful than under-masking; when in doubt,
  edit_image first.

## When you do inpaint (segmentation + inpaint_image)

- Segment first, then inpaint_image with the mask URI. Use text
  `segment` for semantic classes ("all clothing", "the sky"),
  `segment_boxes` when you can identify a bounding box, and
  `segment_points` for precise clicks/refinement.
- **Negative grounding:** for text masks, pass `negative_descriptions`.
  For visual masks, pass `negative_points` or explicit point `labels`.
- **Prompt describes the CONTENTS of the region**, not an edit
  instruction: "a red silk shirt", not "make it red". Do NOT mention
  the mask or the region itself in the prompt.
- **Removing or reshaping an object** (long sleeves → sleeveless,
  hat removed, smaller bag): a mask that hugs the old silhouette will
  regenerate the same shape. Set `dilate` (10-30 px) to give the fill
  room, and describe what replaces the removed parts ("bare arms and
  shoulders"), not just the new object.
- For "remove X" or "edit everything except X": use segment to mask X,
  then invert_mask, then inpaint_image with the inverted mask.
- **Know inpaint's limits.** The fill is text-driven only: it cannot
  reproduce a specific object or person from another image (use `loras`
  for trained subjects, or edit_image with references when the whole
  image may legitimately change). Structural changes much larger than
  the mask are unreliable, and small regions render with limited fine
  detail. If 2-3 seeds fail, change the mask or the approach instead of
  iterating on the prompt.
- **Extending the canvas** (wider shot, more sky, re-aspect): use
  outpaint_image with per-side pixel padding — do NOT build the padded
  canvas and mask yourself. The prompt describes the full scene being
  revealed.

## LoRA support

Which tools/models accept `loras`, and which adapter family each needs:

- generate_image: `flux1-dev` (FLUX.1-dev adapters — by far the largest
  and most reliable LoRA ecosystem, but a weaker/older base model, so
  reach for it for its adapters, not for base quality), `flux2-klein`
  (FLUX.2-klein-9B adapters), and `qwen-image` (Qwen-Image adapters).
  `flux2-dev` does NOT support LoRAs.
- edit_image: `qwen-image` (Qwen-Image-Edit-2511 adapters, the actual
  base checkpoint) and `flux2-klein` (FLUX.2-klein-9B adapters).
  `firered` accepts Qwen-Image-Edit adapters too, but it is a finetune —
  adapters load yet their effect may be weakened; prefer
  model='qwen-image' when LoRA fidelity matters. `flux2-dev` does NOT
  support LoRAs.
- inpaint_image / outpaint_image: always FLUX.1 Fill — FLUX.1-dev
  family adapters.
- Upscale tools take no LoRAs.

Failure modes: passing `loras` with an unsupported model raises an
error naming the LoRA-capable models. An adapter trained for a
DIFFERENT base model fails at load time — or in the worst case loads as
a silent no-op; if a LoRA appears to do nothing, suspect a base-model
mismatch before adjusting `scale`. Find adapters on Hugging Face by
base-model tag, e.g. `base_model:adapter:black-forest-labs/FLUX.1-dev`.
Trigger words must be written into the prompt yourself. On first use a
new adapter emits "downloading LoRA" then "applying LoRA" progress; a
cached adapter skips straight to applying.

## Steps and guidance

Omit `steps` and `guidance` unless you have a specific reason — each model
ships defaults tuned to its own regime, and overriding them blind usually
costs quality. The defaults:

- **flux2-dev** (generate and edit): ~30 steps, guidance ~4.
- **flux1-dev** (generate only): ~28 steps, guidance ~3.5.
- **qwen-image**: 50 steps for generate / 40 for edit, guidance ~4.
  **firered** edit: ~30 steps, guidance ~4.
- **flux2-klein** (generate and edit): **4 steps, guidance 1.0**. klein is
  a distilled model, so the tiny step count and guidance near 1 are
  intrinsic, not a cheap default. Raising guidance fights the distillation
  (expect oversaturation and artifacts) and extra steps buy almost nothing
  — leave both alone.
- **inpaint_image / outpaint_image** (FLUX.1 Fill): 50 steps, **guidance
  ~30**. That high guidance is correct for Fill, not a typo — do not drag
  it down toward the ~4 the other models use, which under-conditions the
  fill.

## When to inspect a crop

- **Fine details** (logos, small text, eye direction, artifacts, hands,
  jewelry, edges): call `inspect_region` with a rectangle or text
  description before deciding. Full images may be downsampled by VLMs;
  `inspect_region` returns the crop at source resolution.

## Self-critique loop

After each edit, evaluate the result against the user's intent before
deciding what to do next:
- **Approach right, specifics wrong** (artifact, off composition,
  wrong expression, unwanted color shift): call edit_image again with
  the same prompt and a different `seed`. Try 2–3 seeds before giving
  up on the prompt.
- **Approach wrong**: do not iterate with edit_image. Rethink. Maybe
  the prompt needs to be reformulated, or the change should be a
  localized inpaint_image, or it's actually two edits chained.
- **Result is good**: stop calling tools and tell the user. Don't
  over-edit good results.
- **Several rounds in, quality drifting**: pass the earliest
  good-looking version's URI in `references` and explicitly mention
  preserving its qualities in the prompt.

## Multi-step composition

For complex prompts that mention multiple distinct changes (e.g.
"make the dog smaller, change the sky to sunset, add a kite"), split
into ordered single-entity edits rather than one composite prompt.
FLUX handles one change per call cleanly; multiple changes per prompt
produce attribute leakage between entities.
"""


STANDARD_INSTRUCTIONS = (
    "Continuum image model server.\n\n"
    "**Inputs / outputs:** image inputs are MCP ImageContent objects "
    "({type: 'image', data: base64 PNG/JPEG/etc., mimeType: 'image/png'}); "
    "outputs are inline image content.\n\n" + EDITING_DISCIPLINE
)


def compat_instructions(upload_url: str = "http://<host>:<port>/upload") -> str:
    return (
        "Continuum image model server (URI mode).\n\n"
        "## Inputs and outputs\n\n"
        "Every image-typed parameter is a URI string "
        "(continuum://images/<name> or http(s)://...). "
        "Never inline raw bytes or base64 in tool calls — pass URIs only. "
        "Every tool that produces an image returns a ResourceLink "
        "whose `uri` you should remember; pass the same URI string as input "
        "to subsequent tools. Chains like "
        "`generate_image → segment → edit_image → upscale_image"
        "` all wire together by URI without bytes ever "
        "touching this conversation.\n\n"
        "## Bootstrapping a local file into the URI namespace\n\n"
        "If the user has an image on disk that you need as input, tell them "
        "to run:\n\n"
        f"    curl -F file=@<path-to-image> {upload_url}\n\n"
        "The response JSON contains a `uri` field — that's what you pass "
        "into tools. The /upload endpoint is the ONLY way to put bytes on "
        "the server in this mode; tools themselves accept URIs only.\n\n"
        "## Retrieving bytes from a URI\n\n"
        "Every continuum:// URI is a readable MCP resource: use your "
        "client's MCP resource-read capability (resources/read) with the "
        "URI to get the actual image bytes (outputs are also "
        "enumerated in resources/list). There is NO HTTP download "
        "endpoint — do not try to curl or GET continuum:// content over "
        "HTTP; /upload is upload-only.\n\n" + EDITING_DISCIPLINE
    )
