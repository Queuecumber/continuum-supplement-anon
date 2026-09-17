"""SAM3 image segmentation for masked editing.

Provides object selection via text descriptions using Meta's SAM3 PCS model
and point/box visual prompts using the SAM3 tracker/PVS model.
Lazy-loaded to avoid VRAM usage until needed. Integrated with ModelManager
for eviction support (single-GPU: evict before loading, dual-GPU: TBD).

Requires transformers >= 5.0.0 (SAM3 support added in v5.0.0).
Gated model — requires `huggingface-cli login` with access to facebook/sam3.
~848M params, ~4GB VRAM at inference.
"""

import gc

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

from continuum.progress import ProgressCallback, emit_progress, patch_tqdm_progress


class SAM3Segmenter:
    """Text-mode segmentation using SAM3 (Segment Anything Model 3).

    Accepts a text description and returns a combined binary mask of all
    matching instances as a PIL Image (mode 'L').

    Supports caching vision embeddings so multiple text prompts on the
    same image skip redundant ViT forward passes.
    """

    MODEL_ID = "facebook/sam3"

    def __init__(self, device: str = "auto", model_path: str | None = None):
        self._model = None
        self._processor = None
        self._device = device
        self._model_path = model_path or self.MODEL_ID
        # Cached vision embeddings for multi-prompt efficiency
        self._cached_vision_embeds = None
        self._cached_original_sizes = None
        self._cached_image_id: int | None = None  # id() of the PIL Image

    def load(self, load_progress: ProgressCallback | None = None) -> None:
        self._load(load_progress=load_progress)

    def _load(self, load_progress: ProgressCallback | None = None) -> None:
        if self._model is not None:
            return

        emit_progress(load_progress, "setting up model", 0, None)
        if self._device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        with patch_tqdm_progress(load_progress, "loading processor"):
            self._processor = Sam3Processor.from_pretrained(self._model_path)
        with patch_tqdm_progress(load_progress, "loading model"):
            self._model = Sam3Model.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        self._model = self._model.to(self._device)

    def segment(
        self,
        image: Image.Image,
        description: str,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> Image.Image:
        """Text-mode segmentation. Returns combined binary mask as PIL Image.

        All instances matching the description are merged into one mask.

        Args:
            image: PIL Image to segment (RGB).
            description: Text description of the object to select
                (e.g. 'the red car', 'her hair', 'sky').
            threshold: Score threshold for filtering instances.
            mask_threshold: Binarization threshold for mask logits.

        Returns:
            PIL Image in mode 'L' (grayscale), same size as input.
            255 = selected region, 0 = background.
        """
        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")
        img_id = id(image)

        # Reuse cached vision embeddings if same image
        if self._cached_image_id == img_id and self._cached_vision_embeds is not None:
            vision_embeds = self._cached_vision_embeds
            original_sizes = self._cached_original_sizes
            text_inputs = self._processor(
                text=description,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    vision_embeds=vision_embeds,
                    **text_inputs,
                )
        else:
            inputs = self._processor(
                images=image,
                text=description,
                return_tensors="pt",
            ).to(self._device)
            original_sizes = inputs.get("original_sizes")

            with torch.no_grad():
                # Cache vision embeddings for subsequent prompts on same image
                self._cached_vision_embeds = self._model.get_vision_features(
                    pixel_values=inputs["pixel_values"],
                )
                self._cached_image_id = img_id
                self._cached_original_sizes = original_sizes

                outputs = self._model(
                    vision_embeds=self._cached_vision_embeds,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

        # Post-process into instance masks
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=original_sizes.tolist(),
        )[0]

        masks = results["masks"]  # (N, H, W) binary tensor

        if len(masks) == 0:
            # No instances found — return empty mask
            return Image.new("L", image.size, 0)

        # Merge all instances into one combined mask
        combined = masks.any(dim=0).cpu().numpy().astype(np.uint8) * 255
        return Image.fromarray(combined, mode="L")

    def clear_cache(self) -> None:
        """Clear cached vision embeddings (call when switching images)."""
        self._cached_vision_embeds = None
        self._cached_original_sizes = None
        self._cached_image_id = None

    def unload(self) -> None:
        """Free model from memory."""
        self.clear_cache()
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        try:
            if self._device and self._device not in ("auto", "cpu"):
                with torch.cuda.device(self._device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        except Exception:
            pass


class SAM3TrackerSegmenter:
    """Point/box segmentation using SAM3 Tracker (PVS).

    This is the SAM2-style promptable visual segmentation path. It is
    separate from SAM3Segmenter because the text PCS and visual PVS models
    are different architectures even though they share the facebook/sam3 repo.
    """

    MODEL_ID = "facebook/sam3"

    def __init__(self, device: str = "auto", model_path: str | None = None):
        self._model = None
        self._processor = None
        self._device = device
        self._model_path = model_path or self.MODEL_ID

    def load(self, load_progress: ProgressCallback | None = None) -> None:
        self._load(load_progress=load_progress)

    def _load(self, load_progress: ProgressCallback | None = None) -> None:
        if self._model is not None:
            return

        emit_progress(load_progress, "setting up model", 0, None)
        if self._device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        with patch_tqdm_progress(load_progress, "loading processor"):
            self._processor = Sam3TrackerProcessor.from_pretrained(self._model_path)
        with patch_tqdm_progress(load_progress, "loading model"):
            self._model = Sam3TrackerModel.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16,
            )
        emit_progress(load_progress, "setting up model", 0, None)
        self._model = self._model.to(self._device)

    @staticmethod
    def _scale_point(point: list[float], size: tuple[int, int], normalized: bool) -> list[float]:
        if len(point) != 2:
            raise ValueError(f"point must be [x, y], got {point!r}")
        x, y = float(point[0]), float(point[1])
        if normalized:
            w, h = size
            x *= w
            y *= h
        return [x, y]

    @staticmethod
    def _scale_box(box: list[float], size: tuple[int, int], normalized: bool) -> list[float]:
        if len(box) != 4:
            raise ValueError(f"box must be [x1, y1, x2, y2], got {box!r}")
        x1, y1, x2, y2 = [float(v) for v in box]
        if normalized:
            w, h = size
            x1 *= w
            x2 *= w
            y1 *= h
            y2 *= h
        return [x1, y1, x2, y2]

    @staticmethod
    def _combine_masks(masks, outputs, mask_index: int | None) -> Image.Image:
        masks = masks.detach().cpu()
        scores = getattr(outputs, "iou_scores", None)
        if scores is not None:
            scores = scores.detach().cpu()

        if masks.ndim == 2:
            selected = masks.unsqueeze(0)
        elif masks.ndim == 3:
            # Single object with multiple candidates, or multiple objects
            # with one candidate. Prefer IoU if it disambiguates candidates.
            if (
                scores is not None
                and scores.ndim == 3
                and scores.shape[1] == 1
                and scores.shape[2] == masks.shape[0]
            ):
                idx = int(mask_index) if mask_index is not None else int(scores[0, 0].argmax())
                idx = max(0, min(idx, masks.shape[0] - 1))
                selected = masks[idx].unsqueeze(0)
            else:
                selected = masks
        elif masks.ndim == 4:
            num_objects, num_candidates = masks.shape[:2]
            if mask_index is not None:
                idx = torch.full(
                    (num_objects,),
                    max(0, min(int(mask_index), num_candidates - 1)),
                    dtype=torch.long,
                )
            elif scores is not None and scores.ndim == 3:
                idx = scores[0].argmax(dim=-1).to(torch.long)
                idx = idx.clamp(0, num_candidates - 1)
            else:
                idx = torch.zeros(num_objects, dtype=torch.long)
            selected = masks[torch.arange(num_objects), idx]
        else:
            raise ValueError(f"unexpected SAM3 tracker mask shape: {tuple(masks.shape)}")

        combined = selected.float().gt(0).any(dim=0).numpy().astype(np.uint8) * 255
        return Image.fromarray(combined, mode="L")

    def segment_visual(
        self,
        image: Image.Image,
        *,
        input_points: list | None = None,
        input_labels: list | None = None,
        input_boxes: list | None = None,
        multimask_output: bool = True,
        mask_index: int | None = None,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        apply_non_overlapping_constraints: bool = False,
    ) -> Image.Image:
        """Run SAM3 Tracker with already-batched visual prompts."""
        if input_points is None and input_boxes is None:
            raise ValueError("provide input_points or input_boxes")

        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(
                **inputs,
                multimask_output=bool(multimask_output),
            )

        original_sizes = inputs["original_sizes"]
        if hasattr(original_sizes, "cpu"):
            original_sizes = original_sizes.cpu()

        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes,
            mask_threshold=float(mask_threshold),
            binarize=True,
            max_hole_area=float(max_hole_area),
            max_sprinkle_area=float(max_sprinkle_area),
            apply_non_overlapping_constraints=bool(apply_non_overlapping_constraints),
        )[0]
        return self._combine_masks(masks, outputs, mask_index)

    def segment_points(
        self,
        image: Image.Image,
        points: list[list[float]],
        *,
        labels: list[int] | None = None,
        negative_points: list[list[float]] | None = None,
        normalized: bool = False,
        multimask_output: bool = True,
        mask_index: int | None = None,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        apply_non_overlapping_constraints: bool = False,
    ) -> Image.Image:
        """Segment one target from positive/negative point clicks."""
        if not points:
            raise ValueError("points must contain at least one [x, y] point")
        scaled_points = [self._scale_point(point, image.size, normalized) for point in points]
        if labels is None:
            point_labels = [1] * len(scaled_points)
        else:
            if len(labels) != len(scaled_points):
                raise ValueError("labels length must match points length")
            point_labels = [int(label) for label in labels]

        if negative_points:
            scaled_points.extend(
                self._scale_point(point, image.size, normalized) for point in negative_points
            )
            point_labels.extend([0] * len(negative_points))

        return self.segment_visual(
            image,
            input_points=[[scaled_points]],
            input_labels=[[point_labels]],
            multimask_output=multimask_output,
            mask_index=mask_index,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
            apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        )

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        *,
        normalized: bool = False,
        multimask_output: bool = False,
        mask_index: int | None = None,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        apply_non_overlapping_constraints: bool = False,
    ) -> Image.Image:
        """Segment and merge one or more box-prompted targets."""
        if not boxes:
            raise ValueError("boxes must contain at least one [x1, y1, x2, y2] box")
        scaled_boxes = [self._scale_box(box, image.size, normalized) for box in boxes]
        return self.segment_visual(
            image,
            input_boxes=[scaled_boxes],
            multimask_output=multimask_output,
            mask_index=mask_index,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
            apply_non_overlapping_constraints=apply_non_overlapping_constraints,
        )

    def unload(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        try:
            if self._device and self._device not in ("auto", "cpu"):
                with torch.cuda.device(self._device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        except Exception:
            pass


def composite_masked_edit(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    feather_radius: int = 5,
) -> Image.Image:
    """Composite edited image onto original using a feathered mask.

    Post-hoc compositing strategy: runs a full-image edit, then blends
    only the masked region back onto the working image.

    Args:
        original: PIL Image — the current working image.
        edited: PIL Image — the full-image edit result.
        mask: PIL Image mode 'L' — binary mask (255 = edit region).
        feather_radius: Gaussian blur radius for mask edge feathering.

    Returns:
        PIL Image with edited content blended into masked region.
    """
    # Ensure all images are the same size
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.LANCZOS)

    # Feather mask edges for smooth blending
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    # Composite: original * (1 - mask) + edited * mask
    return Image.composite(edited, original, mask)
