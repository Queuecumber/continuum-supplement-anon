"""Model manager: load on demand, evict on mode switch."""

import gc
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
from huggingface_hub import snapshot_download

from continuum.backends.flux1 import Flux1Generator
from continuum.backends.flux2 import Flux2Editor
from continuum.backends.flux_fill import FluxFillInpainter
from continuum.backends.klein import Flux2KleinEditor
from continuum.backends.qwen import FireRedEditor, QwenImageGenerator
from continuum.backends.sr import release_sr_pipeline, upscale_faithdiff
from continuum.gpu import get as get_gpu_config
from continuum.progress import (
    ProgressCallback,
    emit_progress,
    make_hf_snapshot_download_tqdm_class,
    make_silent_tqdm_class,
)
from continuum.segmentation import SAM3Segmenter, SAM3TrackerSegmenter

RAMDISK_CACHE = os.environ.get("CONTINUUM_MODEL_CACHE") or "/tmp/continuum_models"

FLUX2_DEV_REPO = "black-forest-labs/FLUX.2-dev"
# Klein 9B: larger/better than the distilled 4B, and the base model the
# public FLUX.2 LoRA ecosystem targets (LoRAs are architecture-specific, so
# 9B LoRAs will not load into 4B). 4B has been dropped.
FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-9B"
FLUX1_DEV_REPO = "black-forest-labs/FLUX.1-dev"
FLUX1_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
QWEN_IMAGE_REPO = "Qwen/Qwen-Image"
FIRERED_REPO = "FireRedTeam/FireRed-Image-Edit-1.1"
QWEN_IMAGE_EDIT_REPO = "Qwen/Qwen-Image-Edit-2511"
SAM3_REPO = "facebook/sam3"


def _mcp_log(message: str, *, verbose: bool = False) -> None:
    if verbose:
        return
    print(f"[mcp] {message}", flush=True)


def normalize_image_model_name(model: str | None) -> str:
    return (model or "flux2").strip().lower().replace("_", "-")


@dataclass(frozen=True)
class ImageBackendSpec:
    """One hosted image model: canonical key, weights repo, aliases, builder.

    aliases: every accepted spelling for this backend. "qwen-image" is
    deliberately ambiguous — it appears in both the edit and generate
    entries and resolves by intent (see canonical_image_model_name vs the
    generate-side resolution in core.ops).
    """

    key: str
    repo: str
    aliases: frozenset[str]
    build: "Callable[[ModelManager, str], Any]"
    # Base model that LoRA adapters must be trained for, or None if the
    # backend cannot take LoRAs (flux2-dev is FP8-quantized at load; LoRA
    # application onto quantized weights is unreliable).
    lora_base: str | None = None


def _build_flux2(mgr: "ModelManager", local: str):
    # Dual-GPU: park the text encoder on the secondary card so the
    # transformer can run pure BF16 on the primary. Single-GPU: same
    # device for both, FP8 path kicks in.
    text_encoder_device = mgr._gpu.aux_device if mgr._gpu.dual_gpu else mgr._gpu.gen_device
    return Flux2Editor(
        model_id=local,
        device=mgr._gpu.gen_device,
        text_encoder_device=text_encoder_device,
        hf_token=mgr._hf_token,
    )


def _simple_builder(cls):
    """Builder for backends with the uniform (model_id, device, token) ctor."""

    def build(mgr: "ModelManager", local: str):
        return cls(
            model_id=local,
            device=mgr._gpu.gen_device,
            hf_token=mgr._hf_token,
        )

    return build


IMAGE_BACKENDS: dict[str, ImageBackendSpec] = {
    spec.key: spec
    for spec in (
        ImageBackendSpec(
            key="flux2",
            repo=FLUX2_DEV_REPO,
            aliases=frozenset(
                {
                    "flux2",
                    "flux-2",
                    "flux.2",
                    "flux2-dev",
                    "flux-2-dev",
                    "flux.2-dev",
                }
            ),
            build=_build_flux2,
        ),
        ImageBackendSpec(
            key="flux2-klein",
            repo=FLUX2_KLEIN_REPO,
            aliases=frozenset(
                {
                    "klein",
                    "flux-klein",
                    "flux2-klein",
                    "flux-2-klein",
                    "flux.2-klein",
                    "flux2-klein-4b",
                    "flux.2-klein-4b",
                    "flux2-klein-base-4b",
                    "flux.2-klein-base-4b",
                    "flux2-klein-9b",
                    "flux.2-klein-9b",
                    "flux2-klein-base-9b",
                    "flux.2-klein-base-9b",
                }
            ),
            build=_simple_builder(Flux2KleinEditor),
            lora_base=FLUX2_KLEIN_REPO,
        ),
        ImageBackendSpec(
            key="flux1-fill",
            repo=FLUX1_FILL_REPO,
            aliases=frozenset(
                {
                    "flux1-fill",
                    "flux-1-fill",
                    "flux.1-fill",
                    "flux1-fill-dev",
                    "flux.1-fill-dev",
                    "flux-fill",
                    "fill",
                }
            ),
            build=_simple_builder(FluxFillInpainter),
            lora_base=FLUX1_DEV_REPO,
        ),
        ImageBackendSpec(
            key="flux1-dev",
            repo=FLUX1_DEV_REPO,
            aliases=frozenset(
                {
                    "flux1",
                    "flux-1",
                    "flux.1",
                    "flux1-dev",
                    "flux-1-dev",
                    "flux.1-dev",
                    "flux1-generate",
                }
            ),
            build=_simple_builder(Flux1Generator),
            lora_base=FLUX1_DEV_REPO,
        ),
        ImageBackendSpec(
            key="firered",
            repo=FIRERED_REPO,
            aliases=frozenset(
                {
                    "firered",
                    "fire-red",
                    "firered-image-edit",
                }
            ),
            build=_simple_builder(FireRedEditor),
            lora_base=QWEN_IMAGE_EDIT_REPO,
        ),
        ImageBackendSpec(
            key="qwen-image-edit",
            repo=QWEN_IMAGE_EDIT_REPO,
            aliases=frozenset(
                {
                    "qwen",
                    "qwen-image",
                    "qwen-image-edit",
                    "qwen-image-edit-2511",
                }
            ),
            build=_simple_builder(FireRedEditor),
            lora_base=QWEN_IMAGE_EDIT_REPO,
        ),
        ImageBackendSpec(
            key="qwen-image-generate",
            repo=QWEN_IMAGE_REPO,
            aliases=frozenset(
                {
                    "qwen-image",
                    "qwen-image-2512",
                    "qwen-image-generate",
                    "qwen-image-t2i",
                    "qwen-image-text-to-image",
                }
            ),
            build=_simple_builder(QwenImageGenerator),
            lora_base=QWEN_IMAGE_REPO,
        ),
    )
}

# Edit-intent resolution order for ambiguous aliases ("qwen-image" is an
# edit model here; the generate side resolves it in core.ops).
_EDIT_RESOLUTION_ORDER = (
    "flux2",
    "flux2-klein",
    "flux1-fill",
    "firered",
    "qwen-image-edit",
)

# Per-tool default models. Used as the tool parameter defaults so the agent
# sees them on schema inspection, and applied server-side for null. Generate
# favors the more-creative flux2-dev; edit favors the cheap firered.
DEFAULT_GENERATE_MODEL = "flux2-dev"
DEFAULT_EDIT_MODEL = "firered"


def _in_aliases(key: str, model: str | None) -> bool:
    return normalize_image_model_name(model) in IMAGE_BACKENDS[key].aliases


def is_flux2_model(model: str | None) -> bool:
    return _in_aliases("flux2", model)


def is_flux2_klein_model(model: str | None) -> bool:
    return _in_aliases("flux2-klein", model)


def is_flux1_fill_model(model: str | None) -> bool:
    return _in_aliases("flux1-fill", model)


def is_flux1_model(model: str | None) -> bool:
    return _in_aliases("flux1-dev", model)


def is_qwen_image_edit_model(model: str | None) -> bool:
    return _in_aliases("qwen-image-edit", model)


def is_qwen_image_model(model: str | None) -> bool:
    return _in_aliases("qwen-image-generate", model)


def canonical_image_model_name(model: str | None) -> str:
    key = normalize_image_model_name(model)
    for backend_key in _EDIT_RESOLUTION_ORDER:
        if key in IMAGE_BACKENDS[backend_key].aliases:
            return backend_key
    return key


def configure_hf_cache() -> str:
    """Ensure Hugging Face hub cache points at a writable directory.

    Cluster/container environments sometimes inject HF_HOME=/var/lib/hf even
    when that path is not mounted in the job. huggingface_hub then fails with
    a bare FileNotFoundError for /var/lib/hf/hub. Keep explicit usable cache
    env vars, but fall back to a writable cache next to CONTINUUM_MODEL_CACHE
    when the configured path cannot be created.
    """
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    hf_home = os.environ.get("HF_HOME")
    configured = explicit or (
        os.path.join(hf_home, "hub") if hf_home else os.path.expanduser("~/.cache/huggingface/hub")
    )

    def _can_use(path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            return False
        return os.path.isdir(path) and os.access(path, os.W_OK)

    if _can_use(configured):
        os.environ.setdefault("HF_HUB_CACHE", configured)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", configured)
        return configured

    fallback_home = os.path.join(RAMDISK_CACHE, ".hf")
    fallback_hub = os.path.join(fallback_home, "hub")
    os.makedirs(fallback_hub, exist_ok=True)
    os.environ["HF_HOME"] = fallback_home
    os.environ["HF_HUB_CACHE"] = fallback_hub
    os.environ["HUGGINGFACE_HUB_CACHE"] = fallback_hub
    _mcp_log(
        "models: WARNING Hugging Face cache path is unavailable; "
        f"using {fallback_hub} instead of {configured}",
    )
    return fallback_hub


def _dry_run_download_size(files: list[Any]) -> int:
    total = 0
    for info in files:
        if not getattr(info, "will_download", True):
            continue
        size = getattr(info, "file_size", None)
        if isinstance(size, int) and size > 0:
            total += size
    return total


def _looks_like_missing_tqdm_class_support(error: TypeError) -> bool:
    message = str(error)
    return "tqdm_class" in message and (
        "unexpected keyword" in message or "unexpected argument" in message or "got an unexpected" in message
    )


def ensure_local(
    repo_id: str,
    token: str | None = None,
    progress: ProgressCallback | None = None,
) -> str:
    """Copy HF model from network cache to ramdisk for fast loading.

    Uses huggingface_hub.snapshot_download with local_dir to copy all model
    files to a fast local path. If already copied (sentinel exists), returns
    immediately. Returns the local path to pass to from_pretrained().
    """
    safe_name = repo_id.replace("/", "--")
    local_dir = os.path.join(RAMDISK_CACHE, safe_name)
    sentinel = os.path.join(local_dir, ".ready")

    if os.path.exists(sentinel) or os.path.exists(os.path.join(local_dir, ".continuum_ready")):
        return local_dir

    hub_cache = configure_hf_cache()
    os.makedirs(local_dir, exist_ok=True)

    _mcp_log(f"models: caching {repo_id} to {local_dir}", verbose=True)
    expected_bytes = 0
    try:
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": local_dir,
            "cache_dir": hub_cache,
            "token": token,
        }
        try:
            dry_run_files = snapshot_download(**kwargs, dry_run=True)
            if isinstance(dry_run_files, list):
                expected_bytes = _dry_run_download_size(dry_run_files)
        except TypeError:
            pass
        except Exception as e:
            _mcp_log(f"models: WARNING Hugging Face dry-run failed for {repo_id}: {e}")

        emit_progress(
            progress,
            "downloading model",
            0,
            expected_bytes if expected_bytes > 0 else None,
        )
        tqdm_class = make_hf_snapshot_download_tqdm_class(
            progress,
            "downloading model",
            expected_bytes,
        )
        if tqdm_class is None:
            tqdm_class = make_silent_tqdm_class()
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        try:
            snapshot_download(**kwargs)
        except TypeError as e:
            if "tqdm_class" not in kwargs or not _looks_like_missing_tqdm_class_support(e):
                raise
            # Older huggingface_hub versions did not expose tqdm_class.
            kwargs.pop("tqdm_class", None)
            snapshot_download(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to cache Hugging Face repo {repo_id!r} into {local_dir!r} "
            f"using hub cache {hub_cache!r}: {e}"
        ) from e
    if expected_bytes > 0:
        emit_progress(progress, "downloading model", expected_bytes, expected_bytes)
    else:
        emit_progress(progress, "downloading model", 1, 1)

    with open(sentinel, "w") as f:
        f.write("ok")

    return local_dir


def _clear_vram(device: str | None = None) -> None:
    """Free cached VRAM on a specific device, or all devices if None."""
    try:
        gc.collect()
        if device is not None and device != "cpu":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    except Exception:
        gc.collect()


class ModelManager:
    """Manages GPU model lifecycle.

    Single-GPU: one generation model family in VRAM at a time.
    Dual-GPU: FLUX.2 may place its text encoder on the secondary GPU.
    """

    def __init__(self):
        configure_hf_cache()
        self._gpu = get_gpu_config()
        self._image_editor = None
        self._image_model_name: str | None = None
        self._segmenter = None  # SAM3 PCS text segmentation
        self._tracker_segmenter = None  # SAM3 Tracker PVS point/box segmentation
        self._mode: str | None = None
        self._hf_token: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # -- Public properties --

    @property
    def image_editor(self):
        return self._image_editor


    @property
    def mode(self) -> str | None:
        return self._mode

    # -- Mode switching --

    def enter_image_mode(
        self,
        model: str = "flux2",
        load_progress: ProgressCallback | None = None,
    ) -> None:
        """Evict competing models, load image editor."""
        model = canonical_image_model_name(model)
        if self._mode == "image" and self._image_model_name == model and self._image_editor is not None:
            return
        if self._image_editor is not None and self._image_model_name != model:
            self._evict_image()
        self._ensure_image_editor(model, load_progress=load_progress)
        self._mode = "image"


    # -- Image models --

    def _ensure_image_editor(
        self,
        model: str = "flux2",
        load_progress: ProgressCallback | None = None,
    ) -> None:
        if self._image_editor is not None:
            return

        model = canonical_image_model_name(model)
        spec = IMAGE_BACKENDS.get(model)
        if spec is None:
            # Generate-intent keys arrive already resolved (e.g.
            # "qwen-image-t2i"); accept any alias of a registered backend.
            for candidate in IMAGE_BACKENDS.values():
                if model in candidate.aliases:
                    spec = candidate
                    break
        if spec is None:
            known = ", ".join(sorted(IMAGE_BACKENDS))
            raise ValueError(f"unknown image model {model!r}; expected one of: {known}")

        local = ensure_local(spec.repo, token=self._hf_token, progress=load_progress)
        self._image_editor = spec.build(self, local)
        self._image_editor.load(load_progress=load_progress)
        self._image_model_name = spec.key

    def _evict_image(self) -> None:
        if self._image_editor is not None:
            split = getattr(self._image_editor, "split_devices", False)
            te_device = getattr(self._image_editor, "text_encoder_device", None)
            del self._image_editor
            self._image_editor = None
            self._image_model_name = None
            _clear_vram(self._gpu.gen_device)
            if split and te_device and te_device != self._gpu.gen_device:
                _clear_vram(te_device)

    # -- SAM3 segmenter --

    @property
    def segmenter(self):
        """SAM3 PCS text segmenter. Loaded on demand (~4GB VRAM)."""
        return self.get_segmenter()

    @property
    def tracker_segmenter(self):
        """SAM3 Tracker PVS segmenter. Loaded on demand (~4GB VRAM)."""
        return self.get_tracker_segmenter()

    def get_segmenter(self, load_progress: ProgressCallback | None = None):
        if self._segmenter is None:
            self._ensure_segmenter(load_progress=load_progress)
        return self._segmenter

    def get_tracker_segmenter(self, load_progress: ProgressCallback | None = None):
        if self._tracker_segmenter is None:
            self._ensure_tracker_segmenter(load_progress=load_progress)
        return self._tracker_segmenter

    def _ensure_segmenter(
        self,
        load_progress: ProgressCallback | None = None,
    ) -> None:
        if self._segmenter is not None:
            return
        local = ensure_local(
            SAM3_REPO,
            token=self._hf_token,
            progress=load_progress,
        )
        self._segmenter = SAM3Segmenter(
            device=self._gpu.gen_device,
            model_path=local,
        )
        self._segmenter.load(load_progress=load_progress)

    def _evict_segmenter(self) -> None:
        if self._segmenter is not None:
            self._segmenter.unload()
            del self._segmenter
            self._segmenter = None
            _clear_vram(self._gpu.gen_device)

    def _ensure_tracker_segmenter(
        self,
        load_progress: ProgressCallback | None = None,
    ) -> None:
        if self._tracker_segmenter is not None:
            return
        local = ensure_local(
            SAM3_REPO,
            token=self._hf_token,
            progress=load_progress,
        )
        self._tracker_segmenter = SAM3TrackerSegmenter(
            device=self._gpu.gen_device,
            model_path=local,
        )
        self._tracker_segmenter.load(load_progress=load_progress)

    def _evict_tracker_segmenter(self) -> None:
        if self._tracker_segmenter is not None:
            self._tracker_segmenter.unload()
            del self._tracker_segmenter
            self._tracker_segmenter = None
            _clear_vram(self._gpu.gen_device)


    def evict_for_sr(self) -> None:
        """Evict image editor to free VRAM for SR upscaler.

        The editor will lazy-reload on the next generation.
        """
        if self._image_editor is not None:
            split = getattr(self._image_editor, "split_devices", False)
            te_device = getattr(self._image_editor, "text_encoder_device", None)
            # Clear the cached pipeline so it reloads next time
            if hasattr(self._image_editor, "_pipe"):
                self._image_editor._pipe = None
            del self._image_editor
            self._image_editor = None
            self._image_model_name = None
            _clear_vram(self._gpu.gen_device)
            if split and te_device and te_device != self._gpu.gen_device:
                _clear_vram(te_device)
        self._mode = None

    def upscale(
        self,
        img,
        target_w: int,
        target_h: int,
        on_step=None,
        load_progress: ProgressCallback | None = None,
        steps: int = 20,
        guidance: float = 5.0,
        target_tile_size: int = 1024,
        overlap: float = 0.5,
        keep_loaded: bool = False,
    ):
        """Run FaithDiff SR. Evicts current models as needed.

        keep_loaded holds the pipeline across calls, for callers that upscale
        many images in a row: SR is ~10GB of weights, and reloading it per
        image costs more than the upscale does. The caller owns releasing it
        afterwards via release_sr.
        """
        self.evict_for_sr()
        return upscale_faithdiff(
            img,
            target_w,
            target_h,
            on_step=on_step,
            device=self._gpu.gen_device,
            load_progress=load_progress,
            steps=steps,
            guidance=guidance,
            target_tile_size=target_tile_size,
            overlap=overlap,
            keep_loaded=keep_loaded,
        )

    def release_sr(self) -> None:
        """Free the cached SR pipeline held by keep_loaded."""
        release_sr_pipeline()

    # -- Unload everything --

    def unload_all(self) -> None:
        """Unload all models and free VRAM."""
        self._evict_image()
        self._evict_segmenter()
        self._evict_tracker_segmenter()
        self._mode = None

    # -- Status --

    def status(self) -> dict[str, Any]:
        """Return loaded models + VRAM usage."""
        info: dict[str, Any] = {
            "mode": self._mode,
            "loaded": [],
            "gpu_config": {
                "num_gpus": self._gpu.num_gpus,
                "dual_gpu": self._gpu.dual_gpu,
                "gen_device": self._gpu.gen_device,
                "aux_device": self._gpu.aux_device,
            },
        }

        if self._image_editor is not None:
            info["loaded"].append(f"image:{self._image_model_name}")
            if hasattr(self._image_editor, "runtime_info"):
                info["image_editor"] = self._image_editor.runtime_info()
        if self._segmenter is not None:
            info["loaded"].append("sam3-pcs")
        if self._tracker_segmenter is not None:
            info["loaded"].append("sam3-tracker")

        try:
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_mem / 1e9
                    gpus.append(
                        {
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "vram_used_gb": round(mem, 1),
                            "vram_total_gb": round(total, 1),
                        }
                    )
                info["gpus"] = gpus
                # Backward compat
                if gpus:
                    info["vram_used_gb"] = gpus[0]["vram_used_gb"]
                    info["vram_total_gb"] = gpus[0]["vram_total_gb"]
                    info["gpu"] = gpus[0]["name"]
        except Exception:
            pass

        return info
