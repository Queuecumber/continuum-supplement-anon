"""Progress helpers shared by MCP model loading paths."""

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType

from tqdm.auto import tqdm as base_tqdm

ProgressCallback = Callable[[str, float, float | None], None]
TQDM_PROGRESS_INTERVAL_SECONDS = 0.5
TQDM_PROGRESS_FRACTION_DELTA = 0.01


def emit_progress(
    callback: ProgressCallback | None,
    message: str,
    progress: float = 0,
    total: float | None = None,
) -> None:
    """Call a progress callback if one was supplied."""
    if callback is None:
        return
    callback(message, float(progress), None if total is None else float(total))


def make_progress_tqdm_class(
    callback: ProgressCallback | None,
    message: str,
    *,
    emit_terminal_complete: bool = True,
):
    """Return a tqdm subclass that mirrors tqdm updates to MCP progress.

    Hugging Face Hub accepts a ``tqdm_class`` for downloads. Diffusers and
    Transformers use module-level tqdm symbols for component/shard loading; see
    ``patch_tqdm_progress`` below.
    """
    if callback is None:
        return None

    class ProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            # huggingface_hub passes an internal progress-group name through
            # tqdm_class; tqdm itself rejects unknown kwargs. Also suppress
            # terminal rendering; MCP progress notifications are the UI.
            kwargs.pop("name", None)
            kwargs.setdefault("disable", True)
            self._last_emit_time = 0.0
            self._last_emit_fraction = -1.0
            self._min_interval_s = TQDM_PROGRESS_INTERVAL_SECONDS
            self._min_fraction = TQDM_PROGRESS_FRACTION_DELTA
            super().__init__(*args, **kwargs)
            self._emit_force()

        def _emit(self) -> None:
            total = getattr(self, "total", None)
            current = getattr(self, "n", 0)
            if total:
                total_f = float(total)
                current_f = min(float(current), total_f)
                if not emit_terminal_complete and current_f >= total_f:
                    return
                fraction = current_f / total_f if total_f > 0 else 0.0
                now = time.monotonic()
                if (
                    current_f <= 0
                    or (emit_terminal_complete and current_f >= total_f)
                    or fraction - self._last_emit_fraction >= self._min_fraction
                    or now - self._last_emit_time >= self._min_interval_s
                ):
                    self._last_emit_time = now
                    self._last_emit_fraction = fraction
                    emit_progress(callback, message, current_f, total_f)
            else:
                now = time.monotonic()
                if now - self._last_emit_time >= self._min_interval_s:
                    self._last_emit_time = now
                    emit_progress(callback, message, 0, None)

        def _emit_force(self) -> None:
            total = getattr(self, "total", None)
            current = getattr(self, "n", 0)
            if total:
                if not emit_terminal_complete and float(current) >= float(total):
                    return
                emit_progress(callback, message, min(float(current), float(total)), float(total))
            else:
                emit_progress(callback, message, 0, None)

        def update(self, n=1):
            if getattr(self, "disable", False):
                self.n += n
                result = None
            else:
                result = super().update(n)
            self._emit()
            return result

        def close(self):
            self._emit_force()
            return super().close()

    return ProgressTqdm


def make_silent_tqdm_class():
    """Return a tqdm class that suppresses terminal output and emits nothing."""

    class SilentTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs.setdefault("disable", True)
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if getattr(self, "disable", False):
                self.n += n
                return None
            return super().update(n)

    return SilentTqdm


def make_hf_snapshot_download_tqdm_class(
    callback: ProgressCallback | None,
    message: str,
    expected_total: int,
):
    """Return a tqdm class that reports HF snapshot byte progress.

    ``snapshot_download`` passes ``tqdm_class`` to two progress bars: a
    thread-map/file-count bar and a parent byte bar named
    ``huggingface_hub.snapshot_download``. The parent byte bar's internal
    ``total`` starts incomplete and grows as files are discovered, so use the
    dry-run total supplied by ``ensure_local`` as the stable denominator.
    """
    if callback is None or expected_total <= 0:
        return make_silent_tqdm_class()

    class HFSnapshotDownloadTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            name = kwargs.pop("name", None)
            unit = kwargs.get("unit")
            desc = kwargs.get("desc")
            kwargs.setdefault("disable", True)
            self._tracked = unit == "B" and (
                name == "huggingface_hub.snapshot_download"
                or (name is None and isinstance(desc, str) and desc.startswith("Downloading"))
            )
            self._last_emit_time = 0.0
            self._last_emit_fraction = -1.0
            self._expected_total = float(expected_total)
            self._min_interval_s = TQDM_PROGRESS_INTERVAL_SECONDS
            self._min_fraction = TQDM_PROGRESS_FRACTION_DELTA
            super().__init__(*args, **kwargs)
            if self._tracked:
                emit_progress(callback, message, 0, self._expected_total)

        def update(self, n=1):
            if getattr(self, "disable", False):
                self.n += n
                result = None
            else:
                result = super().update(n)
            if self._tracked:
                current = max(0.0, min(float(self.n), self._expected_total))
                fraction = current / self._expected_total
                now = time.monotonic()
                if (
                    current <= 0
                    or fraction - self._last_emit_fraction >= self._min_fraction
                    or now - self._last_emit_time >= self._min_interval_s
                ):
                    self._last_emit_time = now
                    self._last_emit_fraction = fraction
                    emit_progress(callback, message, current, self._expected_total)
            return result

    return HFSnapshotDownloadTqdm


@contextmanager
def patch_tqdm_progress(
    callback: ProgressCallback | None,
    message: str,
) -> Iterator[None]:
    """Patch common tqdm symbols during opaque model-loading calls."""
    tqdm_class = make_progress_tqdm_class(callback, message)
    if tqdm_class is None:
        yield
        return

    patches: list[tuple[ModuleType, str, object]] = []

    def patch(module: ModuleType | None, attr: str) -> None:
        if module is None or not hasattr(module, attr):
            return
        patches.append((module, attr, getattr(module, attr)))
        setattr(module, attr, tqdm_class)

    patch(sys.modules.get("tqdm.auto"), "tqdm")
    patch(sys.modules.get("diffusers.pipelines.pipeline_utils"), "tqdm")
    patch(sys.modules.get("transformers.modeling_utils"), "tqdm")
    patch(sys.modules.get("transformers.utils.logging"), "tqdm")

    try:
        yield
    finally:
        for module, attr, original in reversed(patches):
            setattr(module, attr, original)
