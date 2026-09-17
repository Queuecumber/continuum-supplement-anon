from pathlib import Path

from continuum import models
from continuum.progress import (
    make_hf_snapshot_download_tqdm_class,
    make_progress_tqdm_class,
    make_silent_tqdm_class,
)


class DryRunInfo:
    def __init__(
        self,
        *,
        file_size: int,
        filename: str = "weights.bin",
        will_download: bool = True,
    ) -> None:
        self.file_size = file_size
        self.filename = filename
        self.will_download = will_download


def test_ensure_local_reports_first_use_download_progress(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(models, "RAMDISK_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    def fake_snapshot_download(*, local_dir, tqdm_class=None, dry_run=False, **_kwargs):
        if dry_run:
            return [DryRunInfo(file_size=100)]
        if tqdm_class is not None:
            # HF's parent byte bar starts with an incomplete total, then grows
            # as file metadata is discovered. MCP progress must keep the
            # dry-run denominator, not report 1/2 = 50%.
            bar = tqdm_class(
                total=0,
                unit="B",
                name="huggingface_hub.snapshot_download",
                desc="Downloading (incomplete total...)",
            )
            bar.total += 2
            bar.update(1)
            bar.total += 98
            bar.update(1)
            # The thread-map/file-count bar is also constructed from the same
            # tqdm_class and should be ignored.
            file_bar = tqdm_class(total=2, desc="Fetching 2 files")
            file_bar.update(1)
            file_bar.close()
            bar.close()
        Path(local_dir, "weights.bin").write_text("weights")
        return local_dir

    # Patch the module-level binding, not huggingface_hub itself.
    monkeypatch.setattr(models, "snapshot_download", fake_snapshot_download)

    events: list[tuple[str, float, float | None]] = []

    local = models.ensure_local(
        "owner/repo",
        progress=lambda message, current, total: events.append((message, current, total)),
    )

    assert local == str(tmp_path / "owner--repo")
    assert events[0] == ("downloading model", 0.0, 100.0)
    assert ("downloading model", 1.0, 2.0) not in events
    assert ("downloading model", 1.0, 100.0) in events
    assert events[-1] == ("downloading model", 100.0, 100.0)
    assert (tmp_path / "owner--repo" / ".ready").exists()
    captured = capsys.readouterr()
    assert "Downloading" not in captured.err


def test_ensure_local_skips_download_progress_when_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(models, "RAMDISK_CACHE", str(tmp_path))
    local_dir = tmp_path / "owner--repo"
    local_dir.mkdir()
    (local_dir / ".ready").write_text("ok")

    events: list[tuple[str, float, float | None]] = []

    local = models.ensure_local(
        "owner/repo",
        progress=lambda message, current, total: events.append((message, current, total)),
    )

    assert local == str(local_dir)
    assert events == []


def test_progress_tqdm_accepts_hf_name_and_tracks_when_disabled() -> None:
    events: list[tuple[str, float, float | None]] = []
    tqdm_class = make_progress_tqdm_class(
        lambda message, current, total: events.append((message, current, total)),
        "downloading model",
    )
    assert tqdm_class is not None

    bar = tqdm_class(total=3, name="huggingface_hub.snapshot_download")
    bar.update(1)
    bar.update(2)
    bar.close()

    assert bar.disable is True
    assert ("downloading model", 3.0, 3.0) in events


def test_hf_snapshot_download_tqdm_tracks_only_snapshot_byte_bar() -> None:
    events: list[tuple[str, float, float | None]] = []
    tqdm_class = make_hf_snapshot_download_tqdm_class(
        lambda message, current, total: events.append((message, current, total)),
        "downloading model",
        100,
    )
    assert tqdm_class is not None

    bar = tqdm_class(
        total=0,
        unit="B",
        name="huggingface_hub.snapshot_download",
    )
    bar.total += 2
    bar.update(1)
    bar.total += 98
    bar.update(1)

    ignored = tqdm_class(total=2, desc="Fetching 2 files")
    ignored.update(1)

    assert ("downloading model", 1.0, 2.0) not in events
    assert ("downloading model", 1.0, 100.0) in events
    assert ("downloading model", 2.0, 100.0) in events
    assert ignored.n == 1


def test_progress_tqdm_can_suppress_terminal_completion() -> None:
    events: list[tuple[str, float, float | None]] = []
    tqdm_class = make_progress_tqdm_class(
        lambda message, current, total: events.append((message, current, total)),
        "downloading model",
        emit_terminal_complete=False,
    )
    assert tqdm_class is not None

    bar = tqdm_class(total=3, name="huggingface_hub.snapshot_download")
    bar.update(1)
    bar.update(2)
    bar.close()

    assert ("downloading model", 3.0, 3.0) not in events


def test_silent_tqdm_accepts_hf_name_and_emits_nothing() -> None:
    tqdm_class = make_silent_tqdm_class()
    assert tqdm_class is not None

    bar = tqdm_class(total=2, name="huggingface_hub.snapshot_download")
    bar.update(1)

    assert bar.disable is True
    assert bar.n == 1
