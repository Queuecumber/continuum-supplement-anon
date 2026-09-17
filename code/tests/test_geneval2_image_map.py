"""The scoring handoff must reject incomplete runs before writing any maps."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/geneval2/make_image_map.py"


def prepare(tmp_path, *, status="ok", stop_reason="completed", sample=0, image_exists=True):
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(json.dumps({"id": "a", "prompt": "a red bird"}) + "\n")
    run = tmp_path / "run"
    shard = run / "shard-0"
    shard.mkdir(parents=True)
    image = tmp_path / "image.png"
    if image_exists:
        image.write_bytes(b"image contents are not read by the mapper")
    row = {"item_id": "a", "sample_index": sample, "status": status,
           "stop_reason": stop_reason, "output_image": str(image)}
    (shard / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    return manifest, run, image


def invoke(tmp_path, manifest, run, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run", str(run), "--manifest", str(manifest),
         "--out", str(tmp_path / "maps"), *extra],
        capture_output=True, text=True,
    )


@pytest.mark.parametrize("stop_reason", ["completed", "direct", ""])
def test_complete_run_writes_expected_map(tmp_path, stop_reason):
    manifest, run, image = prepare(tmp_path, stop_reason=stop_reason)
    result = invoke(tmp_path, manifest, run, "--require-complete", "--samples", "1")
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "maps/sample_00.json").read_text()) == {"a red bird": str(image)}


@pytest.mark.parametrize("kwargs", [
    {"status": "failed"}, {"stop_reason": "max_steps"}, {"image_exists": False},
    {"sample": 1},
])
def test_incomplete_run_writes_nothing(tmp_path, kwargs):
    manifest, run, _ = prepare(tmp_path, **kwargs)
    result = invoke(tmp_path, manifest, run, "--require-complete", "--samples", "1")
    assert result.returncode != 0
    assert "run is incomplete" in result.stderr
    assert not (tmp_path / "maps").exists()


def test_absent_sample_preserves_existing_maps(tmp_path):
    manifest, run, _ = prepare(tmp_path)
    maps = tmp_path / "maps"
    maps.mkdir()
    existing = maps / "sample_00.json"
    existing.write_text("previous mapping")
    result = invoke(tmp_path, manifest, run, "--require-complete", "--samples", "2")
    assert result.returncode != 0
    assert "sample 1: 1/1 prompts missing" in result.stderr
    assert existing.read_text() == "previous mapping"


def test_missing_prompt_is_rejected(tmp_path):
    manifest, run, _ = prepare(tmp_path)
    with manifest.open("a") as handle:
        handle.write(json.dumps({"id": "b", "prompt": "a blue bird"}) + "\n")
    result = invoke(tmp_path, manifest, run, "--require-complete", "--samples", "1")
    assert result.returncode != 0
    assert "sample 0: 1/2 prompts missing" in result.stderr


def test_partial_maps_remain_available_without_strict_flag(tmp_path):
    manifest, run, _ = prepare(tmp_path)
    result = invoke(tmp_path, manifest, run)
    assert result.returncode == 0, result.stderr


def test_samples_must_be_positive(tmp_path):
    manifest, run, _ = prepare(tmp_path)
    result = invoke(tmp_path, manifest, run, "--samples", "0")
    assert result.returncode != 0
    assert "--samples must be positive" in result.stderr
