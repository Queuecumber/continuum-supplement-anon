import json
from pathlib import Path

import pytest

from benchmarks.make_geneval2_table import build_rows, render, summarize


def write_samples(tmp_path, rows):
    for sample in range(4):
        (tmp_path / f"sample_{sample:02d}.json").write_text(json.dumps(rows))


def test_prompt_means_are_not_pooled_atoms_and_zeros_are_preserved(tmp_path):
    write_samples(tmp_path, [[1.0], [0.0, 0.0, 0.0]])
    benchmark = [{"vqa_list": [None]}, {"vqa_list": [None] * 3}]
    assert summarize(tmp_path, benchmark) == {"am": 50.0, "gm": 50.0}


def test_every_sample_contributes(tmp_path):
    write_samples(tmp_path, [[1.0]])
    (tmp_path / "sample_03.json").write_text("[[0.0]]")
    assert summarize(tmp_path, [{"vqa_list": [None]}]) == {"am": 75.0, "gm": 75.0}


@pytest.mark.parametrize("rows", [[], [[1.0, 1.0]], [[-0.1]], [[float('nan')]], [[1.1]]])
def test_invalid_score_files_are_rejected(tmp_path, rows):
    write_samples(tmp_path, rows)
    with pytest.raises(ValueError):
        summarize(tmp_path, [{"vqa_list": [None]}])


def test_incomplete_arm_is_not_silently_averaged(tmp_path):
    write_samples(tmp_path, [[1.0]])
    (tmp_path / "sample_03.json").unlink()
    with pytest.raises(ValueError, match="exactly 4"):
        summarize(tmp_path, [{"vqa_list": [None]}])


def test_rows_sort_by_gm_and_keep_published_comparisons_distinct(tmp_path):
    for arm, value in (("k3-klein", 0.9), ("qwen38-klein", 0.7), ("direct-klein", 0.4)):
        path = tmp_path / arm
        path.mkdir()
        write_samples(path, [[value]])
    published = {"rows": [{"method": "EPIC", "generator": "FLUX.2-klein-9B",
                           "gm": 80.0, "am": 95.0, "source": "epic"}]}
    rows = build_rows(tmp_path, [{"vqa_list": [None]}], published)
    assert [row["method"] for row in rows] == ["Kimi K3", "EPIC", "Qwen3.8-27B", "Direct (ours)"]
    tex = render(rows)
    assert r"\textbf{Kimi K3}" in tex
    assert r"EPIC$^{\dagger}$" in tex
    assert r"\textbf{EPIC}" not in tex
    assert "official leaderboard ranks" in tex


def test_source_snapshot_uses_automated_scores():
    source = Path(__file__).resolve().parents[1] / "benchmarks/geneval2/published_results.json"
    rows = json.loads(source.read_text())["rows"]
    # Adjacent columns in GenEval 2 Table 4 are human scores (4.3 / 45.3);
    # accidentally copying those would mix metrics in the comparison.
    sdxl = next(row for row in rows if row["generator"] == "SDXL")
    assert (sdxl["gm"], sdxl["am"]) == (9.1, 50.1)


def test_full_matrix_includes_every_generator_and_its_direct_baseline(tmp_path):
    for generator in ("klein", "dev", "qwenimage"):
        for reasoner in ("k3", "qwen38", "direct"):
            arm = tmp_path / f"{reasoner}-{generator}"
            arm.mkdir()
            write_samples(arm, [[0.8]])
    benchmark = [{"vqa_list": [None]}]
    published = {"rows": []}
    # Default stays usable while the newly requested cells are pending.
    assert len(build_rows(tmp_path, benchmark, published)) == 3
    rows = build_rows(tmp_path, benchmark, published, ("klein", "dev", "qwenimage"))
    assert len(rows) == 9
    for name in ("FLUX.2-klein-9B", "FLUX.2-dev", "Qwen-Image"):
        methods = {r["method"] for r in rows if r["generator"] == name}
        assert methods == {"Kimi K3", "Qwen3.8-27B", "Direct (ours)"}
    (tmp_path / "direct-qwenimage/sample_03.json").unlink()
    with pytest.raises(ValueError, match="exactly 4"):
        build_rows(tmp_path, benchmark, published, ("klein", "dev", "qwenimage"))
