import json

from benchmarks.make_paper_tables import load_rows, main_table, trace_stats, ugb_cell


def test_historical_ugb_names_map_to_the_correct_runs():
    assert ugb_cell("klein", "k3") == "agent"
    assert ugb_cell("klein", "direct") == "direct"
    assert ugb_cell("qwenimage", "qwen38") == "qwen38-qwenimage"


def test_pending_scores_are_not_reported_as_zero():
    text = main_table([{"generator": "FLUX.2-dev", "agent": "Kimi K3", "ugb": 94.14,
                        "geneval2": None}])
    assert text.count(r"\textit{pending}") == 2
    assert "0.00" not in text
    assert "ten primary-dimension accuracies" in text


def test_completed_table_has_no_pending_disclaimer():
    text = main_table([{"generator": "FLUX.2-dev", "agent": "Kimi K3", "ugb": 94.14,
                        "geneval2": {"gm": 78.682, "am": 92.782}}])
    assert "78.68" in text and "92.78" in text
    assert "pending" not in text.lower()


def test_precision_baseline_override_is_explicit_and_scoped(tmp_path, monkeypatch):
    from benchmarks import make_paper_tables as tables
    paths = []
    def read(path):
        paths.append(path.name)
        return {}, {}
    monkeypatch.setattr(tables.ugb, 'load', read)
    monkeypatch.setattr(tables.ugb, 'overall', lambda _: 80.0)
    rows = load_rows(tmp_path, tmp_path, [], 'direct-dev-fp8-final')
    assert 'direct-dev-fp8-final_en.json' in paths
    assert 'direct-dev_en.json' not in paths
    assert 'k3-dev_en.json' in paths and 'qwen38-dev_en.json' in paths
    assert 'direct_en.json' in paths and 'direct-qwenimage_en.json' in paths
    assert rows[3]['ugb_cell'] == 'direct-dev-fp8-final'


def test_trace_census_deduplicates_attempts_and_counts_successful_image_calls(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps([
        {"role": "assistant", "reasoning_content": "inspect", "tool_calls": [
            {"id": "a", "function": {"name": "generate_image"}},
            {"id": "b", "function": {"name": "inspect_region"}},
            {"id": "c", "function": {"name": "edit_image"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "Produced image img_0."},
        {"role": "tool", "tool_call_id": "b", "content": "Produced image img_1."},
        {"role": "tool", "tool_call_id": "c", "content": "Error executing edit_image"},
        {"role": "assistant", "content": "Done."},
    ]))
    old = {"item_id": "a", "sample_index": 0, "status": "failed", "transcript": "absent-old-file"}
    new = {**old, "status": "ok", "transcript": str(transcript), "stop_reason": "completed"}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n")
    stats = trace_stats(tmp_path)
    assert stats["traces"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["turns"] == 2
    assert stats["image_calls"] == 1
    assert stats["reasoning_percent"] == 50
