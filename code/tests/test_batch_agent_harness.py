import json
from io import BytesIO

import pytest
from PIL import Image

from continuum.harnesses import agent_loop, batch_agent


def _png(color="red") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeProc:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture(autouse=True)
def _never_launch_a_real_server(monkeypatch):
    """Safety net: no test spawns a real server, waits on a port, or needs openai
    (the loop is mocked, so the openai preflight is satisfied by default)."""
    monkeypatch.setattr(batch_agent.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(batch_agent, "_wait_for_port", lambda *a, **k: True)
    monkeypatch.setattr(batch_agent.time, "sleep", lambda *a: None)
    monkeypatch.setattr(batch_agent.importlib.util, "find_spec", lambda name: object())


# ---- item discovery / manifest -------------------------------------------------


def test_discover_items_uses_sorted_images_and_prompt_fallback(tmp_path) -> None:
    assert batch_agent.discover_items() == [
        batch_agent.BatchItem(index=0, item_id="0000-prompt", input_path=None)
    ]

    (tmp_path / "b.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "z.png").write_bytes(b"png")
    (tmp_path / "a.JPG").write_bytes(b"jpg")

    items = batch_agent.discover_items(image_dir=tmp_path)
    assert [item.item_id for item in items] == ["0000-a", "0001-z"]


def test_read_manifest_items_per_item_prompts_and_aliases(tmp_path) -> None:
    img_root = tmp_path / "imgs"
    img_root.mkdir()
    (img_root / "a.png").write_bytes(b"png")
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text(
        '{"id": "cat", "instruction": "make it snow", "source": "a.png"}\n'
        "\n"  # blank lines are skipped
        '{"caption": "a red bike"}\n',
        encoding="utf-8",
    )

    items = batch_agent.read_manifest_items(manifest, image_root=img_root)

    assert [item.item_id for item in items] == ["cat", "0001"]
    assert items[0].prompt == "make it snow"
    assert items[0].input_path == (img_root / "a.png").resolve()
    assert items[1].prompt == "a red bike"
    assert items[1].input_path is None


def test_read_manifest_items_prompt_fallback_and_errors(tmp_path) -> None:
    manifest = tmp_path / "b.jsonl"

    manifest.write_text('{"id": "x"}\n', encoding="utf-8")
    assert batch_agent.read_manifest_items(manifest, default_prompt="global")[0].prompt == "global"

    with pytest.raises(batch_agent.BatchHarnessError):  # no prompt, no fallback
        batch_agent.read_manifest_items(manifest)

    manifest.write_text('{"id": "x", "prompt": "a"}\n{"id": "x", "prompt": "b"}\n', encoding="utf-8")
    with pytest.raises(batch_agent.BatchHarnessError):  # duplicate id
        batch_agent.read_manifest_items(manifest)


def test_parse_pin_size() -> None:
    assert batch_agent.parse_pin_size(None) is None
    assert batch_agent.parse_pin_size("1024x1024") == (1024, 1024)
    assert batch_agent.parse_pin_size(" 768 x 512 ") == (768, 512)
    with pytest.raises(batch_agent.BatchHarnessError):
        batch_agent.parse_pin_size("big")


# ---- run_batch drives the agent loop ------------------------------------------


def _fake_loop(png: bytes, seen: list[dict]):
    async def run(**kwargs):
        seen.append(kwargs)
        return agent_loop.LoopResult(
            status="ok",
            output_image=png,
            output_image_id="img_1",
            steps=3,
            transcript=[{"role": "user", "content": kwargs["task_prompt"]}],
        )

    return run


def test_run_batch_drives_loop_and_writes_outputs(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text(
        '{"id": "cat", "prompt": "a cat"}\n{"id": "dog", "prompt": "a dog"}\n', encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    png = _png()
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(png, seen))

    exit_code = batch_agent.main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir), "--model", "gpt-x", "--no-agents-md"]
    )

    assert exit_code == 0
    # eval-ready flat outputs + per-item transcript
    assert (output_dir / "outputs" / "cat.png").read_bytes() == png
    assert (output_dir / "outputs" / "dog.png").read_bytes() == png
    assert (output_dir / "cat" / "transcript.json").exists()
    records = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert records["cat"]["status"] == "ok"
    assert records["cat"]["steps"] == 3
    assert records["cat"]["output_image"].endswith("cat.png")
    # per-item prompts reached the loop
    assert sorted(k["task_prompt"] for k in seen) == ["a cat", "a dog"]


def test_run_batch_samples_and_system_prompt(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text('{"id": "cat", "prompt": "a cat"}\n', encoding="utf-8")
    system = tmp_path / "system.md"
    system.write_text("Do not set a seed.", encoding="utf-8")
    output_dir = tmp_path / "out"
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    exit_code = batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--system-prompt-file",
            str(system),
            "--samples",
            "3",
            "--pin-size",
            "512x512",
            "--output-dir",
            str(output_dir),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )

    assert exit_code == 0
    # 3 samples -> outputs/cat/0.png..2.png
    assert {p.name for p in (output_dir / "outputs" / "cat").iterdir()} == {"0.png", "1.png", "2.png"}
    records = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(r["sample_index"] for r in records) == [0, 1, 2]
    # system prompt + forced size reached every call
    assert all(k["system_prompt"] == "Do not set a seed." for k in seen)
    assert all(k["pin_size"] == (512, 512) for k in seen)


def test_run_batch_saves_images_and_relinks_transcript(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "cat", "prompt": "a cat"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    png0, png1 = _png("red"), _png("blue")

    async def run(**kwargs):
        return agent_loop.LoopResult(
            status="ok",
            output_image=png1,
            output_image_id="img_1",
            steps=2,
            transcript=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Result img_1:"},
                        {"type": "image_url", "image_url": {"url": "img_1"}},
                    ],
                }
            ],
            images={"img_0": png0, "img_1": png1},
        )

    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", run)
    batch_agent.main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir), "--model", "gpt-x", "--no-agents-md"]
    )

    # every image in the chain is written under the item's images/ dir
    assert (output_dir / "cat" / "images" / "img_0.png").read_bytes() == png0
    assert (output_dir / "cat" / "images" / "img_1.png").read_bytes() == png1
    # the transcript points at those files, not a data-url or a bare id
    transcript = json.loads((output_dir / "cat" / "transcript.json").read_text(encoding="utf-8"))
    assert transcript[0]["content"][1]["image_url"]["url"] == "images/img_1.png"


def test_verbose_streams_loop_progress(tmp_path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []

    async def run(**kwargs):
        seen.append(kwargs)
        if kwargs.get("on_event"):
            kwargs["on_event"]("step 1/20  1.0s  5000→80 tok")
        return agent_loop.LoopResult(
            status="ok", output_image=_png(), output_image_id="img_1", steps=1, transcript=[]
        )

    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", run)
    base = ["--manifest", str(manifest), "--model", "gpt-x", "--no-agents-md"]

    batch_agent.main([*base, "--output-dir", str(tmp_path / "quiet")])
    assert seen[-1]["on_event"] is None  # no callback without -v
    quiet_out = capsys.readouterr().out
    assert "starting x" in quiet_out  # heartbeat always prints
    assert "step 1/20" not in quiet_out  # but no per-step stream

    batch_agent.main([*base, "--output-dir", str(tmp_path / "loud"), "-v"])
    assert callable(seen[-1]["on_event"])
    loud_out = capsys.readouterr().out
    assert "    [x] step 1/20  1.0s  5000→80 tok" in loud_out  # labeled, streamed


def test_progress_summary_prints_final_tally(tmp_path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), []))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )

    assert "done: 1 ok, 0 failed, 0 pending" in capsys.readouterr().out


def test_run_batch_dry_run_skips_the_loop(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text('{"id": "cat", "prompt": "a cat"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"

    def _boom(**kwargs):
        raise AssertionError("loop should not run on --dry-run")

    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _boom)

    exit_code = batch_agent.main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir), "--dry-run", "--no-agents-md"]
    )

    assert exit_code == 0
    records = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["status"] == "dry-run"


# ---- resume / stateless autoresume --------------------------------------------


def _failing_loop(seen: list[dict]):
    async def run(**kwargs):
        seen.append(kwargs)
        return agent_loop.LoopResult(
            status="failed",
            output_image=None,
            output_image_id=None,
            steps=0,
            transcript=[],
            error="boom",
            stop_reason="error",
        )

    return run


def _write_progress(output_dir, rows) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_read_progress_tolerates_corrupt_lines(tmp_path) -> None:
    m = tmp_path / "manifest.jsonl"
    m.write_text(
        json.dumps({"item_id": "a", "sample_index": 0, "status": "ok"})
        + "\n\n"  # trailing blank line
        + json.dumps({"item_id": "a", "sample_index": 0, "status": "failed"})
        + "\n"
        + '{"item_id": "b", "sample_index": 0, "status":',  # truncated final row (preemption)
        encoding="utf-8",
    )
    progress = batch_agent.read_progress(m)
    assert [a["status"] for a in progress[("a", 0)]] == ["ok", "failed"]  # history preserved in order
    assert ("b", 0) not in progress  # corrupt row skipped, not a crash
    assert batch_agent.read_progress(tmp_path / "nope.jsonl") == {}  # missing file -> empty


def test_disposition_rules() -> None:
    def att(status, stop_reason=""):
        return {"status": status, "stop_reason": stop_reason}

    # accepted output (ok + completed, or a legacy ok row with no stop_reason) -> done
    assert batch_agent._disposition([att("failed"), att("ok", "completed")], 0) == "done"
    assert batch_agent._disposition([att("ok")], 2) == "done"  # legacy row, ok == accepted
    # an ok run cut off by max_steps was NOT accepted -> keep going
    assert batch_agent._disposition([att("ok", "max_steps")], 0) == "pending"
    assert batch_agent._disposition([], 0) == "pending"
    assert batch_agent._disposition([att("failed")], 0) == "pending"  # unlimited retries by default
    assert batch_agent._disposition([att("failed"), att("failed")], 2) == "exhausted"
    assert batch_agent._disposition([att("ok", "max_steps"), att("failed")], 2) == "exhausted"
    assert batch_agent._disposition([att("failed")], 2) == "pending"


def test_resume_default_skips_ok_and_retries_failed(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text(
        '{"id": "cat", "prompt": "a cat"}\n{"id": "dog", "prompt": "a dog"}\n', encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    _write_progress(
        output_dir,
        [
            {"item_id": "cat", "sample_index": 0, "status": "ok"},
            {"item_id": "dog", "sample_index": 0, "status": "failed"},
        ],
    )
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    exit_code = batch_agent.main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir), "--model", "gpt-x", "--no-agents-md"]
    )

    assert exit_code == 0
    assert [k["task_prompt"] for k in seen] == ["a dog"]  # cat skipped (ok), dog retried


def test_resume_retries_max_steps_truncated_runs(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text('{"id": "done", "prompt": "p1"}\n{"id": "cut", "prompt": "p2"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    _write_progress(
        output_dir,
        [
            {"item_id": "done", "sample_index": 0, "status": "ok", "stop_reason": "completed"},
            {"item_id": "cut", "sample_index": 0, "status": "ok", "stop_reason": "max_steps"},
        ],
    )
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir), "--model", "gpt-x", "--no-agents-md"]
    )

    # 'done' accepted its output -> skipped; 'cut' was truncated by max_steps -> retried
    assert [k["task_prompt"] for k in seen] == ["p2"]


def test_no_resume_reruns_everything(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text(
        '{"id": "cat", "prompt": "a cat"}\n{"id": "dog", "prompt": "a dog"}\n', encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    _write_progress(output_dir, [{"item_id": "cat", "sample_index": 0, "status": "ok"}])
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--no-resume",
        ]
    )

    assert sorted(k["task_prompt"] for k in seen) == ["a cat", "a dog"]  # prior progress discarded


def test_max_attempts_exhausts_and_lets_requeue_terminate(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text('{"id": "dog", "prompt": "a dog"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    _write_progress(
        output_dir,
        [
            {"item_id": "dog", "sample_index": 0, "status": "failed"},
            {"item_id": "dog", "sample_index": 0, "status": "failed"},
        ],
    )
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _failing_loop(seen))

    exit_code = batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--max-attempts",
            "2",
        ]
    )

    assert seen == []  # 2 prior failures >= cap -> exhausted, not retried
    assert exit_code == 0  # nothing left to attempt -> requeue can stop


def test_resume_exit_code_signals_pending_work(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "bench.jsonl"
    manifest.write_text('{"id": "dog", "prompt": "a dog"}\n', encoding="utf-8")
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _failing_loop(seen))

    # unlimited: a failure leaves work pending -> exit 1 so a scheduler requeues
    assert (
        batch_agent.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "a"),
                "--model",
                "gpt-x",
                "--no-agents-md",
            ]
        )
        == 1
    )
    # capped at 1: the single failure exhausts the run -> exit 0 so requeue stops
    assert (
        batch_agent.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "b"),
                "--model",
                "gpt-x",
                "--no-agents-md",
                "--max-attempts",
                "1",
            ]
        )
        == 0
    )


# ---- in-process retry queue + retry/delay knobs -------------------------------


def _flaky_loop(seen: list[dict], fail_first: int):
    """Fail the first `fail_first` calls, then accept."""
    calls = {"n": 0}

    async def run(**kwargs):
        seen.append(kwargs)
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return agent_loop.LoopResult(
                status="failed",
                output_image=None,
                output_image_id=None,
                steps=0,
                transcript=[],
                error="boom",
                stop_reason="error",
            )
        return agent_loop.LoopResult(
            status="ok",
            output_image=_png(),
            output_image_id="img_1",
            steps=1,
            transcript=[],
            stop_reason="completed",
        )

    return run


def test_in_process_requeue_retries_failures_within_one_invocation(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _flaky_loop(seen, fail_first=2))

    exit_code = batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )

    assert len(seen) == 3  # failed twice, requeued each time, accepted on the 3rd — all in one run
    assert exit_code == 0


def test_attempts_per_invocation_bounds_in_process_retries(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _failing_loop(seen))

    exit_code = batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--attempts-per-invocation",
            "2",
        ]
    )

    assert len(seen) == 2  # tried twice this invocation, then deferred
    assert exit_code == 1  # still pending -> the scheduler requeues


def test_max_steps_run_is_deferred_not_requeued(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []

    async def run(**kwargs):
        seen.append(kwargs)
        return agent_loop.LoopResult(
            status="ok",
            output_image=_png(),
            output_image_id="img_1",
            steps=20,
            transcript=[],
            stop_reason="max_steps",
        )

    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", run)
    exit_code = batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )

    assert len(seen) == 1  # structural truncation is NOT requeued in-process
    assert exit_code == 1  # but it is pending -> resume will retry (raise --max-steps)


def test_failed_run_still_writes_the_image_it_had(tmp_path, monkeypatch) -> None:
    """A run that failed after producing an image keeps that image.

    The loop resolves output_image before returning, so an abort — a step limit,
    or the endpoint refusing an oversized request — still holds the model's
    marked best, or the last image it made. Discarding it lost the whole item
    for a reason unrelated to the image. The attempt stays unaccepted, so a
    resume retries it; this only means it is not empty in the meantime.
    """
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"

    async def run(**kwargs):  # a failure that still made a partial image
        return agent_loop.LoopResult(
            status="failed",
            output_image=_png("blue"),
            output_image_id="img_1",
            steps=2,
            transcript=[],
            error="boom",
            stop_reason="error",
            images={"img_0": _png("red")},
        )

    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", run)
    # cap attempts so the failing run terminates instead of requeuing
    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--max-attempts",
            "1",
        ]
    )

    assert (output_dir / "outputs" / "x.png").exists(), "the salvaged image is a deliverable"
    assert (output_dir / "x" / "output.png").exists()
    assert (output_dir / "x" / "images" / "img_0.png").exists()  # partial chain kept for debugging
    row = [
        json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ][-1]
    assert row["output_image"] is not None
    assert row["status"] == "failed", "salvaging must not disguise the failure"
    assert not batch_agent._is_accepted(row), "and must not stop a resume retrying it"


def test_tool_filter_flags_thread_to_loop(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--tools",
            "generate_image, edit_image",
            "--exclude-tools",
            "upscale_image",
        ]
    )

    assert seen[0]["allow_tools"] == {"generate_image", "edit_image"}  # whitespace stripped
    assert seen[0]["block_tools"] == {"upscale_image"}
    # unset -> None (all tools)
    seen.clear()
    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "o2"),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )
    assert seen[0]["allow_tools"] is None and seen[0]["block_tools"] is None


def test_retry_and_delay_flags_thread_to_loop(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--max-retries",
            "9",
            "--request-delay",
            "1.5",
        ]
    )

    assert seen[0]["max_retries"] == 9
    assert seen[0]["request_delay"] == 1.5


def test_manifest_records_model_provenance(tmp_path, monkeypatch) -> None:
    # Licence review / training-data filtering needs to know which model made what.
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"

    async def run(**kwargs):
        return agent_loop.LoopResult(
            status="ok",
            output_image=_png(),
            output_image_id="img_2",
            steps=3,
            transcript=[],
            stop_reason="completed",
            images={"img_0": _png(), "img_1": _png(), "img_2": _png()},
            image_models={"img_0": "flux2-klein", "img_1": "firered", "img_2": "firered"},
        )

    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", run)
    batch_agent.main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir), "--model", "gpt-x", "--no-agents-md"]
    )

    row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["models"] == ["firered", "flux2-klein"]  # every model touched, deduped + sorted
    assert row["output_model"] == "firered"  # the one that made the saved output


# ---- --direct baseline (no agent) ---------------------------------------------


def test_direct_mode_calls_tool_without_llm(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "a neon sign reading OPEN"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    seen: list[dict] = []

    async def direct(**kwargs):
        seen.append(kwargs)
        return agent_loop.LoopResult(
            status="ok",
            output_image=_png(),
            output_image_id="img_0",
            steps=1,
            transcript=[],
            stop_reason="direct",
        )

    def _no_agent(**kwargs):
        raise AssertionError("--direct must not run the agent loop")

    monkeypatch.setattr(batch_agent.agent_loop, "run_direct_tool_call", direct)
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _no_agent)
    # no --model, and openai absent: --direct needs neither
    monkeypatch.setattr(batch_agent.importlib.util, "find_spec", lambda name: None)

    exit_code = batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--direct",
            "--image-model",
            "qwen-image",
            "--pin-size",
            "1024x1024",
        ]
    )

    assert exit_code == 0
    assert seen[0]["task_prompt"] == "a neon sign reading OPEN"  # verbatim, not rewritten
    assert seen[0]["image_model"] == "qwen-image"
    assert seen[0]["tool_name"] == "generate_image"
    assert seen[0]["pin_size"] == (1024, 1024)
    assert (output_dir / "outputs" / "x.png").exists()
    row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["stop_reason"] == "direct"


def test_direct_runs_count_as_accepted_on_resume(tmp_path, monkeypatch) -> None:
    # A 'direct' row must satisfy resume, or a requeue loop would rerun it forever.
    assert batch_agent._disposition([{"status": "ok", "stop_reason": "direct"}], 0) == "done"

    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    _write_progress(
        output_dir, [{"item_id": "x", "sample_index": 0, "status": "ok", "stop_reason": "direct"}]
    )

    def _boom(**kwargs):
        raise AssertionError("already-accepted direct run must not rerun")

    monkeypatch.setattr(batch_agent.agent_loop, "run_direct_tool_call", _boom)
    assert batch_agent.main(["--manifest", str(manifest), "--output-dir", str(output_dir), "--direct"]) == 0


def test_require_agent_deps_errors_clearly_without_openai(monkeypatch) -> None:
    monkeypatch.setattr(batch_agent.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(batch_agent.BatchHarnessError, match="openai"):
        batch_agent._require_agent_deps()


def test_find_agents_md_walks_up_from_start(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("shared policy", encoding="utf-8")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert batch_agent._find_agents_md(deep) == tmp_path / "AGENTS.md"
    assert batch_agent._find_agents_md(tmp_path) == tmp_path / "AGENTS.md"


def test_agents_md_auto_discovered_and_composed_with_system_prompt(tmp_path, monkeypatch) -> None:
    (tmp_path / "AGENTS.md").write_text("SHARED policy", encoding="utf-8")
    sysfile = tmp_path / "run.md"
    sysfile.write_text("RUN specifics", encoding="utf-8")
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # AGENTS.md discovered from CWD
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--system-prompt-file",
            str(sysfile),
            "--model",
            "gpt-x",
        ]
    )

    # AGENTS.md (shared) first, --system-prompt-file (run-specific) appended.
    assert seen[0]["system_prompt"] == "SHARED policy\n\nRUN specifics"


def test_server_instructions_flag_threads_to_loop(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "a"),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )
    assert seen[0]["include_server_instructions"] is True

    seen.clear()
    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "b"),
            "--model",
            "gpt-x",
            "--no-agents-md",
            "--no-server-instructions",
        ]
    )
    assert seen[0]["include_server_instructions"] is False


def test_mcp_server_launches_when_url_omitted(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    proc = _FakeProc()
    monkeypatch.setattr(batch_agent.subprocess, "Popen", lambda cmd, **k: (captured.update(cmd=cmd), proc)[1])
    monkeypatch.setattr(batch_agent, "_free_port", lambda: 45678)

    args = batch_agent.build_parser().parse_args(["--manifest", "m", "--output-dir", str(tmp_path)])
    with batch_agent._mcp_server(args, tmp_path) as url:
        assert url == "http://127.0.0.1:45678/mcp"
    assert proc.terminated  # torn down on exit
    assert "continuum.mcp_server" in " ".join(captured["cmd"])
    assert "45678" in captured["cmd"] and "--transport" in captured["cmd"]


def test_mcp_server_uses_provided_url_without_launching(tmp_path, monkeypatch) -> None:
    def _no_launch(*a, **k):
        raise AssertionError("must not launch a server when --mcp-url is given")

    monkeypatch.setattr(batch_agent.subprocess, "Popen", _no_launch)

    args = batch_agent.build_parser().parse_args(
        ["--manifest", "m", "--output-dir", str(tmp_path), "--mcp-url", "http://ext:9/mcp"]
    )
    with batch_agent._mcp_server(args, tmp_path) as url:
        assert url == "http://ext:9/mcp"


def test_agents_md_discovered_from_cwd_not_manifest_dir(tmp_path, monkeypatch) -> None:
    # The geneval regression: run from the benchmark dir (which has the right
    # AGENTS.md) with a manifest that lives elsewhere (whose dir has a decoy —
    # e.g. the repo-root "Repo Rules"). The CWD's AGENTS.md must win.
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "AGENTS.md").write_text("RIGHT (benchmark)", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "AGENTS.md").write_text("WRONG (decoy next to the manifest)", encoding="utf-8")
    manifest = elsewhere / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    monkeypatch.chdir(bench)
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(["--manifest", str(manifest), "--output-dir", str(tmp_path / "out"), "--model", "gpt-x"])

    assert seen[0]["system_prompt"] == "RIGHT (benchmark)"


def test_no_agents_md_disables_discovery(tmp_path, monkeypatch) -> None:
    (tmp_path / "AGENTS.md").write_text("SHARED", encoding="utf-8")
    manifest = tmp_path / "b.jsonl"
    manifest.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    seen: list[dict] = []
    monkeypatch.setattr(batch_agent.agent_loop, "run_agent_loop", _fake_loop(_png(), seen))

    batch_agent.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gpt-x",
            "--no-agents-md",
        ]
    )

    assert seen[0]["system_prompt"] is None


def _shard_items(n):
    return [
        batch_agent.BatchItem(index=k, item_id=f"i{k}", input_path=None, prompt=f"p{k}") for k in range(n)
    ]


def test_parse_shard_accepts_and_rejects():
    assert batch_agent.parse_shard("0/8") == (0, 8)
    assert batch_agent.parse_shard("7/8") == (7, 8)
    for bad in ("8/8", "-1/4", "1/0", "half", "1/2/3", ""):
        with pytest.raises(batch_agent.BatchHarnessError):
            batch_agent.parse_shard(bad)


def test_shards_partition_the_manifest_exactly_once():
    """Every item lands in exactly one shard: no gaps, no duplicates."""
    items = _shard_items(2000)
    seen = [i.item_id for k in range(8) for i in batch_agent.select_shard(items, k, 8)]
    assert sorted(seen) == sorted(i.item_id for i in items)
    assert len(seen) == len(set(seen))


def test_shards_are_strided_and_balanced():
    """Strided, so category-grouped manifests spread across shards evenly."""
    items = _shard_items(2000)
    sizes = {len(batch_agent.select_shard(items, k, 8)) for k in range(8)}
    assert sizes == {250}
    # Contiguous slicing would give shard 0 items i0..i249; strided gives every 8th.
    assert [i.item_id for i in batch_agent.select_shard(items, 0, 8)][:3] == ["i0", "i8", "i16"]


def test_uneven_shards_differ_by_at_most_one():
    items = _shard_items(2003)
    sizes = [len(batch_agent.select_shard(items, k, 8)) for k in range(8)]
    assert sum(sizes) == 2003
    assert max(sizes) - min(sizes) <= 1


def test_reasoning_effort_defaults_from_the_environment(monkeypatch) -> None:
    """The strength that used to select an endpoint is now a request parameter.

    kimi-k3 replaced the -low/-high/-max-preview endpoints, so the benchmark
    config sets the model once and the strength beside it; the flag reads that
    env var so a run does not silently drop to the default effort.
    """
    monkeypatch.setenv("CONTINUUM_LLM_REASONING_EFFORT", "max")
    args = batch_agent.build_parser().parse_args(["--manifest", "m.jsonl", "--out", "o"])
    assert args.reasoning_effort == "max"


def test_reasoning_effort_is_unset_without_configuration(monkeypatch) -> None:
    # Unset means the field is left out of the request entirely: an endpoint
    # that does not know it may reject the call rather than ignore it.
    monkeypatch.delenv("CONTINUUM_LLM_REASONING_EFFORT", raising=False)
    args = batch_agent.build_parser().parse_args(["--manifest", "m.jsonl", "--out", "o"])
    assert args.reasoning_effort is None


def test_a_failed_run_still_writes_the_image_it_had() -> None:
    """An abort keeps what the model marked, rather than losing the item.

    The loop resolves output_image before returning, so a run cut off by a step
    limit or a 413 still carries an image. Throwing it away costs the whole item
    for a reason unrelated to its quality — and with mark_best that image is one
    the model explicitly chose.
    """
    from continuum.harnesses import agent_loop

    loop = agent_loop.LoopResult(
        status="failed",
        output_image=b"salvaged",
        output_image_id="img_1",
        steps=12,
        transcript=[],
        error="APIStatusError: 413",
        stop_reason="error",
        kept_id="img_1",
    )
    assert loop.output_image is not None
    # The attempt is not accepted, so a resume retries it rather than treating
    # a salvaged failure as a finished item.
    assert not batch_agent._is_accepted({"status": loop.status, "stop_reason": loop.stop_reason})
