import asyncio
import base64
import io
import threading

import pytest
from PIL import Image

from continuum import mcp_server
from continuum.mcp_server import _build_server


class FakeContext:
    def __init__(self) -> None:
        self.progress: list[tuple[float, float | None, str | None]] = []

    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        self.progress.append((progress, total, message))


def _image_content_from_pil(img: Image.Image) -> dict[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mimeType": "image/png",
    }


def test_model_queued_progress_is_indeterminate(monkeypatch) -> None:
    ctx = FakeContext()
    tool = _build_server(compat_mode=False)._tool_manager._tools["segment"]

    async def fail_after_progress(*_args, **_kwargs) -> None:
        raise RuntimeError("stop before model load")

    from continuum.core import runtime

    monkeypatch.setattr(runtime, "run_locked", fail_after_progress)

    async def run_tool() -> None:
        await tool.run(
            {
                "image": _image_content_from_pil(Image.new("RGB", (8, 8), "white")),
                "description": "white square",
            },
            context=ctx,
        )

    with pytest.raises(Exception):
        asyncio.run(run_tool())

    assert ctx.progress[0] == (0, None, "queued")


def test_loading_progress_flushes_when_token_present(monkeypatch) -> None:
    ctx = FakeContext()
    ctx.request_context = type(
        "RequestContext",
        (),
        {"meta": type("Meta", (), {"progressToken": "token-1"})()},
    )()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(mcp_server.asyncio, "sleep", fake_sleep)

    asyncio.run(mcp_server._report_loading_progress(ctx, "edit_image", "loading model"))

    assert ctx.progress == [(0, None, "loading model")]
    assert sleeps == [0.05]


def test_load_progress_bridge_reports_normalized_determinate_payload() -> None:
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "edit_image")
        cb("downloading model", 7, 10)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert ctx.progress == [(70.0, 100.0, "downloading model")]


def test_load_progress_bridge_renames_loading_weights() -> None:
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "generate_image")
        cb("Loading weights", 1, 2)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert ctx.progress == [(50.0, 100.0, "loading model")]


def test_load_progress_bridge_throttles_repeated_small_updates() -> None:
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "generate_image")
        cb("downloading model", 0, 1000)
        cb("downloading model", 1, 1000)
        cb("downloading model", 2, 1000)
        cb("downloading model", 1000, 1000)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert ctx.progress == [
        (0.0, 100.0, "downloading model"),
        (100.0, 100.0, "downloading model"),
    ]


def test_load_progress_bridge_suppresses_same_message_regression() -> None:
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "generate_image")
        cb("loading model", 100, 100)
        cb("loading model", 1, 2)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert ctx.progress == [(100.0, 100.0, "loading model")]


def test_load_progress_bridge_flushes_from_worker_thread() -> None:
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "generate_image")
        thread = threading.Thread(
            target=lambda: cb("loading model", 1, 2),
        )
        thread.start()
        while thread.is_alive():
            await asyncio.sleep(0.01)
        thread.join()

    asyncio.run(run())

    assert ctx.progress == [(50.0, 100.0, "loading model")]


def test_generation_progress_message_is_generating() -> None:
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._progress_bridge(ctx, "generate_image")
        cb(3, 10)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert ctx.progress == [(3, 10, "generating")]


def test_run_locked_logs_worker_exceptions(capsys) -> None:
    async def run() -> None:
        def fail() -> None:
            raise RuntimeError("model exploded")

        with pytest.raises(RuntimeError):
            await mcp_server._run_locked(fail, tool_name="edit_image")

    asyncio.run(run())

    out = capsys.readouterr().out
    assert "[mcp] edit_image: ERROR RuntimeError: model exploded" in out
    assert "Traceback" in out


def test_load_progress_is_whole_percent() -> None:
    """A ratio that does not divide evenly still reports a round number.

    The scale is bytes downloaded, so the percentage is a full-precision float.
    Hosts print the number they are handed, and a model download reported as
    "37.14285714285714/100" is what that produces.
    """
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "generate_image")
        cb("downloading model", 13, 35)  # 37.142857...%
        await asyncio.sleep(0)

    asyncio.run(run())

    progress, total, _ = ctx.progress[0]
    assert progress == 37.0, progress
    assert total == 100.0
    assert progress == int(progress), "a fractional percent reaches the host verbatim"


def test_load_progress_still_reaches_a_hundred() -> None:
    """Rounding must not leave a completed download short of complete."""
    ctx = FakeContext()

    async def run() -> None:
        cb = mcp_server._load_progress_bridge(ctx, "generate_image")
        cb("downloading model", 35, 35)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert ctx.progress[-1][0] == 100.0
