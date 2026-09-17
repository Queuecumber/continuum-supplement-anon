"""Adapters between MCP request context and the core layer's Hooks.

Everything here is view-side: it turns ctx.report_progress into plain
callbacks the transport-free ops can drive from worker threads.
"""

import asyncio
import functools
import threading
import time
from concurrent.futures import TimeoutError as _FuturesTimeoutError

from mcp.server.fastmcp import Context

from continuum.core.ops import Hooks
from continuum.core.runtime import log, log_error


def progress_token(ctx: Context) -> object | None:
    request_context = getattr(ctx, "request_context", None)
    meta = getattr(request_context, "meta", None)
    return getattr(meta, "progressToken", None) if meta else None


def _progress_message(message: str | None) -> str | None:
    if message is None:
        return None
    if message.strip().lower().rstrip(".") == "loading weights":
        return "loading model"
    return message


async def report_loading_progress(ctx: Context, tool_name: str, message: str) -> None:
    """Emit initial indeterminate progress and let the transport flush it."""
    message = _progress_message(message) or message
    token = progress_token(ctx)
    log(f"{tool_name}: {message}")
    log(f"{tool_name}: progressToken={token!r}", verbose=True)
    await ctx.report_progress(0, None, message)
    if token is None:
        await asyncio.sleep(0)
        return

    await asyncio.sleep(0.05)


def load_progress_bridge(
    ctx: Context,
    tool_name: str = "",
    cancel: threading.Event | None = None,
):
    """Return a sync loader callback that schedules MCP progress updates."""
    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()
    token = progress_token(ctx)
    log(f"{tool_name}: load progressToken={token!r}", verbose=True)

    fired = [0]
    last_emit_time = [0.0]
    last_message: list[str | None] = [None]
    last_progress = [-1.0]
    min_interval_s = 1.0
    min_delta = 0.5
    send_timeout_s = 0.25

    def cb(message: str, progress: float = 0, total: float | None = None) -> None:
        if cancel is not None and cancel.is_set():
            raise KeyboardInterrupt(f"{tool_name} cancelled by host")
        message = _progress_message(message) or message
        now = time.monotonic()
        out_progress = float(progress)
        out_total = None if total is None else float(total)
        if out_total and out_total > 0:
            # Whole percent. The scale is bytes downloaded or shards loaded, so
            # the ratio is a full-precision float — hosts render the number they
            # are given, and "downloading model 37.14285714285714/100" is what
            # that looks like. A percent is finer than any progress bar shows.
            out_progress = float(round(max(0.0, min(100.0, (out_progress / out_total) * 100.0))))
            out_total = 100.0
        else:
            out_progress = 0.0

        changed_message = message != last_message[0]
        if (
            not changed_message
            and out_total is not None
            and last_progress[0] >= 0.0
            and out_progress < last_progress[0]
        ):
            return
        terminal = out_total is not None and out_progress >= out_total
        initial = out_total is not None and out_progress <= 0.0 and last_progress[0] < 0.0
        enough_progress = out_total is not None and out_progress - last_progress[0] >= min_delta
        enough_time = now - last_emit_time[0] >= min_interval_s
        if not (changed_message or initial or terminal or enough_progress or enough_time):
            return

        last_emit_time[0] = now
        last_message[0] = message
        last_progress[0] = out_progress
        fired[0] += 1
        if changed_message:
            log(f"{tool_name}: {message}")
        log(
            f"{tool_name}: load #{fired[0]} ({out_progress}/{out_total}) {message}",
            verbose=True,
        )
        fut = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(out_progress, out_total, message),
            loop,
        )

        def _check(f):
            try:
                f.result(timeout=0)
            except Exception as e:
                log(f"{tool_name}: load report_progress error: {e!r}")

        fut.add_done_callback(_check)
        if threading.get_ident() != loop_thread_id:
            try:
                fut.result(timeout=max(0.0, send_timeout_s))
            except _FuturesTimeoutError:
                pass
            except Exception as e:
                log(f"{tool_name}: load report_progress error: {e!r}")

    return cb


def progress_bridge(
    ctx: Context,
    tool_name: str = "",
    cancel: threading.Event | None = None,
    message: str = "generating",
    min_interval_s: float = 0.0,
):
    """Return a sync (step, total) callback that schedules
    ctx.report_progress on the running loop. If `cancel` is set when
    the callback fires, raises KeyboardInterrupt to abort the diffusion
    loop. Safe to call from worker threads.

    `message` names the stage, because more than one thing in a tool call
    takes minutes and the host shows this text to the user.

    `min_interval_s` rate-limits the reports. Diffusion steps are seconds
    apart and want none, but a frequent callback can fire a hundred times a
    second — which is noise on the wire and churn in the host's display. The
    final step is always sent, so the bar still lands on complete."""
    loop = asyncio.get_running_loop()

    # Diagnose silent-progress complaints: log once whether the host opted
    # into progress notifications. fastmcp.report_progress is a no-op when
    # the request meta has no progressToken — so if this prints `None`,
    # nothing we send will ever surface to the host.
    token = progress_token(ctx)
    log(f"{tool_name}: progressToken={token!r}", verbose=True)

    fired = [0]
    last_emit_time = [0.0]

    def cb(step: int, total: int) -> None:
        if cancel is not None and cancel.is_set():
            raise KeyboardInterrupt(f"{tool_name} cancelled by host")
        if min_interval_s:
            now = time.monotonic()
            terminal = total is not None and step >= total
            if not terminal and now - last_emit_time[0] < min_interval_s:
                return
            last_emit_time[0] = now
        fired[0] += 1
        log(
            f"{tool_name}: on_step #{fired[0]} ({step}/{total})",
            verbose=True,
        )
        fut = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(step, total, message),
            loop,
        )

        # Surface any exception (closed session, etc.) instead of swallowing
        def _check(f):
            try:
                f.result(timeout=0)
            except Exception as e:
                log(f"{tool_name}: report_progress error: {e!r}")

        fut.add_done_callback(_check)

    return cb


def make_hooks(ctx: Context, tool_name: str) -> Hooks:
    """Bundle the ctx-derived callbacks for a core op call."""
    cancel = threading.Event()
    return Hooks(
        on_step=progress_bridge(ctx, tool_name, cancel=cancel),
        on_load=load_progress_bridge(ctx, tool_name, cancel=cancel),
        cancel=cancel,
        notify=lambda message: report_loading_progress(ctx, tool_name, message),
    )


def log_tool_errors(fn):
    """Log tool exceptions server-side under the wrapped function's name."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            log_error(fn.__name__, e)
            raise

    return wrapper
