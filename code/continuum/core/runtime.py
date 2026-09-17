"""Process-wide runtime state: model manager singleton, GPU lock, logging."""

import asyncio
import sys
import threading
import traceback

from continuum.models import ModelManager

_models: ModelManager | None = None


def get_models() -> ModelManager:
    global _models
    if _models is None:
        _models = ModelManager()
    return _models


# Single lock serializes all GPU-bound tool calls. Two reasons:
#   1. Diffusers schedulers maintain an internal step counter; running two
#      diffusion loops concurrently against the same pipeline blows past
#      num_inference_steps and OOBs on sigmas[N+1].
#   2. The image pipelines and segmenter all use shared GPU
#      memory; serializing avoids OOM races.
# Hosts that fire concurrent tool calls (Claude Code does) will queue here.
gpu_lock = asyncio.Lock()


def log(message: str, *, verbose: bool = False) -> None:
    if verbose:
        return
    print(f"[mcp] {message}", flush=True)


def log_error(tool_name: str, exc: BaseException) -> None:
    if getattr(exc, "_continuum_mcp_logged", False):
        return
    try:
        setattr(exc, "_continuum_mcp_logged", True)
    except Exception:
        pass
    log(f"{tool_name}: ERROR {type(exc).__name__}: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)


async def run_locked(
    fn,
    cancel: threading.Event | None = None,
    tool_name: str | None = None,
):
    """Run a sync GPU-bound function on the executor while holding gpu_lock.

    If `cancel` is provided and the awaiting coroutine is cancelled
    (asyncio.CancelledError, e.g. when the MCP host sends a cancellation
    notification), the event is set and we wait for the executor thread
    to drain before releasing the lock — so the next caller starts on a
    clean pipeline. The blocking function should poll the event via its
    on_step callback and raise KeyboardInterrupt to abort the diffusion
    loop early; without that the thread runs to completion and only the
    next call benefits from cancellation.
    """
    async with gpu_lock:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, fn)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if cancel is not None:
                cancel.set()
            # Drain inside the lock so the next caller doesn't race with
            # the executor thread still tearing down the previous job.
            try:
                await asyncio.shield(future)
            except BaseException:
                pass
            raise
        except BaseException as e:
            if tool_name is not None:
                log_error(tool_name, e)
            raise
