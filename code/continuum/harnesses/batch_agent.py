"""Batch agent harness: run an OpenAI-compatible agent over many items.

Two input modes: a global prompt over a directory of images, or a JSONL
benchmark manifest with a per-item prompt (and optional image) per line. Per
item (and per --samples) the harness drives an OpenAI-compatible LLM through
Continuum's MCP tools via continuum.harnesses.agent_loop — the harness owns
the loop and the client-side id<->bytes registry against a STATELESS
standard-mode Continuum server. Outputs an eval-ready id -> output image
mapping plus a per-run transcript.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from continuum.harnesses import agent_loop

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - progress bar is an optional nicety
    _tqdm = None

IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class BatchItem:
    index: int
    item_id: str
    input_path: Path | None
    prompt: str | None = None  # per-item prompt (manifest mode); None falls back to the global --prompt


@dataclass(frozen=True)
class AgentRunResult:
    item: BatchItem
    status: str
    prompt: str
    sample_index: int
    output_path: Path | None
    steps: int
    transcript_path: Path | None
    started_at: float
    finished_at: float
    error: str | None = None
    stop_reason: str = ""  # completed | max_steps | error | dry-run (see LoopResult.stop_reason)
    # Provenance for licence review / training-data filtering.
    models: list[str] = field(default_factory=list)  # every model used in this run
    output_model: str | None = None  # the model that produced the saved output
    # Which image the model explicitly kept via the keep tool, and why. None
    # means it never kept anything, so the output is whatever was produced last.
    kept_id: str | None = None
    kept_reason: str = ""


class BatchHarnessError(RuntimeError):
    """Raised when the batch harness cannot complete a requested operation."""


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    value = value.strip(".-")
    return value[:80] or "item"


def discover_items(image_dir: Path | None = None, images: Sequence[Path] = ()) -> list[BatchItem]:
    paths: list[Path] = []
    if image_dir is not None:
        if not image_dir.exists():
            raise BatchHarnessError(f"image directory does not exist: {image_dir}")
        if not image_dir.is_dir():
            raise BatchHarnessError(f"image path is not a directory: {image_dir}")
        paths.extend(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    paths.extend(images)
    paths = sorted({path.resolve() for path in paths}, key=lambda path: str(path).lower())

    if not paths:
        return [BatchItem(index=0, item_id="0000-prompt", input_path=None)]

    items: list[BatchItem] = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise BatchHarnessError(f"input image is not a file: {path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise BatchHarnessError(f"input file does not look like an image: {path}")
        items.append(
            BatchItem(index=index, item_id=f"{index:04d}-{sanitize_name(path.stem)}", input_path=path)
        )
    return items


def read_prompt(args: argparse.Namespace) -> str | None:
    """The global prompt, or None. Optional in --manifest mode (per-item prompts)."""
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


# Manifest field aliases so common benchmark schemas load without conversion.
_MANIFEST_PROMPT_KEYS = ("prompt", "instruction", "edit", "caption")
_MANIFEST_IMAGE_KEYS = ("image", "source", "input", "input_image")


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse `I/N` into a zero-based shard index and a shard count."""
    try:
        index_text, count_text = spec.split("/", 1)
        index, count = int(index_text), int(count_text)
    except ValueError:
        raise BatchHarnessError(f"--shard must look like 0/8, got {spec!r}") from None
    if count < 1:
        raise BatchHarnessError("--shard count must be at least 1")
    if not 0 <= index < count:
        raise BatchHarnessError(f"--shard index must be in 0..{count - 1}, got {index}")
    return index, count


def select_shard(items: Sequence[BatchItem], index: int, count: int) -> list[BatchItem]:
    """Every `count`-th item, starting at `index`.

    Strided rather than contiguous because manifests are grouped by category: a
    contiguous slice would give one shard every hard prompt and another every
    easy one, so a partial run would not represent the whole set and shards
    would finish at wildly different times.
    """
    return list(items[index::count])


def read_manifest_items(
    manifest_path: Path,
    *,
    image_root: Path | None = None,
    default_prompt: str | None = None,
) -> list[BatchItem]:
    """Parse a JSONL benchmark manifest into per-item BatchItems.

    One JSON object per line: `prompt` (or instruction/edit/caption), optional
    `image` (or source/input, resolved against image_root), optional `id`. A
    row without a prompt falls back to default_prompt.
    """
    if not manifest_path.is_file():
        raise BatchHarnessError(f"manifest does not exist: {manifest_path}")
    items: list[BatchItem] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchHarnessError(f"{manifest_path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise BatchHarnessError(f"{manifest_path}:{lineno}: expected a JSON object")
        prompt = next((row[key] for key in _MANIFEST_PROMPT_KEYS if row.get(key)), None) or default_prompt
        if not prompt:
            raise BatchHarnessError(f"{manifest_path}:{lineno}: row has no prompt and no --prompt fallback")
        image_value = next((row[key] for key in _MANIFEST_IMAGE_KEYS if row.get(key)), None)
        input_path: Path | None = None
        if image_value:
            candidate = Path(str(image_value))
            if not candidate.is_absolute() and image_root is not None:
                candidate = image_root / candidate
            input_path = candidate.resolve()
            if not input_path.is_file():
                raise BatchHarnessError(f"{manifest_path}:{lineno}: image not found: {input_path}")
        index = len(items)
        raw_id = row.get("id", row.get("key"))
        item_id = sanitize_name(str(raw_id)) if raw_id is not None else f"{index:04d}"
        if item_id in seen_ids:
            raise BatchHarnessError(f"{manifest_path}:{lineno}: duplicate item id {item_id!r}")
        seen_ids.add(item_id)
        items.append(BatchItem(index=index, item_id=item_id, input_path=input_path, prompt=str(prompt)))
    if not items:
        raise BatchHarnessError(f"manifest produced no items: {manifest_path}")
    return items


def _resolve_items(args: argparse.Namespace, default_prompt: str | None) -> list[BatchItem]:
    if args.manifest:
        if args.image_dir or args.image:
            raise BatchHarnessError("--manifest cannot be combined with --image-dir/--image")
        image_root = Path(args.image_root).resolve() if args.image_root else None
        return read_manifest_items(Path(args.manifest), image_root=image_root, default_prompt=default_prompt)
    if not default_prompt:
        raise BatchHarnessError("provide --prompt/--prompt-file, or --manifest with per-item prompts")
    images = [Path(path) for path in args.image]
    image_dir = Path(args.image_dir) if args.image_dir else None
    return discover_items(image_dir=image_dir, images=images)


def _parse_tool_list(value: str | None) -> set[str] | None:
    """Comma-separated tool names -> set, or None if unset/empty."""
    if not value:
        return None
    names = {name.strip() for name in value.split(",") if name.strip()}
    return names or None


def parse_pin_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", value)
    if not match:
        raise BatchHarnessError(f"--pin-size must look like 1024x1024, got {value!r}")
    return int(match.group(1)), int(match.group(2))


def _run_label(item_id: str, sample_index: int, num_samples: int) -> str:
    return item_id if num_samples == 1 else f"{item_id}#{sample_index}"


def _item_dir(output_root: Path, item_id: str, sample_index: int, num_samples: int) -> Path:
    base = output_root / item_id
    return base if num_samples == 1 else base / f"sample_{sample_index:02d}"


def _output_target(output_root: Path, item_id: str, sample_index: int, num_samples: int, suffix: str) -> Path:
    outputs_dir = output_root / "outputs"
    if num_samples > 1:
        return outputs_dir / item_id / f"{sample_index}{suffix}"
    return outputs_dir / f"{item_id}{suffix}"


def _save_images(images_dir: Path, images: dict[str, bytes]) -> dict[str, str]:
    """Write each id -> bytes to images_dir/<id>.png; return id -> relative path."""
    if not images:
        return {}
    images_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for img_id, data in images.items():
        (images_dir / f"{img_id}.png").write_bytes(data)
        saved[img_id] = f"images/{img_id}.png"  # relative to the item dir
    return saved


def _relink_transcript(transcript: list[dict[str, Any]], saved: dict[str, str]) -> list[dict[str, Any]]:
    """Rewrite the transcript's bare image-id refs to their saved file paths."""
    out = json.loads(json.dumps(transcript))  # small (data-urls already stripped)
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url")
                if url in saved:
                    part["image_url"]["url"] = saved[url]
    return out


def run_item_sample(
    *,
    item: BatchItem,
    item_prompt: str,
    sample_index: int,
    num_samples: int,
    output_root: Path,
    system_prompt: str | None,
    args: argparse.Namespace,
    pin_size: tuple[int, int] | None,
    mcp_url: str,
) -> AgentRunResult:
    item_dir = _item_dir(output_root, item.item_id, sample_index, num_samples)
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "prompt.txt").write_text(item_prompt, encoding="utf-8")
    started_at = time.time()

    if args.dry_run:
        return AgentRunResult(
            item=item,
            status="dry-run",
            prompt=item_prompt,
            sample_index=sample_index,
            output_path=None,
            steps=0,
            transcript_path=None,
            started_at=started_at,
            finished_at=time.time(),
            stop_reason="dry-run",
        )

    input_images = [item.input_path.read_bytes()] if item.input_path is not None else []
    on_event = None
    if getattr(args, "verbose", False):
        label = _run_label(item.item_id, sample_index, num_samples)

        def on_event(message: str, _label: str = label) -> None:
            print(f"    [{_label}] {message}", flush=True)

    try:
        if args.direct:
            # Baseline: prompt straight to the generator, no LLM in the loop.
            loop = asyncio.run(
                agent_loop.run_direct_tool_call(
                    task_prompt=item_prompt,
                    mcp_url=mcp_url,
                    input_images=input_images,
                    tool_name=args.direct_tool,
                    image_model=args.image_model,
                    pin_size=pin_size,
                    strip_seed=args.strip_seed,
                    on_event=on_event,
                )
            )
        else:
            loop = asyncio.run(
                agent_loop.run_agent_loop(
                    task_prompt=item_prompt,
                    system_prompt=system_prompt,
                    input_images=input_images,
                    mcp_url=mcp_url,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
                    max_steps=args.max_steps,
                    max_image_side=args.max_image_side,
                    pin_size=pin_size,
                    strip_seed=args.strip_seed,
                    include_server_instructions=not args.no_server_instructions,
                    on_event=on_event,
                    max_retries=args.max_retries,
                    request_delay=args.request_delay,
                    max_inline_images=args.max_inline_images,
                    reasoning_effort=args.reasoning_effort,
                    image_model=args.image_model,
                    allow_tools=_parse_tool_list(args.tools),
                    block_tools=_parse_tool_list(args.exclude_tools),
                    fail_on_truncation=getattr(args, "fail_on_truncation", False),
                )
            )
    except Exception as exc:
        return AgentRunResult(
            item=item,
            status="failed",
            prompt=item_prompt,
            sample_index=sample_index,
            output_path=None,
            steps=0,
            transcript_path=None,
            started_at=started_at,
            finished_at=time.time(),
            error=agent_loop._format_exc(exc),
            stop_reason="error",
        )

    saved_images = _save_images(item_dir / "images", loop.images)
    transcript_path = item_dir / "transcript.json"
    transcript = _relink_transcript(loop.transcript, saved_images)
    transcript_path.write_text(json.dumps(transcript, indent=2, default=str), encoding="utf-8")

    # A run that produced an image yields a deliverable even if it later failed:
    # the loop resolves output_image before returning, so an abort — a step
    # limit, or the endpoint refusing an oversized request — still carries the
    # model's marked best, or the last image it made if it marked nothing.
    # Discarding that loses real work for a reason that has nothing to do with
    # the image's quality.
    #
    # status stays as it was, so _is_accepted still reads False and a resume
    # retries the item; this only means the attempt is not empty in the
    # meantime.
    output_path: Path | None = None
    if loop.output_image is not None:
        (item_dir / "output.png").write_bytes(loop.output_image)
        output_path = _output_target(output_root, item.item_id, sample_index, num_samples, ".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(loop.output_image)

    return AgentRunResult(
        item=item,
        status=loop.status,
        prompt=item_prompt,
        sample_index=sample_index,
        output_path=output_path,
        steps=loop.steps,
        transcript_path=transcript_path,
        started_at=started_at,
        finished_at=time.time(),
        kept_id=loop.kept_id,
        kept_reason=loop.kept_reason,
        error=loop.error,
        stop_reason=loop.stop_reason,
        models=sorted(set(loop.image_models.values())),
        output_model=loop.image_models.get(loop.output_image_id or ""),
    )


def manifest_record(result: AgentRunResult) -> dict[str, Any]:
    return {
        "item_id": result.item.item_id,
        "sample_index": result.sample_index,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "models": result.models,
        "output_model": result.output_model,
        "prompt": result.prompt,
        "input_image": str(result.item.input_path) if result.item.input_path else None,
        "output_image": str(result.output_path) if result.output_path else None,
        # Which image the model explicitly kept, so a later analysis can ask
        # whether keeping changed the outcome at all.
        "kept_id": result.kept_id,
        "kept_reason": result.kept_reason,
        "transcript": str(result.transcript_path) if result.transcript_path else None,
        "steps": result.steps,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": round(result.finished_at - result.started_at, 3),
        "error": result.error,
    }


def append_manifest(manifest_path: Path, result: AgentRunResult) -> None:
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_record(result), sort_keys=True) + "\n")


def read_progress(manifest_path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    """Per-(item_id, sample_index) attempt history from a prior manifest.jsonl.

    Each attempt is {"status", "stop_reason"}. Tolerant of blank and corrupt
    lines — a job killed mid-write can leave a truncated final row — so
    autoresume survives preemption. Multiple rows for one key (a retried run)
    accumulate in order.
    """
    history: dict[tuple[str, int], list[dict[str, str]]] = {}
    if not manifest_path.is_file():
        return history
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            key = (str(row["item_id"]), int(row["sample_index"]))
            attempt = {"status": str(row["status"]), "stop_reason": str(row.get("stop_reason", ""))}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # skip malformed / partially-written rows
        history.setdefault(key, []).append(attempt)
    return history


def _is_accepted(attempt: dict[str, str]) -> bool:
    """Did the model accept its output on this attempt?

    True only when it returned control (stop_reason 'completed') with an image
    (status 'ok') — a run that --max-steps cut off mid-work is NOT accepted, so
    resume retries it. 'direct' (--direct baseline, no agent) is accepted by
    definition: one blind call, no iteration to resume. Legacy rows without a
    stop_reason fall back to status ok.
    """
    if attempt.get("status") != "ok":
        return False
    return attempt.get("stop_reason", "") in ("", "completed", "direct")


def _disposition(attempts: list[dict[str, str]], max_attempts: int) -> str:
    """done | exhausted | pending, from a run's prior attempt history.

    done: the model accepted an output — never rerun. exhausted: only when
    max_attempts > 0 and that many attempts have not accepted — give up so the
    requeue can terminate. pending: retry on this invocation. With
    max_attempts == 0 (default) an unaccepted run is retried indefinitely.
    """
    if any(_is_accepted(a) for a in attempts):
        return "done"
    if max_attempts and len(attempts) >= max_attempts:
        return "exhausted"
    return "pending"


def _find_agents_md(start: Path) -> Path | None:
    """Nearest AGENTS.md walking up from `start` (the AGENTS.md convention)."""
    start = start.resolve()
    for parent in (start, *start.parents):
        candidate = parent / "AGENTS.md"
        if candidate.is_file():
            return candidate
    return None


def resolve_system_prompt(args: argparse.Namespace) -> str | None:
    """The shared agent system prompt.

    AGENTS.md (explicit --agents-md, else the nearest one walking up from the
    CWD — run the harness from your benchmark directory — unless --no-agents-md)
    is the shared base; --system-prompt-file is appended as run-specific policy.
    The resolved source is logged so a wrong or missing pickup is never silent.
    """
    parts: list[str] = []
    if not args.no_agents_md:
        if args.agents_md:
            agents_md: Path | None = Path(args.agents_md)
            if not agents_md.is_file():
                raise BatchHarnessError(f"--agents-md not found: {agents_md}")
        else:
            agents_md = _find_agents_md(Path.cwd())
        if agents_md is not None:
            print(f"system prompt: AGENTS.md at {agents_md}", flush=True)
            parts.append(agents_md.read_text(encoding="utf-8").strip())
        else:
            print(f"system prompt: no AGENTS.md found (searched up from {Path.cwd()})", flush=True)
    if args.system_prompt_file:
        print(f"system prompt: + {args.system_prompt_file}", flush=True)
        parts.append(Path(args.system_prompt_file).read_text(encoding="utf-8").strip())
    return "\n\n".join(p for p in parts if p) or None


def _require_agent_deps() -> None:
    """Fail fast (before launching a server) if the LLM client isn't installed."""
    if importlib.util.find_spec("openai") is None:
        raise BatchHarnessError(
            "the agent loop needs the 'openai' package — install with "
            "`pip install 'continuum[agent]'` (or `pip install openai`)."
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@contextmanager
def _mcp_server(args: argparse.Namespace, output_root: Path):
    """Yield the MCP URL to drive: the provided --mcp-url, or a managed
    standard-mode continuum server launched on a free local port and torn
    down afterward. Dry-run never touches a server.
    """
    if args.mcp_url or args.dry_run:
        yield args.mcp_url or "http://127.0.0.1:8742/mcp"
        return

    port = _free_port()
    log_path = output_root / "mcp-server.log"
    print(f"launching continuum MCP server on 127.0.0.1:{port} (log: {log_path})", flush=True)
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "continuum.mcp_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--transport",
                "http",
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        try:
            if not _wait_for_port("127.0.0.1", port, args.mcp_startup_timeout):
                raise BatchHarnessError(
                    f"launched continuum MCP server never became ready on port {port} (see {log_path})"
                )
            time.sleep(0.5)  # small grace for FastMCP to finish wiring routes
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_batch(args: argparse.Namespace) -> int:
    default_prompt = read_prompt(args)
    # --direct has no agent, so no system prompt / LLM / openai dependency.
    system_prompt = None if args.direct else resolve_system_prompt(args)
    if not args.dry_run and not args.direct and not args.model:
        raise BatchHarnessError("provide --model (the OpenAI-compatible model that drives the agent)")
    if not args.dry_run and not args.direct:
        _require_agent_deps()
    pin_size = parse_pin_size(args.pin_size)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    num_samples = max(1, int(args.samples))
    items = _resolve_items(args, default_prompt)
    if args.shard:
        shard_index, shard_count = parse_shard(args.shard)
        items = select_shard(items, shard_index, shard_count)
        print(f"shard {shard_index}/{shard_count}: {len(items)} items", flush=True)
    manifest_path = output_root / "manifest.jsonl"
    max_attempts = max(0, int(args.max_attempts))

    # Resume is the default: skip runs already recorded ok, retry the rest.
    # Append-only + tolerant reads make re-invoking the same command safe after
    # preemption (stateless autoresume under a scheduler). --no-resume starts clean.
    if args.no_resume and manifest_path.exists():
        manifest_path.unlink()
    progress = read_progress(manifest_path)

    planned: list[tuple[BatchItem, int]] = []
    done = exhausted = 0
    for item in items:
        for sample_index in range(num_samples):
            disp = _disposition(progress.get((item.item_id, sample_index), []), max_attempts)
            if disp == "done":
                done += 1
            elif disp == "exhausted":
                exhausted += 1
            else:
                planned.append((item, sample_index))
    total_runs = len(items) * num_samples
    if done or exhausted:
        print(
            f"resume: {done} done, {exhausted} exhausted, {len(planned)} to run ({total_runs} total)",
            flush=True,
        )
    if not planned:
        print("nothing to do — every run is complete", flush=True)
        return 0

    attempts_per_invocation = max(1, int(args.attempts_per_invocation))
    prior_attempts = {
        (item.item_id, s): len(progress.get((item.item_id, s), []))
        for item in items
        for s in range(num_samples)
    }
    in_proc: dict[tuple[str, int], int] = {}
    queue: deque[tuple[BatchItem, int]] = deque(planned)
    run_index = 0
    still_pending = 0
    counts = {"ok": 0, "failed": 0, "pending": 0}
    # A live bar on an interactive terminal; disable=None auto-disables it on a
    # non-TTY (a scheduler log), where the per-run lines below are the output.
    # Skipped under --verbose (which streams its own per-step lines) and --dry-run.
    show_bar = _tqdm is not None and not args.verbose and not args.dry_run
    bar = _tqdm(total=len(planned), unit="run", file=sys.stdout, disable=None) if show_bar else None

    def line(msg: str) -> None:
        bar.write(msg) if bar is not None else print(msg, flush=True)

    def advance(bucket: str) -> None:
        counts[bucket] += 1
        if bar is not None:
            bar.set_postfix(counts, refresh=False)
            bar.update(1)

    with _mcp_server(args, output_root) as mcp_url:
        while queue:
            item, sample_index = queue.popleft()
            key = (item.item_id, sample_index)
            item_prompt = item.prompt or default_prompt
            if not item_prompt:
                raise BatchHarnessError(f"item {item.item_id} has no prompt and no --prompt fallback")
            run_index += 1
            label = _run_label(item.item_id, sample_index, num_samples)
            attempt = prior_attempts.get(key, 0) + in_proc.get(key, 0) + 1
            done = counts["ok"] + counts["failed"] + counts["pending"]
            line(f"[{done}/{len(planned)}] starting {label} (run {run_index}, attempt {attempt})")
            result = run_item_sample(
                item=item,
                item_prompt=item_prompt,
                sample_index=sample_index,
                num_samples=num_samples,
                output_root=output_root,
                system_prompt=system_prompt,
                args=args,
                pin_size=pin_size,
                mcp_url=mcp_url,
            )
            in_proc[key] = in_proc.get(key, 0) + 1
            append_manifest(manifest_path, result)
            elapsed = f"{result.finished_at - result.started_at:.1f}s"
            line(
                f"    {result.status}/{result.stop_reason} ({result.steps} steps, {elapsed}) "
                f"-> {result.output_path or '(no output)'}"
            )

            # Same rule resume uses, so the two can't drift apart.
            accepted = _is_accepted({"status": result.status, "stop_reason": result.stop_reason})
            if accepted or result.status == "dry-run":
                advance("ok")  # the model accepted its output (or this is a dry run) — resolved
                continue
            if result.error:
                line(f"    error: {result.error}")

            if result.status == "failed":
                # Transient error — kick it back onto the queue for another try this
                # invocation, bounded so the process still terminates.
                total = prior_attempts.get(key, 0) + in_proc[key]
                if max_attempts and total >= max_attempts:
                    line(f"    giving up on {label} after {total} attempts (exhausted)")
                    advance("failed")  # terminal — resume won't retry it either
                    continue
                if in_proc[key] < attempts_per_invocation:
                    queue.append((item, sample_index))
                    line(f"    requeued {label} for retry")
                    continue
                line(f"    deferring {label} to the next resume")
                still_pending += 1
                advance("pending")
                continue

            # Not accepted, not a transient failure (max_steps / no image): retrying
            # as-is won't help, so leave it for a resume rather than burn attempts.
            hint = " — raise --max-steps" if result.stop_reason == "max_steps" else ""
            line(f"    {label} did not accept an output{hint}; deferring to resume")
            still_pending += 1
            advance("pending")

    if bar is not None:
        bar.close()
    total_done = counts["ok"] + counts["failed"] + counts["pending"]
    print(
        f"[{total_done}/{len(planned)}] done: "
        f"{counts['ok']} ok, {counts['failed']} failed, {counts['pending']} pending",
        flush=True,
    )

    # Exit 0 only when nothing is left to retry, so a scheduler can stop requeuing.
    return 1 if still_pending else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continuum-batch-agent",
        description="Run an OpenAI-compatible agent over a benchmark/batch using Continuum MCP tools.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument(
        "--prompt",
        help="Prompt for each item. Required unless --manifest supplies per-item prompts (then it is the fallback).",
    )
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing the prompt.")
    parser.add_argument(
        "--agents-md",
        help="AGENTS.md used as the shared system prompt (default: nearest AGENTS.md walking up from the CWD).",
    )
    parser.add_argument(
        "--no-agents-md",
        action="store_true",
        help="Do not auto-discover AGENTS.md for the system prompt.",
    )
    parser.add_argument(
        "--no-server-instructions",
        action="store_true",
        help="Do not prepend continuum's MCP server instructions (editing discipline) to the system prompt.",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="Markdown/text file appended to AGENTS.md as run-specific policy (e.g. don't set a seed, keep iterating).",
    )
    parser.add_argument(
        "--shard",
        help="Process one strided slice of the manifest, as I/N (e.g. 0/8). Each "
        "shard needs its own --output-dir: outputs/ may be shared safely, but "
        "concurrent appends to one manifest.jsonl would interleave.",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "JSONL benchmark manifest: one JSON object per line with a per-item prompt "
            "(prompt/instruction/edit/caption), optional image (image/source/input), and optional id."
        ),
    )
    parser.add_argument(
        "--image-root", help="Base directory for resolving relative image paths in --manifest."
    )
    parser.add_argument("--image-dir", help="Directory of images to process (non-manifest mode).")
    parser.add_argument("--image", action="append", default=[], help="Additional image file to process.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for per-item outputs and manifest.jsonl."
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Run the agent this many times per item (e.g. 4 for GenEval)."
    )
    # Agent / LLM
    parser.add_argument("--model", help="OpenAI-compatible model id that drives the agent.")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL / openai)."
    )
    parser.add_argument(
        "--api-key", help="API key (default: OPENAI_API_KEY; pass a dummy for keyless local servers)."
    )
    parser.add_argument(
        "--mcp-url",
        default=None,
        help="Continuum standard-mode MCP streamable HTTP URL. If omitted, the harness "
        "launches its own standard-mode continuum server and tears it down afterward.",
    )
    parser.add_argument(
        "--mcp-startup-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for a harness-launched continuum server to become ready.",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent tool-use turns per item.")
    parser.add_argument("--fail-on-truncation", action="store_true",
                        help="Record a token-limited agent response as an error rather than normal completion.")
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("CONTINUUM_LLM_REASONING_EFFORT") or None,
        help="Reasoning strength for models that support it (e.g. low/medium/high/max on "
        "kimi-k3, which replaced the -low/-high/-max-preview endpoints). Omitted from the "
        "request when unset, since endpoints that do not know the field may reject it. "
        "Defaults to $CONTINUUM_LLM_REASONING_EFFORT.",
    )
    parser.add_argument(
        "--max-inline-images",
        type=int,
        default=-1,
        help="Keep at most this many images inline in the conversation; older ones are "
        "replaced with a placeholder. Default -1 keeps them all, which is what the agent "
        "reasons best with. Set a positive cap if the endpoint starts rejecting the "
        "request: every request re-sends the history, so payload grows with step count "
        "(HTTP 413 was observed at step 15 before this defaulted off). Full-resolution "
        "bytes are unaffected either way.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Baseline mode: send each prompt straight to the generator in one blind tool call "
        "and accept the result — no LLM, no agent loop. Needs no --model/--base-url and no "
        "'openai' install. Pair with an agentic run over the same manifest to measure uplift.",
    )
    parser.add_argument(
        "--direct-tool",
        default="generate_image",
        help="MCP tool to call in --direct mode (default: generate_image).",
    )
    parser.add_argument(
        "--image-model",
        help="Continuum image model to use (e.g. qwen-image, flux2-klein). In "
        "--direct mode it is the model called; in agentic mode it is forced onto "
        "every generate_image and edit_image call, overriding the agent. Omit to "
        "let the agent choose, which means the server default whenever it omits "
        "the argument.",
    )
    parser.add_argument(
        "--tools",
        help="Comma-separated allowlist of MCP tools the agent may use "
        "(e.g. 'generate_image,edit_image,inspect_region'). Default: all tools the server offers.",
    )
    parser.add_argument(
        "--exclude-tools",
        help="Comma-separated MCP tools to hide from the agent (applied after --tools).",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=768,
        help="Downscale the model's view of each image to this long edge.",
    )
    parser.add_argument(
        "--pin-size", help="Force generation size WxH (e.g. 1024x1024) regardless of the model."
    )
    parser.add_argument(
        "--strip-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip any seed the model sets so samples genuinely vary (default: on).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream per-step progress (timing, per-call token usage, each tool call and result).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start clean: discard any prior manifest.jsonl and rerun everything "
        "(default is to resume — skip runs already recorded ok, retry the rest).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Cap total attempts per run across all invocations (0 = unlimited, the default). "
        "After this many failed attempts a run is abandoned so the requeue loop can terminate.",
    )
    parser.add_argument(
        "--attempts-per-invocation",
        type=int,
        default=3,
        help="How many times to retry a failed run within one invocation (kicked back onto "
        "the work queue between tries) before deferring it to the next resume.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Per-LLM-call retry budget (openai SDK) — absorbs transient 5xx/429 mid-loop "
        "so a blip doesn't fail the whole run.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Minimum seconds between LLM calls, to ease a shared or rate-limited endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve items and write manifest stubs without calling the agent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except BatchHarnessError as exc:
        parser.error(str(exc))
        return 2
