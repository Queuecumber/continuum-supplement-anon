"""LoRA loading helpers shared by the image pipelines.

Spec format matches the generate/edit pipelines: each entry is a source
string or an object {"source": ..., "scale": <float, default 1.0>,
"weight_name": <optional file within the repo>}.
"""

import os
from typing import Any
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download, snapshot_download

from continuum.progress import ProgressCallback, emit_progress


def parse_lora_source(source: str) -> tuple[str, str | None]:
    """Resolve a LoRA source to (repo_or_path, weight_name).

    Accepts:
      - a Hugging Face repo id: "owner/name"
      - a repo id with a file: "owner/name/adapter.safetensors"
      - a Hugging Face URL: https://huggingface.co/owner/name[/(blob|resolve|
        tree|raw)/<ref>/<path/to/adapter.safetensors>]
      - a local path to a .safetensors file or a directory
    """
    s = source.strip()
    if not s:
        raise ValueError("LoRA source must not be empty")

    if os.path.isfile(s) or os.path.isdir(s):
        return s, None

    if s.startswith("http://") or s.startswith("https://"):
        path = urlparse(s).path.lstrip("/")
    else:
        path = s

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"invalid LoRA source {source!r}: expected 'owner/name' or an "
            "https://huggingface.co/owner/name[/...] URL"
        )
    repo = "/".join(parts[:2])
    rest = parts[2:]
    if rest and rest[0] in {"blob", "resolve", "tree", "raw"} and len(rest) >= 2:
        rest = rest[2:]  # drop the "<kind>/<ref>" segment
    weight_name = "/".join(rest) if rest else None
    return repo, weight_name


def normalize_lora_specs(loras: Any) -> list[dict[str, Any]]:
    """Normalize a list of LoRA specs into dicts with repo/weight_name/scale."""
    specs: list[dict[str, Any]] = []
    for i, item in enumerate(loras or []):
        if isinstance(item, str):
            source, scale, weight_override = item, 1.0, None
        elif isinstance(item, dict):
            source = item.get("source") or item.get("repo") or item.get("path")
            if not source:
                raise ValueError(f"LoRA #{i} is missing a 'source'")
            scale = float(item.get("scale", item.get("weight", 1.0)))
            weight_override = item.get("weight_name")
        else:
            # Structured spec (schema.LoraSpec, or any object with the fields).
            source = getattr(item, "source", None)
            if not source:
                raise ValueError(f"LoRA #{i} must be a string, a dict, or a LoraSpec")
            scale = float(getattr(item, "scale", 1.0))
            weight_override = getattr(item, "weight_name", None)
        repo, parsed_weight = parse_lora_source(str(source))
        specs.append(
            {
                "name": f"lora_{i}",
                "repo": repo,
                "weight_name": weight_override or parsed_weight,
                "scale": scale,
                "source": str(source),
            }
        )
    return specs


def _prefetch_lora(repo: str, weight_name: str | None, hf_token: str | None) -> None:
    """Best-effort pre-download of the adapter weights into the HF cache.

    Splitting the download out from ``load_lora_weights`` lets the progress
    bridge report a real "downloading" phase distinct from "applying"; the
    subsequent load then resolves from cache. Failures are swallowed on
    purpose — ``load_lora_weights`` re-attempts the fetch and surfaces the
    canonical error (bad repo, gated model, wrong family).
    """
    if os.path.isfile(repo) or os.path.isdir(repo):
        return  # local source; nothing to fetch
    token = hf_token or None
    try:
        if weight_name:
            hf_hub_download(repo, weight_name, token=token)
        else:
            snapshot_download(repo, allow_patterns=["*.safetensors"], token=token)
    except Exception:
        pass


def apply_loras(
    owner: Any,
    pipe: Any,
    specs: list[dict[str, Any]],
    hf_token: str | None = None,
    load_progress: ProgressCallback | None = None,
) -> None:
    """Load/set or reset LoRA adapters on a long-lived cached pipeline.

    The pipeline is reused across calls, so adapters are swapped only when
    the requested set changes (tracked on *owner* as _active_lora_sig).
    Adapters are set (not fused) so they can be cleanly replaced next call.

    ``load_progress`` receives "downloading LoRA" / "applying LoRA" phase
    notices so the client can see the mechanics fire on first use. When it is
    None the path is byte-identical to a plain ``load_lora_weights`` (no
    separate pre-download), which keeps the common no-progress call cheap.
    """
    sig = tuple((s["repo"], s["weight_name"], round(float(s["scale"]), 6)) for s in specs)
    if sig == getattr(owner, "_active_lora_sig", None):
        return
    if getattr(owner, "_active_lora_sig", None):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    owner._active_lora_sig = None
    if specs:
        n = len(specs)
        names: list[str] = []
        scales: list[float] = []
        for i, s in enumerate(specs):
            suffix = f" {i + 1}/{n}" if n > 1 else ""
            emit_progress(load_progress, f"downloading LoRA{suffix}: {s['repo']}", 0, None)
            # Only pre-download when someone is watching: it is purely to
            # split the "downloading" phase from "applying".
            if load_progress is not None:
                _prefetch_lora(s["repo"], s["weight_name"], hf_token)
            emit_progress(load_progress, f"applying LoRA{suffix}: {s['repo']}", 0, None)
            load_kwargs: dict[str, Any] = {"adapter_name": s["name"]}
            if s["weight_name"]:
                load_kwargs["weight_name"] = s["weight_name"]
            if hf_token:
                load_kwargs["token"] = hf_token
            try:
                pipe.load_lora_weights(s["repo"], **load_kwargs)
            except TypeError:
                load_kwargs.pop("token", None)
                pipe.load_lora_weights(s["repo"], **load_kwargs)
            names.append(s["name"])
            scales.append(float(s["scale"]))
        pipe.set_adapters(names, adapter_weights=scales)
    owner._active_lora_sig = sig if specs else None
