"""Continuum MCP server — entry point and CLI.

The actual architecture lives in two packages:

- continuum.core: the transport-free model layer (operations, GPU lock,
  model-manager singleton). PIL images and plain params in/out.
- continuum.server: two concrete MCP views over the same operations —
  `standard` (inline MCP content I/O) and `compat` (URI-string I/O with a
  /tmp resource store and /upload endpoint).

Tools exposed (either view):

- generate_image(prompt, ...) -> image
- edit_image(image, prompt, references?, ...) -> image  (whole-image, no mask)
- inpaint_image(image, mask, prompt, ...) -> image  (masked region)
- outpaint_image(image, prompt, left/right/top/bottom, ...) -> image
- inspect_region(image, roi|description) -> cropped image
- segment(image, description) -> mask image
- segment_points(image, points, negative_points?, ...) -> mask image
- segment_boxes(image, boxes, ...) -> mask image
- invert_mask(mask) -> mask image
- upscale_image(image, scale|target_w/h) -> image

Launch::

    continuum-mcp                                # streamable HTTP on :8742
    continuum-mcp --host 0.0.0.0 --port 9000     # custom bind
    continuum-mcp --precache                     # cache weights before serving
    continuum-mcp --transport stdio              # stdio (rare)
"""

import argparse
import asyncio  # noqa: F401  (kept importable for tests that patch asyncio.sleep)
import logging
import os
import socket

from continuum import models

# Backwards-compatible aliases: tests and older callers import these from
# here. The implementations live in continuum.core / continuum.server.
from continuum.core.ops import (  # noqa: F401
    bbox_from_mask as _bbox_from_mask,
)
from continuum.core.ops import (
    canonical_edit_model_name as _canonical_edit_model_name,
)
from continuum.core.ops import (
    canonical_generate_model_name as _canonical_generate_model_name,
)
from continuum.core.ops import (
    coerce_roi_box as _coerce_roi_box,
)
from continuum.core.ops import (
    crop_image_roi as _crop_image_roi,
)
from continuum.core.ops import (
    resolve_edit_defaults as _resolve_edit_defaults,
)
from continuum.core.ops import (
    resolve_generate_defaults as _resolve_generate_defaults,
)
from continuum.core.ops import (
    resolve_seed as _resolve_seed,
)
from continuum.core.runtime import (  # noqa: F401
    log as _mcp_log,
)
from continuum.core.runtime import (
    log_error as _mcp_log_error,
)
from continuum.core.runtime import (
    run_locked as _run_locked,
)
from continuum.models import (
    FIRERED_REPO,
    FLUX2_DEV_REPO,
    FLUX2_KLEIN_REPO,
    QWEN_IMAGE_EDIT_REPO,
    QWEN_IMAGE_REPO,
    SAM3_REPO,
)
from continuum.server import build_server
from continuum.server.bridge import (  # noqa: F401
    load_progress_bridge as _load_progress_bridge,
)
from continuum.server.bridge import (
    progress_bridge as _progress_bridge,
)
from continuum.server.bridge import (
    report_loading_progress as _report_loading_progress,
)


def _build_server(compat_mode: bool, upload_url: str | None = None):
    return build_server(compat_mode=compat_mode, upload_url=upload_url)


# Module-level default-mode instance for tests / direct imports
mcp = _build_server(compat_mode=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _configure_server_logging()

    parser = argparse.ArgumentParser(
        prog="continuum-mcp",
        description="Stateless MCP server for Continuum image-editing primitives.",
    )
    parser.add_argument(
        "--transport",
        choices=("http", "sse", "stdio"),
        default="http",
        help="MCP transport (default: http / streamable HTTP)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP/SSE bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8742, help="HTTP/SSE bind port (default: 8742)")
    parser.add_argument(
        "--precache",
        action="store_true",
        help="Copy model weights to ramdisk on startup instead of first use",
    )
    parser.add_argument("--no-precache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--client-compatibility-mode",
        action="store_true",
        help=(
            "Switch to URI-only I/O. Image inputs become continuum:// or "
            "http(s):// URIs, outputs become ResourceLinks (served from "
            "/tmp/continuum-mcp/outputs), and POST /upload is exposed for "
            "out-of-band byte upload. Use this for hosts that can't put "
            "bytes in the LLM conversation. Default mode is bytes-only "
            "and assumes your harness does pointer substitution."
        ),
    )
    args = parser.parse_args()

    if args.precache and not args.no_precache:
        _precache_models()

    addrs = _resolve_addresses(args.host) if args.transport in ("http", "sse") else []
    canonical = addrs[0] if addrs else args.host
    upload_url = (
        f"http://{canonical}:{args.port}/upload"
        if args.transport == "http" and args.client_compatibility_mode
        else None
    )

    global mcp
    mcp = _build_server(
        compat_mode=args.client_compatibility_mode,
        upload_url=upload_url,
    )

    if args.transport in ("http", "sse"):
        # Settings is built at FastMCP() instantiation, so env-var injection
        # here is too late — assign directly.
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        # DNS-rebinding protection allow-lists only loopback by default,
        # which makes any non-localhost client get 421 Misdirected Request.
        # The user explicitly opted into a network transport — turn it off.
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        _print_urls(args.transport, args.host, args.port, addrs, show_upload=args.client_compatibility_mode)

    transport = "streamable-http" if args.transport == "http" else args.transport
    mcp.run(transport=transport)


def _resolve_addresses(host: str) -> list[str]:
    """Enumerate reachable IPv4 addresses if `host` is 0.0.0.0,
    otherwise just return `[host]`. Sorted; loopback-first removed
    so the canonical address is something a remote client can hit."""
    if host != "0.0.0.0":
        return [host]
    try:
        hostname = socket.gethostname()
        addrs = sorted({addr[4][0] for addr in socket.getaddrinfo(hostname, None, socket.AF_INET)})
    except OSError:
        return [host]
    if not addrs:
        return [host]
    # Push 127.x to the back so the first address is one a remote client
    # can actually use. If only loopback resolves, that's all we have.
    non_loopback = [a for a in addrs if not a.startswith("127.")]
    return non_loopback + [a for a in addrs if a.startswith("127.")] if non_loopback else addrs


def _precache_models() -> None:
    """Copy model weights to ramdisk before serving so first request is fast.

    Only the models the MCP tools actually use are cached.
    """
    repos = [
        FLUX2_DEV_REPO,
        FLUX2_KLEIN_REPO,
        QWEN_IMAGE_REPO,
        FIRERED_REPO,
        QWEN_IMAGE_EDIT_REPO,
        SAM3_REPO,
        "SG161222/RealVisXL_V4.0",  # upscale (FaithDiff base)
        "madebyollin/sdxl-vae-fp16-fix",  # upscale (VAE)
        "jychen9811/FaithDiff",  # upscale (SR weights)
    ]
    repos = list(dict.fromkeys(repos))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    _mcp_log("precache: caching model weights")
    for repo in repos:
        _mcp_log(f"precache: {repo}", verbose=True)
        models.ensure_local(repo, token)  # via module attr: tests patch it


def _print_urls(transport: str, host: str, port: int, addrs: list[str], *, show_upload: bool = False) -> None:
    """Print every reachable URL the host can be hit at."""
    path = mcp.settings.streamable_http_path if transport == "http" else mcp.settings.sse_path

    print("\n=== Continuum MCP server ===", flush=True)
    for a in addrs:
        print(f"  MCP:    http://{a}:{port}{path}", flush=True)
        if show_upload and transport == "http":
            print(f"  upload: http://{a}:{port}/upload  (curl -F file=@img.png ...)", flush=True)
    print(flush=True)


def _configure_server_logging() -> None:
    """Keep default server output focused on MCP events and errors."""
    for name in (
        "accelerate",
        "diffusers",
        "huggingface_hub",
        "httpcore",
        "httpx",
        "PIL",
        "torch",
        "transformers",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)


if __name__ == "__main__":
    main()
