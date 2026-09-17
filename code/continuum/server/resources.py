"""Compat-view resource store: /tmp-backed continuum:// URIs.

Only the compat view knows URIs exist; this module owns the on-disk store,
URI resolution, and MCP resource registration. The standard view never
imports it.
"""

import uuid
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import FileResource
from mcp.types import ResourceLink
from PIL.Image import Image as PILImage

IMAGES_DIR = Path("/tmp/continuum-mcp/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def fetch_uri(uri: str) -> bytes:
    """Resolve a URI to raw bytes. Supports continuum://images/,
    and http(s):// schemes."""
    if uri.startswith("continuum://images/"):
        path = IMAGES_DIR / uri[len("continuum://images/") :]
    elif uri.startswith(("http://", "https://")):
        with httpx.Client(timeout=60.0) as client:
            r = client.get(uri)
            r.raise_for_status()
            return r.content
    else:
        raise ValueError(f"unsupported URI scheme: {uri!r}")
    if not path.exists():
        raise FileNotFoundError(f"resource not found on server: {uri}")
    return path.read_bytes()


def save_image_as_resource(img: PILImage, name_hint: str = "out") -> ResourceLink:
    """Write an image under the store and return its ResourceLink."""
    fname = f"{name_hint}_{uuid.uuid4().hex[:12]}.png"
    path = IMAGES_DIR / fname
    img.save(path, format="PNG")
    return ResourceLink(
        type="resource_link",
        uri=f"continuum://images/{fname}",
        name=fname,
        mimeType="image/png",
        size=path.stat().st_size,
    )




def register_file_resource(server: FastMCP, path: Path, uri: str, mime_type: str) -> None:
    """Register an on-disk file as a concrete MCP resource so it shows up in
    resources/list. Idempotent — silently skips if already registered."""
    try:
        server.add_resource(
            FileResource(
                uri=uri,  # type: ignore[arg-type]
                name=path.name,
                mime_type=mime_type,
                path=path,
                is_binary=True,
            )
        )
    except Exception:
        # Most likely a duplicate URI; warn_on_duplicate_resources may raise.
        pass


def scan_existing_resources(server: FastMCP) -> None:
    """At startup, list anything already on disk under /tmp/continuum-mcp/
    so files persisted across restarts are still browsable."""
    for p in IMAGES_DIR.glob("*"):
        if p.is_file():
            register_file_resource(
                server,
                p,
                f"continuum://images/{p.name}",
                "image/png",
            )
