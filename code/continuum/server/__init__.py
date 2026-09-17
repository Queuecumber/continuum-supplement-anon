"""MCP views over the continuum.core model layer.

Two concrete views consume the same operations:

- standard: inline MCP content I/O (ImageContent in, MCPImage out)
- compat:   URI-string I/O backed by a /tmp resource store + /upload

Each view is a flat FastMCP module with its server instance at module
level (`continuum.server.standard.mcp` / `continuum.server.compat.mcp`);
the choice between them is made once at startup.
"""

from mcp.server.fastmcp import FastMCP


def build_server(compat_mode: bool = False, upload_url: str | None = None) -> FastMCP:
    """Return the requested view's server instance.

    Views are module-level singletons; this just selects one (and, for
    compat, injects the real /upload URL into the instructions).
    """
    if compat_mode:
        from continuum.server import compat

        if upload_url:
            compat.set_upload_url(upload_url)
        return compat.mcp
    from continuum.server import standard

    return standard.mcp
