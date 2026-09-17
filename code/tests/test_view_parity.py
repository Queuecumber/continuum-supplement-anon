"""The two MCP views must expose the same tool surface.

Views differ only in transport types (ImageContent vs URI string,
MCPImage vs ResourceLink); tool names, parameter names, and parameter
defaults must match exactly. This is the guard against a knob being added
to one view and forgotten in the other.
"""

from continuum.server import build_server


def _tool_params(server) -> dict[str, dict]:
    return {name: tool.parameters.get("properties", {}) for name, tool in server._tool_manager._tools.items()}


def test_views_expose_identical_tool_surfaces() -> None:
    standard = _tool_params(build_server(compat_mode=False))
    compat = _tool_params(build_server(compat_mode=True))

    assert set(standard) == set(compat)
    for tool in standard:
        assert set(standard[tool]) == set(compat[tool]), tool
        for param in standard[tool]:
            assert standard[tool][param].get("default") == compat[tool][param].get("default"), (tool, param)


def test_every_tool_has_a_description() -> None:
    # Docstrings double as MCP descriptions; catch a tool losing its
    # docstring (FastMCP would ship an empty description).
    for compat_mode in (False, True):
        server = build_server(compat_mode=compat_mode)
        for name, tool in server._tool_manager._tools.items():
            assert tool.description and tool.description.strip(), (compat_mode, name)


def test_image_params_use_transport_native_types() -> None:
    standard = _tool_params(build_server(compat_mode=False))
    compat = _tool_params(build_server(compat_mode=True))

    assert standard["edit_image"]["image"].get("$ref") == "#/$defs/ImageContent"
    assert compat["edit_image"]["image"].get("type") == "string"
