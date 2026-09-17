"""Transport-free model layer: operations, runtime state, and media helpers.

Nothing in this package imports MCP types — inputs and outputs are PIL
images, plain params, and raw bytes. The MCP views in continuum.server
adapt these operations to their transports.
"""
