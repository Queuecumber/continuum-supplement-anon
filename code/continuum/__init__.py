"""Continuum: MCP server for GPU-hosted image tools."""

import os

# Configure the quantization backend before importing model pipelines.
os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
