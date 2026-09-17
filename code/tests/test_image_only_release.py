"""The distributed artifact exposes a complete, consistent image tool set."""

import asyncio
import json
from pathlib import Path

from continuum import models
from continuum.server import build_server, instructions

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    'generate_image', 'edit_image', 'inpaint_image', 'outpaint_image',
    'inspect_region', 'segment', 'segment_points', 'segment_boxes',
    'invert_mask', 'upscale_image',
}


def test_both_transports_expose_only_the_image_tool_set():
    for compat_mode in (False, True):
        server = build_server(compat_mode=compat_mode)
        tools = asyncio.run(server.list_tools())
        assert {tool.name for tool in tools} == TOOLS


def test_distributed_schemas_and_instructions_match_the_running_server():
    tools = asyncio.run(build_server(compat_mode=False).list_tools())
    live = [tool.model_dump(mode='json', exclude_none=True) for tool in tools]
    assert json.loads((ROOT / 'tool-schemas.json').read_text()) == live
    assert (ROOT / 'mcp-instructions.md').read_text().strip() == instructions.STANDARD_INSTRUCTIONS.strip()


def test_image_model_switch_and_unload(monkeypatch):
    monkeypatch.setattr(models, 'configure_hf_cache', lambda: '/tmp')
    manager = models.ModelManager()
    loaded = []

    def load(model, load_progress=None):
        loaded.append(model)
        manager._image_editor = object()
        manager._image_model_name = model

    monkeypatch.setattr(manager, '_ensure_image_editor', load)
    manager.enter_image_mode('flux2-klein')
    manager.enter_image_mode('flux2-klein')
    assert len(loaded) == 1
    manager.enter_image_mode('flux2-dev')
    assert len(loaded) == 2
    assert manager.mode == 'image'
    assert len(manager.status()['loaded']) == 1
    manager.unload_all()
    assert manager.mode is None
    assert manager.status()['loaded'] == []
