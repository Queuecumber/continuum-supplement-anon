"""Model backends: one thin adapter per hosted model, plus shared plumbing.

Backends wrap diffusers pipelines with the load policy (device, dtype,
quantization) and call adaptation for each model. Pixel policy lives in
continuum.imaging; orchestration lives in continuum.core.ops.
"""

from continuum.backends.base import EditParams
from continuum.backends.flux1 import Flux1Generator
from continuum.backends.flux2 import Flux2Editor
from continuum.backends.flux_fill import FluxFillInpainter
from continuum.backends.klein import Flux2KleinEditor
from continuum.backends.qwen import FireRedEditor, QwenImageGenerator

__all__ = [
    "EditParams",
    "FireRedEditor",
    "Flux1Generator",
    "Flux2Editor",
    "Flux2KleinEditor",
    "FluxFillInpainter",
    "QwenImageGenerator",
]
