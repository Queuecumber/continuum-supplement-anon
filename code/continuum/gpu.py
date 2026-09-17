"""GPU topology detection and device role assignment.

Detected once at startup, cached for the process lifetime. Every module
that needs a device string imports from here instead of hardcoding "cuda".
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUConfig:
    """Immutable GPU role assignment."""

    num_gpus: int
    gen_device: str  # "cuda:0" (or "cpu" if no GPU)
    aux_device: str  # "cuda:1" if available, same as gen_device otherwise
    dual_gpu: bool  # True when a secondary GPU is available


_config: GPUConfig | None = None


def detect() -> GPUConfig:
    """Detect GPU topology and assign roles. Cached after first call."""
    global _config
    if _config is not None:
        return _config

    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0

    if n >= 2:
        _config = GPUConfig(
            num_gpus=n,
            gen_device="cuda:0",
            aux_device="cuda:1",
            dual_gpu=True,
        )
    elif n == 1:
        _config = GPUConfig(
            num_gpus=1,
            gen_device="cuda:0",
            aux_device="cuda:0",
            dual_gpu=False,
        )
    else:
        _config = GPUConfig(
            num_gpus=0,
            gen_device="cpu",
            aux_device="cpu",
            dual_gpu=False,
        )

    return _config


def get() -> GPUConfig:
    """Return cached GPUConfig, calling detect() if needed."""
    if _config is None:
        return detect()
    return _config
