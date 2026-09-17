"""Shared backend plumbing: params, device/dtype policy, small utilities."""

import inspect
from dataclasses import dataclass
from typing import Literal

import torch

DeviceName = Literal["auto", "cuda", "cpu", "mps"]
DTypeName = Literal["auto", "bf16", "fp16", "fp32"]


def model_log(message: str) -> None:
    return None


@dataclass(frozen=True)
class EditParams:
    prompt: str
    guidance_scale: float = 4.0
    num_inference_steps: int = 30
    seed: int | None = None
    width: int | None = None
    height: int | None = None


def pick_device(device: DeviceName) -> str:
    if device != "auto":
        return device

    try:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def pick_dtype(dtype: DTypeName, device: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16

    # auto - prefer bf16
    if device.startswith("cuda"):
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def accepts_kw(callable_obj, name: str) -> bool:
    """True if callable accepts keyword `name` (or **kwargs).

    Used to ride diffusers git-main: pipeline __call__ signatures drift
    between revs, so optional kwargs are feature-detected per call.
    """
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
