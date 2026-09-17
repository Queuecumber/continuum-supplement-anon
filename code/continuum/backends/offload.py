"""Explicit, opt-in CPU offloading for image pipelines on smaller GPUs."""
import os


def image_offload_mode():
    mode = os.environ.get('CONTINUUM_IMAGE_OFFLOAD', 'none')
    if mode not in ('none', 'model', 'sequential'):
        raise ValueError('CONTINUUM_IMAGE_OFFLOAD must be none, model, or sequential')
    return mode


def place_image_pipeline(pipe, device, mode):
    if mode == 'none':
        pipe.to(device)
    elif not str(device).startswith('cuda'):
        raise ValueError('image CPU offload requires a CUDA execution device')
    elif mode == 'model':
        pipe.enable_model_cpu_offload(device=device)
    elif mode == 'sequential':
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        raise ValueError(f'unknown image offload mode: {mode}')
