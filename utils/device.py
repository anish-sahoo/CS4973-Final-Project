"""
Utility: Flexible device selection for training. Switch between 'cuda', 'mps', and 'cpu' via `config.DEVICE`.
"""
from typing import Tuple
import torch
import config
import os


def get_device() -> torch.device:
    """Return the appropriate torch.device depending on `config.DEVICE`.

    If config.DEVICE == 'auto', prefer cuda > mps > cpu.
    If user sets an explicit device, try that and fall back to cpu.
    """
    # Allow environment variable override (e.g. DEVICE=cuda) for quick switching
    env_device = os.environ.get('DEVICE')
    preferred = env_device.lower() if env_device else (config.DEVICE.lower() if isinstance(config.DEVICE, str) else 'auto')

    if preferred == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print('[device] CUDA requested but not available; falling back to CPU')
            return torch.device('cpu')
    elif preferred in ['mps', 'metal']:
        # PyTorch exposes MPS (Apple Metal Performance Shaders) as 'mps'
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        else:
            print('[device] MPS requested but not available; falling back to CPU')
            return torch.device('cpu')
    elif preferred == 'cpu':
        return torch.device('cpu')
    else:  # 'auto' or unrecognized
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        return torch.device('cpu')


def device_name() -> str:
    return str(get_device())
