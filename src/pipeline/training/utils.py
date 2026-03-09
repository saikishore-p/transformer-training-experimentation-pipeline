"""
Device detection and reproducibility utilities.

Central place for all device/seed logic so every module gets
consistent behaviour — especially important for MPS on Apple Silicon.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """
    Return the best available device for this machine.

    Priority: MPS (Apple Silicon GPU) > CPU
    Override with env var FORCE_CPU=1 to stay on CPU regardless
    (useful when an op isn't yet supported on MPS).
    """
    if os.getenv("FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Seed all random number generators for full reproducibility.

    Covers: Python stdlib, NumPy, PyTorch (CPU + MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
