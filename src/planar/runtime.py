"""Runtime helpers for deterministic execution and filesystem management."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def configure_runtime_environment(cache_root: str | Path = ".runtime_cache") -> None:
    """Configure cache and thread settings for stable portable runs.

    Args:
        cache_root: Root directory used for matplotlib/font caches.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    cache_dir = Path(cache_root)
    mpl_dir = cache_dir / "matplotlib"
    font_dir = cache_dir / "fontconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    font_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    os.environ.setdefault("FC_CACHEDIR", str(font_dir.resolve()))


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return its `Path`.

    Args:
        path: Directory path.

    Returns:
        Created or existing directory path.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    """Set random seeds across supported libraries.

    Args:
        seed: Seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Deterministic kernels improve exact reproducibility at a runtime cost.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch can be optional for lightweight utility calls.
        pass


def get_device(requested: str = "auto") -> str:
    """Resolve runtime device from user preference.

    Args:
        requested: `auto`, `cpu`, `cuda`, or `mps`.

    Returns:
        Device string.
    """
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with stable formatting.

    Args:
        payload: Serializable dictionary.
        path: Output path.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
