"""Repository-local runner for `python -m planar` without installation.

This file also acts as a safe package shim so imports like
`from planar.models import ...` work in test environments where the repo
root is on `sys.path` (which would otherwise shadow the real package).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PKG = SRC / "planar"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("KMP_DISABLE_MMAP", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_USE_SHM", "0")

def _load_package_shim() -> None:
    """Load the real package __init__ into this module namespace."""
    if not PKG.exists():
        return
    spec = importlib.util.spec_from_file_location(__name__, PKG / "__init__.py")
    if spec is None or spec.loader is None:
        return
    module = sys.modules[__name__]
    # Mark this module as a package so submodules can be imported.
    module.__package__ = __name__
    module.__path__ = [str(PKG)]
    spec.submodule_search_locations = [str(PKG)]
    spec.loader.exec_module(module)


_load_package_shim()

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("planar.cli", PKG / "cli.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load CLI module for planar.")
    cli_mod = importlib.util.module_from_spec(spec)
    sys.modules["planar.cli"] = cli_mod
    spec.loader.exec_module(cli_mod)
    cli_mod.main()
