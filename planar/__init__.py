"""Shim package to expose the src/planar package without installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "planar"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Make this package resolve submodules from src/planar.
__path__ = [str(PKG)]
