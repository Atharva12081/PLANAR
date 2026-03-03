"""Logging configuration for PLANAR command-line workflows."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Initialize root logging.

    Args:
        level: Logging level string (`DEBUG`, `INFO`, ...).
    """
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
