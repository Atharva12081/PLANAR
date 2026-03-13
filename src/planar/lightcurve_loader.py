"""Load observational light curves from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class LightCurveLoadError(RuntimeError):
    """Raised when an observational light curve cannot be parsed."""


def _iter_csv_paths(folder: str | Path) -> Iterable[Path]:
    root = Path(folder)
    yield from sorted(root.glob("*.csv"))


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, int | None]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise LightCurveLoadError(f"Empty CSV: {path}")

    colnames = {name.lower(): name for name in data.dtype.names or []}
    if "time" not in colnames or "flux" not in colnames:
        raise LightCurveLoadError(f"Missing required columns time/flux in {path}")

    time = np.asarray(data[colnames["time"]], dtype=np.float32)
    flux = np.asarray(data[colnames["flux"]], dtype=np.float32)

    label = None
    if "label" in colnames:
        label = int(np.asarray(data[colnames["label"]])[0])
    elif "has_planet" in colnames:
        label = int(np.asarray(data[colnames["has_planet"]])[0])

    return time, flux, label


def _resample(time: np.ndarray, flux: np.ndarray, num_points: int) -> np.ndarray:
    if len(time) == num_points:
        return flux.astype(np.float32)

    t_min = float(np.min(time))
    t_max = float(np.max(time))
    target = np.linspace(t_min, t_max, num_points, dtype=np.float32)
    return np.interp(target, time, flux).astype(np.float32)


def load_observational_lightcurves(
    folder: str | Path,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Load observational light curves from a folder of CSVs.

    Each CSV must include columns: time, flux, and optionally label/has_planet.

    Returns:
        X: shape (N, num_points)
        y: labels or None if unavailable
        skipped: list of skip reasons
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    skipped: list[str] = []

    for path in _iter_csv_paths(folder):
        try:
            time, flux, label = _read_csv(path)
            flux_resampled = _resample(time, flux, num_points=num_points)
            X.append(flux_resampled)
            if label is not None:
                y.append(label)
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")

    if not X:
        return np.empty((0, num_points), dtype=np.float32), None, skipped

    X_mat = np.stack(X, axis=0).astype(np.float32)
    y_arr = None
    if len(y) == len(X):
        y_arr = np.asarray(y, dtype=np.int64)

    return X_mat, y_arr, skipped
