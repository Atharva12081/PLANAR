"""FITS ingestion utilities for ALMA disk image datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


class FitsLoadError(RuntimeError):
    """Raised when a FITS file cannot be parsed into a 2D image."""


def _to_2d_image(array: np.ndarray, layer: int = 0) -> np.ndarray:
    """Reduce a FITS array into a 2D image plane.

    Args:
        array: Raw FITS array.
        layer: Layer index to select for 3D cubes.

    Returns:
        Image array with shape `(H, W)`.

    Raises:
        FitsLoadError: If array cannot be reduced to 2D.
    """
    arr = np.asarray(array, dtype=np.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim < 2:
        raise FitsLoadError(f"Expected at least 2 dimensions, got {arr.ndim}")

    arr = arr[layer]
    while arr.ndim > 2:
        arr = arr[0]

    if arr.ndim != 2:
        raise FitsLoadError(f"Could not reduce array to 2D, final ndim={arr.ndim}")

    return np.asarray(arr, dtype=np.float32)


def load_fits_image(path: str | Path, layer: int = 0) -> np.ndarray:
    """Load a FITS file and extract one 2D image.

    Args:
        path: FITS file path.
        layer: Layer index for data cubes.

    Returns:
        2D image array.

    Raises:
        FitsLoadError: If reading or conversion fails.
    """
    path = Path(path)

    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
    except Exception as exc:
        raise FitsLoadError(f"Failed to open FITS: {path}") from exc

    if data is None:
        raise FitsLoadError(f"FITS file has no primary data array: {path}")

    image = _to_2d_image(data, layer=layer)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image


def iter_fits_paths(folder: str | Path, recursive: bool = False) -> Iterable[Path]:
    """Iterate through FITS paths in sorted order.

    Args:
        folder: Directory containing FITS files.
        recursive: If `True`, recurse into subdirectories.

    Yields:
        FITS file paths.
    """
    root = Path(folder)
    if recursive:
        yield from sorted(root.rglob("*.fits"))
    else:
        yield from sorted(root.glob("*.fits"))


def load_dataset(
    folder: str | Path,
    layer: int = 0,
    expected_shape: tuple[int, int] | None = None,
    recursive: bool = False,
    max_files: int | None = None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Load all FITS images from a directory.

    Args:
        folder: FITS directory path.
        layer: Layer index for cubes.
        expected_shape: Optional exact expected shape.
        recursive: If `True`, search recursively.
        max_files: Optional cap on number of loaded files.

    Returns:
        Tuple of loaded images, filenames, and skipped-file reasons.
    """
    images: list[np.ndarray] = []
    filenames: list[str] = []
    skipped: list[str] = []

    all_paths = list(iter_fits_paths(folder, recursive=recursive))
    if max_files is not None:
        all_paths = all_paths[:max_files]

    for path in all_paths:
        try:
            img = load_fits_image(path, layer=layer)
            if expected_shape and img.shape != expected_shape:
                skipped.append(f"{path.name}: shape {img.shape} != {expected_shape}")
                continue

            images.append(img)
            filenames.append(path.name)
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")

    return images, filenames, skipped
