"""Tests for FITS loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from planar.data_loader import load_dataset, load_fits_image


def test_load_fits_image_from_cube(tmp_path: Path) -> None:
    """Loader should reduce cube to 2D image."""
    cube = np.zeros((1, 64, 64), dtype=np.float32)
    cube[0, 10:20, 10:20] = 1.0
    path = tmp_path / "sample.fits"
    fits.PrimaryHDU(cube).writeto(path)

    image = load_fits_image(path)
    assert image.shape == (64, 64)
    assert float(image.max()) == 1.0


def test_load_dataset_collects_files(tmp_path: Path) -> None:
    """Dataset loader should return aligned images and filenames."""
    for i in range(2):
        arr = np.random.rand(1, 32, 32).astype(np.float32)
        fits.PrimaryHDU(arr).writeto(tmp_path / f"disk_{i:02d}.fits")

    images, filenames, skipped = load_dataset(tmp_path)
    assert len(images) == 2
    assert len(filenames) == 2
    assert skipped == []
