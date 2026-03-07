"""Generate synthetic ALMA-like protoplanetary disk FITS for proposal experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def _make_disk_image(
    shape: tuple[int, int],
    n_rings: int,
    gap_positions: list[float],
    inclination_deg: float,
    pa_deg: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create one synthetic disk image with ring/gap structure."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    x = x.astype(float) - cx
    y = y.astype(float) - cy

    inclination = np.deg2rad(inclination_deg)
    pa = np.deg2rad(pa_deg)
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)

    x_rot = x * cos_pa + y * sin_pa
    y_rot = (-x * sin_pa + y * cos_pa) / max(cos_i, 0.1)
    r = np.sqrt(x_rot**2 + y_rot**2)
    r_norm = r / (min(h, w) * 0.4)

    img = np.exp(-r_norm**2) * 0.8
    for ring_center in np.linspace(0.2, 0.8, n_rings):
        ring_width = 0.05
        img += 0.3 * np.exp(-((r_norm - ring_center) ** 2) / (2 * ring_width**2))

    for gap in gap_positions:
        gap_width = 0.04
        img *= 1.0 - 0.9 * np.exp(-((r_norm - gap) ** 2) / (2 * gap_width**2))

    if noise_std > 0:
        img += rng.normal(0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img, 0, None).astype(np.float32)
    return img


def generate_fits_dataset(
    out_dir: str | Path,
    n_samples: int = 200,
    shape: tuple[int, int] = (600, 600),
    seed: int = 42,
) -> None:
    """Generate synthetic ALMA-like disk FITS files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_planet_options = [0, 1, 2, 3]
    for i in range(n_samples):
        n_rings = int(rng.integers(2, 6))
        n_gaps = int(rng.integers(0, 4))
        gap_positions = [float(rng.uniform(0.2, 0.8)) for _ in range(n_gaps)]
        inclination = float(rng.uniform(60, 90))
        pa = float(rng.uniform(0, 180))
        noise = float(rng.uniform(0.001, 0.01))

        img = _make_disk_image(
            shape=shape,
            n_rings=n_rings,
            gap_positions=gap_positions,
            inclination_deg=inclination,
            pa_deg=pa,
            noise_std=noise,
            rng=rng,
        )
        hdu = fits.PrimaryHDU(img)
        path = out_dir / f"disk_{i:04d}.fits"
        hdu.writeto(path, overwrite=True)

    print(f"Generated {n_samples} FITS files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw", type=str)
    parser.add_argument("--n-samples", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    generate_fits_dataset(args.out_dir, n_samples=args.n_samples, seed=args.seed)
