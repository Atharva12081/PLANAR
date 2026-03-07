"""Synthetic transit light-curve simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransitSampleMeta:
    """Metadata for one simulated transit sample."""

    has_planet: int
    period: float
    radius_ratio: float
    inclination_deg: float
    noise_sigma: float
    red_noise_sigma: float
    missing_fraction: float
    missing_segments: int
    stress_mode: bool


def _smooth_box(time: np.ndarray, center: float, duration: float, depth: float) -> np.ndarray:
    """Create a smooth box-like transit profile.

    Args:
        time: Time axis.
        center: Transit center.
        duration: Transit duration.
        depth: Transit depth.

    Returns:
        Multiplicative flux profile.
    """
    edge = max(duration * 0.08, 1e-3)
    ingress = 1.0 / (1.0 + np.exp(-(time - (center - duration / 2.0)) / edge))
    egress = 1.0 / (1.0 + np.exp(-(time - (center + duration / 2.0)) / edge))
    profile = ingress - egress
    return 1.0 - depth * profile


def _generate_red_noise(
    size: int,
    sigma: float,
    rng: np.random.Generator,
    phi: float = 0.96,
) -> np.ndarray:
    """Generate AR(1)-style red noise.

    Args:
        size: Number of points.
        sigma: White innovation standard deviation.
        rng: Random generator.
        phi: AR(1) coefficient.

    Returns:
        Red-noise vector.
    """
    if sigma <= 0.0:
        return np.zeros(size, dtype=np.float32)

    eps = rng.normal(0.0, sigma, size=size).astype(np.float32)
    red = np.zeros(size, dtype=np.float32)
    for i in range(1, size):
        red[i] = phi * red[i - 1] + eps[i]
    return red


def _apply_missing_segments(
    flux: np.ndarray,
    rng: np.random.Generator,
    max_segments: int = 3,
    segment_frac_range: tuple[float, float] = (0.02, 0.07),
) -> tuple[np.ndarray, int, float]:
    """Simulate missing-cadence segments and interpolate them.

    Args:
        flux: Input flux vector.
        rng: Random generator.
        max_segments: Maximum missing segments.
        segment_frac_range: Segment length range as fraction of sequence.

    Returns:
        Tuple of reconstructed flux, number of segments, and missing fraction.
    """
    n = len(flux)
    if max_segments <= 0:
        return flux, 0, 0.0

    out = flux.copy()
    mask = np.ones(n, dtype=bool)
    n_seg = int(rng.integers(0, max_segments + 1))

    for _ in range(n_seg):
        frac = float(rng.uniform(segment_frac_range[0], segment_frac_range[1]))
        seg_len = max(2, int(frac * n))
        start = int(rng.integers(0, max(1, n - seg_len)))
        mask[start : start + seg_len] = False

    if mask.sum() < max(5, int(0.35 * n)):
        return out, 0, 0.0

    if not np.all(mask):
        idx = np.arange(n)
        out[~mask] = np.interp(idx[~mask], idx[mask], out[mask])

    missing_fraction = float(np.mean(~mask))
    return out, n_seg, missing_fraction


def simulate_transit(
    period: float,
    radius_ratio: float,
    inclination_deg: float = 90.0,
    noise_sigma: float = 0.001,
    num_points: int = 500,
    limb_darkening_u1: float = 0.3,
    variability_amp: float = 0.0005,
    red_noise_sigma: float = 0.0,
    add_missing_segments: bool = False,
    max_missing_segments: int = 2,
    irregular_sampling: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Generate one noisy synthetic light curve.

    Args:
        period: Orbital period.
        radius_ratio: Planet-to-star radius ratio.
        inclination_deg: Orbital inclination in degrees.
        noise_sigma: White-noise sigma.
        num_points: Number of time samples.
        limb_darkening_u1: Simple limb darkening coefficient.
        variability_amp: Stellar variability amplitude.
        red_noise_sigma: Red-noise sigma.
        add_missing_segments: Whether to inject missing segments.
        max_missing_segments: Maximum number of missing segments.
        irregular_sampling: If `True`, sample irregularly in time.
        rng: Random generator.

    Returns:
        Time vector, flux vector, missing segment count, missing fraction.
    """
    rng = rng or np.random.default_rng()

    if irregular_sampling:
        time = np.sort(rng.uniform(0.0, period, size=num_points).astype(np.float32))
    else:
        time = np.linspace(0.0, period, num_points, dtype=np.float32)

    baseline = np.ones_like(time, dtype=np.float32)

    b = np.cos(np.deg2rad(np.clip(inclination_deg, 0.0, 90.0)))
    transit_visible = b < 0.2 and radius_ratio > 0.0

    if transit_visible:
        depth = radius_ratio**2
        depth *= 1.0 - 0.5 * limb_darkening_u1
        duration = np.clip(0.10 * period * (1.0 + radius_ratio), 0.02 * period, 0.18 * period)
        center = 0.5 * period
        transit_flux = _smooth_box(time, center=center, duration=duration, depth=depth)
        baseline *= transit_flux.astype(np.float32)

    phase = rng.uniform(0, 2 * np.pi)
    variability = variability_amp * np.sin(2 * np.pi * time / period + phase)
    noise = rng.normal(0.0, noise_sigma, size=num_points)
    red_noise = _generate_red_noise(num_points, sigma=red_noise_sigma, rng=rng)

    flux = baseline + variability.astype(np.float32) + noise.astype(np.float32) + red_noise
    flux = flux.astype(np.float32)

    missing_segments = 0
    missing_fraction = 0.0
    if add_missing_segments:
        flux, missing_segments, missing_fraction = _apply_missing_segments(
            flux=flux,
            rng=rng,
            max_segments=max_missing_segments,
        )

    return time.astype(np.float32), flux.astype(np.float32), missing_segments, missing_fraction


def generate_transit_dataset(
    n_samples: int = 5000,
    transit_fraction: float = 0.5,
    num_points: int = 500,
    seed: int = 42,
    stress_mode: bool = False,
    stress_profile: str = "moderate",
) -> tuple[np.ndarray, np.ndarray, list[TransitSampleMeta]]:
    """Generate a binary transit/non-transit dataset.

    Args:
        n_samples: Number of generated samples.
        transit_fraction: Positive class fraction.
        num_points: Sequence length.
        seed: Random seed.
        stress_mode: If `True`, use harder noise/variability settings.
        stress_profile: Stress severity profile (`mild`, `moderate`, `severe`, `extreme`).

    Returns:
        Feature matrix `(N, T)`, labels `(N,)`, and per-sample metadata.
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, num_points), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    metas: list[TransitSampleMeta] = []

    for i in range(n_samples):
        has_planet = int(rng.random() < transit_fraction)
        period = float(rng.uniform(0.8, 1.2))

        if stress_mode:
            profile = stress_profile.lower()
            if profile == "mild":
                inclination = float(rng.uniform(80.0, 90.0))
                noise_sigma = float(rng.uniform(0.0007, 0.0018))
                variability_amp = float(rng.uniform(0.0002, 0.0010))
                red_noise_sigma = float(rng.uniform(0.0001, 0.0005))
                missing_cap = 1
            elif profile == "severe":
                inclination = float(rng.uniform(76.0, 90.0))
                noise_sigma = float(rng.uniform(0.0010, 0.0030))
                variability_amp = float(rng.uniform(0.0004, 0.0016))
                red_noise_sigma = float(rng.uniform(0.0002, 0.0009))
                missing_cap = 2
            elif profile == "extreme":
                inclination = float(rng.uniform(74.0, 90.0))
                noise_sigma = float(rng.uniform(0.0012, 0.0036))
                variability_amp = float(rng.uniform(0.0006, 0.0020))
                red_noise_sigma = float(rng.uniform(0.0003, 0.0012))
                missing_cap = 3
            else:
                inclination = float(rng.uniform(78.0, 90.0))
                noise_sigma = float(rng.uniform(0.0007, 0.0024))
                variability_amp = float(rng.uniform(0.0002, 0.0012))
                red_noise_sigma = float(rng.uniform(0.0001, 0.0006))
                missing_cap = 1
            add_missing_segments = True
            max_missing_segments = missing_cap
        else:
            inclination = float(rng.uniform(84.0, 90.0))
            noise_sigma = float(rng.uniform(0.0005, 0.0020))
            variability_amp = float(rng.uniform(0.0001, 0.0010))
            red_noise_sigma = float(rng.uniform(0.0, 0.0003))
            add_missing_segments = False
            max_missing_segments = 0

        radius_ratio = float(rng.uniform(0.07, 0.15)) if has_planet else 0.0

        _, flux, missing_segments, missing_fraction = simulate_transit(
            period=period,
            radius_ratio=radius_ratio,
            inclination_deg=inclination,
            noise_sigma=noise_sigma,
            num_points=num_points,
            limb_darkening_u1=float(rng.uniform(0.2, 0.5)),
            variability_amp=variability_amp,
            red_noise_sigma=red_noise_sigma,
            add_missing_segments=add_missing_segments,
            max_missing_segments=max_missing_segments,
            irregular_sampling=True,
            rng=rng,
        )

        X[i] = flux
        y[i] = has_planet
        metas.append(
            TransitSampleMeta(
                has_planet=has_planet,
                period=period,
                radius_ratio=radius_ratio,
                inclination_deg=inclination,
                noise_sigma=noise_sigma,
                red_noise_sigma=red_noise_sigma,
                missing_fraction=missing_fraction,
                missing_segments=missing_segments,
                stress_mode=stress_mode,
            )
        )

    return X, y, metas
