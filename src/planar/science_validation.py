"""Scientific validation utilities for cluster interpretation and bias checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from planar.preprocessing import center_crop, radial_intensity_profile

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.stats import kruskal

    HAS_SCIPY = True
except Exception:
    gaussian_filter1d = None
    find_peaks = None
    kruskal = None
    HAS_SCIPY = False


def brightness_proxy(images: list[np.ndarray], crop_size: int) -> np.ndarray:
    """Compute mean intensity on centered crops before normalization.

    Args:
        images: Raw images.
        crop_size: Center crop size.

    Returns:
        Brightness proxy per image.
    """
    vals = [float(np.mean(center_crop(img, size=crop_size))) for img in images]
    return np.asarray(vals, dtype=np.float32)


def axis_ratio_proxy(img: np.ndarray) -> float:
    """Estimate axis-ratio proxy from weighted second moments.

    Args:
        img: Processed image.

    Returns:
        Axis ratio in `[0, 1]` where lower indicates more elongated structure.
    """
    h, w = img.shape
    y, x = np.indices((h, w))

    weights = img - float(np.min(img))
    total = float(np.sum(weights))
    if total <= 1e-8:
        return 1.0

    x_mean = float(np.sum(x * weights) / total)
    y_mean = float(np.sum(y * weights) / total)

    x_c = x - x_mean
    y_c = y - y_mean

    cov_xx = float(np.sum(weights * x_c * x_c) / total)
    cov_xy = float(np.sum(weights * x_c * y_c) / total)
    cov_yy = float(np.sum(weights * y_c * y_c) / total)

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov)
    lmin = max(float(eigvals[0]), 1e-12)
    lmax = max(float(eigvals[1]), 1e-12)
    return float(np.sqrt(lmin / lmax))


def axis_ratio_proxies(processed_images: np.ndarray) -> np.ndarray:
    """Batch version of axis-ratio proxy.

    Args:
        processed_images: Array of processed images `(N, H, W)`.

    Returns:
        Axis-ratio proxy per image.
    """
    return np.asarray([axis_ratio_proxy(img) for img in processed_images], dtype=np.float32)


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute eta-squared effect size for proxy-vs-cluster association.

    Eta-squared estimates what fraction of total proxy variance is explained
    by cluster identity. Low values support the claim that clustering is not
    dominated by that nuisance proxy (brightness/orientation).

    Args:
        values: Proxy values.
        labels: Cluster labels.

    Returns:
        Eta-squared effect size or `None` when undefined.
    """
    mask = labels != -1
    values = values[mask]
    labels = labels[mask]

    if len(values) < 2:
        return None

    unique = np.unique(labels)
    if len(unique) < 2:
        return None

    grand_mean = float(np.mean(values))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total <= 1e-12:
        return 0.0

    ss_between = 0.0
    for cid in unique:
        vals = values[labels == cid]
        if len(vals) == 0:
            continue
        ss_between += len(vals) * float((np.mean(vals) - grand_mean) ** 2)

    return float(ss_between / ss_total)


def _kruskal_pvalue(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute Kruskal-Wallis p-value across cluster groups.

    Args:
        values: Proxy values.
        labels: Cluster labels.

    Returns:
        p-value or `None` when SciPy is unavailable/invalid.
    """
    if not HAS_SCIPY or kruskal is None:
        return None

    mask = labels != -1
    values = values[mask]
    labels = labels[mask]

    unique = np.unique(labels)
    if len(unique) < 2:
        return None

    groups = [values[labels == cid] for cid in unique if np.sum(labels == cid) > 0]
    if len(groups) < 2:
        return None

    _, p_value = kruskal(*groups)
    return float(p_value)


def cluster_bias_summary(
    labels: np.ndarray,
    brightness: np.ndarray,
    axis_ratio: np.ndarray,
    effect_threshold: float = 0.60,
) -> dict[str, Any]:
    """Assess brightness/orientation dominance in discovered clusters.

    Args:
        labels: Cluster labels.
        brightness: Brightness proxy values.
        axis_ratio: Orientation proxy values.
        effect_threshold: Eta-squared threshold considered dominant.

    Returns:
        Summary dictionary with effect sizes and dominance flags.
    """
    brightness_eta2 = _eta_squared(brightness, labels)
    axis_eta2 = _eta_squared(axis_ratio, labels)

    return {
        "brightness_eta_squared": brightness_eta2,
        "axis_ratio_eta_squared": axis_eta2,
        "brightness_kruskal_p": _kruskal_pvalue(brightness, labels),
        "axis_ratio_kruskal_p": _kruskal_pvalue(axis_ratio, labels),
        "dominance_flags": {
            "brightness_dominated": bool(brightness_eta2 is not None and brightness_eta2 >= effect_threshold),
            "orientation_dominated": bool(axis_eta2 is not None and axis_eta2 >= effect_threshold),
        },
    }


def estimate_rings_and_gaps(profile: np.ndarray, prominence: float = 0.015) -> tuple[int, int]:
    """Estimate ring/gap counts from radial profile extrema.

    Args:
        profile: Radial profile.
        prominence: Peak prominence threshold.

    Returns:
        Tuple `(ring_count, gap_count)`.
    """
    p = np.asarray(profile, dtype=np.float32)
    if len(p) < 5:
        return 0, 0

    if HAS_SCIPY and gaussian_filter1d is not None and find_peaks is not None:
        smooth = gaussian_filter1d(p, sigma=2)
        peak_idx, _ = find_peaks(smooth, prominence=prominence)
        gap_idx, _ = find_peaks(-smooth, prominence=prominence)
        return int(len(peak_idx)), int(len(gap_idx))

    k = 5
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(p, kernel, mode="same")
    d = np.diff(smooth)
    signs = np.sign(d)
    peak_count = int(np.sum((signs[:-1] > 0) & (signs[1:] < 0)))
    gap_count = int(np.sum((signs[:-1] < 0) & (signs[1:] > 0)))
    return peak_count, gap_count


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float32)
    if HAS_SCIPY and gaussian_filter1d is not None:
        return gaussian_filter1d(p, sigma=2)

    kernel = np.ones(5, dtype=np.float32) / 5.0
    return np.convolve(p, kernel, mode="same")


def _find_extrema(profile: np.ndarray, prominence: float = 0.015) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = _smooth_profile(profile)

    if HAS_SCIPY and find_peaks is not None:
        peaks, _ = find_peaks(smooth, prominence=prominence)
        gaps, _ = find_peaks(-smooth, prominence=prominence)
        return peaks.astype(int), gaps.astype(int), smooth

    d = np.diff(smooth)
    signs = np.sign(d)
    peaks = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1
    gaps = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0] + 1
    return peaks.astype(int), gaps.astype(int), smooth


def estimate_gap_widths(profile: np.ndarray, prominence: float = 0.015) -> list[float]:
    """Estimate gap widths from radial troughs via half-depth crossings.

    Args:
        profile: Radial profile.
        prominence: Prominence threshold used for extrema detection.

    Returns:
        List of estimated gap widths in pixels.
    """
    peaks, gaps, smooth = _find_extrema(profile, prominence=prominence)
    if len(gaps) == 0 or len(peaks) < 2:
        return []

    widths: list[float] = []

    for g in gaps:
        left_candidates = peaks[peaks < g]
        right_candidates = peaks[peaks > g]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            continue

        lp = int(left_candidates[-1])
        rp = int(right_candidates[0])
        if rp <= lp + 1:
            continue

        peak_level = float(min(smooth[lp], smooth[rp]))
        trough_level = float(smooth[g])
        depth = peak_level - trough_level
        if depth <= 0:
            continue

        target = trough_level + 0.5 * depth

        li = int(g)
        while li > lp and smooth[li] < target:
            li -= 1
        ri = int(g)
        while ri < rp and smooth[ri] < target:
            ri += 1

        width = float(max(ri - li, 1))
        widths.append(width)

    return widths


def derivative_peak_count(profile: np.ndarray, prominence: float = 0.005) -> int:
    """Count high-gradient transitions in a radial profile.

    Args:
        profile: Radial profile.
        prominence: Peak prominence on absolute derivative.

    Returns:
        Number of derivative peaks.
    """
    smooth = _smooth_profile(profile)
    deriv = np.gradient(smooth)
    abs_deriv = np.abs(deriv)

    if HAS_SCIPY and find_peaks is not None:
        idx, _ = find_peaks(abs_deriv, prominence=prominence)
        return int(len(idx))

    d = np.diff(abs_deriv)
    signs = np.sign(d)
    idx = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1
    return int(len(idx))


def morphology_label(
    ring_count: int,
    gap_count: int,
    mean_gap_width_px: float | None,
    profile_len: int,
) -> str:
    """Assign coarse morphology label from ring/gap statistics.

    Args:
        ring_count: Estimated ring count.
        gap_count: Estimated gap count.
        mean_gap_width_px: Mean gap width in pixels.
        profile_len: Length of radial profile.

    Returns:
        Human-readable morphology label.
    """
    width = float(mean_gap_width_px) if mean_gap_width_px is not None else 0.0
    broad_threshold = 0.08 * profile_len
    narrow_threshold = 0.04 * profile_len

    if gap_count == 0 and ring_count <= 2:
        return "smooth disk"

    if gap_count == 1 and width >= broad_threshold:
        return "single broad gap"

    if (ring_count >= 3 or gap_count >= 2) and width <= max(narrow_threshold, 1.0):
        return "multiple narrow rings"

    if ring_count >= 3 or gap_count >= 2:
        return "multi-ring/gap structure"

    if gap_count == 1:
        return "single narrow gap"

    return "mixed morphology"


def cluster_interpretation_rows(
    processed_images: np.ndarray,
    labels: np.ndarray,
    brightness: np.ndarray,
    axis_ratio: np.ndarray,
) -> list[dict[str, Any]]:
    """Build per-cluster scientific interpretation rows.

    Args:
        processed_images: Processed images `(N, H, W)`.
        labels: Cluster labels.
        brightness: Brightness proxy values.
        axis_ratio: Axis-ratio proxy values.

    Returns:
        List of dictionaries suitable for CSV/JSON reporting.
    """
    rows: list[dict[str, Any]] = []
    unique = [cid for cid in sorted(np.unique(labels).tolist()) if cid != -1]

    for cid in unique:
        mask = labels == cid
        cluster_imgs = processed_images[mask]
        if len(cluster_imgs) == 0:
            continue

        mean_img = cluster_imgs.mean(axis=0)
        profile = radial_intensity_profile(mean_img)
        rings, gaps = estimate_rings_and_gaps(profile)
        gap_widths = estimate_gap_widths(profile)
        mean_gap_width = float(np.mean(gap_widths)) if gap_widths else None
        max_gap_width = float(np.max(gap_widths)) if gap_widths else None
        d_peaks = derivative_peak_count(profile)
        morph = morphology_label(rings, gaps, mean_gap_width, len(profile))

        rows.append(
            {
                "cluster_id": int(cid),
                "count": int(np.sum(mask)),
                "mean_brightness": float(np.mean(brightness[mask])),
                "std_brightness": float(np.std(brightness[mask])),
                "mean_axis_ratio": float(np.mean(axis_ratio[mask])),
                "std_axis_ratio": float(np.std(axis_ratio[mask])),
                "estimated_ring_count": int(rings),
                "estimated_gap_count": int(gaps),
                "derivative_peak_count": int(d_peaks),
                "mean_gap_width_px": mean_gap_width,
                "max_gap_width_px": max_gap_width,
                "gap_widths_px": [round(float(width), 3) for width in gap_widths],
                "morphology_label": morph,
            }
        )

    return rows
