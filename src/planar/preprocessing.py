"""Image preprocessing utilities for protoplanetary disk observations."""

from __future__ import annotations

import numpy as np


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Z-score normalize a disk image.

    Args:
        img: Input image.

    Returns:
        Normalized image.
    """
    img = np.asarray(img, dtype=np.float32)
    return (img - img.mean()) / (img.std() + 1e-8)


def minmax_scale(img: np.ndarray) -> np.ndarray:
    """Scale image values to [0, 1].

    Args:
        img: Input image.

    Returns:
        Scaled image.
    """
    min_v = float(np.min(img))
    max_v = float(np.max(img))
    return (img - min_v) / (max_v - min_v + 1e-8)


def center_crop(img: np.ndarray, size: int = 512) -> np.ndarray:
    """Center-crop an image to `size x size` (padding when needed).

    Args:
        img: Input image.
        size: Target square size.

    Returns:
        Cropped image.
    """
    h, w = img.shape

    if h < size or w < size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img = np.pad(img, ((top, bottom), (left, right)), mode="constant")
        h, w = img.shape

    start_x = w // 2 - size // 2
    start_y = h // 2 - size // 2
    return img[start_y : start_y + size, start_x : start_x + size]


def radial_average_image(img: np.ndarray) -> np.ndarray:
    """Compute azimuthal averaging in Cartesian image space.

    Radial averaging intentionally suppresses azimuthal orientation cues so that
    clustering focuses on radial morphology (rings/gaps) instead of viewing angle.

    Args:
        img: Input image.

    Returns:
        Orientation-suppressed image rebuilt from the radial profile.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    radial_sum = np.bincount(r.ravel(), weights=img.ravel())
    radial_count = np.bincount(r.ravel())
    profile = radial_sum / np.maximum(radial_count, 1)
    return profile[r]


def preprocess_image(
    img: np.ndarray,
    crop_size: int = 512,
    use_radial_average: bool = False,
) -> np.ndarray:
    """Preprocess one image for training and clustering.

    Args:
        img: Input image.
        crop_size: Center crop size.
        use_radial_average: Whether to apply radial averaging.

    Returns:
        Preprocessed image.
    """
    out = normalize_image(img)
    out = center_crop(out, size=crop_size)

    if use_radial_average:
        out = radial_average_image(out)

    out = minmax_scale(out)
    return out.astype(np.float32)


def preprocess_dataset(
    images: list[np.ndarray],
    crop_size: int = 512,
    use_radial_average: bool = False,
) -> np.ndarray:
    """Preprocess a list of images into `(N, H, W)` format.

    Args:
        images: Raw image list.
        crop_size: Crop size.
        use_radial_average: Whether to apply radial averaging.

    Returns:
        Stacked preprocessed array.
    """
    processed = [
        preprocess_image(img, crop_size=crop_size, use_radial_average=use_radial_average)
        for img in images
    ]
    return np.stack(processed, axis=0).astype(np.float32)


def radial_intensity_profile(img: np.ndarray) -> np.ndarray:
    """Compute azimuthally averaged radial profile.

    Args:
        img: Input image.

    Returns:
        1D radial intensity profile.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    radial_sum = np.bincount(r.ravel(), weights=img.ravel())
    radial_count = np.bincount(r.ravel())
    return radial_sum / np.maximum(radial_count, 1)
