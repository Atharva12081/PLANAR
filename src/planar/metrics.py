"""Metrics for reconstruction quality and clustering reliability."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim  # type: ignore

    HAS_MS_SSIM = True
except Exception:
    ms_ssim = None
    HAS_MS_SSIM = False


def reconstruction_components(
    output: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.8,
    ssim_weight: float = 0.2,
    data_range: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total reconstruction loss plus detached components.

    Args:
        output: Reconstructed images.
        target: Ground-truth images.
        mse_weight: Weight for MSE term.
        ssim_weight: Weight for MS-SSIM term.
        data_range: Dynamic range used by MS-SSIM.

    Returns:
        Tuple of total loss, detached MSE, and detached `(1 - MS-SSIM)`.
    """
    mse = F.mse_loss(output, target)

    if HAS_MS_SSIM and ms_ssim is not None:
        ssim_loss = 1.0 - ms_ssim(output, target, data_range=data_range)
        total = mse_weight * mse + ssim_weight * ssim_loss
    else:
        ssim_loss = torch.zeros((), device=output.device)
        total = mse

    return total, mse.detach(), ssim_loss.detach()


def reconstruction_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for project-standard reconstruction loss.

    Args:
        output: Reconstructed images.
        target: Ground-truth images.

    Returns:
        Total reconstruction loss.
    """
    total, _, _ = reconstruction_components(output, target)
    return total


def clustering_quality_scores(latent_vectors: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute core clustering quality statistics.

    Args:
        latent_vectors: Latent vectors of shape `(N, D)`.
        labels: Cluster labels where `-1` denotes noise/outliers.

    Returns:
        Dictionary with number of clusters, noise fraction, and silhouette.
    """
    from sklearn.metrics import silhouette_score

    unique = np.unique(labels)
    valid = unique[unique != -1]

    scores: dict[str, Any] = {
        "num_clusters": int(len(valid)),
        "noise_fraction": float(np.mean(labels == -1)) if len(labels) else 0.0,
    }

    if len(valid) >= 2:
        mask = labels != -1
        scores["silhouette"] = float(silhouette_score(latent_vectors[mask], labels[mask]))
    else:
        scores["silhouette"] = None

    return scores


def clustering_stability_scores(
    latent_vectors: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 10,
    n_clusters: int = 6,
    num_runs: int = 5,
    noise_std: float = 0.01,
    random_state: int = 42,
) -> dict[str, Any]:
    """Quantify clustering stability under latent perturbations.

    Args:
        latent_vectors: Latent vectors of shape `(N, D)`.
        method: Clustering backend.
        min_cluster_size: HDBSCAN minimum cluster size.
        n_clusters: KMeans/GMM cluster count.
        num_runs: Number of perturbation runs.
        noise_std: Relative perturbation level (scaled by latent std).
        random_state: Reproducibility seed.

    Returns:
        Dictionary with ARI summary statistics and cluster count stability.
    """
    if num_runs < 2:
        return {
            "runs": int(num_runs),
            "noise_std": float(noise_std),
            "ari_mean": None,
            "ari_std": None,
            "ari_min": None,
            "ari_max": None,
            "cluster_count_mean": None,
            "cluster_count_std": None,
            "methods_used": [],
        }

    from sklearn.metrics import adjusted_rand_score

    from planar.models.clustering import cluster_latent_space

    rng = np.random.default_rng(random_state)
    scale = float(np.std(latent_vectors))
    sigma = max(scale * noise_std, 1e-8)

    labels_runs: list[np.ndarray] = []
    methods_used: list[str] = []
    cluster_counts: list[int] = []

    for i in range(num_runs):
        noise = rng.normal(0.0, sigma, size=latent_vectors.shape).astype(np.float32)
        perturbed = latent_vectors + noise
        labels, _, used = cluster_latent_space(
            perturbed,
            method=method,
            min_cluster_size=min_cluster_size,
            n_clusters=n_clusters,
            random_state=random_state + i,
        )
        labels_runs.append(labels)
        methods_used.append(used)
        valid = np.unique(labels[labels != -1])
        cluster_counts.append(int(len(valid)))

    pairwise_ari: list[float] = []
    for i in range(len(labels_runs)):
        for j in range(i + 1, len(labels_runs)):
            # ARI compares partitions after relabeling, so it tracks structural
            # agreement across noisy reruns without requiring label identity.
            pairwise_ari.append(float(adjusted_rand_score(labels_runs[i], labels_runs[j])))

    return {
        "runs": int(num_runs),
        "noise_std": float(noise_std),
        "ari_mean": float(np.mean(pairwise_ari)) if pairwise_ari else None,
        "ari_std": float(np.std(pairwise_ari)) if pairwise_ari else None,
        "ari_min": float(np.min(pairwise_ari)) if pairwise_ari else None,
        "ari_max": float(np.max(pairwise_ari)) if pairwise_ari else None,
        "cluster_count_mean": float(np.mean(cluster_counts)) if cluster_counts else None,
        "cluster_count_std": float(np.std(cluster_counts)) if cluster_counts else None,
        "methods_used": sorted(set(methods_used)),
    }
