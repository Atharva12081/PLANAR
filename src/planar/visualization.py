"""Plotting utilities for training, clustering, and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from planar.preprocessing import radial_intensity_profile


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_metric_curves(
    train_values: list[float],
    val_values: list[float],
    title: str,
    ylabel: str,
    save_path: str | Path,
) -> None:
    """Plot train/validation metric curves.

    Args:
        train_values: Training metric history.
        val_values: Validation metric history.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Output image path.
    """
    _ensure_parent(save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=f"Train {ylabel}")
    plt.plot(val_values, label=f"Val {ylabel}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_embedding_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    title: str = "Latent Space Embedding",
) -> None:
    """Plot 2D embedding with cluster coloring.

    Args:
        embedding: 2D embedding array.
        labels: Cluster labels.
        save_path: Output image path.
        title: Plot title.
    """
    _ensure_parent(save_path)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab20",
        s=20,
        alpha=0.9,
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_cluster_means(
    images: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    max_clusters: int = 16,
) -> None:
    """Plot mean image per cluster.

    Args:
        images: Processed images `(N, H, W)`.
        labels: Cluster labels.
        save_path: Output image path.
        max_clusters: Maximum clusters to display.
    """
    _ensure_parent(save_path)

    unique = [cluster_id for cluster_id in sorted(np.unique(labels).tolist()) if cluster_id != -1]
    unique = unique[:max_clusters]
    n = len(unique)

    if n == 0:
        return

    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, cluster_id in enumerate(unique):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        mean_img = images[labels == cluster_id].mean(axis=0)
        ax.imshow(mean_img, cmap="magma")
        ax.set_title(f"Cluster {cluster_id}")
        ax.axis("off")

    for j in range(n, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_radial_intensity_profiles(
    images: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    max_clusters: int = 12,
) -> None:
    """Plot radial intensity profiles of cluster mean images.

    Args:
        images: Processed images `(N, H, W)`.
        labels: Cluster labels.
        save_path: Output image path.
        max_clusters: Maximum number of clusters to plot.
    """
    _ensure_parent(save_path)

    unique = [cluster_id for cluster_id in sorted(np.unique(labels).tolist()) if cluster_id != -1][:max_clusters]
    if not unique:
        return

    plt.figure(figsize=(8, 6))
    for cluster_id in unique:
        cluster_mean = images[labels == cluster_id].mean(axis=0)
        profile = radial_intensity_profile(cluster_mean)
        plt.plot(profile, label=f"Cluster {cluster_id}")

    plt.title("Cluster Mean Radial Intensity Profiles")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_reconstructions(
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    save_path: str | Path,
    n_examples: int = 8,
) -> None:
    """Plot input/reconstruction comparison grid.

    Args:
        inputs: Input image array.
        reconstructions: Reconstructed image array.
        save_path: Output image path.
        n_examples: Number of examples to visualize.
    """
    _ensure_parent(save_path)

    n = min(n_examples, len(inputs))
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5))

    for i in range(n):
        axes[0, i].imshow(inputs[i], cmap="magma")
        axes[0, i].set_title("Input")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructions[i], cmap="magma")
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, save_path: str | Path) -> float:
    """Plot ROC curve and return AUC.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities.
        save_path: Output image path.

    Returns:
        Area-under-curve score.
    """
    _ensure_parent(save_path)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(6.5, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Transit Classifier ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

    return roc_auc


def plot_proxy_by_cluster(
    proxy_values: np.ndarray,
    labels: np.ndarray,
    ylabel: str,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot boxplots of proxy values per cluster.

    Args:
        proxy_values: Proxy vector.
        labels: Cluster labels.
        ylabel: Y-axis label.
        title: Plot title.
        save_path: Output image path.
    """
    _ensure_parent(save_path)

    unique = [cluster_id for cluster_id in sorted(np.unique(labels).tolist()) if cluster_id != -1]
    if not unique:
        return

    data = [proxy_values[labels == cluster_id] for cluster_id in unique]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=[str(cluster_id) for cluster_id in unique], showfliers=False)
    plt.title(title)
    plt.xlabel("Cluster ID")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_bias_scatter(
    brightness: np.ndarray,
    axis_ratio: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot brightness-orientation scatter colored by cluster.

    Args:
        brightness: Brightness proxy values.
        axis_ratio: Axis-ratio proxy values.
        labels: Cluster labels.
        save_path: Output image path.
    """
    _ensure_parent(save_path)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        brightness,
        axis_ratio,
        c=labels,
        cmap="tab20",
        s=28,
        alpha=0.9,
    )
    plt.xlabel("Brightness Proxy (mean crop intensity)")
    plt.ylabel("Axis Ratio Proxy")
    plt.title("Brightness vs Orientation Proxy by Cluster")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
