"""Tests for clustering outputs."""

from __future__ import annotations

import numpy as np

from planar.models.clustering import cluster_latent_space


def test_cluster_output_shape_density() -> None:
    """Clustering should return one label per latent vector."""
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(40, 12)).astype(np.float32)
    labels, _, method = cluster_latent_space(
        latent,
        method="hdbscan",
        min_cluster_size=5,
        random_state=42,
    )

    assert labels.shape == (40,)
    assert method in {"hdbscan", "dbscan_fallback"}
