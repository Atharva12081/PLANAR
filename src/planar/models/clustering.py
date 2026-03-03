"""Latent-space clustering and dimensionality reduction backends."""

from __future__ import annotations

import numpy as np


def reduce_dim(
    latent_vectors: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, object, str]:
    """Reduce latent vectors to a low-dimensional embedding.

    Args:
        latent_vectors: Latent vectors with shape `(N, D)`.
        method: Reduction backend (`pca` or `umap`).
        n_components: Number of embedding dimensions.
        random_state: Reproducibility seed.

    Returns:
        Embedding, reducer instance, and reducer name used.
    """
    method = method.lower()

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1,
            )
            embedding = reducer.fit_transform(latent_vectors)
            return embedding, reducer, "umap"
        except Exception:
            pass

    from sklearn.decomposition import PCA

    reducer = PCA(n_components=n_components)
    embedding = reducer.fit_transform(latent_vectors)
    name = "pca" if method == "pca" else "pca_fallback"
    return embedding, reducer, name


def cluster_latent_space(
    latent_vectors: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 10,
    n_clusters: int = 6,
    random_state: int = 42,
) -> tuple[np.ndarray, object, str]:
    """Cluster latent vectors using a selected backend.

    Args:
        latent_vectors: Latent vectors with shape `(N, D)`.
        method: Clustering backend (`hdbscan`, `kmeans`, or `gmm`).
        min_cluster_size: Minimum cluster size for HDBSCAN.
        n_clusters: Number of clusters for partitioning methods.
        random_state: Reproducibility seed.

    Returns:
        Labels, fitted clusterer, and method name used.

    Raises:
        ValueError: If method is unsupported.
    """
    method = method.lower()

    if method == "hdbscan":
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(latent_vectors)
            # HDBSCAN can assign `-1` for points in low-density regions.
            # In astronomical morphology studies, this often captures rare or
            # ambiguous systems rather than an algorithmic failure.
            return labels, clusterer, "hdbscan"
        except Exception:
            from sklearn.cluster import DBSCAN

            clusterer = DBSCAN(eps=0.7, min_samples=max(5, min_cluster_size // 2))
            labels = clusterer.fit_predict(latent_vectors)
            return labels, clusterer, "dbscan_fallback"

    if method == "kmeans":
        from sklearn.cluster import KMeans

        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = clusterer.fit_predict(latent_vectors)
        return labels, clusterer, "kmeans"

    if method in {"gmm", "gaussian_mixture", "gaussian-mixture"}:
        from sklearn.mixture import GaussianMixture

        for reg_covar in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                clusterer = GaussianMixture(
                    n_components=n_clusters,
                    random_state=random_state,
                    reg_covar=reg_covar,
                )
                labels = clusterer.fit_predict(latent_vectors)
                tag = "gmm" if reg_covar == 1e-6 else f"gmm_reg_{reg_covar:g}"
                return labels, clusterer, tag
            except Exception:
                continue

        from sklearn.cluster import KMeans

        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = clusterer.fit_predict(latent_vectors)
        return labels, clusterer, "gmm_fallback_kmeans"

    raise ValueError(f"Unsupported clustering method: {method}")
