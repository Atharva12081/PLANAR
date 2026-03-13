# PLANAR Run Report

Generated: 2026-03-13 10:16 UTC

## Autoencoder
- Train size: 120
- Val size: 30
- Best val loss: 0.0376

## Clustering
- Method: hdbscan_all_noise_fallback_kmeans
- Reducer: pca
- Silhouette: 0.3363
- Noise fraction: 0.0000
- Stability ARI mean: 0.7652
- Brightness eta^2: 0.6921
- Orientation eta^2: 0.6480

### Morphology Snapshot
- Cluster 0: smooth disk (rings=1, gaps=0)
- Cluster 1: smooth disk (rings=1, gaps=0)
- Cluster 2: smooth disk (rings=0, gaps=0)
- Cluster 3: smooth disk (rings=1, gaps=0)
- Cluster 4: smooth disk (rings=1, gaps=0)

## Transit
- Best val AUC: n/a
- Test AUC: 0.9962
- Stress AUC: 0.9612

## Inference
- Loaded images: 150
- Method: hdbscan_all_noise_fallback_kmeans

## Reproducibility Sweep
- Seeds: [42, 43, 44]
- Silhouette: 0.5275 ± 0.0075 (n=3)
- Stability ARI: 0.9482 ± 0.0285 (n=3)
- Orientation eta^2: 0.0169 ± 0.0154 (n=3)
- Transit test AUC: n/a ± n/a (n=0)
- Transit stress AUC: n/a ± n/a (n=0)
- NegControl (shuffled labels): -0.0053 ± 0.0298 (n=3)
