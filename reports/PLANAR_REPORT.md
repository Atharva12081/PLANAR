# PLANAR Run Report

Generated: 2026-03-20 11:48 UTC

## Autoencoder
- Train size: 120
- Val size: 30
- Best val loss: 0.0376

## Clustering
- Method: hdbscan
- Reducer: pca
- Silhouette: 0.6976
- Noise fraction: 0.0467
- Stability ARI mean (3-seed aggregate): 0.9482
- Brightness eta^2 (3-seed aggregate): 0.0524
- Orientation eta^2 (3-seed aggregate): 0.0169
- Radial-average audit available: use_radial_average=False

### Morphology Snapshot
- Cluster 0: smooth disk (rings=1, gaps=0)
- Cluster 1: smooth disk (rings=1, gaps=0)

## Transit
- Best val AUC: n/a
- Test AUC: 0.9962
- Stress AUC: 0.9612

## Inference
- Loaded images: 150
- Method: hdbscan

## Reproducibility Sweep
- Seeds: [42, 43, 44]
- Silhouette: 0.5275 ± 0.0075 (n=3)
- Stability ARI: 0.9482 ± 0.0285 (n=3)
- Brightness eta^2: 0.0524 ± 0.0394 (n=3.0000)
- Orientation eta^2: 0.0169 ± 0.0154 (n=3)
- Transit test AUC: 0.9984 ± 0.0015 (n=3.0000)
- Transit stress AUC: 0.9610 ± 0.0006 (n=3.0000)
- NegControl (shuffled labels): -0.0053 ± 0.0298 (n=3)
