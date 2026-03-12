# PLANAR Run Report

Generated: 2026-03-12 22:28 UTC

Artifacts + pretrained weights:
https://drive.google.com/drive/u/0/folders/1x3jiMVj2Iyeu9quEI53SVg6-EF7tMFqx

## Autoencoder
- Train size: 120
- Val size: 30
- Best val loss: 0.0376

## Clustering
- Method: hdbscan
- Reducer: pca
- Silhouette: 0.6976
- Noise fraction: 0.0467
- Stability ARI mean: 0.9905
- Brightness eta^2: 0.5822
- Orientation eta^2: 0.0096

### Morphology Snapshot
- Cluster 0: smooth disk (rings=1, gaps=0)
- Cluster 1: smooth disk (rings=1, gaps=0)

## Transit
- Best val AUC: 0.9994
- Test AUC: 0.9962
- Stress AUC: 0.9612

## Inference
- Loaded images: 150
- Method: hdbscan

## Reproducibility Sweep
- Seeds: [42, 43, 44]
- Silhouette: 0.5275 ± 0.0075 (n=3)
- Stability ARI: 0.9482 ± 0.0285 (n=3)
- Orientation eta^2: 0.0169 ± 0.0154 (n=3)
- Transit test AUC: n/a ± n/a (n=0)
- Transit stress AUC: n/a ± n/a (n=0)
- NegControl (shuffled labels): -0.0053 ± 0.0298 (n=3)
