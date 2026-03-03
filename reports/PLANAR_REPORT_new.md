# PLANAR Run Report

Generated: 2026-03-03 08:44 UTC

## Autoencoder
- Train size: 256
- Val size: 64
- Best val loss: 0.0408

## Clustering
- Method: kmeans
- Reducer: pca
- Silhouette: 0.1807
- Noise fraction: 0.0000
- Stability ARI mean: 0.5422
- Brightness eta^2: 0.3320
- Orientation eta^2: 0.4291

### Morphology Snapshot
- Cluster 0: multi-ring/gap structure (rings=2, gaps=2)
- Cluster 1: single narrow gap (rings=1, gaps=1)
- Cluster 2: single narrow gap (rings=1, gaps=1)
- Cluster 3: single narrow gap (rings=1, gaps=1)
- Cluster 4: multi-ring/gap structure (rings=2, gaps=2)

## Transit
- Best val AUC: 0.9987
- Test AUC: 0.9975
- Stress AUC: 0.9598

## Inference
- Loaded images: 320
- Method: kmeans
