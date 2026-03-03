# PLANAR Run Report

Generated: 2026-03-03 07:34 UTC

## Autoencoder (Image Test)
- Train size: 256
- Val size: 64
- Best val loss: 0.0408
- MS-SSIM available: True

## Clustering (General Test)
- Images: 320
- Method: kmeans
- Reducer: pca
- Num clusters: 6
- Noise fraction: 0.0000
- Silhouette: 0.1807

### Stability Under Perturbations
- Runs: 7
- Noise std (relative): 0.0100
- Pairwise ARI mean: 0.5422
- Pairwise ARI std: 0.1110
- Cluster count mean: 6.0000

### Bias Dominance Checks
- Brightness eta^2: 0.3320
- Orientation eta^2: 0.4291
- Brightness dominated: False
- Orientation dominated: False

### Morphology Interpretation
- Cluster 0: multi-ring/gap structure (rings=2, gaps=2, derivative_peaks=4, mean_gap_width_px=67.0000)
- Cluster 1: single narrow gap (rings=1, gaps=1, derivative_peaks=2, mean_gap_width_px=n/a)
- Cluster 2: single narrow gap (rings=1, gaps=1, derivative_peaks=1, mean_gap_width_px=n/a)
- Cluster 3: single narrow gap (rings=1, gaps=1, derivative_peaks=2, mean_gap_width_px=n/a)
- Cluster 4: multi-ring/gap structure (rings=2, gaps=2, derivative_peaks=2, mean_gap_width_px=51.0000)
- Cluster 5: single narrow gap (rings=1, gaps=1, derivative_peaks=2, mean_gap_width_px=n/a)

### Radial-Average Mitigation Run
- Silhouette: 0.2208
- Num clusters: 6
- Noise fraction: 0.0000
- Brightness eta^2: 0.2356
- Orientation eta^2: 0.1182
- Brightness dominated: False
- Orientation dominated: False
- Pairwise ARI mean: 0.7188

### Config Sweep Recommendation
- Variant: radial
- Method: hdbscan
- Silhouette: 0.4089
- Noise fraction: 0.3750
- Stability ARI mean: 0.9923
- Rank score: 0.4540

## Transit Classifier (Sequential Test)
- Samples: 4000
- Best val AUC: 0.9987
- Test AUC: 0.9975
- Stress eval samples: 2000
- Stress test AUC: 0.9598

## Inference
- Loaded images: 320
- Method: kmeans
- Reducer: pca
