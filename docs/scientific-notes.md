# Scientific Notes

## Why radial averaging

Radial averaging reduces orientation-dependent azimuthal features so latent clustering is less likely to group by viewing angle.

## Why HDBSCAN

Disk populations may be uneven and include rare morphologies. HDBSCAN models variable-density clusters and explicitly labels ambiguous points as noise (`-1`).

## Silhouette interpretation

Silhouette quantifies separation vs compactness (higher is better). It is reported on non-noise points for density clustering.

## ARI stability

ARI is computed pairwise across perturbation reruns in latent space. High ARI indicates cluster assignments are robust to small embedding noise.

## Noise fraction

A high HDBSCAN noise fraction can be scientifically acceptable when the dataset contains transitional or mixed morphologies that do not form dense groups.
