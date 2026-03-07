# Reproducibility Guide

PLANAR supports deterministic, multi-seed reproducibility audits from a single command.

## One-command run

```bash
planar reproduce --config configs/reproduce.yaml
```

This generates:

- Per-seed artifacts under `artifacts/reproducibility/seed_<seed>/...`
- Aggregate summary at `artifacts/reproducibility/reproducibility_summary.json`

## What is reported

- Autoencoder: best validation loss (mean ± std)
- Clustering: silhouette, ARI stability, noise fraction (mean ± std)
- Bias audits: brightness/orientation eta² (mean ± std)
- Negative controls:
  - silhouette with shuffled labels
  - silhouette after clustering on permuted latent vectors
- Transit classifier: test AUC and stress AUC (mean ± std)

## Leakage checks

Stage summaries include split integrity checks:

- Autoencoder: train/val overlap check
- Transit: train/val/test overlap and coverage checks

If `leakage_detected` is `true`, run-level claims should be treated as invalid until fixed.
