# PLANAR: Planetary Latent Analysis & Representation

PLANAR is a bias-aware, stability-validated unsupervised morphology discovery framework
for protoplanetary disk observations.

It is designed not merely to cluster images, but to test whether discovered
structure reflects physical morphology rather than nuisance factors such as
brightness scaling or orientation.

[![CI](https://github.com/Atharva12081/PLANAR/actions/workflows/ci.yml/badge.svg)](https://github.com/Atharva12081/PLANAR/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Install](https://img.shields.io/badge/install-pip%20install%20planar-2ea44f)

## Key Features

- End-to-end pipeline for image representation learning, latent clustering, and transit classification.
- Config-driven execution with deterministic seeding and structured artifact outputs.
- Scientific diagnostics for brightness/orientation dominance (`eta^2`, Kruskal tests).
- Stability analysis via perturbation-based pairwise ARI.
- Morphology interpretation layer with radial derivative peaks and gap-width proxies.
- PyPI-style package layout (`src/`), CLI entrypoint, tests, and documentation scaffold.

## Scientific Motivation

Protoplanetary disk images contain both physically informative structure (rings, gaps, asymmetries) and nuisance variation (brightness scaling, inclination/orientation). PLANAR is designed to separate these effects by learning latent representations and explicitly auditing cluster bias against nuisance proxies. The goal is clustering that is scientifically meaningful, not merely visually separable.

## Design Philosophy

PLANAR was built under three guiding principles:

1. Determinism before optimization.
2. Scientific validation before visual appeal.
3. Reproducibility before performance claims.

Every clustering result must pass:

- Stability under perturbation
- Bias audit against nuisance proxies
- Transparent metric reporting

## Architecture Overview

![Latent embedding example](docs/assets/latent_umap_example.png)

```text
FITS Loader -> Preprocessing -> ConvAutoencoder -> Latent Vectors -> Clustering
   |              |                 |                 |                |
   |              |                 |                 |                +-> HDBSCAN / KMeans / GMM
   |              |                 |                 +-> Embedding (PCA/UMAP)
   |              |                 +-> Reconstructions
   |              +-> Radial averaging (optional)
   +-> Validation + shape checks

Transit Simulator -> 1D CNN Transit Classifier -> ROC/AUC + Stress Evaluation
```

## Repository Layout

```text
PLANAR/
├── src/planar/
│   ├── models/
│   ├── pipelines/
│   ├── cli.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── science_validation.py
│   └── transit_sim.py
├── configs/
├── scripts/
├── tests/
├── docs/
├── notebooks/
├── reports/
├── pyproject.toml
├── requirements.txt
└── environment.yml
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
planar run --config configs/default.yaml
```

## CLI Usage

```bash
# Full pipeline
planar run --config configs/default.yaml

# Deterministic multi-seed reproducibility sweep
planar reproduce --config configs/reproduce.yaml

# Autoencoder only
planar train-ae --config configs/default.yaml

# Clustering with explicit checkpoint
planar cluster --config configs/default.yaml --model-path artifacts/autoencoder/autoencoder.pth

# Inference on new FITS folder
planar infer --config configs/default.yaml --data-dir path/to/fits
```

## Sample Output Snippet

```text
2026-03-08 03:13:17 | INFO | planar.pipelines.autoencoder | Autoencoder artifacts written to artifacts/reproducibility/seed_42/autoencoder
2026-03-08 03:13:29 | INFO | planar.pipelines.clustering | Clustering complete: method=hdbscan reducer=pca silhouette=0.6921 noise=0.033
2026-03-08 03:20:06 | INFO | planar.pipelines.transit | Transit training complete. test_auc=0.9962 stress_auc=0.9612
2026-03-08 03:59:19 | INFO | planar.pipelines.reproducibility | Reproducibility summary written to artifacts/reproducibility/reproducibility_summary.json
```

## Benchmark Snapshot (Current Artifacts)

Metrics below are from the latest seed-controlled runs generated on March 8, 2026:

| Task | Configuration | Result |
|---|---|---|
| Clustering (debiased latent, 3 seeds) | Radial preprocessing + HDBSCAN + nuisance regression | Silhouette `0.5275 ± 0.0075` |
| Clustering stability (debiased latent) | 7 perturbation reruns, 3 seeds | ARI mean `0.9482 ± 0.0285` |
| Bias audit (debiased latent) | Brightness `eta^2` / Orientation `eta^2` | `0.0524 ± 0.0394` / `0.0169 ± 0.0154` |
| HDBSCAN behavior (debiased latent) | Same 900-image run family | Noise fraction `0.5756 ± 0.0645` |
| Negative control | Shuffled labels silhouette | `-0.0053 ± 0.0298` |
| Transit classifier (3 seeds) | Test split AUC | `0.9984 ± 0.0015` |
| Transit stress test (3 seeds) | Red noise + variability + missing segments | AUC `0.9610 ± 0.0006` |
| Autoencoder (3 seeds) | Best val loss (MSE + MS-SSIM objective) | `0.0379 ± 0.0013` |

Artifact sources: `artifacts/reproducibility/reproducibility_summary.json`, `artifacts/reproducibility/seed_*/`.

For reviewer-grade claims, use reproducibility summaries generated by:
`artifacts/reproducibility/reproducibility_summary.json` (mean ± std across seeds + negative controls).

## Reproducibility

- Python version is pinned to `3.11.9` in `.python-version`, `pyproject.toml`, and `environment.yml`.
- All stages consume a YAML config (`configs/default.yaml` or `configs/research_top.yaml`).
- Global seed is set once and propagated to NumPy and PyTorch.
- Deterministic cuDNN mode is enabled when PyTorch is available.
- Run outputs are written to versioned artifact folders with JSON summaries for auditability.
- Reproducibility pipeline supports seed sweeps with aggregate statistics and controls:
  `planar reproduce --config configs/reproduce.yaml`.
- Negative controls are recorded in clustering artifacts (`negative_controls.json`) to validate that structure exceeds shuffled/permuted baselines.
- Train/validation/test leakage checks are emitted in stage summaries under `split_integrity`.

### How headline results are generated

```bash
planar reproduce --config configs/reproduce.yaml
planar report --config configs/research_top.yaml
```

Primary evidence files:

- `artifacts/reproducibility/reproducibility_summary.json`
- `artifacts/clustering*/cluster_stability_summary.json`
- `artifacts/clustering*/cluster_bias_summary.json`
- `artifacts/transit/train_summary.json`
- `reports/PLANAR_REPORT.md`

## Methodological Notes

- **Why radial averaging:** suppresses azimuthal orientation effects so clustering emphasizes radial morphology (rings/gaps).
- **Why HDBSCAN:** supports variable-density structure and marks ambiguous objects as noise (`-1`) rather than forcing assignment.
- **What silhouette means:** measures within-cluster compactness versus between-cluster separation (higher is better).
- **What ARI stability means:** agreement of cluster partitions under latent perturbations; high ARI indicates robust structure.
- **Why high noise fraction is not always bad:** in density clustering, noise can represent rare/transitional morphologies rather than failure.

## Why This Matters

Astrophysical ML often optimizes predictive performance without testing whether learned structure aligns with physical hypotheses. PLANAR addresses this gap by combining representation learning with explicit scientific validation layers, making unsupervised outcomes more useful for downstream disk-physics interpretation.

## Limitations

- Current transit data are simulated; domain shift to real survey light curves still requires calibration.
- Ring/gap estimators are proxy-based and do not replace radiative transfer modeling.
- HDBSCAN sensitivity to feature scaling and sample density can alter cluster counts.
- Current latent model uses a single autoencoder family; contrastive/self-supervised alternatives are not yet integrated.

## Future Work

- Integrate contrastive pretraining and encoder ensembles.
- Add uncertainty-aware clustering and calibrated outlier scoring.
- Incorporate physically grounded simulators and real mission light curves in transit training.
- Add benchmark datasets and continuous integration for regression tracking.
- Expand docs with API references and experiment registry templates.

## Development and Tests

```bash
pytest -q
```

## License

This project is licensed under the MIT License. See `LICENSE`.

## Citation

If you use PLANAR in research, please cite:

```bibtex
@software{planar2026,
  title        = {PLANAR: Planetary Latent Analysis \& Representation},
  author       = {Atharva Parande},
  year         = {2026},
  url          = {https://github.com/Atharva12081/PLANAR},
  version      = {0.1.0},
  license      = {MIT}
}
```
