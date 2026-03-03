# Architecture

```text
FITS Loader -> Preprocessing -> ConvAutoencoder -> Latent Space
                                           |            |
                                           |            +-> HDBSCAN / KMeans / GMM
                                           |                       |
                                           +-> Reconstruction       +-> Embeddings + Scientific diagnostics

Transit Simulator -> 1D CNN Transit Classifier -> ROC/AUC + Stress evaluation
```

Primary package layout:

- `src/planar/models`: neural models + clustering backends.
- `src/planar/pipelines`: train/cluster/transit/inference orchestration.
- `src/planar/science_validation.py`: eta² bias checks, ring/gap proxies.
- `configs/`: reproducible YAML configs.
- `tests/`: fast regression checks.
