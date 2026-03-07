# Configurations

- `default.yaml`: baseline full pipeline run.
- `research_top.yaml`: top-performing radial + HDBSCAN configuration on 900-image experiments.
- `reproduce.yaml`: deterministic multi-seed reproducibility sweep configuration.

Use with:

```bash
planar run --config configs/default.yaml
planar reproduce --config configs/reproduce.yaml
```
