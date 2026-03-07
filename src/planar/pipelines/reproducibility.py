"""Deterministic multi-seed reproducibility pipeline for PLANAR."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from planar.config import PlanarConfig
from planar.pipelines.autoencoder import run_autoencoder_pipeline
from planar.pipelines.clustering import run_clustering_pipeline
from planar.pipelines.reporting import generate_markdown_report
from planar.pipelines.transit import run_transit_pipeline
from planar.runtime import ensure_dir, save_json

LOGGER = logging.getLogger(__name__)


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    """Compute mean/std for finite numeric values.

    Args:
        values: List of optional floats.

    Returns:
        Dictionary with `mean`, `std`, and `n`.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "n": int(len(clean)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_reproducibility_pipeline(config: PlanarConfig) -> Path:
    """Run selected PLANAR stages across multiple seeds and aggregate metrics.

    Args:
        config: Global PLANAR configuration.

    Returns:
        Path to aggregated reproducibility summary JSON.
    """
    out_root = ensure_dir(Path(config.paths.artifacts_dir) / config.reproducibility.out_subdir)

    per_seed: list[dict[str, Any]] = []
    for seed in config.reproducibility.seeds:
        run_cfg = copy.deepcopy(config)
        run_cfg.project.seed = int(seed)
        run_cfg.paths.artifacts_dir = str(ensure_dir(out_root / f"seed_{seed}"))

        LOGGER.info("Reproducibility run for seed=%d", seed)

        model_path: Path | None = None
        ae_summary: dict[str, Any] = {}
        cl_summary: dict[str, Any] = {}
        tr_summary: dict[str, Any] = {}

        if config.reproducibility.run_autoencoder and run_cfg.autoencoder.enabled:
            ae_artifacts = run_autoencoder_pipeline(run_cfg)
            model_path = ae_artifacts.checkpoint_path
            ae_summary = _read_json(ae_artifacts.summary_path)

        if config.reproducibility.run_clustering and run_cfg.clustering.enabled:
            cl_artifacts = run_clustering_pipeline(run_cfg, model_path=model_path)
            cl_summary = _read_json(cl_artifacts.summary_path)

        if config.reproducibility.run_transit and run_cfg.transit.enabled:
            tr_artifacts = run_transit_pipeline(run_cfg)
            tr_summary = _read_json(tr_artifacts.summary_path)

        if run_cfg.run.run_report:
            generate_markdown_report(run_cfg, output_path=Path(run_cfg.paths.reports_dir) / f"PLANAR_REPORT_seed_{seed}.md")

        per_seed.append(
            {
                "seed": int(seed),
                "artifacts_dir": run_cfg.paths.artifacts_dir,
                "autoencoder": ae_summary,
                "clustering": cl_summary,
                "transit": tr_summary,
            }
        )

    clustering_rows = [entry.get("clustering", {}) for entry in per_seed]
    transit_rows = [entry.get("transit", {}) for entry in per_seed]
    ae_rows = [entry.get("autoencoder", {}) for entry in per_seed]

    aggregate = {
        "autoencoder_best_val_loss": _mean_std([row.get("best_val_loss") for row in ae_rows]),
        "clustering_silhouette": _mean_std([row.get("metrics", {}).get("silhouette") for row in clustering_rows]),
        "clustering_ari_mean": _mean_std([row.get("stability_summary", {}).get("ari_mean") for row in clustering_rows]),
        "clustering_noise_fraction": _mean_std([row.get("metrics", {}).get("noise_fraction") for row in clustering_rows]),
        "brightness_eta_squared": _mean_std([row.get("bias_summary", {}).get("brightness_eta_squared") for row in clustering_rows]),
        "orientation_eta_squared": _mean_std([row.get("bias_summary", {}).get("axis_ratio_eta_squared") for row in clustering_rows]),
        "negative_control_silhouette_shuffled_labels": _mean_std(
            [row.get("negative_controls", {}).get("shuffled_labels", {}).get("silhouette") for row in clustering_rows]
        ),
        "negative_control_silhouette_permuted_latent": _mean_std(
            [
                row.get("negative_controls", {}).get("permuted_latent_refit", {}).get("metrics", {}).get("silhouette")
                for row in clustering_rows
            ]
        ),
        "transit_test_auc": _mean_std([row.get("test_auc") for row in transit_rows]),
        "transit_stress_auc": _mean_std([row.get("stress_test_auc") for row in transit_rows]),
    }

    summary = {
        "project": config.project.name,
        "seeds": [int(seed) for seed in config.reproducibility.seeds],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }

    summary_path = out_root / config.reproducibility.summary_filename
    save_json(summary, summary_path)
    LOGGER.info("Reproducibility summary written to %s", summary_path)
    return summary_path
