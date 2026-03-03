"""Top-level orchestration for end-to-end PLANAR execution."""

from __future__ import annotations

import logging
from pathlib import Path

from planar.config import PlanarConfig
from planar.pipelines.autoencoder import run_autoencoder_pipeline
from planar.pipelines.clustering import run_clustering_pipeline
from planar.pipelines.inference import run_inference_pipeline
from planar.pipelines.reporting import generate_markdown_report
from planar.pipelines.transit import run_transit_pipeline

LOGGER = logging.getLogger(__name__)


def run_full_pipeline(config: PlanarConfig, data_dir_override: str | Path | None = None) -> Path | None:
    """Run enabled PLANAR stages sequentially.

    Args:
        config: Global PLANAR configuration.
        data_dir_override: Optional runtime data directory override.

    Returns:
        Path to generated markdown report if requested, else `None`.
    """
    if data_dir_override is not None:
        config.paths.data_dir = str(data_dir_override)

    model_path: Path | None = None

    if config.autoencoder.enabled:
        ae_artifacts = run_autoencoder_pipeline(config)
        model_path = ae_artifacts.checkpoint_path

    if config.clustering.enabled:
        run_clustering_pipeline(config, model_path=model_path)

    if config.transit.enabled:
        run_transit_pipeline(config)

    if config.inference.enabled:
        run_inference_pipeline(config, model_path=model_path, data_dir=config.paths.data_dir)

    if config.run.run_report:
        report_path = generate_markdown_report(config)
        LOGGER.info("Run report written to %s", report_path)
        return report_path

    return None
