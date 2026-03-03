"""Pipeline entrypoints for PLANAR training and inference workflows."""

from __future__ import annotations

from .autoencoder import run_autoencoder_pipeline
from .clustering import run_clustering_pipeline
from .full import run_full_pipeline
from .inference import run_inference_pipeline
from .reporting import generate_markdown_report
from .transit import run_transit_pipeline

__all__ = [
    "run_autoencoder_pipeline",
    "run_clustering_pipeline",
    "run_full_pipeline",
    "run_inference_pipeline",
    "run_transit_pipeline",
    "generate_markdown_report",
]
