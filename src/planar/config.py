"""YAML configuration models and loader for PLANAR workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Top-level project runtime controls."""

    name: str = "PLANAR"
    seed: int = 42
    device: str = "auto"
    log_level: str = "INFO"
    python_version: str = "3.11.9"


@dataclass
class PathsConfig:
    """Input/output path configuration."""

    data_dir: str = "data/raw"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    max_files: int | None = None
    expected_shape: list[int] | None = field(default_factory=lambda: [600, 600])


@dataclass
class AutoencoderConfig:
    """Autoencoder training configuration."""

    enabled: bool = True
    out_subdir: str = "autoencoder"
    crop_size: int = 512
    latent_dim: int = 64
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    val_split: float = 0.2
    patience: int = 8
    use_radial_average: bool = False
    augment_rot90: bool = False
    num_workers: int = 0


@dataclass
class ClusteringConfig:
    """Latent-space clustering configuration."""

    enabled: bool = True
    out_subdir: str = "clustering"
    method: str = "hdbscan"
    embedder: str = "pca"
    min_cluster_size: int = 10
    n_clusters: int = 6
    batch_size: int = 16
    use_radial_average: bool = False
    stability_runs: int = 5
    stability_noise_std: float = 0.01


@dataclass
class TransitConfig:
    """Transit classifier training configuration."""

    enabled: bool = True
    out_subdir: str = "transit"
    samples: int = 4000
    num_points: int = 500
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    val_split: float = 0.2
    test_split: float = 0.2
    patience: int = 7
    stress_eval_size: int = 2000
    stress_seed: int = 4242


@dataclass
class InferenceConfig:
    """Inference-time clustering and reconstruction configuration."""

    enabled: bool = True
    out_subdir: str = "inference"
    method: str = "hdbscan"
    embedder: str = "pca"
    min_cluster_size: int = 10
    n_clusters: int = 6
    batch_size: int = 16
    use_radial_average: bool = False


@dataclass
class RunConfig:
    """Pipeline orchestration settings."""

    run_report: bool = True


@dataclass
class PlanarConfig:
    """Full PLANAR configuration model."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    transit: TransitConfig = field(default_factory=TransitConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    run: RunConfig = field(default_factory=RunConfig)


DEFAULT_CONFIG = PlanarConfig()


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries.

    Args:
        base: Base dictionary.
        updates: Overlay dictionary.

    Returns:
        Merged dictionary.
    """
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_to_dict(config: PlanarConfig) -> dict[str, Any]:
    """Convert config dataclass to a plain dictionary.

    Args:
        config: Config object.

    Returns:
        Dictionary representation.
    """
    return asdict(config)


def load_config(path: str | Path) -> PlanarConfig:
    """Load a PLANAR YAML config file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed `PlanarConfig` object.

    Raises:
        FileNotFoundError: If config path does not exist.
        ValueError: If YAML is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    merged = _deep_update(config_to_dict(DEFAULT_CONFIG), raw)

    return PlanarConfig(
        project=ProjectConfig(**merged["project"]),
        paths=PathsConfig(**merged["paths"]),
        autoencoder=AutoencoderConfig(**merged["autoencoder"]),
        clustering=ClusteringConfig(**merged["clustering"]),
        transit=TransitConfig(**merged["transit"]),
        inference=InferenceConfig(**merged["inference"]),
        run=RunConfig(**merged["run"]),
    )
