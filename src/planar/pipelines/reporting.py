"""Markdown report generation from PLANAR artifact summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from planar.config import PlanarConfig
from planar.runtime import ensure_dir


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON if present, otherwise return empty mapping.

    Args:
        path: JSON path.

    Returns:
        Parsed JSON dictionary or empty dictionary.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: object, default: str = "n/a") -> str:
    """Format values for markdown rendering.

    Args:
        value: Value to render.
        default: Text used for missing values.

    Returns:
        Formatted string.
    """
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def generate_markdown_report(config: PlanarConfig, output_path: str | Path | None = None) -> Path:
    """Generate a compact markdown report across pipeline stages.

    Args:
        config: Global PLANAR configuration.
        output_path: Optional output override.

    Returns:
        Path to generated report.
    """
    artifacts_root = Path(config.paths.artifacts_dir)

    ae = _load_json(artifacts_root / config.autoencoder.out_subdir / "train_summary.json")
    cl = _load_json(artifacts_root / config.clustering.out_subdir / "clustering_summary.json")
    cb = _load_json(artifacts_root / config.clustering.out_subdir / "cluster_bias_summary.json")
    cs = _load_json(artifacts_root / config.clustering.out_subdir / "cluster_stability_summary.json")
    ci = _load_json(artifacts_root / config.clustering.out_subdir / "cluster_interpretation.json")
    tr = _load_json(artifacts_root / config.transit.out_subdir / "train_summary.json")
    inf = _load_json(artifacts_root / config.inference.out_subdir / "inference_summary.json")
    rp = _load_json(artifacts_root / config.reproducibility.out_subdir / config.reproducibility.summary_filename)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# {config.project.name} Run Report")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append("## Autoencoder")
    lines.append(f"- Train size: {_fmt(ae.get('train_size'))}")
    lines.append(f"- Val size: {_fmt(ae.get('val_size'))}")
    lines.append(f"- Best val loss: {_fmt(ae.get('best_val_loss'))}")
    lines.append("")

    metrics = cl.get("metrics", {}) if isinstance(cl, dict) else {}
    lines.append("## Clustering")
    lines.append(f"- Method: {_fmt(cl.get('method_used'))}")
    lines.append(f"- Reducer: {_fmt(cl.get('reducer'))}")
    lines.append(f"- Silhouette: {_fmt(metrics.get('silhouette'))}")
    lines.append(f"- Noise fraction: {_fmt(metrics.get('noise_fraction'))}")
    lines.append(f"- Stability ARI mean: {_fmt(cs.get('ari_mean'))}")
    lines.append(f"- Brightness eta^2: {_fmt(cb.get('brightness_eta_squared'))}")
    lines.append(f"- Orientation eta^2: {_fmt(cb.get('axis_ratio_eta_squared'))}")

    clusters = ci.get("clusters") if isinstance(ci, dict) else None
    if isinstance(clusters, list) and clusters:
        lines.append("")
        lines.append("### Morphology Snapshot")
        for row in sorted(clusters, key=lambda item: int(item.get("cluster_id", 0)))[:5]:
            lines.append(
                f"- Cluster {row.get('cluster_id')}: {row.get('morphology_label')} "
                f"(rings={row.get('estimated_ring_count')}, gaps={row.get('estimated_gap_count')})"
            )

    lines.append("")
    lines.append("## Transit")
    lines.append(f"- Best val AUC: {_fmt(tr.get('best_val_auc'))}")
    lines.append(f"- Test AUC: {_fmt(tr.get('test_auc'))}")
    lines.append(f"- Stress AUC: {_fmt(tr.get('stress_test_auc'))}")
    lines.append("")
    lines.append("## Inference")
    lines.append(f"- Loaded images: {_fmt(inf.get('num_loaded'))}")
    lines.append(f"- Method: {_fmt(inf.get('method_used'))}")

    agg = rp.get("aggregate") if isinstance(rp, dict) else None
    if isinstance(agg, dict):
        lines.append("")
        lines.append("## Reproducibility Sweep")

        def _ms(key: str) -> str:
            stat = agg.get(key, {})
            if not isinstance(stat, dict):
                return "n/a"
            return f"{_fmt(stat.get('mean'))} ± {_fmt(stat.get('std'))} (n={_fmt(stat.get('n'))})"

        lines.append(f"- Seeds: {rp.get('seeds', [])}")
        lines.append(f"- Silhouette: {_ms('clustering_silhouette')}")
        lines.append(f"- Stability ARI: {_ms('clustering_ari_mean')}")
        lines.append(f"- Orientation eta^2: {_ms('orientation_eta_squared')}")
        lines.append(f"- Transit test AUC: {_ms('transit_test_auc')}")
        lines.append(f"- Transit stress AUC: {_ms('transit_stress_auc')}")
        lines.append(f"- NegControl (shuffled labels): {_ms('negative_control_silhouette_shuffled_labels')}")

    reports_dir = ensure_dir(config.paths.reports_dir)
    out_path = Path(output_path) if output_path is not None else reports_dir / "PLANAR_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
