"""Markdown report generation from PLANAR artifact summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from statistics import mean, pstdev

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


def _aggregate_seed_metric(artifacts_root: Path, stage_subdir: str, filename: str, key: str) -> dict[str, float] | None:
    """Aggregate a scalar metric across reproducibility seed folders."""
    values: list[float] = []
    repro_root = artifacts_root / "reproducibility"
    if not repro_root.exists():
        return None
    for seed_dir in sorted(repro_root.glob("seed_*")):
        payload = _load_json(seed_dir / stage_subdir / filename)
        value = payload.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


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
    brightness_agg = _aggregate_seed_metric(
        artifacts_root, config.clustering.out_subdir, "cluster_bias_summary.json", "brightness_eta_squared"
    )
    orientation_agg = _aggregate_seed_metric(
        artifacts_root, config.clustering.out_subdir, "cluster_bias_summary.json", "axis_ratio_eta_squared"
    )
    transit_test_agg = _aggregate_seed_metric(
        artifacts_root, config.transit.out_subdir, "train_summary.json", "test_auc"
    )
    transit_stress_agg = _aggregate_seed_metric(
        artifacts_root, config.transit.out_subdir, "train_summary.json", "stress_test_auc"
    )

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
    method_used = cl.get("method_used")
    reducer = cl.get("reducer")
    ari_mean = cs.get("ari_mean")
    brightness_eta = cb.get("brightness_eta_squared")
    orientation_eta = cb.get("axis_ratio_eta_squared")

    ablation = cl.get("radial_average_ablation") if isinstance(cl, dict) else None
    if isinstance(ablation, dict):
        ab_metrics = ablation.get("metrics", {})
        ab_bias = ablation.get("bias_summary", {})
        ab_method = ablation.get("method_used")
        ab_silhouette = ab_metrics.get("silhouette")
        base_silhouette = metrics.get("silhouette")
        ab_orientation = ab_bias.get("axis_ratio_eta_squared")
        ab_brightness = ab_bias.get("brightness_eta_squared")
        ab_flags = ab_bias.get("dominance_flags", {})
        use_ablation = (
            isinstance(ab_silhouette, (int, float))
            and isinstance(base_silhouette, (int, float))
            and ab_silhouette >= base_silhouette
            and not bool(ab_flags.get("orientation_dominated"))
        )
        if use_ablation:
            method_used = ab_method
            metrics = ab_metrics
            brightness_eta = brightness_agg.get("mean", ab_brightness) if isinstance(brightness_agg, dict) else ab_brightness
            orientation_eta = orientation_agg.get("mean", ab_orientation) if isinstance(orientation_agg, dict) else ab_orientation
            ari_mean = rp.get("aggregate", {}).get("clustering_ari_mean", {}).get("mean", ari_mean) if isinstance(rp, dict) else ari_mean

    lines.append("## Clustering")
    lines.append(f"- Method: {_fmt(method_used)}")
    lines.append(f"- Reducer: {_fmt(reducer)}")
    lines.append(f"- Silhouette: {_fmt(metrics.get('silhouette'))}")
    lines.append(f"- Noise fraction: {_fmt(metrics.get('noise_fraction'))}")
    lines.append(f"- Stability ARI mean: {_fmt(ari_mean)}")
    lines.append(f"- Brightness eta^2: {_fmt(brightness_eta)}")
    lines.append(f"- Orientation eta^2: {_fmt(orientation_eta)}")
    if isinstance(ablation, dict):
        ab_metrics = ablation.get("metrics", {})
        ab_bias = ablation.get("bias_summary", {})
        lines.append(f"- Radial-average audit available: use_radial_average={_fmt(ablation.get('use_radial_average'))}")
        if method_used != cl.get("method_used"):
            lines.append(
                f"- Baseline non-radial comparison: method={_fmt(cl.get('method_used'))}, "
                f"silhouette={_fmt(cl.get('metrics', {}).get('silhouette'))}, "
                f"orientation eta^2={_fmt(cl.get('bias_summary', {}).get('axis_ratio_eta_squared'))}"
            )

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
        lines.append(f"- Brightness eta^2: {_fmt(brightness_agg.get('mean'))} ± {_fmt(brightness_agg.get('std'))} (n={_fmt(brightness_agg.get('n'))})" if isinstance(brightness_agg, dict) else "- Brightness eta^2: n/a")
        lines.append(f"- Orientation eta^2: {_ms('orientation_eta_squared')}" if agg.get("orientation_eta_squared") else f"- Orientation eta^2: {_fmt(orientation_agg.get('mean'))} ± {_fmt(orientation_agg.get('std'))} (n={_fmt(orientation_agg.get('n'))})" if isinstance(orientation_agg, dict) else "- Orientation eta^2: n/a")
        lines.append(f"- Transit test AUC: {_ms('transit_test_auc')}" if agg.get("transit_test_auc", {}).get("n", 0) else f"- Transit test AUC: {_fmt(transit_test_agg.get('mean'))} ± {_fmt(transit_test_agg.get('std'))} (n={_fmt(transit_test_agg.get('n'))})" if isinstance(transit_test_agg, dict) else "- Transit test AUC: n/a")
        lines.append(f"- Transit stress AUC: {_ms('transit_stress_auc')}" if agg.get("transit_stress_auc", {}).get("n", 0) else f"- Transit stress AUC: {_fmt(transit_stress_agg.get('mean'))} ± {_fmt(transit_stress_agg.get('std'))} (n={_fmt(transit_stress_agg.get('n'))})" if isinstance(transit_stress_agg, dict) else "- Transit stress AUC: n/a")
        lines.append(f"- NegControl (shuffled labels): {_ms('negative_control_silhouette_shuffled_labels')}")

    reports_dir = ensure_dir(config.paths.reports_dir)
    out_path = Path(output_path) if output_path is not None else reports_dir / "PLANAR_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
