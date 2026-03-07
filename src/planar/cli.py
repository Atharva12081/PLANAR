"""Command-line interface for PLANAR."""

from __future__ import annotations

import argparse
from pathlib import Path

from planar.config import load_config
from planar.logging_utils import setup_logging
from planar.runtime import configure_runtime_environment


def build_parser() -> argparse.ArgumentParser:
    """Build the PLANAR CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="planar",
        description="PLANAR: Planetary Latent Analysis & Representation CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full PLANAR pipeline from a YAML config.",
        description="Run all enabled stages: autoencoder, clustering, transit, inference, and report.",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )
    run_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for input FITS directory.",
    )

    ae_parser = subparsers.add_parser(
        "train-ae",
        help="Train only the autoencoder stage.",
        description="Train the convolutional autoencoder and write reconstruction artifacts.",
    )
    ae_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Run only latent-space clustering.",
        description="Encode images with a trained autoencoder, then cluster latent vectors.",
    )
    cluster_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )
    cluster_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional checkpoint override (defaults to autoencoder output path).",
    )

    transit_parser = subparsers.add_parser(
        "train-transit",
        help="Train only the transit classifier stage.",
        description="Generate synthetic light curves and train the transit detection CNN.",
    )
    transit_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run only inference on FITS images.",
        description="Encode, cluster, and reconstruct new FITS images using trained models.",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional checkpoint override (defaults to autoencoder output path).",
    )
    infer_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for FITS input directory.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate markdown report from artifacts.",
        description="Aggregate JSON summaries into a single markdown report.",
    )
    report_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )
    report_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for generated markdown report.",
    )

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Run deterministic multi-seed reproducibility sweep.",
        description=(
            "Execute selected PLANAR stages over configured seeds, then aggregate "
            "mean/std metrics and negative-control diagnostics."
        ),
    )
    reproduce_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to PLANAR YAML configuration file.",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.project.log_level)
    configure_runtime_environment()

    if args.command == "run":
        from planar.pipelines.full import run_full_pipeline

        run_full_pipeline(config=config, data_dir_override=args.data_dir)
        return

    if args.command == "train-ae":
        from planar.pipelines.autoencoder import run_autoencoder_pipeline

        run_autoencoder_pipeline(config)
        return

    if args.command == "cluster":
        from planar.pipelines.clustering import run_clustering_pipeline

        run_clustering_pipeline(config, model_path=args.model_path)
        return

    if args.command == "train-transit":
        from planar.pipelines.transit import run_transit_pipeline

        run_transit_pipeline(config)
        return

    if args.command == "infer":
        from planar.pipelines.inference import run_inference_pipeline

        run_inference_pipeline(config, model_path=args.model_path, data_dir=args.data_dir)
        return

    if args.command == "report":
        from planar.pipelines.reporting import generate_markdown_report

        output = generate_markdown_report(config, output_path=Path(args.out) if args.out else None)
        print(output)
        return

    if args.command == "reproduce":
        from planar.pipelines.reproducibility import run_reproducibility_pipeline

        output = run_reproducibility_pipeline(config)
        print(output)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
