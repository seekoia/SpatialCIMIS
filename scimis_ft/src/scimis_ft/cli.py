from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig, parse_config_file
from .processing import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scimis-ft",
        description="Spatial CIMIS weekly climatology and Fourier decomposition pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to plain-text configuration file (key=value per line).",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing NetCDF outputs.")
    parser.add_argument("--no-metrics", action="store_true", help="Skip RMSE computation.")
    parser.add_argument("--no-plots", action="store_true", help="Skip component plotting.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = parse_config_file(args.config)
    results = run_pipeline(
        config,
        save_results=not args.no_save,
        compute_metrics=not args.no_metrics,
        make_plots=not args.no_plots,
        show_plots=args.show_plots,
    )

    data_shape = results.get("data_shape")
    data_dtype = results.get("data_dtype")

    summary_lines = []
    if data_shape is not None:
        dtype_text = f", dtype={data_dtype}" if data_dtype else ""
        summary_lines.append(f"Loaded data shape: {data_shape}{dtype_text}")
    summary_lines.append(f"Weekly climatology shape: {results['weekly_climatology'].shape}")
    summary_lines.append(
        f"Fourier components frequencies: {results['fourier_components'].frequency.values}"
    )
    if results.get("plot_path"):
        summary_lines.append(f"Component plot saved to: {results['plot_path']}")
    if results.get("timeseries_plot"):
        summary_lines.append(f"Timeseries plot saved to: {results['timeseries_plot']}")
    if results.get("rmse_plot"):
        summary_lines.append(f"RMSE map saved to: {results['rmse_plot']}")
    if results.get("histogram_plot"):
        summary_lines.append(f"Histogram plot saved to: {results['histogram_plot']}")
    if results.get("rmse") is not None:
        summary_lines.append("RMSE dataset computed.")

    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

