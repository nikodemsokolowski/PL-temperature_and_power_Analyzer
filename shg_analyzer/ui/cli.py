"""Command-line interface for the SHG analyzer."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from shg_analyzer.core.analysis import AnalysisOptions
from shg_analyzer.plotting.report import build_summary_table
from shg_analyzer.ui.controller import SHGAnalyzerController


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polarization-resolved SHG analyzer")
    parser.add_argument("directory", type=Path, help="Directory containing SHG measurement files")
    parser.add_argument("--background", choices=["global_min", "local_min", "linear", "polynomial", "none"], default="global_min")
    parser.add_argument("--integration-window", type=float, default=0.02, help="Integration window width (energy units)")
    parser.add_argument("--method", action="append", dest="methods", help="Intensity methods to run. Can be set multiple times.")
    parser.add_argument("--normalization", choices=["raw", "normalized", "equalized"], default="raw")
    parser.add_argument("--plot", action="store_true", help="Generate polar plot after analysis")
    parser.add_argument("--plot-output", type=Path, help="Output path (without extension) for polar plot")
    parser.add_argument("--twist", nargs=2, metavar=("SERIES_A", "SERIES_B"), help="Compute twist angle between two series labels")
    parser.add_argument("--orientation-method", default="peak_max")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    controller = SHGAnalyzerController()
    controller.load_directory(args.directory)

    methods = args.methods or ["peak_max", "integrated"]
    method_params = {"integrated": {"window": args.integration_window}}
    background = None if args.background == "none" else args.background

    options = AnalysisOptions(
        intensity_methods=methods,
        method_params=method_params,
        background_method=background,
        normalization_mode=args.normalization,
        orientation_method=args.orientation_method,
    )

    controller.run_analysis(options)

    summary = build_summary_table(controller.series)
    if not summary.empty:
        print(summary.to_string(index=False))

    if args.plot:
        controller.plot(method=args.orientation_method, normalization=args.normalization, output=args.plot_output)
        if args.plot_output:
            print(f"Polar plot exported to {args.plot_output}")

    if args.twist:
        message = controller.compute_twist(args.twist)
        print(message)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
