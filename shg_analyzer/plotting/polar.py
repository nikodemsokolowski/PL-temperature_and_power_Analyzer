"""Polar plotting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from shg_analyzer.config import DEFAULT_CONFIG, AppConfig
from shg_analyzer.core.analysis import aggregate_intensity_matrix
from shg_analyzer.core.models import SampleSeries
from shg_analyzer.core.twist import normalize_intensities


def plot_polar_series(
    series: Iterable[SampleSeries],
    method: str,
    normalization: str = "raw",
    show_fit: bool = True,
    config: AppConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
):
    collection = list(series)
    if not collection:
        raise ValueError("No series supplied for plotting.")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=config.polar_plot.figure_size)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    data_matrix = aggregate_intensity_matrix(collection, method, normalization)
    colors = config.polar_plot.color_cycle

    for index, item in enumerate(collection):
        rows = data_matrix.get(item.label, [])
        if not rows:
            continue
        angles = np.deg2rad([row["angle_deg"] for row in rows])
        intensities = [row["normalized_intensity"] if normalization != "raw" else row["intensity"] for row in rows]
        ax.scatter(angles, intensities, label=item.label, color=colors[index % len(colors)], s=40)
        ax.plot(angles, intensities, color=colors[index % len(colors)], alpha=config.polar_plot.normalized_alpha)

        if show_fit and item.orientation is not None:
            theta = np.linspace(0.0, 2 * np.pi, 360)
            theta_deg = np.rad2deg(theta)
            amplitude = item.orientation.amplitude
            offset = item.orientation.diagnostics.get("offset", 0.0)
            model = amplitude * (np.cos(3.0 * (theta - np.deg2rad(item.orientation.theta0_deg))) ** 2) + offset
            if normalization != "raw":
                model = normalize_intensities(model.tolist(), normalization)
            ax.plot(theta, model, linestyle="--", color=colors[index % len(colors)], linewidth=1.0)

    ax.set_title(f"Polar Plot — {method} ({normalization})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in config.polar_plot.export_formats:
            fig.savefig(output_path.with_suffix(f".{ext}"), dpi=config.polar_plot.dpi, bbox_inches="tight")

    return fig, ax


__all__ = ["plot_polar_series"]
