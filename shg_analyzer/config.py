"""Configuration defaults for the SHG analyzer."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BackgroundConfig:
    """Default parameters for background correction strategies."""

    linear_fit_range: Tuple[float, float] = (1.9, 2.1)
    local_window_width: float = 0.05
    polynomial_degree: int = 2


@dataclass(frozen=True)
class IntensityExtractionConfig:
    """Configuration for intensity extraction methods."""

    integration_window: float = 0.02
    gaussian_initial_sigma: float = 0.01
    lorentz_initial_gamma: float = 0.01
    bootstrap_samples: int = 200


@dataclass(frozen=True)
class PolarPlotConfig:
    """Styling and export defaults for polar plots."""

    dpi: int = 200
    figure_size: Tuple[float, float] = (6.0, 6.0)
    export_formats: Tuple[str, ...] = ("png", "pdf")
    color_cycle: Tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    normalized_alpha: float = 0.7


@dataclass(frozen=True)
class FileDiscoveryConfig:
    """Settings for locating and parsing measurement files."""

    extensions: Tuple[str, ...] = (".csv", ".txt")
    recursive: bool = True
    decimal_separators: Tuple[str, ...] = (".", ",")


@dataclass(frozen=True)
class TwistClassificationConfig:
    """Thresholds used to label stacking type."""

    r_type_tolerance: float = 15.0
    h_type_target: float = 60.0
    h_type_tolerance: float = 15.0


@dataclass(frozen=True)
class AppConfig:
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    intensity: IntensityExtractionConfig = field(default_factory=IntensityExtractionConfig)
    polar_plot: PolarPlotConfig = field(default_factory=PolarPlotConfig)
    discovery: FileDiscoveryConfig = field(default_factory=FileDiscoveryConfig)
    twist: TwistClassificationConfig = field(default_factory=TwistClassificationConfig)
    cache_dir: Path = Path(".shg_sessions")


DEFAULT_CONFIG = AppConfig()
