"""High-level controller coordinating SHG analysis workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from shg_analyzer.config import DEFAULT_CONFIG, AppConfig
from shg_analyzer.core.analysis import AnalysisOptions, analyze_series
from shg_analyzer.core.models import SampleSeries
from shg_analyzer.core.twist import compute_twist
from shg_analyzer.data.loader import ingest_directory
from shg_analyzer.data.registry import SampleRegistry
from shg_analyzer.plotting.polar import plot_polar_series


class SHGAnalyzerController:
    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.registry = SampleRegistry()
        self.series: List[SampleSeries] = []

    def load_directory(self, directory: Path) -> List[SampleSeries]:
        self.series = ingest_directory(Path(directory), config=self.config)
        self._apply_overrides()
        return self.series

    def set_override(self, label: str, **metadata) -> None:
        self.registry.set_override(label, **metadata)
        self._apply_overrides()

    def clear_override(self, label: str) -> None:
        self.registry.clear_override(label)
        self._apply_overrides()

    def _apply_overrides(self) -> None:
        for item in self.series:
            self.registry.apply(item)

    def run_analysis(self, options: AnalysisOptions | None = None) -> List[SampleSeries]:
        options = options or AnalysisOptions()
        analyzed = []
        for item in self.series:
            analyzed.append(analyze_series(item, options, config=self.config))
        self.series = analyzed
        return self.series

    def compute_twist(self, labels: Iterable[str]) -> Optional[str]:
        selected = [series for series in self.series if series.label in labels]
        if len(selected) != 2:
            raise ValueError("Twist computation requires exactly two series labels.")
        orientations = [series.orientation for series in selected]
        if any(orientation is None for orientation in orientations):
            raise ValueError("Both series need successful orientation fits before computing twist.")
        twist = compute_twist(orientations, [series.label for series in selected], config=self.config)
        return f"Twist: {twist.twist_angle_deg:.2f}° ({twist.classification}-type)"

    def plot(self, method: str = "peak_max", normalization: str = "raw", output: Optional[Path] = None):
        return plot_polar_series(self.series, method=method, normalization=normalization, config=self.config, output_path=output)


__all__ = ["SHGAnalyzerController"]
