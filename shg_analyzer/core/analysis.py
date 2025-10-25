"""Series-level analysis orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from shg_analyzer.config import DEFAULT_CONFIG, AppConfig
from shg_analyzer.core.background import apply_background
from shg_analyzer.core.intensity import run_intensity_strategy
from shg_analyzer.core.models import AngleMeasurement, SampleSeries
from shg_analyzer.core.twist import fit_orientation, normalize_intensities


@dataclass
class AnalysisOptions:
    intensity_methods: List[str] = field(default_factory=lambda: ["peak_max", "integrated"])
    method_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    background_method: Optional[str] = "global_min"
    background_params: Dict[str, float] = field(default_factory=dict)
    normalization_mode: str = "raw"
    orientation_method: str = "peak_max"
    fit_orientation_enabled: bool = True


def analyze_series(series: SampleSeries, options: AnalysisOptions, config: AppConfig = DEFAULT_CONFIG) -> SampleSeries:
    intensity_tables: Dict[str, List[Dict[str, float]]] = {}

    for measurement in series.measurements:
        working_spectrum = measurement.raw_spectrum
        background_result = None
        if options.background_method:
            background_result = apply_background(
                measurement.raw_spectrum,
                options.background_method,
                **options.background_params,
            )
            measurement.corrected_spectrum = background_result.corrected
            working_spectrum = measurement.corrected_spectrum
        else:
            measurement.corrected_spectrum = measurement.raw_spectrum.copy()

        for method in options.intensity_methods:
            params = options.method_params.get(method, {})
            result = run_intensity_strategy(working_spectrum, method, background=None, **params)
            if background_result is not None:
                result.background_applied = options.background_method
                result.diagnostics.update({f"background_{k}": v for k, v in background_result.diagnostics.items()})
            measurement.add_intensity_result(result)
            row = intensity_tables.setdefault(method, [])
            angle = measurement.metadata.polarization_angle_deg
            row.append(
                {
                    "angle_deg": angle,
                    "intensity": result.value,
                    "uncertainty": result.uncertainty,
                }
            )

    series.overrides.setdefault("intensity_tables", {}).update(intensity_tables)

    if options.fit_orientation_enabled:
        orientation_data = [entry for entry in intensity_tables.get(options.orientation_method, []) if entry["angle_deg"] is not None]
        if len(orientation_data) >= 3:
            angles = [entry["angle_deg"] for entry in orientation_data]
            intensities = [entry["intensity"] for entry in orientation_data]
            normalized = normalize_intensities(intensities, options.normalization_mode)
            series.overrides.setdefault("normalized_intensities", {})[options.orientation_method] = [
                {"angle_deg": angle, "value": value}
                for angle, value in zip(angles, normalized)
            ]
            try:
                series.orientation = fit_orientation(angles, normalized)
            except Exception as exc:
                series.overrides.setdefault("fit_errors", []).append(str(exc))

    return series


def aggregate_intensity_matrix(series: List[SampleSeries], method: str, normalization: str) -> Dict[str, List[Dict[str, float]]]:
    matrix: Dict[str, List[Dict[str, float]]] = {}
    for item in series:
        rows = []
        for measurement in item.sorted_measurements():
            result = measurement.intensity_results.get(method)
            if result and measurement.metadata.polarization_angle_deg is not None:
                rows.append(
                    {
                        "angle_deg": measurement.metadata.polarization_angle_deg,
                        "intensity": result.value,
                        "series": item.label,
                    }
                )
        values = [row["intensity"] for row in rows]
        scaled = normalize_intensities(values, normalization)
        for row, value in zip(rows, scaled):
            row["normalized_intensity"] = value
        matrix[item.label] = rows
    return matrix


__all__ = ["AnalysisOptions", "analyze_series", "aggregate_intensity_matrix"]
