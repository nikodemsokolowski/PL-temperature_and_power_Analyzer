"""Core data models for the SHG analyzer."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from shg_analyzer.data.naming import ParsedMetadata


@dataclass
class IntensityResult:
    method: str
    value: float
    uncertainty: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    background_applied: Optional[str] = None


@dataclass
class AngleMeasurement:
    path: Path
    metadata: ParsedMetadata
    raw_spectrum: pd.DataFrame
    corrected_spectrum: Optional[pd.DataFrame] = None
    intensity_results: Dict[str, IntensityResult] = field(default_factory=dict)

    def add_intensity_result(self, result: IntensityResult) -> None:
        self.intensity_results[result.method] = result


@dataclass
class OrientationResult:
    theta0_deg: float
    amplitude: float
    uncertainty: Optional[float]
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwistResult:
    twist_angle_deg: float
    classification: str
    uncertainty: Optional[float]
    reference_labels: List[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleSeries:
    label: str
    sample_type: Optional[str]
    material: Optional[str]
    series_id: Optional[str]
    measurements: List[AngleMeasurement] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)
    orientation: Optional[OrientationResult] = None

    def sorted_measurements(self) -> List[AngleMeasurement]:
        return sorted(
            self.measurements,
            key=lambda m: m.metadata.polarization_angle_deg or -1.0,
        )

    def add_measurement(self, measurement: AngleMeasurement) -> None:
        self.measurements.append(measurement)


__all__ = [
    "AngleMeasurement",
    "IntensityResult",
    "OrientationResult",
    "TwistResult",
    "SampleSeries",
]
