"""Measurement discovery and loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from shg_analyzer.config import DEFAULT_CONFIG, AppConfig
from shg_analyzer.core.models import AngleMeasurement, SampleSeries
from shg_analyzer.data.naming import ParsedMetadata, parse_filename


@dataclass
class LoadedMeasurement:
    series_label: str
    measurement: AngleMeasurement


def discover_measurement_files(root: Path, extensions: Sequence[str], recursive: bool) -> List[Path]:
    if recursive:
        candidates = [p for ext in extensions for p in root.rglob(f"*{ext}")]
    else:
        candidates = [p for ext in extensions for p in root.glob(f"*{ext}")]
    return sorted({path.resolve() for path in candidates})


def _try_read_csv(path: Path, decimal_separators: Sequence[str]) -> pd.DataFrame:
    errors = {}
    for decimal in decimal_separators:
        try:
            df = pd.read_csv(path, comment="#", decimal=decimal)
            if df.empty:
                continue
            df.columns = [col.strip().lower() for col in df.columns]
            return df
        except Exception as exc:  # pragma: no cover - diagnostic branch
            errors[decimal] = exc
    raise ValueError(f"Unable to read {path} with provided decimal separators: {errors}")


def load_measurement(path: Path, config: AppConfig = DEFAULT_CONFIG) -> LoadedMeasurement:
    metadata = parse_filename(path.name)
    dataframe = _try_read_csv(path, config.discovery.decimal_separators)
    measurement = AngleMeasurement(path=path, metadata=metadata, raw_spectrum=dataframe)

    label = metadata.sample_label or path.stem
    series = SampleSeries(
        label=label,
        sample_type=metadata.sample_type,
        material=metadata.material,
        series_id=metadata.series_hint,
        measurements=[measurement],
    )
    return LoadedMeasurement(series_label=label, measurement=measurement)


def ingest_directory(root: Path, config: AppConfig = DEFAULT_CONFIG) -> List[SampleSeries]:
    files = discover_measurement_files(root, config.discovery.extensions, config.discovery.recursive)
    series_map = {}
    for path in files:
        metadata = parse_filename(path.name)
        dataframe = _try_read_csv(path, config.discovery.decimal_separators)
        measurement = AngleMeasurement(path=path, metadata=metadata, raw_spectrum=dataframe)
        label = metadata.sample_label or path.stem
        series = series_map.setdefault(
            label,
            SampleSeries(
                label=label,
                sample_type=metadata.sample_type,
                material=metadata.material,
                series_id=metadata.series_hint,
            ),
        )
        if series.sample_type is None and metadata.sample_type:
            series.sample_type = metadata.sample_type
        if series.material is None and metadata.material:
            series.material = metadata.material
        if series.series_id is None and metadata.series_hint:
            series.series_id = metadata.series_hint
        series.add_measurement(measurement)
    return list(series_map.values())


__all__ = ["ingest_directory", "discover_measurement_files", "load_measurement", "LoadedMeasurement"]
