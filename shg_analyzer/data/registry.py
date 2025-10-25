"""Session registry for sample overrides and metadata adjustments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from shg_analyzer.core.models import SampleSeries


@dataclass
class OverrideRecord:
    label: str
    updates: Dict[str, Any] = field(default_factory=dict)


class SampleRegistry:
    def __init__(self) -> None:
        self._overrides: Dict[str, OverrideRecord] = {}

    def set_override(self, label: str, **updates: Any) -> None:
        record = self._overrides.setdefault(label, OverrideRecord(label=label))
        record.updates.update({k: v for k, v in updates.items() if v is not None})

    def clear_override(self, label: str) -> None:
        self._overrides.pop(label, None)

    def apply(self, series: SampleSeries) -> SampleSeries:
        record = self._overrides.get(series.label)
        if not record:
            return series
        if "sample_type" in record.updates:
            series.sample_type = record.updates["sample_type"]
        if "material" in record.updates:
            series.material = record.updates["material"]
        if "series_id" in record.updates:
            series.series_id = record.updates["series_id"]
        series.overrides.update(record.updates)
        return series

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {label: record.updates for label, record in self._overrides.items()}


__all__ = ["SampleRegistry", "OverrideRecord"]
