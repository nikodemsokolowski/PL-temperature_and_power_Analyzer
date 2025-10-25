"""Reporting helpers for SHG analysis."""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from shg_analyzer.core.models import SampleSeries


def build_summary_table(series: Iterable[SampleSeries]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for item in series:
        orientation = item.orientation
        records.append(
            {
                "label": item.label,
                "sample_type": item.sample_type,
                "material": item.material,
                "series_id": item.series_id,
                "theta0_deg": orientation.theta0_deg if orientation else None,
                "amplitude": orientation.amplitude if orientation else None,
                "uncertainty": orientation.uncertainty if orientation else None,
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = ["build_summary_table"]
