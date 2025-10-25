"""Session persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from shg_analyzer.config import DEFAULT_CONFIG, AppConfig
from shg_analyzer.core.models import SampleSeries


class SessionStore:
    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_name: str, series: Dict[str, SampleSeries], overrides: Dict[str, Dict[str, Any]]) -> Path:
        payload = {
            "series": {
                label: {
                    "sample_type": item.sample_type,
                    "material": item.material,
                    "series_id": item.series_id,
                }
                for label, item in series.items()
            },
            "overrides": overrides,
        }
        path = self.config.cache_dir / f"{session_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def load(self, session_name: str) -> Dict[str, Any]:
        path = self.config.cache_dir / f"{session_name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


__all__ = ["SessionStore"]
