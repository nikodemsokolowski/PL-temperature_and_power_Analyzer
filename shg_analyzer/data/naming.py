"""Filename parsing heuristics for SHG measurements."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

_SAMPLE_MONO_TOKENS = {"ml", "mono", "monolayer"}
_SAMPLE_HS_TOKENS = {"hs", "hetero", "stack"}
_MATERIAL_TOKENS = {
    "wse2": "WSe2",
    "ws2": "WS2",
    "wsse": "WSSe",
    "mose2": "MoSe2",
    "mos2": "MoS2",
    "wmo": "WMo",
}
_SERIES_FINDER = re.compile(r"(ml|hs)[-_]?(\d+(?:_\d+)?)", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class ParsedMetadata:
    source_name: str
    sample_type: Optional[str] = None
    sample_label: Optional[str] = None
    material: Optional[str] = None
    series_hint: Optional[str] = None
    hwp_angle_deg: Optional[float] = None
    polarization_angle_deg: Optional[float] = None
    confidence: float = 0.0
    tokens: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def with_warning(self, message: str) -> "ParsedMetadata":
        self.warnings.append(message)
        return self


def _parse_angle(token: str) -> Optional[float]:
    token = token.lower()
    match = re.fullmatch(r"(\d+)(?:p(\d+))?", token)
    if not match:
        return None
    whole = int(match.group(1))
    frac = match.group(2)
    value = float(whole)
    if frac is not None:
        scale = 10 ** len(frac)
        value += int(frac) / scale
    return value


def _detect_sample_type(tokens: List[str]) -> Optional[str]:
    if any(tok in _SAMPLE_MONO_TOKENS for tok in tokens):
        return "monolayer"
    if any(tok in _SAMPLE_HS_TOKENS for tok in tokens):
        return "heterostructure"
    return None


def _detect_material(tokens: List[str]) -> Optional[str]:
    for tok in tokens:
        if tok in _MATERIAL_TOKENS:
            return _MATERIAL_TOKENS[tok]
    return None


def _detect_series(normalized: str) -> Optional[str]:
    match = _SERIES_FINDER.search(normalized)
    if match:
        prefix, suffix = match.groups()
        return f"{prefix.upper()}{suffix}"
    return None


def parse_filename(filename: str) -> ParsedMetadata:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    normalized = stem.lower()
    tokens = [tok for tok in _TOKEN_SPLIT.split(normalized) if tok]

    parsed = ParsedMetadata(source_name=filename, tokens=tokens)

    if not tokens:
        return parsed.with_warning("No alphanumeric tokens found in filename.")

    sample_type = _detect_sample_type(tokens)
    if sample_type:
        parsed.sample_type = sample_type
        parsed.confidence += 0.2
    else:
        parsed.with_warning("Sample type (ML/HS) not detected; user confirmation required.")

    material = _detect_material(tokens)
    if material:
        parsed.material = material
        parsed.confidence += 0.2

    series_hint = _detect_series(normalized)
    if series_hint:
        parsed.series_hint = series_hint
        parsed.confidence += 0.2

    angle_candidates = []
    for token in tokens:
        angle = _parse_angle(token)
        if angle is not None:
            angle_candidates.append(angle)

    if angle_candidates:
        hwp = angle_candidates[-1]
        parsed.hwp_angle_deg = hwp
        parsed.polarization_angle_deg = (hwp * 2.0) % 360.0
        parsed.confidence += 0.2
    else:
        parsed.with_warning("Half-wave-plate angle not detected in filename.")

    if parsed.material is None and parsed.sample_type == "heterostructure":
        parsed.with_warning("Heterostructure detected but material stack unspecified.")

    if parsed.sample_type is None and parsed.material is None:
        parsed.confidence = max(parsed.confidence - 0.1, 0.0)

    parsed.sample_label = _generate_label(parsed, tokens)
    return parsed


def _generate_label(parsed: ParsedMetadata, tokens: List[str]) -> str:
    if parsed.series_hint:
        return parsed.series_hint
    if parsed.material:
        return parsed.material
    for token in tokens:
        if token not in {"from", "scan", "series"}:
            return token.upper()
    return parsed.source_name


__all__ = ["ParsedMetadata", "parse_filename"]
