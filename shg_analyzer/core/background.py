"""Background correction strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BackgroundResult:
    corrected: pd.DataFrame
    diagnostics: Dict[str, float]


def _energy_axis(df: pd.DataFrame) -> pd.Series:
    for column in ("energy_ev", "energy", "wavelength_nm", "wavelength"):
        if column in df.columns:
            return df[column]
    raise KeyError("Spectrum must contain an energy or wavelength column.")


def _intensity_column(df: pd.DataFrame) -> str:
    for column in ("counts", "intensity", "signal", "value"):
        if column in df.columns:
            return column
    return df.columns[-1]


def subtract_global_minimum(spectrum: pd.DataFrame) -> BackgroundResult:
    intensity_col = _intensity_column(spectrum)
    corrected = spectrum.copy()
    min_val = float(corrected[intensity_col].min())
    corrected[intensity_col] -= min_val
    return BackgroundResult(corrected=corrected, diagnostics={"offset": min_val})


def subtract_local_minimum(spectrum: pd.DataFrame, center: float, window: float) -> BackgroundResult:
    intensity_col = _intensity_column(spectrum)
    energy = _energy_axis(spectrum)
    mask = (energy >= center - window / 2.0) & (energy <= center + window / 2.0)
    local_min = float(spectrum.loc[mask, intensity_col].min()) if mask.any() else float(spectrum[intensity_col].min())
    corrected = spectrum.copy()
    corrected[intensity_col] -= local_min
    return BackgroundResult(corrected=corrected, diagnostics={"offset": local_min, "center": center, "window": window})


def subtract_linear_baseline(spectrum: pd.DataFrame, start: float, stop: float) -> BackgroundResult:
    intensity_col = _intensity_column(spectrum)
    energy = _energy_axis(spectrum)
    mask = (energy >= start) & (energy <= stop)
    if not mask.any():
        raise ValueError("Linear baseline range does not overlap with spectrum data.")
    x = energy.loc[mask].to_numpy()
    y = spectrum.loc[mask, intensity_col].to_numpy()
    coeffs = np.polyfit(x, y, deg=1)
    baseline = np.polyval(coeffs, energy.to_numpy())
    corrected = spectrum.copy()
    corrected[intensity_col] -= baseline
    diagnostics = {"slope": float(coeffs[0]), "intercept": float(coeffs[1]), "start": start, "stop": stop}
    return BackgroundResult(corrected=corrected, diagnostics=diagnostics)


def subtract_polynomial_baseline(
    spectrum: pd.DataFrame,
    degree: int = 2,
    start: float | None = None,
    stop: float | None = None,
) -> BackgroundResult:
    intensity_col = _intensity_column(spectrum)
    energy = _energy_axis(spectrum)
    if start is not None and stop is not None:
        mask = (energy >= start) & (energy <= stop)
        if not mask.any():
            raise ValueError("Polynomial baseline range does not overlap with spectrum data.")
        x = energy.loc[mask].to_numpy()
        y = spectrum.loc[mask, intensity_col].to_numpy()
    else:
        x = energy.to_numpy()
        y = spectrum[intensity_col].to_numpy()
    coeffs = np.polyfit(x, y, deg=degree)
    baseline = np.polyval(coeffs, energy.to_numpy())
    corrected = spectrum.copy()
    corrected[intensity_col] -= baseline
    diagnostics = {"degree": degree}
    if start is not None and stop is not None:
        diagnostics.update({"start": start, "stop": stop})
    return BackgroundResult(corrected=corrected, diagnostics=diagnostics)


def apply_background(
    spectrum: pd.DataFrame,
    method: str,
    **params: float,
) -> BackgroundResult:
    method = method.lower()
    if method in {"global_min", "baseline_min", "min"}:
        return subtract_global_minimum(spectrum)
    if method == "local_min":
        center = params.get("center")
        window = params.get("window", 0.02)
        if center is None:
            raise ValueError("local_min requires 'center' parameter (energy/wavelength).")
        return subtract_local_minimum(spectrum, center=center, window=window)
    if method == "linear":
        start = params.get("start")
        stop = params.get("stop")
        if start is None or stop is None:
            raise ValueError("linear background requires 'start' and 'stop' parameters.")
        return subtract_linear_baseline(spectrum, start=start, stop=stop)
    if method == "polynomial":
        degree = int(params.get("degree", 2))
        start = params.get("start")
        stop = params.get("stop")
        return subtract_polynomial_baseline(spectrum, degree=degree, start=start, stop=stop)
    raise ValueError(f"Unknown background method '{method}'.")


__all__ = [
    "BackgroundResult",
    "apply_background",
    "subtract_global_minimum",
    "subtract_linear_baseline",
    "subtract_local_minimum",
    "subtract_polynomial_baseline",
]
