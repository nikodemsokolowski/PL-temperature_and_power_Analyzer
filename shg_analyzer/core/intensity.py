"""Intensity extraction strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from shg_analyzer.core.background import BackgroundResult, apply_background
from shg_analyzer.core.models import IntensityResult


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


def _pick_peak_region(energy: pd.Series, peak_center: float | None, window: float) -> pd.Series:
    if peak_center is None:
        return energy
    return (energy >= peak_center - window / 2.0) & (energy <= peak_center + window / 2.0)


def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) + offset


def _lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float, offset: float) -> np.ndarray:
    return amplitude * (gamma ** 2 / ((x - center) ** 2 + gamma ** 2)) + offset


def _curve_fit_wrapper(func, xdata, ydata, p0) -> Tuple[np.ndarray, np.ndarray | None]:
    try:
        params, covariance = curve_fit(func, xdata, ydata, p0=p0, maxfev=10000)
        return params, covariance
    except Exception as exc:
        raise RuntimeError(f"Curve fitting failed: {exc}") from exc


def extract_peak_max(spectrum: pd.DataFrame) -> IntensityResult:
    intensity_col = _intensity_column(spectrum)
    value = float(spectrum[intensity_col].max())
    return IntensityResult(method="peak_max", value=value)


def extract_integrated_intensity(spectrum: pd.DataFrame, window: float, peak_center: float | None = None) -> IntensityResult:
    intensity_col = _intensity_column(spectrum)
    energy = _energy_axis(spectrum)
    mask = _pick_peak_region(energy, peak_center, window)
    selected_energy = energy[mask].to_numpy()
    selected_counts = spectrum.loc[mask, intensity_col].to_numpy()
    if selected_energy.size < 2:
        raise ValueError("Not enough points for integration window.")
    value = float(np.trapz(selected_counts, selected_energy))
    return IntensityResult(method="integrated", value=value, diagnostics={"window": window, "center": peak_center})


def extract_gaussian_fit(spectrum: pd.DataFrame, initial_sigma: float = 0.01) -> IntensityResult:
    intensity_col = _intensity_column(spectrum)
    energy = _energy_axis(spectrum)
    counts = spectrum[intensity_col].to_numpy()
    x = energy.to_numpy()
    peak_index = int(np.argmax(counts))
    peak_center = float(x[peak_index])
    peak_value = float(counts[peak_index])
    p0 = [peak_value, peak_center, initial_sigma, float(np.median(counts[:10]))]
    params, covariance = _curve_fit_wrapper(_gaussian, x, counts, p0)
    amplitude, center, sigma, offset = params
    area = float(amplitude * sigma * np.sqrt(2.0 * np.pi))
    uncertainty = float(np.sqrt(covariance[0, 0])) if covariance is not None else None
    diag = {
        "center": float(center),
        "sigma": float(sigma),
        "offset": float(offset),
        "area": area,
    }
    return IntensityResult(method="gaussian", value=float(amplitude), uncertainty=uncertainty, diagnostics=diag)


def extract_lorentzian_fit(spectrum: pd.DataFrame, initial_gamma: float = 0.01) -> IntensityResult:
    intensity_col = _intensity_column(spectrum)
    energy = _energy_axis(spectrum)
    counts = spectrum[intensity_col].to_numpy()
    x = energy.to_numpy()
    peak_index = int(np.argmax(counts))
    peak_center = float(x[peak_index])
    peak_value = float(counts[peak_index])
    p0 = [peak_value, peak_center, initial_gamma, float(np.median(counts[:10]))]
    params, covariance = _curve_fit_wrapper(_lorentzian, x, counts, p0)
    amplitude, center, gamma, offset = params
    area = float(np.pi * amplitude * gamma)
    uncertainty = float(np.sqrt(covariance[0, 0])) if covariance is not None else None
    diag = {
        "center": float(center),
        "gamma": float(gamma),
        "offset": float(offset),
        "area": area,
    }
    return IntensityResult(method="lorentzian", value=float(amplitude), uncertainty=uncertainty, diagnostics=diag)


def run_intensity_strategy(
    spectrum: pd.DataFrame,
    method: str,
    background: str | None = None,
    background_params: Dict[str, float] | None = None,
    **method_kwargs,
) -> IntensityResult:
    working = spectrum
    applied_background: BackgroundResult | None = None
    if background:
        applied_background = apply_background(spectrum, background, **(background_params or {}))
        working = applied_background.corrected

    method = method.lower()
    if method == "peak_max":
        result = extract_peak_max(working)
    elif method in {"integrated", "integration"}:
        window = float(method_kwargs.get("window", 0.02))
        peak_center = method_kwargs.get("center")
        result = extract_integrated_intensity(working, window=window, peak_center=peak_center)
    elif method == "gaussian":
        sigma = float(method_kwargs.get("initial_sigma", 0.01))
        result = extract_gaussian_fit(working, initial_sigma=sigma)
    elif method == "lorentzian":
        gamma = float(method_kwargs.get("initial_gamma", 0.01))
        result = extract_lorentzian_fit(working, initial_gamma=gamma)
    else:
        raise ValueError(f"Unknown intensity extraction method '{method}'.")

    if applied_background is not None:
        result.background_applied = background
        result.diagnostics.update({f"background_{k}": v for k, v in applied_background.diagnostics.items()})
    return result


__all__ = [
    "extract_gaussian_fit",
    "extract_integrated_intensity",
    "extract_lorentzian_fit",
    "extract_peak_max",
    "run_intensity_strategy",
]
