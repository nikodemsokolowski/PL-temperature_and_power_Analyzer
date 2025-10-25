"""Intensity extraction tests."""
import numpy as np
import pandas as pd

from shg_analyzer.core.intensity import (
    extract_gaussian_fit,
    extract_peak_max,
    run_intensity_strategy,
)


def _synthetic_gaussian():
    energy = np.linspace(1.95, 2.05, 101)
    amplitude = 100.0
    center = 2.0
    sigma = 0.01
    counts = amplitude * np.exp(-((energy - center) ** 2) / (2.0 * sigma ** 2))
    return pd.DataFrame({"energy_ev": energy, "counts": counts})


def test_extract_peak_max_matches_expected():
    spectrum = _synthetic_gaussian()
    result = extract_peak_max(spectrum)
    assert abs(result.value - 100.0) < 1e-6


def test_gaussian_fit_recovers_amplitude():
    spectrum = _synthetic_gaussian()
    result = extract_gaussian_fit(spectrum, initial_sigma=0.02)
    assert abs(result.value - 100.0) < 1.0


def test_run_intensity_strategy_with_background():
    spectrum = _synthetic_gaussian()
    # Add background offset to verify subtraction
    spectrum["counts"] += 5.0
    result = run_intensity_strategy(spectrum, method="peak_max", background="global_min")
    assert result.background_applied == "global_min"
    assert abs(result.value - 100.0) < 1.0
