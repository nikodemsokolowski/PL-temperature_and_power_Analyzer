"""Background correction tests."""
import pandas as pd

from shg_analyzer.core.background import (
    apply_background,
    subtract_global_minimum,
)


def _sample_spectrum():
    return pd.DataFrame({
        "energy_ev": [1.95, 1.96, 1.97, 1.98, 1.99],
        "counts": [10.0, 12.0, 50.0, 12.0, 11.0],
    })


def test_subtract_global_minimum():
    spectrum = _sample_spectrum()
    result = subtract_global_minimum(spectrum)
    assert result.corrected["counts"].min() == 0.0


def test_apply_background_linear_range():
    spectrum = _sample_spectrum()
    result = apply_background(spectrum, method="linear", start=1.95, stop=1.96)
    assert "slope" in result.diagnostics
