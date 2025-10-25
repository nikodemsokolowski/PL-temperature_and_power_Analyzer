"""Tests for filename parsing heuristics."""
from shg_analyzer.data.naming import parse_filename


def test_parse_hs_filename_with_angle():
    parsed = parse_filename("Martas_HS1_7uW_1s_from_HS_2_0p0.csv")
    assert parsed.sample_type == "heterostructure"
    assert parsed.series_hint == "HS2"
    assert parsed.hwp_angle_deg == 0.0
    assert parsed.polarization_angle_deg == 0.0


def test_parse_ml_series_with_decimal_angle():
    parsed = parse_filename("sample_ml1_2_12p5.csv")
    assert parsed.sample_type == "monolayer"
    assert parsed.series_hint == "ML1_2"
    assert abs(parsed.hwp_angle_deg - 12.5) < 1e-6
    assert abs(parsed.polarization_angle_deg - 25.0) < 1e-6
