"""Orientation fitting and twist angle utilities."""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import curve_fit

from shg_analyzer.config import DEFAULT_CONFIG, AppConfig
from shg_analyzer.core.models import OrientationResult, SampleSeries, TwistResult


def _polar_model(theta_deg: np.ndarray, amplitude: float, theta0_deg: float, offset: float) -> np.ndarray:
    radians = np.deg2rad(theta_deg - theta0_deg)
    return amplitude * (np.cos(3.0 * radians) ** 2) + offset


def fit_orientation(
    angles_deg: Iterable[float],
    intensities: Iterable[float],
    method: str = "cos2",
) -> OrientationResult:
    angles = np.asarray(list(angles_deg), dtype=float)
    values = np.asarray(list(intensities), dtype=float)
    if angles.size < 3:
        raise ValueError("Need at least three angle measurements to fit orientation.")
    if method != "cos2":
        raise ValueError(f"Unsupported orientation model '{method}'.")

    max_idx = int(np.argmax(values))
    amp_guess = float(values[max_idx])
    theta_guess = float(angles[max_idx])
    offset_guess = float(np.min(values))

    params, covariance = curve_fit(
        _polar_model,
        angles,
        values,
        p0=[amp_guess, theta_guess, offset_guess],
        maxfev=20000,
    )
    amp, theta0, offset = params
    uncertainty = float(np.sqrt(covariance[1, 1])) if covariance is not None else None
    return OrientationResult(
        theta0_deg=float(theta0 % 60.0),
        amplitude=float(amp),
        uncertainty=uncertainty,
        method="cos2",
        diagnostics={"offset": float(offset)},
    )


def compute_twist(
    orientations: List[OrientationResult],
    labels: List[str],
    config: AppConfig = DEFAULT_CONFIG,
) -> TwistResult:
    if len(orientations) != 2 or len(labels) != 2:
        raise ValueError("Twist calculation requires two orientation results (two monolayers).")
    theta_a = orientations[0].theta0_deg
    theta_b = orientations[1].theta0_deg
    twist = min((theta_b - theta_a) % 60.0, (theta_a - theta_b) % 60.0)

    classification = "undetermined"
    if twist <= config.twist.r_type_tolerance:
        classification = "R"
    elif abs(twist - config.twist.h_type_target) <= config.twist.h_type_tolerance:
        classification = "H"

    unc_a = orientations[0].uncertainty or 0.0
    unc_b = orientations[1].uncertainty or 0.0
    combined_unc = math.sqrt(unc_a ** 2 + unc_b ** 2) if (unc_a or unc_b) else None

    return TwistResult(
        twist_angle_deg=float(twist),
        classification=classification,
        uncertainty=combined_unc,
        reference_labels=labels,
        diagnostics={"theta_a": theta_a, "theta_b": theta_b},
    )


def normalize_intensities(intensities: List[float], mode: str) -> List[float]:
    if not intensities:
        return intensities
    mode = mode.lower()
    values = np.asarray(intensities, dtype=float)
    if mode == "raw":
        return values.tolist()
    if mode == "normalized":
        peak = np.max(values)
        return (values / peak if peak else values).tolist()
    if mode == "equalized":
        mean = np.mean(values)
        return (values / mean if mean else values).tolist()
    raise ValueError(f"Unknown normalization mode '{mode}'.")


__all__ = ["fit_orientation", "compute_twist", "normalize_intensities"]
