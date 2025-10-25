#!/usr/bin/env python3
"""
Generate synthetic circularly polarised PL spectra with a known g-factor.

Each B-field value produces two spectra (σ+ and σ-) with Gaussian peaks that
split according to ΔE = g * μ_B * B. Files are written with energy/counts
columns that match the PL Analyzer filename conventions.
"""
from __future__ import annotations

import argparse
import pathlib
import numpy as np
import pandas as pd

MU_B_MEV_PER_T = 0.05788  # Bohr magneton in meV/T


def simulate_spectrum(
    energy_axis: np.ndarray,
    center_ev: float,
    amplitude: float,
    width_ev: float,
    baseline: float,
    noise_scale: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Return a synthetic Gaussian spectrum with optional noise and baseline."""
    profile = amplitude * np.exp(-0.5 * ((energy_axis - center_ev) / width_ev) ** 2)
    noise = rng.normal(loc=0.0, scale=amplitude * noise_scale, size=energy_axis.size)
    return profile + baseline + noise


def format_bfield(bfield: float) -> str:
    """Represent B-field for filenames (e.g. 0T, 1T, 2p5T)."""
    if abs(bfield - int(bfield)) < 1e-6:
        return f"{int(bfield)}T"
    return f"{str(bfield).replace('.', 'p')}T"


def save_spectrum(path: pathlib.Path, energy_ev: np.ndarray, counts: np.ndarray) -> None:
    df = pd.DataFrame({"energy_ev": energy_ev, "counts": counts})
    df.to_csv(path, index=False)


def generate_dataset(
    output_dir: pathlib.Path,
    temperatures: list[float],
    powers: list[float],
    bfields: list[float],
    g_factor: float,
    base_energy_ev: float,
    amplitude: float,
    width_ev: float,
    baseline: float,
    noise_scale: float,
    energy_min: float,
    energy_max: float,
    energy_points: int,
    seed: int
) -> None:
    rng = np.random.default_rng(seed)
    energies = np.linspace(energy_min, energy_max, energy_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    for temp in temperatures:
        for power in powers:
            for bfield in bfields:
                delta_mev = g_factor * MU_B_MEV_PER_T * bfield
                delta_ev = delta_mev / 1000.0
                center_plus = base_energy_ev + delta_ev / 2.0
                center_minus = base_energy_ev - delta_ev / 2.0

                for pol, center in (("sigma+", center_plus), ("sigma-", center_minus)):
                    counts = simulate_spectrum(
                        energies,
                        center,
                        amplitude=amplitude,
                        width_ev=width_ev,
                        baseline=baseline,
                        noise_scale=noise_scale,
                        rng=rng,
                    )
                    filename = (
                        f"sample_{temp:g}K_{power:g}uW_{format_bfield(bfield)}_{pol}.csv"
                    )
                    save_spectrum(output_dir / filename, energies, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic PL datasets with known g-factor.")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("test_data/bfield_cp"),
                        help="Output directory for generated spectra (default: %(default)s)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[5.0],
                        help="Temperatures (K) to encode in filenames (default: %(default)s)")
    parser.add_argument("--powers", type=float, nargs="+", default=[100.0],
                        help="Powers (uW) to encode in filenames (default: %(default)s)")
    parser.add_argument("--bfields", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0],
                        help="Magnetic field values (Tesla) (default: %(default)s)")
    parser.add_argument("--g-factor", type=float, default=2.0,
                        help="Target g-factor for Zeeman splitting (default: %(default)s)")
    parser.add_argument("--base-energy", type=float, default=2.000,
                        help="Zero-field transition energy in eV (default: %(default)s)")
    parser.add_argument("--amplitude", type=float, default=1500.0,
                        help="Peak amplitude (default: %(default)s)")
    parser.add_argument("--width", type=float, default=0.003,
                        help="Gaussian width (standard deviation) in eV (default: %(default)s)")
    parser.add_argument("--baseline", type=float, default=50.0,
                        help="Constant background level added to each spectrum (default: %(default)s)")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Relative noise level (fraction of amplitude) (default: %(default)s)")
    parser.add_argument("--energy-min", type=float, default=1.95,
                        help="Minimum energy (eV) for spectra (default: %(default)s)")
    parser.add_argument("--energy-max", type=float, default=2.05,
                        help="Maximum energy (eV) for spectra (default: %(default)s)")
    parser.add_argument("--points", type=int, default=1200,
                        help="Number of energy sampling points (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_dir=args.output,
        temperatures=args.temperatures,
        powers=args.powers,
        bfields=args.bfields,
        g_factor=args.g_factor,
        base_energy_ev=args.base_energy,
        amplitude=args.amplitude,
        width_ev=args.width,
        baseline=args.baseline,
        noise_scale=args.noise,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        energy_points=args.points,
        seed=args.seed,
    )
    print(f"Generated synthetic dataset in {args.output.resolve()}")
    print(f"Configured g-factor: {args.g_factor:.3f}")


if __name__ == "__main__":
    main()
