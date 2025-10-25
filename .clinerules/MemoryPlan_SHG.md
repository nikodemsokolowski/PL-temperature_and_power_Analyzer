# Memory Plan — SHG Analyzer

## Must-Remember Concepts
- Primary package lives in `shg_analyzer/` with submodules for data loading, analysis, plotting, UI, and storage.
- Filename parsing via `parse_filename` extracts ML/HS, material tokens, series IDs (e.g., `ML1_2`), and half-wave-plate angles (`2p0` → 2° HWP → 4° polarization).
- Background strategies (`global_min`, `local_min`, `linear`, `polynomial`) return diagnostics that downstream modules attach to intensity results.
- Intensity methods (peak, integrated, Gaussian, Lorentzian) are orchestrated through `AnalysisOptions`; orientation fitting uses normalized intensities.
- Twist classification compares two fitted orientations: `R` when |Δθ| < 15°, `H` when |Δθ − 60°| < 15°.

## Critical Paths
1. `SHGAnalyzerController.load_directory` → `ingest_directory` → `SampleSeries` objects
2. `controller.run_analysis` → `analyze_series` → background + intensity strategies → orientation fit
3. `controller.plot` → `plot_polar_series` for visualization and export
4. `controller.compute_twist` → `compute_twist` for stacking classification

## Open Work to Track
- Implement GUI in `shg_analyzer/ui/qt/main_window.py` once backend stabilizes.
- Add persistence glue using `SessionStore` so overrides and analyses survive reloads.
- Expand tests (orientation, twist, controller) once `pytest` is installed.

## Usage Tips
- Use `scripts/run_shg_cli.py <directory> --plot` for quick end-to-end run on datasets.
- For ambiguous filenames, call `controller.set_override(label, sample_type="monolayer", material="WS2")` before rerunning analysis.
- When fits fail, inspect `series.overrides['fit_errors']` for diagnostics.

## Data Assumptions to Revisit
- CSV columns expected to include `energy_ev` or `wavelength_nm` and intensity column (`counts`).
- HWP angles encoded as `<degrees>p<decimals>`; adjust parser if alternative schemes appear.
- Equalization currently scales by mean; revisit once experimental requirements clarified.
