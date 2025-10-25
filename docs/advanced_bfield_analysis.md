# Advanced B-Field Analysis

This document describes the Phase 2 / Module 2 features added to the PL Analyzer application. The enhancements focus on magnetic-field driven analysis workflows, interactive fitting, and tooling for polarization sums.

## Enhanced Integrate vs B-Field

The new **Enhanced Integrate vs B-Field** window replaces the previous one-series plot with a multi-trace experience:

- Plots `sigma+`, `sigma-`, and their sum on the same axes. Missing channels are skipped gracefully.
- Optional Degree of Circular Polarization (DCP = (I\_+ - I\_-)/(I\_+ + I\_-)) subplot that shares the magnetic-field axis.
- Display options:
  - Toggle individual traces (sigma+, sigma-, sum, DCP).
  - Normalise intensities to the global maximum when comparing amplitudes.
  - Style selector (lines, markers, or both).
- CSV export includes B-field, all intensity channels, DCP, and (when normalised) additional columns for the scaled intensities. A short header records the integration range, normalisation flag, and current style.
- Integration range validation continues to use the B-Field Analysis tab entries.

When the action runs, the window remains cached; reinvoking the button updates the content while preserving user-selected toggles.

## Advanced G-Factor Analysis Window

The **GFactorAnalysisWindow** provides a publication-ready workflow for Zeeman splitting analysis.

- **Plots:**
  - Top panel: energy vs B-field scatter for sigma+ (red) and sigma- (blue).
  - Middle panel: DeltaE (E\_+ - E\_-) vs B-field with the fitted line and optional manual overlay.
  - Bottom panel: residuals for the fitted data points with a zero-reference line.
- **Peak detection:** Choose maximum, centroid, or Gaussian fit within an energy window. Optional moving-average smoothing is available per extraction.
- **Interactive fitting:**
  - Fit range: limit the B-field domain that feeds the linear regression.
  - Manual overlay: supply a custom g-factor and optional intercept to compare visually against the fitted result.
- **Reported statistics:** g, g-uncertainty, intercept, intercept-uncertainty, and R². Failed or insufficient fits surface user-friendly error messages.
- **Exports:**
  - Data CSV (`B`, `E_sigma+`, `E_sigma-`, `DeltaE`, fitted DeltaE, residuals, method, smoothing).
  - Figure export (PNG/PDF/SVG).

The window caches its state; re-opening through the action refreshes the analysis for the currently active dataset while maintaining manual overlays if desired.

## Virtual Sum Polarisation Datasets

A new **Create Sum Dataset** button appears in the left control panel. It creates (or refreshes) a virtual dataset containing `sigma+ + sigma-` spectra pairs:

- Pairs require matching temperature, power, and B-field metadata. Interpolation aligns the energy grids before summing.
- The dataset is named `{source_name}_sum`. Re-running the action refreshes an existing sum dataset for the same source; otherwise, a new dataset is created.
- Each virtual spectrum stores:
  - Copy of temperature, power, and B-field values.
  - `polarization = "sum"` flag and a `virtual_sum` metadata boolean.
  - References to the source file IDs (sigma+ and sigma-).
- Sum datasets behave like any other dataset in tabs, integrate with all existing analysis tools, and can be exported. Metadata columns expose the virtual nature, although no dedicated badge is shown in the UI yet.
- If no complete pairs are found, the action warns the user. Incomplete pairs are counted and reported. A warning about deleted source spectra is not yet implemented.

## Testing Notes

- Synthetic smoke tests confirm sum-dataset generation, including interpolation and metadata tracking.
- `python -m compileall pl_analyzer` passes with the new changes.
- Manual verification with the `test_data/bfield_cp/` set is recommended to exercise peak detection and CSV exports.

> Save/load analysis presets and automatic source-deletion warnings are deferred; the code path is ready for future expansion if needed.

