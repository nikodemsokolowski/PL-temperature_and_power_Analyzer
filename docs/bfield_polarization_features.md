# B-Field and Circular Polarization Features

## Overview
Enhancements supporting B-field-dependent, polarization-resolved photoluminescence analysis are summarised below. All functional phases through dual-colour intensity maps are complete; validation notes will be added after the upcoming test pass.

## Phase Summaries

### Phase 1 – Enhanced Data Parsing
- `core/file_parser.py` extracts B-field (`bfield_t`) and polarization tokens from filenames (supports forms such as `5T`, `B=1T`, `1p5T`).
- `core/data_handler.py` and associated metadata infrastructure persist the new fields.
- `utils/config.py` provides persistence helpers for user-defined polarization string mappings (`save_polarization_config`, `load_polarization_config`).

### Phase 2 – Polarization Configuration UI
- Added the Polarization Settings dialog with live detection preview and validation.
- Hooked into the left panel and file actions to re-parse datasets when mappings change.
- Configuration is applied automatically on application start.

### Phase 3 - Polarization-Aware Plotting
- Right-panel selector exposes All, sigma+, sigma-, Both, and Sum modes.
- `_create_sum_spectra()` pairs sigma+/sigma- spectra, interpolating where necessary and reporting incomplete pairs.
- Plot legends/colours indicate polarization (sigma+ = red, sigma- = blue); optional background subtraction is available for all plots.
- `plot_canvas.py` accepts explicit colour sequences for caller-defined palettes.

### Phase 3 Update (2025) - Sweep Controls & Temperature Analysis
- Magnetic Sweep Settings dialog now supports multiple acquisition time ranges with add/remove controls and a sweep-direction selector (Low->High or High->Low). Settings persist to `config.json` and can be applied to existing datasets.
- `utils/config.py` stores `time_ranges` and `sweep_direction`, while `core/file_parser.py` applies per-interval acquisition times and honours reverse sweeps when parsing filenames.
- Temperature Dependence Analysis window has been refactored to a dual-plot layout (Arrhenius + intensity vs. T) with show/hide toggles, colour/marker/line-width controls, in-window integration range editing, and 600 DPI export (PNG/PDF/SVG).
- Polarization UI now auto-resets to “All Data” when magnetic/polarization support is disabled, ensuring plot state stays consistent with the available controls.
- Dedicated headless tests confirm config round-tripping and integration routines for the new window logic.

### Phase 4 - B-Field Analysis Module
- Added a B-Field Analysis tab (integration range inputs, polarization selector, action buttons).
- `core/analysis.py` now includes `calculate_centroid`, `bfield_intensity_analysis`, and `calculate_g_factor` utilities.
- `gui/actions/bfield_analysis_actions.py` and `gui/bfield_analysis_view.py` deliver intensity vs B curves, Zeeman splitting fits, and g-factor reporting.

### Phase 5 - Dual-Colour Intensity Maps
- Intensity map tab offers mode selection: Single (grayscale), Additive RGB, Alpha Overlay, Diverging (sigma+ − sigma-).
- `plot_actions.py` aggregates polarization channels and renders multi-channel maps; diverging mode automatically falls back when both channels are unavailable.
- `plot_canvas.py` gained an RGB plotting helper for overlay and additive modes.

### Magnetic Sweep Defaults & B-Field Series
- Magnetic sweep settings dialog captures sweep ranges (min/max, step) plus optional temperature/power/time defaults and ROI polarisation mapping.
- Filenames matching formats such as `pl_00-Roi-1` auto-populate B-field and polarisation metadata using the configured defaults; existing datasets can be re-parsed via the dialog to refresh metadata.
- Added the "Plot B-Field Series" button to plot spectra versus magnetic field (supports sigma modes and Sum mode).

### Phase 6 – Global Background Subtraction
- `processing.subtract_background()` removes the minimum from spectra; UI toggle added to plot controls.
- Subtraction is applied before line plots and intensity map generation.

### Phase 7 – File Table Display
- File table now lists B-field and polarization columns for quick inspection.

### Phase 8 – Testing & Validation (in progress)
- Synthetic dataset script (`scripts/generate_bfield_cp_testdata.py`) complete; default run (g = 2.0) produced g ≈ 1.94 ± 0.74 with R² ≈ 0.70 after centroid-based fitting.
- Next step: execute the full validation workflow, capture screenshots/metrics, and update documentation accordingly.

### Phase 9 – Plot Temp Series Bug Fix
- Resolved missing `file_ids` initialisation in `plot_temp_series_action()`.

## File Naming Guidance
- B-field example: `Sample_5K_100uW_1s_3T_pol1.txt`
- Polarization example: `Sample_5K_100uW_sigma+.txt`
- Combined example: `WS2_5K_100uW_1s_3T_pol2.txt`
Files lacking B-field/polarization metadata continue to load with `None` entries.

## Sum Mode Quick Reference
1. Select spectra containing both sigma+ and sigma- variants (matching temperature/power/B-field).
2. Choose **“Sum (sigma++sigma-)”** in the polarization selector.
3. Trigger any plot action (Add, Power Series, Temp Series). Matching pairs are summed; unmatched entries raise a warning.
4. Apply expected styling options (log scale, stacking, etc.) as desired.

## Dual-Colour Intensity Map Modes
- **Single (grayscale):** Original behaviour using the selected colormap.
- **Additive RGB:** sigma+ rendered in red, sigma- in blue, and overlap/unpolarized channels in green.
- **Alpha Overlay:** Combined intensity acts as a luminance base with tinted sigma+/sigma- overlays.
- **Diverging (sigma+ − sigma-):** Displays the signed difference using a centred coolwarm colour scale (requires both channels).
Background subtraction and intensity normalisation run before colour compositing.

## Next Steps
- Run full Phase 8 validation with the synthetic dataset; document observations, screenshots, and quantitative results.
- Update this document and `docs/IMPLEMENTATION_STATUS.md` with validation artefacts.
- Evaluate any post-validation polish (e.g., export/report options for B-field analysis outputs).

## Testing Checklist
1. `python scripts/generate_bfield_cp_testdata.py`
2. Load dataset and confirm file table columns (B-field, polarization) populate correctly.
3. Exercise polarization modes (sigma+, sigma-, Both, Sum) and verify background subtraction behaviour.
4. Cycle through all intensity map modes; confirm colouring/fallback rules.
5. Plot a magnetic field sweep via "Plot B-Field Series" (with/without Sum mode) and verify the traces order by B-field.
6. Adjust magnetic sweep settings and ensure ROI-based filenames populate B-field/polarisation metadata accordingly.
7. Use the B-Field Analysis tab (Integrate vs B-Field, Calculate g-Factor) and verify g ≈ 2.0 on synthetic data.
8. Record findings for the Phase 8 validation summary.

---

## Phase 2 Module 1 Updates (Complete)

### General Options Tab
Added dedicated "General Options" tab in the analysis section for application-level settings:
- UI scaling controls (75%, 85%, 100%, 110%, 125%)
- Window resolution configuration
- Informational messages about restart requirements
- Clean, user-friendly layout

### UI Scaling Feature
**Purpose**: Allow users to scale the entire UI to fit smaller screens or improve readability.

**Implementation**:
- Scaling percentages: 75%, 85%, 100%, 110%, 125%
- Uses CustomTkinter's `set_widget_scaling()` and `set_window_scaling()`
- Applied in `run_app.py` before window creation
- Persists to `config.json` via `save_ui_scale()` and `load_ui_scale()`
- Changes take effect on next restart

**Usage**:
1. Go to General Options tab
2. Select desired UI scale from dropdown
3. Click OK on confirmation dialog
4. Restart application to apply changes

### Legend Optimization
**Problem**: B-field plots with many traces (All Data / Both polarizations) had overcrowded, illegible legends.

**Solution**: Automatic legend optimization in `plot_canvas.py`:
- \>10 traces: 3 columns, fontsize 8, best location
- 6-10 traces: 2 columns, fontsize 9, best location
- <6 traces: single column, normal font

**Impact**: Significantly improves readability of B-field series with many measurements (e.g., 24 traces for 12 B-field values × 2 polarizations).

### Polarization Sum Mode Verification
Confirmed full functionality of Sum mode across all plot types:
- ✅ Power series plotting
- ✅ Temperature series plotting
- ✅ B-field series plotting
- ✅ Proper σ+/σ- pairing at each (T, P, B) condition
- ✅ Interpolation for different energy grids
- ✅ Warning messages for unmatched pairs
- ✅ Label formatting with metadata

### Additional Phase 1 Improvements
Previously implemented but now documented:
- Magnetic/polarization checkbox in left panel (enable/disable related features)
- Plot B-Field Series button relocated next to Polarization dropdown
- Background removal moved to Processing Steps (permanent application)
- Smart file duplicate detection (only flags same T, P, B-field, AND polarization)
- Canvas size preservation fix for B-field plots

---

## Phase 2 Module 2: Advanced Analysis Features (Pending)

The following advanced features are planned for implementation by CodeX (High Reasoning model). See `docs/PROMPT_FOR_NEXT_AGENT.md` for detailed requirements.

### 1. Enhanced Integrate vs B-Field
- Plot σ+, σ-, and Sum intensities vs B-field on same axes
- Calculate and plot Degree of Circular Polarization (DCP)
- Enhanced UI with customization options
- Export data to CSV

### 2. Advanced G-Factor Analysis Window
- Energy vs B-field plots for both polarizations
- Zeeman splitting plot with interactive fitting
- Multiple peak detection methods
- Adjustable fit parameters
- Residuals plot
- Export capabilities

### 3. Virtual Sum Polarization Datasets
- Create persistent datasets containing σ+ + σ- sums
- Appear as regular datasets in tabs
- Full analysis tool compatibility
- Track source files and handle updates

