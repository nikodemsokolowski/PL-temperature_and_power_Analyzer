# Polarization-Resolved SHG Analyzer — Detailed Plan

## Vision & Scope
- Deliver a standalone application for polarization-resolved SHG experiments with support for monolayer (ML) and heterostructure (HS) samples.
- Provide robust data ingestion, smart filename metadata extraction, background correction, intensity quantification, polar plotting, and crystallographic orientation fitting with twist-angle determination.
- Maintain architecture parity with the existing PL analyzer where it makes sense, but decouple codebases to keep SHG-specific logic isolated.

## Target Users & Workflows
1. **Exploratory Review**: Load multiple measurement series, inspect automatically parsed metadata, perform quick background corrections, visualize polar plots (raw & normalized), and compare series side-by-side.
2. **Quantitative Analysis**: For each series extract SHG intensities via configurable methods (peak/max, integrated window, Gaussian/Lorentzian fits) with optional background parameters. Fit polar responses to infer crystal axes, compute relative twist angles, and classify stacking (R vs H) with uncertainties.
3. **Batch Reporting**: Export processed intensities, polar plots, and fit summaries for selected samples to support publications.
4. **Quality Control**: Detect naming ambiguities, allow manual overrides, and surface data-quality warnings (e.g., insufficient angle coverage, failed fits).

## Data & Metadata Strategy
- **File Naming Heuristics**:
  - Recognize sample segments: `ml`, `mono`, `hs`, explicit material names (`wse2`, `ws2`, `mos2`, `mose2`, `wsse`, etc.), optional numeric suffixes (`ml1`, `hs2`, `ml1_2`).
  - Angle tokens: pattern `<number>[p|P]<digit>` (e.g., `2p0` → 2.0° HWP → 4.0° polarization).
  - Optional power/fluence tokens (e.g., `7uW`, `1s`) are captured but not required.
  - Provide confidence scores and human-readable interpretations; flag ambiguities (e.g., conflicting sample tokens) for user confirmation.
- **Metadata Overrides**:
  - UI dialog / CLI prompt to edit sample name, type (ML/HS), series ID, power, integration time, notes.
  - Persist overrides to project session file for reproducibility.
- **Data Model**:
  - `SampleSeries` representing one sample & series with multiple `AngleMeasurement` entries.
  - Each `AngleMeasurement` stores raw intensity arrays, background model, and extracted intensity metrics.

## Processing Pipeline
1. **Ingestion**: Load CSV (and future formats) into `pandas.DataFrame` with columns `[wavelength/nm, energy/eV, counts]`. Handle decimal separators (comma vs dot) automatically.
2. **Preprocessing**: Optional resampling/interpolation onto common energy axis per series to enable direct comparisons.
3. **Background Correction** (per polarization angle):
   - `baseline_min`: subtract global minimum.
   - `baseline_local_min`: subtract min within configurable energy window.
   - `baseline_linear`: fit linear trend over selected background range; subtract.
   - `baseline_polynomial`: extendable hook for higher-order corrections.
   - Maintain background parameters per angle; allow global override.
4. **Intensity Extraction Methods** (composable strategy objects):
   - `peak_max`: maximum of corrected counts.
   - `integrated_window`: integrate counts over user-defined or auto-detected peak window.
   - `gaussian_fit`: fit Gaussian; report peak amplitude & area.
   - `lorentzian_fit`: fit Lorentzian; report amplitude & area.
   - Each method returns value, fit diagnostics, and uncertainty (covariance from fit or bootstrapping placeholder).
5. **Normalization & Equalization**:
   - `raw`: no scaling.
   - `normalized`: divide by per-series maximum intensity.
   - `equalized`: scale intensities to match reference series (e.g., first ML) preserving relative amplitudes.
6. **Polar Plotting**:
   - Use Matplotlib polar axes; overlay multiple series with styling presets.
   - Optionally plot theoretical fits alongside data points.
   - Export options (PNG, SVG, PDF) with metadata summary captions.
7. **Crystallographic Fitting**:
   - Fit I(θ) to `I0 * cos^2(3(θ - θ0))` or extended models depending on sample type.
   - Estimate `θ0` and confidence intervals.
   - For HS stacks, compute twist angle Δθ between constituent monolayers; classify stacking (`R` if |Δθ| ≈ 0° (mod 60), `H` if |Δθ - 60°| minimal) and report uncertainty.
8. **Reporting**:
   - Summaries stored in structured data (JSON/CSV) with fit metrics and classification.

## Software Architecture
```
shg_analyzer/
    __init__.py
    config.py                # Defaults for paths, fitting, plotting, tolerances.
    data/
        __init__.py
        loader.py           # File discovery, parsing, dataframe loading.
        naming.py           # Filename heuristics & metadata inference.
        registry.py         # Session persistence, override storage.
    core/
        __init__.py
        background.py       # Background correction strategies.
        intensity.py        # Intensity extraction strategies.
        models.py           # Data classes (SampleSeries, AngleMeasurement).
        analysis.py         # Pipeline orchestration.
        twist.py            # Crystal axis + twist computation utilities.
    plotting/
        __init__.py
        polar.py            # Polar plots (raw, normalized, equalized) & exports.
        report.py           # Multi-series comparison plot helpers.
    ui/
        __init__.py
        controller.py       # High-level orchestration, shared with CLI/GUI.
        cli.py              # Initial command-line interface (placeholder).
        qt/
            __init__.py
            main_window.py  # Future PySide6 GUI skeleton (optional in phase 1).
    storage/
        __init__.py
        session.py          # Session serialization (JSON) for overrides & results.
    tests/
        __init__.py
        test_naming.py
        test_background.py
        test_intensity.py

scripts/
    run_shg_cli.py
    run_shg_gui.py (placeholder)
```

## Phased Implementation Roadmap
1. **Phase 0 — Foundations (current task)**
   - Scaffold package layout, data models, naming heuristics, background & intensity strategy interfaces, beta-level polar plotting.
   - Provide CLI-driven workflow to load directory, configure options via YAML/JSON, and output polar plots & reports.
   - Author detailed developer plan (this document) and update memory bank.
2. **Phase 1 — Analysis Depth**
   - Complete fit models (Gaussian/Lorentzian), error propagation, twist-angle classification, and reporting outputs.
   - Implement session persistence and configuration management.
3. **Phase 2 — GUI & UX**
   - Build PySide6 GUI mirroring PL analyzer patterns (file table, plot canvas, controls) with SHG-specific panels.
   - Add interactive fit refinements and manual overrides.
4. **Phase 3 — Packaging**
   - Prepare PyInstaller spec, bundle resources, produce distributable `.exe`.
   - Document installation, dependencies, and verification tests.

## Immediate Next Steps
- Finalize data classes and strategy interfaces.
- Implement loader + naming heuristics with unit tests covering edge-case filenames.
- Flesh out background correction functions with parameter validation.
- Implement polar plotting utilities with normalization options.
- Create CLI entrypoint to exercise pipeline end-to-end with sample data (to be simulated until real files available).
- Update Memory Bank (`activeContext`, `progress`) to track SHG analyzer efforts.

## Open Questions
- Confirm expected input CSV column schema and units; align with existing measurement exports.
- Determine default background correction windows (based on known spectral ranges?).
- Clarify precision requirements for twist-angle uncertainty (bootstrap vs covariance).
- Decide on baseline directories for session persistence and caching (project-level vs per-analysis).

