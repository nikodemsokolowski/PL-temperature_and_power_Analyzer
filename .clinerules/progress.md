# Progress: SHG Analyzer Bootstrap

## What Works
-   Core `shg_analyzer` package scaffolded with data ingestion, filename heuristics, background correction, intensity extraction, orientation fitting, twist estimation, plotting, and CLI entrypoint.
-   Smart filename parser supports ML/HS detection, material hints, series IDs like `ML1_2`, and half-wave plate angles with decimal syntax (`4p0`).
-   Background correction strategies implemented (`global_min`, `local_min`, `linear`, `polynomial`) with reusable diagnostics.
-   Intensity extraction methods (peak, integrated, Gaussian, Lorentzian) wired into `AnalysisOptions`; uncertainties propagated when available.
-   Polar plotting utility renders multi-series comparisons with optional model overlays; export hooks ready.
-   CLI workflow loads directories, runs analysis, prints summary table, and optionally plots/fits twist angles.
-   Initial unit tests cover naming heuristics, background subtraction, and intensity strategies.
-   Detailed roadmap recorded in `docs/shg_analyzer_plan.md`.

## What's Left to Build
1.  **Orientation Fit Robustness**
    *   Harden covariance handling and add bootstrap uncertainty estimation.
2.  **Twist Classification Workflow**
    *   Surface twist results in CLI/GUI, add per-series pairing UI.
3.  **Normalization Equalization Strategy**
    *   Implement reference-series aware equalization rather than mean scaling.
4.  **GUI Implementation**
    *   Build PySide6 interface mirroring PL analyzer UX with SHG-specific panels.
5.  **Session Persistence**
    *   Integrate `SessionStore` with controller to restore overrides and latest analyses.
6.  **Packaging**
    *   Author PyInstaller spec and resources for `.exe` build once GUI stabilizes.
7.  **Testing Infrastructure**
    *   Add pytest dependency, expand coverage (orientation fits, polar plotting, controller orchestration).

## Current Status
-   Backend analytics foundation in place; CLI usable for early datasets once pytest dependency installed for tests.
-   PL analyzer codebase remains stable; focus shifted to SHG analyzer development.

## Known Issues
-   Orientation fit may fail or wrap angles unexpectedly with sparse/low-SNR data; warning stored in `series.overrides['fit_errors']`.
-   Twist computation currently limited to two-series input; no batching yet.
-   `pytest` missing in environment, so automated tests cannot run until dependency is installed.
