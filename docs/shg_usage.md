# SHG Analyzer Usage Notes

## Quick Start
1. Activate project environment (`venv_build` or preferred Python >=3.10 with dependencies from `requirements.txt`).
2. Collect SHG measurement CSV/TXT files in a directory. Ensure filenames contain sample hints (`ML`, `HS`, `WSe2`, `2p5`, etc.).
3. Run the CLI:
   ```bash
   python scripts/run_shg_cli.py data/shg_runs --plot --plot-output outputs/polar_plot
   ```
4. Review terminal summary for fitted orientations. Optional twist estimation:
   ```bash
   python scripts/run_shg_cli.py data/shg_runs --twist ML1_1 HS2
   ```

## CLI Options
- `--background`: choose `global_min`, `local_min`, `linear`, `polynomial`, or `none`.
- `--integration-window`: sets energy span (eV) for integrated intensity.
- `--method`: repeat flag to specify intensity methods (default: `peak_max`, `integrated`).
- `--normalization`: `raw`, `normalized`, or `equalized` for plotting/orientation fits.
- `--orientation-method`: select which method feeds the orientation fit.

## Overrides & Metadata
- Ambiguous filenames can be corrected via Python shell:
  ```python
  from shg_analyzer.ui.controller import SHGAnalyzerController
  ctrl = SHGAnalyzerController()
  ctrl.load_directory("data/shg_runs")
  ctrl.set_override("HS2", sample_type="heterostructure", material="WS2/WSe2")
  ctrl.run_analysis()
  ```

## Plot Outputs
- `--plot-output` stores polar plot in PNG/PDF (using base path + extension).
- Polar plots display raw or normalized intensities with optional fit overlay.

## Testing
- Install `pytest` to run unit tests: `pip install pytest` then `python -m pytest shg_analyzer/tests -q`.
- Existing tests cover filename parsing, background correction, and intensity extraction. Extend to orientation/twist modules next.

## Next Enhancements
- Hook `SessionStore` into controller for persistent overrides.
- Implement GUI (PySide6) using `shg_analyzer/ui/qt/main_window.py` placeholder.
- Expand reporting (`plotting/report.py`) to produce CSV/JSON summaries automatically.
