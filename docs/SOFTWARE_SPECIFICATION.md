# PL Analyzer - Software Specification

**Version**: 1.0 (Phase 3 Complete)  
**Last Updated**: October 25, 2025  
**Purpose**: AI-friendly specification for understanding and extending the software

---

## 1. Project Overview

### What It Does
PL Analyzer is a desktop application for analyzing photoluminescence (PL) spectroscopy data with support for:
- Power-dependent and temperature-dependent series analysis
- Magnetic field (B-field) dependent measurements
- Circular polarization (σ+/σ-) analysis
- Advanced visualization and data export

### Target Users
- Physics researchers working with 2D materials
- Materials science labs studying optical properties
- Anyone analyzing PL spectroscopy data with complex dependencies

### Key Value Proposition
- Automatic metadata extraction from filenames
- Batch processing of large datasets
- Magnetic field and polarization-aware analysis
- Publication-quality plots and exports

---

## 2. Architecture

### Module Structure

```
pl_analyzer/
├── core/                    # Data processing and analysis
│   ├── file_parser.py      # Parse filenames, extract metadata
│   ├── data_handler.py     # Dataset class, data management
│   ├── processing.py       # Data processing (normalization, smoothing, etc.)
│   ├── analysis.py         # Analysis functions (centroid, g-factor, etc.)
│   ├── export.py           # Data export functionality
│   └── spike_removal.py    # Spike detection and removal algorithms
│
├── gui/                     # User interface
│   ├── main_window.py      # Main application window (CTk)
│   ├── dialogs.py          # Configuration dialogs
│   ├── analysis_view.py    # Power analysis tab
│   ├── temp_analysis_view.py  # Temperature analysis window
│   ├── bfield_analysis_view.py # B-field analysis tab
│   │
│   ├── panels/             # UI panels
│   │   ├── left_panel.py   # File list, processing controls
│   │   └── right_panel.py  # Analysis tabs, plot options
│   │
│   ├── widgets/            # Reusable UI components
│   │   ├── file_table.py   # File list table widget
│   │   └── plot_canvas.py  # Matplotlib canvas wrapper
│   │
│   ├── actions/            # Event handlers (MVC pattern)
│   │   ├── file_actions.py       # Load, save, dataset management
│   │   ├── processing_actions.py # Apply processing steps
│   │   ├── plot_actions.py       # Plotting actions
│   │   ├── analysis_actions.py   # Power/temp analysis
│   │   └── bfield_analysis_actions.py # B-field analysis
│   │
│   └── windows/            # Standalone windows
│       ├── gfactor_analysis_window.py    # G-factor analysis
│       ├── bfield_integration_window.py  # Integrate vs B-field
│       ├── bfield_intensity_map_window.py # B-field intensity maps
│       └── bfield_selection_window.py    # B-field value selector
│
├── utils/                   # Utilities
│   └── config.py           # Configuration management (config.json)
│
└── main.py                 # Application entry point
```

### Design Patterns
- **MVC-like**: GUI actions are separated from core logic
- **Event-driven**: CustomTkinter (CTk) event system
- **Data-centric**: Dataset class holds all file metadata and processing state

---

## 3. Key Features by Phase

### Phase 1: B-Field & Circular Polarization (Complete)
- Filename parsing extracts B-field and polarization (σ+/σ-)
- Polarization mode selector (All, σ+, σ-, Both, Sum)
- B-field series plotting
- Intensity maps with RGB channels for dual polarization
- Magnetic sweep configuration dialog
- Smart duplicate detection in file table

### Phase 2: Advanced Analysis & UI (Complete)
- UI scaling (75%-125%)
- Legend optimization for many traces
- Enhanced integrate-vs-B-field with DCP (Degree of Circular Polarization)
- G-factor analysis window with interactive fitting
- Virtual sum datasets (σ+ + σ-)
- Grid plotting for power series at multiple temperatures
- Spike removal with adaptive detection

### Phase 3: B-Field UI Enhancements (Complete)
- Normalized stacked B-field plots with custom selection
- Dedicated B-field intensity map window with full controls
- RGB intensity maps with separate colorbars for each polarization
- Variable integration times per B-field range
- Sweep direction support (low→high or high→low)
- Enhanced temperature analysis with dual plots (Arrhenius + linear)
- Auto-reset polarization when magnetic data disabled

---

## 4. Data Flow

### Startup
1. `run_app.py` → loads UI scale config → sets CustomTkinter scaling
2. `pl_analyzer/main.py` → loads polarization and magnetic sweep config
3. Applies config to file parser
4. Creates main window → displays empty file table

### Loading Files
1. User clicks "Load Files" → `file_actions.py:load_files_action()`
2. Files parsed by `file_parser.py:parse_filename()`
3. Creates `Dataset` objects in `data_handler.py`
4. Adds to file table → displays metadata (T, P, B, pol)

### Processing
1. User selects files → clicks processing button (e.g., "Normalize by Time")
2. `processing_actions.py` calls appropriate function in `processing.py`
3. Processing modifies Dataset arrays in-place
4. Updates Dataset flags (e.g., `time_normalized = True`)
5. Disables button to prevent re-application

### Plotting
1. User selects plot type (e.g., "Plot Power Series")
2. `plot_actions.py` filters datasets by current temperature
3. Applies polarization filter (`_filter_by_polarization()`)
4. Calls `plot_canvas.py:plot_data()` with formatted data
5. Matplotlib figure displayed in canvas widget

### Analysis
1. User enters integration range → clicks analysis button
2. `analysis_actions.py` collects datasets, calls `analysis.py` functions
3. Results displayed in new window (e.g., Power Dependence Analysis)
4. User can fit data, export results

---

## 5. Configuration System

### config.json Structure
```json
{
  "window_resolution": {"width": 1600, "height": 900},
  "ui_scale": 1.0,
  "polarization": {
    "sigma_plus": ["sigma+", "s+", "sp", "sigma_plus"],
    "sigma_minus": ["sigma-", "s-", "sm", "sigma_minus"]
  },
  "magnetic_sweep": {
    "min_bfield_t": 0.0,
    "max_bfield_t": 12.0,
    "step_t": 0.5,
    "sweep_direction": "low_to_high",
    "time_ranges": [
      {"b_min": 0.0, "b_max": 5.0, "time_s": 30},
      {"b_min": 5.0, "b_max": 12.0, "time_s": 60}
    ],
    "roi_map": {"1": "sigma+", "2": "sigma-"}
  }
}
```

### Config Functions (`utils/config.py`)
- `load_config()` / `save_config()` - Master config operations
- `load_ui_scale()` / `save_ui_scale()` - UI scaling
- `load_polarization_config()` / `save_polarization_config()` - Polarization strings
- `load_magnetic_sweep_config()` / `save_magnetic_sweep_config()` - B-field settings

---

## 6. File Naming Convention

### Standard Format
```
Prefix_<Temperature>K_<Power>uW_<Time>s_<BField>T_<Polarization>.csv
```

**Examples:**
- `sample_5K_100uW_1s.csv` - Basic format
- `WSe2_5K_100uW_0p5s_0T_sigma+.csv` - With B-field and polarization
- `data_10K_50uW_2s_3p5T_sigma-.csv` - Decimal B-field (3.5T)

**Rules:**
- Decimals use 'p' instead of '.' (e.g., `0p5` = 0.5)
- Parameters can appear in any order
- Polarization strings are user-configurable
- Missing parameters default to None

### Sweep Format (Alternative)
```
pl_<index>-Roi-<roi>.csv
```
- B-field inferred from index and sweep config
- ROI mapped to polarization via `roi_map` in config

---

## 7. Key Classes and Functions

### Core Classes

#### `Dataset` (data_handler.py)
```python
class Dataset:
    id: str                  # Unique identifier
    filename: str            # Full path
    energy: np.ndarray       # Energy axis (eV)
    counts: np.ndarray       # Intensity data
    temperature_k: float     # Temperature (K)
    power_uw: float          # Power (μW)
    time_s: float            # Acquisition time (s)
    bfield_t: float          # Magnetic field (T)
    polarization: str        # Polarization (sigma+/sigma-/None)
    
    # Processing flags
    time_normalized: bool
    grey_filter_corrected: bool
    spectrometer_corrected: bool
    background_subtracted: bool
```

### Key Functions

#### File Parsing
- `file_parser.py:parse_filename()` - Extract all metadata from filename
- `file_parser.py:set_polarization_config()` - Update polarization string mappings
- `file_parser.py:set_magnetic_sweep_config()` - Update sweep parameters

#### Analysis
- `analysis.py:calculate_centroid()` - Calculate spectral centroid in range
- `analysis.py:bfield_intensity_analysis()` - Integrate intensity vs B-field
- `analysis.py:calculate_g_factor()` - Extract g-factor from Zeeman splitting

#### Plotting
- `plot_canvas.py:plot_data()` - Main plotting function
- `plot_canvas.py:plot_stacked_data()` - Stacked/offset plots
- `plot_canvas.py:plot_intensity_map()` - 2D intensity maps
- `plot_canvas.py:plot_rgb_intensity_map()` - Multi-channel intensity maps

#### Processing
- `processing.py:normalize_by_time()` - Convert to counts/s
- `processing.py:subtract_background()` - Remove baseline
- `processing.py:smooth_data()` - Apply smoothing
- `spike_removal.py:detect_spikes_adaptive()` - Detect spikes with rolling median/MAD

---

## 8. Dependencies

### Core Dependencies
- **numpy**: Numerical array operations, all data processing
- **matplotlib**: Plotting engine, all visualizations
- **pandas**: CSV reading, data export
- **scipy**: Curve fitting, signal processing, interpolation
- **customtkinter**: Modern UI framework (Tkinter wrapper)

### Why These Packages
- **numpy/scipy**: Standard scientific Python stack, essential for spectroscopy analysis
- **matplotlib**: Publication-quality plots, highly customizable
- **pandas**: Efficient CSV I/O, handles various formats
- **customtkinter**: Modern look, easier than raw Tkinter

### Full List
See `requirements.txt` for versions.

---

## 9. Build Process

### Creating .exe with PyInstaller

**Spec File**: `run_app.spec`
```python
name='PL_analyzer'  # Output executable name
console=False       # No console window (GUI only)
```

**Build Commands:**
```bash
# Activate build environment
.\venv_build\Scripts\activate

# Build
pyinstaller run_app.spec --clean

# Output: dist/PL_analyzer.exe
```

### Distribution
- Single-file executable (all dependencies bundled)
- ~100MB due to numpy/scipy/matplotlib
- Windows 10+ compatible
- No Python installation required for users

---

## 10. Extension Points

### Adding New Analysis Features

**Where to Add:**
1. Create analysis function in `core/analysis.py`
2. Create UI tab/window in `gui/` (e.g., `new_analysis_view.py`)
3. Create action handler in `gui/actions/` (e.g., `new_analysis_actions.py`)
4. Wire button/menu item to action in appropriate panel

**Example Pattern:**
```python
# In core/analysis.py
def new_analysis_function(datasets, params):
    # Your analysis logic
    return results

# In gui/actions/new_analysis_actions.py
def perform_new_analysis_action(app):
    datasets = app.data_manager.get_selected_datasets()
    params = {...}  # From UI
    results = new_analysis_function(datasets, params)
    # Display results in new window
```

### Adding New Plot Types

**Where to Add:**
1. Create plot function in `widgets/plot_canvas.py`
2. Create action in `actions/plot_actions.py`
3. Add button in appropriate panel (left_panel or right_panel)

### Adding New File Formats

**Where to Modify:**
1. Extend `file_parser.py:load_spectrum()` to handle new format
2. Keep `parse_filename()` for metadata extraction
3. May need to add format detection logic

### Adding New Processing Steps

**Where to Add:**
1. Create processing function in `core/processing.py`
2. Create action in `gui/actions/processing_actions.py`
3. Add button in `panels/left_panel.py` processing section
4. Add flag to `Dataset` class if needed (e.g., `new_step_applied`)

---

## 11. Common Modification Scenarios

### Changing Polarization Colors
**File**: `gui/widgets/plot_canvas.py`
**Function**: `plot_data()` and `_filter_by_polarization()`
```python
# Current: red for σ+, blue for σ-
# Modify color assignment in plot_data()
```

### Adding Config Parameters
**File**: `pl_analyzer/utils/config.py`
1. Add default to `DEFAULT_CONFIG`
2. Create `load_your_param()` and `save_your_param()` functions
3. Use in appropriate action/view

### Modifying File Table Columns
**File**: `gui/widgets/file_table.py`
**Class**: `FileTable`
```python
# Add column to self.columns list
# Modify _populate_table() to display new data
```

### Adding Analysis Window
**Pattern**: Copy `gui/windows/gfactor_analysis_window.py`
1. Create new window class inheriting from `ctk.CTkToplevel`
2. Add UI controls in `__init__`
3. Connect buttons to internal methods
4. Call analysis functions from `core/analysis.py`

---

## 12. Important State Management

### Application State
- **Current dataset**: `app.data_manager.datasets[selected_id]`
- **Plot data cache**: `app._last_plot_data` (for exporting)
- **UI state**: Various `ctk.StringVar`, `ctk.IntVar` in panels

### Dataset State
- **Processing flags**: Track which steps applied (prevent double-application)
- **Original data**: Keep copy for "Show Original" feature
- **Metadata**: All experimental parameters from filename

### Configuration State
- **Persistent**: Saved to config.json, loaded on startup
- **Session**: Lost when app closes (e.g., window position)

---

## 13. Testing Recommendations

### Manual Testing Workflow
1. Load test files from `test_data/bfield_cp/`
2. Test each processing step
3. Test each plot type
4. Test each analysis window
5. Verify config persistence (close/reopen)

### Key Test Cases
- Files with/without B-field metadata
- Files with/without polarization
- Power series at multiple temperatures
- B-field series with both polarizations
- Edge cases: single file, missing pairs, etc.

### Test Data
- `test_data/bfield_cp/`: Sample B-field circular polarization data
- Generated by `scripts/generate_bfield_cp_testdata.py`

---

## 14. Known Limitations & Future Work

See `docs/FUTURE_IMPROVEMENTS.md` for detailed bug reports and enhancement requests.

### Current Limitations
1. G-factor "maximum" peak detection not working
2. Gaussian fitting needs smarter parameter initialization
3. Intensity maps lack color interpolation/smoothing
4. Spike removal UI has missing input fields (bug)

### Potential Enhancements
- Batch export of all analysis results
- Custom color schemes for plots
- Multi-file comparison views
- Automated peak tracking across series

---

## 15. Quick Reference for AI

### "Where do I...?"

**...add a new button?**
→ `gui/panels/left_panel.py` or `right_panel.py`

**...add a new analysis function?**
→ `core/analysis.py` + create window in `gui/windows/`

**...modify file parsing?**
→ `core/file_parser.py:parse_filename()`

**...change plot styling?**
→ `gui/widgets/plot_canvas.py`

**...add config option?**
→ `utils/config.py` + update DEFAULT_CONFIG

**...create new dialog?**
→ `gui/dialogs.py` or new file in `gui/windows/`

### "What does this file do?"

| File | Purpose |
|------|---------|
| `main.py` | Application entry point, loads config |
| `main_window.py` | Top-level window, holds left/right panels |
| `data_handler.py` | Dataset class, data management |
| `file_parser.py` | Filename parsing, metadata extraction |
| `plot_canvas.py` | All plotting logic, matplotlib wrapper |
| `analysis.py` | Analysis algorithms (centroid, g-factor, etc) |
| `config.py` | Config file operations (load/save) |
| `*_actions.py` | Event handlers, connect UI to core logic |

---

**End of Specification**

This document should provide enough context for AI assistants to understand the codebase structure and make informed modifications. For implementation details, refer to inline code comments and docstrings.

