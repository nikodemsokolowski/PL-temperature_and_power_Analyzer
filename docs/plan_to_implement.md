# B-Field UI Enhancement Implementation Plan

**Created:** October 24, 2025 - 14:30 UTC  
**Status:** Ready for Implementation  
**Target Agent:** GPT-Codex-5-High (High Reasoning Model)  
**Estimated Time:** 14-15 hours

---

## Overview

Implement 10 major feature enhancements and bug fixes for the PL Analyzer's magnetic field analysis capabilities. Focus on UI improvements for publication-quality data visualization and advanced analysis workflows.

---

## Implementation Priority & Task Breakdown

### CRITICAL BUG FIX (Complete First)

**Task 0: Fix "Integrate vs B-Field" Window Reopen Bug**

- **Issue**: Second opening of integrate window doesn't work
- **Root Cause**: Window exists check in `_ensure_bfield_integration_window()` may be failing
- **Files**: `pl_analyzer/gui/actions/bfield_analysis_actions.py`
- **Solution**: 
  - Check if `window.winfo_exists()` properly validates window state
  - Ensure window is properly destroyed when closed (not just hidden)
  - Add proper window state management (iconify/deiconify vs destroy/recreate)
- **Testing**: Open integrate window, close it, open again - should work without errors

---

### HIGH PRIORITY FEATURES

### Task 1: Normalized Stacked Plot for B-Field with Selection

**Requirements**:
- Similar to existing normalized stacked plot but for B-field series
- **CRITICAL**: Plot both polarizations (σ+ and σ-) on SAME line (not vertically stacked)
  - σ+ in red, σ- in blue, overlaid at same vertical position
- Checkbox/selector UI to include/exclude specific B-field values
- Interval/step selection (e.g., "every 2T" or "every 3T")

**Implementation**:
- **Files to Modify**:
  - `pl_analyzer/gui/panels/right_panel.py`: Add checkbox in top panel near polarization controls
  - `pl_analyzer/gui/panels/right_panel.py`: Extend "Stacked Plot Options" tab with B-field specific controls
  - `pl_analyzer/gui/actions/plot_actions.py`: Create new function `plot_bfield_normalized_stacked()`
  - `pl_analyzer/gui/widgets/plot_canvas.py`: Extend `plot_stacked_data()` to handle dual-polarization same-line overlay

- **UI Design**:
  ```
  Top Panel (near polarization):
  [✓] Normalized Stacked (B-Field)
  
  Stacked Plot Options Tab (new section):
  ┌─ B-Field Selection ──────────────────┐
  │ [✓] 0.0T  [✓] 0.5T  [✓] 1.0T  [✓] 1.5T │
  │ [✓] 2.0T  [✓] 2.5T  [✓] 3.0T  [✓] 3.5T │
  │                                        │
  │ Step Interval: [dropdown: 1T/2T/3T]   │
  │ [Select All] [Deselect All] [Invert]  │
  └────────────────────────────────────────┘
  ```

- **Plotting Logic**:
  1. Filter selected B-field values
  2. For each B-field value, plot BOTH σ+ and σ- at same vertical offset
  3. Use alpha=0.7 for transparency to see overlap
  4. Apply normalization to specified energy range
  5. Label format: "B=2.0T (σ+/σ-)" or separate labels if space permits

- **Key Code Pattern**:
```python
def plot_bfield_normalized_stacked(app):
    # Get selected B-field values from checkboxes
    selected_bfields = _get_selected_bfield_values(app)
    
    # Filter data for both polarizations
    for bfield in selected_bfields:
        sigma_plus_data = filter_data(bfield, 'sigma+')
        sigma_minus_data = filter_data(bfield, 'sigma-')
        
        # Normalize both
        norm_plus = normalize(sigma_plus_data)
        norm_minus = normalize(sigma_minus_data)
        
        # Plot at SAME vertical offset
        offset_value = bfield_index * offset_factor
        plot(norm_plus + offset_value, color='red', alpha=0.7)
        plot(norm_minus + offset_value, color='blue', alpha=0.7)
```

---

### Task 2: B-Field Intensity Map Window

**Requirements**:
- Move ALL intensity map options for magnetic field to B-Field Analysis tab
- Create dedicated sub-window with all intensity map controls
- Add log color scale option
- Add energy range selection
- Add intensity range selection with customizable colorbar

**Implementation**:
- **New File**: `pl_analyzer/gui/windows/bfield_intensity_map_window.py`
- **Files to Modify**:
  - `pl_analyzer/gui/panels/right_panel.py`: Add "Intensity Map" button in B-Field Analysis tab
  - `pl_analyzer/gui/actions/bfield_analysis_actions.py`: Add `open_bfield_intensity_map_action()`

- **Window Design**:
```
┌─ B-Field Intensity Map Settings ─────────────────────┐
│                                                       │
│ Energy Range:  [Min: ____] [Max: ____] eV            │
│                                                       │
│ Intensity Range: [Min: ____] [Max: ____] (auto)      │
│                                                       │
│ Colormap: [dropdown: viridis/plasma/magma/...]       │
│                                                       │
│ [✓] Log Color Scale                                  │
│ [✓] Log Y-Axis (B-field)                             │
│                                                       │
│ Map Mode: [dropdown: Single/RGB/Alpha/Diverging]     │
│                                                       │
│ ── RGB Mode Options (for σ+/σ-) ──                   │
│ [✓] Show σ+ scale bar                                │
│ [✓] Show σ- scale bar                                │
│ [✓] Show combined info box                           │
│ [ ] Hide all scale bars                              │
│                                                       │
│ σ+ Intensity: [0.0 - 1.0] (auto detected)            │
│ σ- Intensity: [0.0 - 1.0] (auto detected)            │
│                                                       │
│ [Generate Map] [Export] [Close]                      │
└───────────────────────────────────────────────────────┘
```

- **Key Features**:
  1. Energy range filtering before plotting
  2. Separate intensity scales for σ+ and σ- in RGB mode
  3. Option to show/hide individual colorbars
  4. Info box showing intensity ranges for each channel
  5. Export with proper legends/colorbars

---

### Task 3: RGB Intensity Map Improvements

**Requirements**:
- Show intensity range scale for EACH polarization separately
- Option to display in info box/legend
- Option to hide/show colorbars independently
- Customizable colors for each channel

**Implementation**:
- **Files to Modify**:
  - `pl_analyzer/gui/widgets/plot_canvas.py`: Enhance `plot_rgb_intensity_map()`
  - Add multi-colorbar support using `matplotlib.colorbar.ColorbarBase`

- **Technical Approach**:
```python
def plot_rgb_intensity_map_enhanced(self, energy, bfield, rgb_data, 
                                    show_sigma_plus_bar=True,
                                    show_sigma_minus_bar=True,
                                    show_info_box=True,
                                    sigma_plus_range=(0, 1),
                                    sigma_minus_range=(0, 1)):
    """Enhanced RGB intensity map with per-channel colorbars."""
    
    # Main plot
    ax = self.figure.add_subplot(111)
    ax.imshow(rgb_data, aspect='auto', extent=...)
    
    # Add σ+ colorbar (red gradient)
    if show_sigma_plus_bar:
        cax1 = self.figure.add_axes([0.92, 0.55, 0.02, 0.35])
        create_colorbar(cax1, 'Reds', sigma_plus_range, label='σ+ Intensity')
    
    # Add σ- colorbar (blue gradient)
    if show_sigma_minus_bar:
        cax2 = self.figure.add_axes([0.92, 0.1, 0.02, 0.35])
        create_colorbar(cax2, 'Blues', sigma_minus_range, label='σ- Intensity')
    
    # Add info box
    if show_info_box:
        info_text = f"σ+: [{sigma_plus_range[0]:.2e}, {sigma_plus_range[1]:.2e}]\n"
        info_text += f"σ-: [{sigma_minus_range[0]:.2e}, {sigma_minus_range[1]:.2e}]"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=9)
```

---

### Task 4: Magnetic Sweep Settings - Variable Integration Times & Sweep Direction

**Requirements**:
- Allow multiple time ranges for different B-field intervals
- **NEW:** Add sweep direction selector (Low→High or High→Low)
- UI with add/remove buttons
- Example: "0-5T: 30s, 5-12T: 60s" with direction "0T→12T" or "12T→0T"

**Implementation**:
- **Files to Modify**:
  - `pl_analyzer/gui/dialogs.py`: Enhance `MagneticSweepSettingsDialog`
  - `pl_analyzer/utils/config.py`: Update config storage for time ranges and sweep direction
  - `pl_analyzer/core/file_parser.py`: Use time ranges and handle both sweep directions

- **UI Design**:
```
┌─ Sweep Direction ─────────────────────────────┐
│ ◉ Low to High (0T → 12T)                      │
│ ○ High to Low (12T → 0T)                      │
└───────────────────────────────────────────────┘

┌─ Acquisition Time Ranges ────────────────────┐
│                                               │
│ B-field From   B-field To   Time (s)   Del   │
│ ──────────────────────────────────────────────│
│ [  0.0  ]      [  5.0  ]    [ 30 ]    [X]    │
│ [  5.0  ]      [ 12.0  ]    [ 60 ]    [X]    │
│                                               │
│ [+ Add Range]                                 │
└───────────────────────────────────────────────┘
```

- **Data Structure**:
```python
# In config.json
"magnetic_sweep": {
    "min_bfield_t": 0.0,
    "max_bfield_t": 12.0,
    "step_t": 0.5,
    "sweep_direction": "low_to_high",  # or "high_to_low"
    "time_ranges": [
        {"b_min": 0.0, "b_max": 5.0, "time_s": 30},
        {"b_min": 5.0, "b_max": 12.0, "time_s": 60}
    ],
    "roi_map": {"1": "sigma+", "2": "sigma-"}
}
```

- **File Parser Logic**:
```python
# In file_parser.py
def parse_sweep_files(files, sweep_config):
    direction = sweep_config.get("sweep_direction", "low_to_high")
    
    if direction == "high_to_low":
        # Reverse the B-field assignment
        # First file gets max_bfield, last file gets min_bfield
        bfield_values = np.arange(max_b, min_b - step, -step)
    else:
        # Standard: first file gets min_bfield, last file gets max_bfield
        bfield_values = np.arange(min_b, max_b + step, step)
    
    # Assign B-field values to files in order
    for file, bfield in zip(files, bfield_values):
        assign_bfield(file, bfield)
```

---

### Task 5: Temperature Analysis Window Enhancement

**Requirements**:
- Show BOTH Arrhenius and linear intensity plots side-by-side
- Customizable plotting options (colors, styles, markers)
- Publication-quality export

**Implementation**:
- **Files to Modify**:
  - `pl_analyzer/gui/temp_analysis_view.py`: Major refactor to dual-panel layout
  - Create side-by-side or stacked subplot layout

- **New Window Layout**:
```
┌─ Temperature Dependence Analysis ────────────────────────┐
│                                                           │
│ Controls:                                                 │
│ [✓] Show Arrhenius   [✓] Show Linear Intensity           │
│ Plot Style: [dropdown]  Color: [picker]  Marker: [picker]│
│                                                           │
│ ┌─ Arrhenius Plot ──────┐ ┌─ Intensity vs T ────────┐   │
│ │                        │ │                          │   │
│ │   I(T) vs 1000/T      │ │   I(T) vs T              │   │
│ │                        │ │                          │   │
│ │   With Ea fit          │ │   Linear/log scale       │   │
│ └────────────────────────┘ └──────────────────────────┘   │
│                                                           │
│ Integration Range: [___] to [___] eV    [Apply]          │
│                                                           │
│ [Fit Arrhenius (1 Exp)] [Fit (2 Exp)] [Export]           │
└───────────────────────────────────────────────────────────┘
```

- **Implementation Approach**:
  1. Use `matplotlib.pyplot.subplots(1, 2)` or GridSpec for flexible layout
  2. Add toggle checkboxes to show/hide each subplot
  3. Add style controls: line color, marker style, marker size, line width
  4. Add integration range selector IN the window (not just from Power Analysis tab)
  5. Export button saves high-DPI figure with both plots

- **Key Code Pattern**:
```python
def _create_dual_plot_layout(self):
    self.figure.clear()
    
    show_arrhenius = self.show_arrhenius_var.get()
    show_linear = self.show_linear_var.get()
    
    if show_arrhenius and show_linear:
        # Side-by-side
        self.ax_arrhenius = self.figure.add_subplot(121)
        self.ax_linear = self.figure.add_subplot(122)
    elif show_arrhenius:
        self.ax_arrhenius = self.figure.add_subplot(111)
        self.ax_linear = None
    elif show_linear:
        self.ax_linear = self.figure.add_subplot(111)
        self.ax_arrhenius = None
    
    self._plot_data_on_axes()
```

---

### MEDIUM PRIORITY FEATURES

### Task 6: Unchecking "Has Magnetic/Polarization Data" Auto-Reset

**Requirements**:
- When user unchecks the "Has Magnetic/Polarization Data" checkbox
- Automatically change polarization dropdown to "All Data"
- Disable polarization-related controls

**Implementation**:
- **Files to Modify**:
  - `pl_analyzer/gui/panels/left_panel.py`: Add callback to magnetic checkbox
  - `pl_analyzer/gui/panels/right_panel.py`: Create function to reset polarization controls

- **Code Location**: Look for `self.magnetic_pol_checkbox` in left_panel.py
- **Solution**:
```python
def _on_magnetic_checkbox_changed(self):
    """Callback when magnetic/polarization checkbox state changes."""
    is_checked = self.magnetic_pol_var.get()
    
    if not is_checked:
        # Auto-reset polarization to "All Data"
        if hasattr(self.app, 'polarization_mode_var'):
            self.app.polarization_mode_var.set("all")
            # Update UI display
            if hasattr(self.app.right_panel, 'pol_menu'):
                self.app.right_panel.pol_menu.set("All Data")
    
    # Enable/disable polarization controls
    self._update_polarization_controls_state(is_checked)
```

---

### Task 7: Energy Range Selection for B-Field Intensity Map

**Requirements**:
- Allow user to specify energy range before generating intensity map
- Applies to both single-color and RGB modes
- Improves plot focus on relevant spectral regions

**Implementation**:
- **Covered in Task 2** (B-Field Intensity Map Window)
- Add Min E / Max E entry fields in window
- Apply range filtering BEFORE interpolation/resampling for intensity map
- Update both `plot_intensity_map()` and `plot_rgb_intensity_map()` to respect range

---

### Task 8: Intensity Range Selection for B-Field Intensity Map

**Requirements**:
- Manual control over colorbar limits
- Option for auto-scaling (default)
- Separate control for σ+ and σ- in RGB mode

**Implementation**:
- **Covered in Task 2 and Task 3**
- Add intensity range controls in B-Field Intensity Map Window
- Implement `vmin`, `vmax` parameters properly
- For RGB mode: separate controls for each channel

---

### Task 9: Additional Testing & Documentation

**Requirements**:
- Frequent testing after each major feature
- Update documentation as features are completed
- Ensure existing features remain functional

**Testing Checklist**:
1. After Task 0: Test integrate window open/close/reopen cycle 5+ times
2. After Task 1: Test normalized stacked with various B-field selections, verify σ+/σ- overlap correctly
3. After Task 2: Test intensity map window with different colormaps, ranges, and modes
4. After Task 3: Verify RGB colorbars appear/hide correctly, info box displays accurate ranges
5. After Task 4: Test magnetic sweep with variable time ranges, verify parsing
6. After Task 5: Test temperature analysis with both plots visible, test export quality
7. After Task 6: Check/uncheck magnetic checkbox, verify polarization resets
8. Full regression test: Run through ALL existing features to ensure no breakage

**Documentation Updates**:
- After each task completion:
  - Update `docs/IMPLEMENTATION_STATUS.md` with task status
  - Add screenshots/examples to `docs/bfield_polarization_features.md`
  - Update user guide sections in `README.md`
- Final documentation:
  - Create comprehensive usage guide for new B-field features
  - Add troubleshooting section
  - Include example workflows

---

## Technical Notes & Best Practices

### Window State Management
- Use `winfo_exists()` to check if window is valid
- Properly destroy windows when closed (not just withdraw)
- Store window references in app object: `app.bfield_intensity_map_window`

### B-Field Value Selection UI
```python
def _create_bfield_selection_checkboxes(self, parent, bfield_values):
    """Dynamically create checkboxes for each B-field value."""
    self.bfield_check_vars = {}
    
    for i, bfield in enumerate(bfield_values):
        var = ctk.StringVar(value="1")  # Checked by default
        checkbox = ctk.CTkCheckBox(parent, text=f"{bfield:.1f}T", variable=var)
        checkbox.grid(row=i//4, column=i%4, padx=5, pady=2, sticky="w")
        self.bfield_check_vars[bfield] = var
```

### Dual-Polarization Same-Line Plotting
```python
def plot_bfield_stacked_dual_polarization(energies_plus, counts_plus,
                                          energies_minus, counts_minus,
                                          offset, label):
    """Plot σ+ and σ- at same vertical position."""
    ax.plot(energies_plus, counts_plus + offset, color='red', 
            alpha=0.7, label=f"{label} σ+")
    ax.plot(energies_minus, counts_minus + offset, color='blue',
            alpha=0.7, label=f"{label} σ-")
```

### RGB Intensity Map with Separate Colorbars
```python
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

def add_separate_colorbars(figure, sigma_plus_range, sigma_minus_range):
    # Create red gradient for σ+
    cax_plus = figure.add_axes([0.92, 0.55, 0.02, 0.35])
    norm_plus = Normalize(vmin=sigma_plus_range[0], vmax=sigma_plus_range[1])
    cb_plus = ColorbarBase(cax_plus, cmap='Reds', norm=norm_plus)
    cb_plus.set_label('σ+ Intensity', fontsize=9)
    
    # Create blue gradient for σ-
    cax_minus = figure.add_axes([0.92, 0.1, 0.02, 0.35])
    norm_minus = Normalize(vmin=sigma_minus_range[0], vmax=sigma_minus_range[1])
    cb_minus = ColorbarBase(cax_minus, cmap='Blues', norm=norm_minus)
    cb_minus.set_label('σ- Intensity', fontsize=9)
```

---

## File Modification Summary

**New Files to Create**:
1. `pl_analyzer/gui/windows/bfield_intensity_map_window.py` - Dedicated intensity map window

**Files to Modify**:
1. `pl_analyzer/gui/actions/bfield_analysis_actions.py` - Fix window bug, add intensity map action
2. `pl_analyzer/gui/panels/right_panel.py` - Add normalized stacked checkbox, B-field selection UI, intensity map button
3. `pl_analyzer/gui/panels/left_panel.py` - Add magnetic checkbox callback
4. `pl_analyzer/gui/actions/plot_actions.py` - Add B-field normalized stacked function
5. `pl_analyzer/gui/widgets/plot_canvas.py` - Enhance RGB plotting with multi-colorbar
6. `pl_analyzer/gui/dialogs.py` - Add variable time ranges to magnetic sweep dialog
7. `pl_analyzer/gui/temp_analysis_view.py` - Major refactor for dual-plot layout
8. `pl_analyzer/utils/config.py` - Update config structure for time ranges
9. `pl_analyzer/core/file_parser.py` - Use variable time ranges when parsing

**Documentation Files to Update**:
1. `docs/IMPLEMENTATION_STATUS.md`
2. `docs/bfield_polarization_features.md`
3. `docs/PROMPT_FOR_NEXT_AGENT.md`
4. `README.md`

---

## Success Criteria

- [ ] All 10 tasks implemented and tested
- [ ] No regression in existing features
- [ ] All new UI elements responsive and intuitive
- [ ] Publication-quality export functionality working
- [ ] Documentation complete and up-to-date
- [ ] Code properly commented and logged
- [ ] User can complete full workflow: load data → analyze B-field → customize plots → export

---

## Estimated Complexity

- **Task 0** (Bug Fix): 30 min
- **Task 1** (Normalized Stacked): 3 hours
- **Task 2** (Intensity Map Window): 2.5 hours
- **Task 3** (RGB Improvements): 2 hours
- **Task 4** (Variable Time Ranges): 1.5 hours
- **Task 5** (Temp Analysis Enhancement): 2.5 hours
- **Task 6** (Auto-reset): 30 min
- **Tasks 7-8** (Covered in 2-3): 0 hours
- **Task 9** (Testing & Docs): 2 hours
- **Total**: ~14-15 hours

---

## Implementation Sequence

1. **Start with Task 0** (critical bug fix) - get integrate window working reliably
2. **Implement Task 1** (normalized stacked with B-field selection) - high priority, high visibility
3. **Implement Tasks 2-3** together (intensity map window + RGB improvements)
4. **Implement Task 4** (variable integration times)
5. **Implement Task 5** (temperature analysis dual-plot)
6. **Implement Task 6** (checkbox auto-reset) - quick win
7. **Test thoroughly** after each major task
8. **Update documentation** after completing each group of tasks
9. **Final regression testing** across all features
10. **Documentation polish** and screenshot updates

---

## Notes for Implementation

- Maintain code consistency with existing style
- Add comprehensive logging for debugging
- Include try/except blocks with user-friendly error messages
- Test with provided test data in `test_data/bfield_cp/`
- Ensure backward compatibility with existing config files
- All new functions should have detailed docstrings
- Use type hints where appropriate

Good luck with the implementation!

