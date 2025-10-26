# Future Improvements & Bug Tracking

**Last Updated**: October 25, 2025 (Updated: Added Issue #1, removed false positive)  
**Purpose**: Track bugs, feature requests, and enhancement ideas for PL Analyzer

---

## High Priority Bugs

### 1. File Sorting in Dataset: Incorrect Numerical Order
**Status**: 🐛 Bug - Incorrect Sorting  
**Priority**: 🔴 Critical (IMPLEMENT FIRST!)  
**Reported**: October 25, 2025

**Issue**: 
Files in the dataset table are sorted alphabetically as strings instead of numerically by power/temperature values. This causes incorrect ordering where, for example, 0.100uW appears before 0.050uW.

**Current Behavior**:
```
1. PL_0.010uW
2. PL_0.100uW    ← Wrong! Should be #3
3. PL_0.050uW    ← Wrong! Should be #2
```

**Expected Behavior**:
```
1. PL_0.010uW
2. PL_0.050uW
3. PL_0.100uW
```

**Impact**:
- Power series plots show data in wrong order
- Temperature series plots show data in wrong order
- Confusing for users analyzing trends
- May lead to misinterpretation of data

**Root Cause**:
Likely sorting by filename string rather than by numerical power/temperature values extracted from metadata.

**Requirements for Fix**:
- Sort datasets by numerical values (power_uw, temperature_k, bfield_t)
- Primary sort key should depend on series type:
  - Power series: sort by `power_uw` ascending
  - Temperature series: sort by `temperature_k` ascending
  - B-field series: sort by `bfield_t` ascending
- Secondary sort keys for ties
- Apply sorting when displaying in file table
- Apply sorting when plotting series

**Suggested Approach**:
1. **Locate sorting code**:
   - Check `pl_analyzer/gui/widgets/file_table.py` for table population
   - Check `pl_analyzer/gui/actions/plot_actions.py` for series filtering

2. **Implement numerical sorting**:
   ```python
   # Current (wrong):
   datasets.sort(key=lambda d: d.filename)
   
   # Fixed (correct):
   # For power series:
   datasets.sort(key=lambda d: (d.temperature_k, d.power_uw, d.bfield_t or 0))
   
   # For temperature series:
   datasets.sort(key=lambda d: (d.power_uw, d.temperature_k, d.bfield_t or 0))
   
   # For B-field series:
   datasets.sort(key=lambda d: (d.temperature_k, d.power_uw, d.bfield_t or 0))
   ```

3. **Apply consistently**:
   - File table display
   - Power series plots
   - Temperature series plots
   - B-field series plots
   - All analysis functions that iterate through datasets

**Files to Modify**:
- `pl_analyzer/gui/widgets/file_table.py` - Table sorting
- `pl_analyzer/gui/actions/plot_actions.py` - Series filtering and plotting
- `pl_analyzer/gui/actions/analysis_actions.py` - Analysis functions
- Possibly `pl_analyzer/core/data_handler.py` - Add sorting helper method

**Testing Checklist**:
- [ ] Load files with powers: 0.01, 0.1, 0.05, 1, 10 (in any order)
- [ ] Verify table shows: 0.01, 0.05, 0.1, 1, 10
- [ ] Plot power series → verify curves appear in correct order
- [ ] Load files with temps: 5, 50, 10, 100 (in any order)
- [ ] Verify table shows: 5, 10, 50, 100
- [ ] Plot temperature series → verify correct order

---

### 2. G-Factor Analysis: Peak Detection "Maximum" Method
**Status**: ❌ Not Working  
**Priority**: 🔴 High  
**Reported**: October 25, 2025

**Issue**: 
The "maximum" peak detection method in g-factor analysis doesn't work correctly. The energy window with peak maximum is very narrow and changes with magnetic field, making reliable peak detection difficult.

**Current Behavior**:
- Peak detection fails or gives incorrect results
- No adaptation to peak position changes with B-field
- σ+ and σ- peaks shift in opposite directions (Zeeman splitting)
- Energy window remains fixed, missing shifted peaks

**Requirements for Fix**:
- Intelligent energy window adjustment that follows peak shift with B-field
- Opposite direction handling for σ+ and σ- polarizations
- Quality indicator to warn if peak detection was unreliable
- Adaptive window sizing based on peak characteristics
- Visual feedback showing detected peak positions

**Suggested Approach**:
1. Track peak position from previous B-field value as initial guess
2. Use centered sliding window that moves with peak
3. Implement confidence score based on:
   - Peak height relative to background
   - Peak width (reject very narrow or very wide)
   - Distance from expected position
4. Add visual markers on spectra showing detected peaks
5. Provide warning dialog if detection quality is poor

**Files to Modify**:
- `pl_analyzer/core/analysis.py:calculate_g_factor()` - Peak detection logic
- `pl_analyzer/gui/windows/gfactor_analysis_window.py` - UI for quality feedback

---

### 3. G-Factor Analysis: Gaussian Fitting Enhancement
**Status**: ⚠️ Needs Improvement  
**Priority**: 🟡 Medium  
**Reported**: October 25, 2025

**Issue**: 
Gaussian fitting for peak position extraction needs intelligent parameter initialization and constraints. Currently each fit is independent, leading to unreliable results and convergence failures.

**Current Behavior**:
- Each spectrum fitted independently
- No parameter constraints between fits
- No way to fix certain parameters (e.g., width) across series
- No visual validation of fit quality
- Difficult to spot bad fits in batch processing

**Requirements for Fix**:
- Use parameters from previous fit as initial guess for next spectrum
- Option to fix certain parameters (e.g., width, amplitude) across entire B-field series
- Visual validation: show fit overlay for each spectrum in review mode
- Compact display showing multiple fits at once for comparison
- Fit quality metrics (R², residuals) displayed

**Suggested Approach**:
1. **Smart Parameter Initialization**:
   ```python
   if first_fit:
       initial_params = auto_guess(spectrum)
   else:
       initial_params = previous_fit_results
       # Adjust center for expected Zeeman shift
   ```

2. **Parameter Constraints UI**:
   - Checkboxes: [✓] Fix Width  [ ] Fix Amplitude
   - Display: "Width = 0.015 eV (locked across all fits)"

3. **Fit Review Window**:
   - Grid layout: 4x3 showing 12 spectra + fits at once
   - Color-coded by R²: green (>0.95), yellow (0.9-0.95), red (<0.9)
   - Click to zoom, keyboard navigation

4. **Results Table**:
   | B-field | Center (eV) | Width (eV) | Amplitude | R² |
   |---------|-------------|------------|-----------|-----|
   | 0T      | 1.650       | 0.015      | 1000      | 0.98|
   | 1T      | 1.653       | 0.015*     | 980       | 0.96|
   
   (* = fixed parameter)

**Files to Modify**:
- `pl_analyzer/core/analysis.py` - Fitting logic
- `pl_analyzer/gui/windows/gfactor_analysis_window.py` - Add constraints UI and review window

---

### 4. Intensity Map Color Smoothing
**Status**: 📝 Enhancement Request  
**Priority**: 🟢 Low
**Reported**: October 25, 2025

**Issue**: 
Intensity maps show rough, pixelated color transitions between data points. Adding interpolation/smoothing would create more visually appealing and publication-ready figures.

**Current Behavior**:
- matplotlib `imshow()` with default interpolation
- Sharp color boundaries between adjacent pixels
- Looks pixelated when data points are sparse

**Desired Behavior**:
- Smooth color gradients between data points
- User-selectable interpolation method
- Applied consistently across all intensity map types

**Requirements**:
- Smooth color transitions in intensity maps
- Apply to all intensity map views:
  - Power vs Temperature intensity maps
  - B-field intensity maps
  - All polarization modes (single, RGB)
- User-configurable smoothing level
- Should work in "Intensity Map Options" panel AND dedicated B-field intensity map window

**Suggested Approach**:
1. Add interpolation dropdown in UI:
   ```
   Interpolation: [None ▼]
   Options: None, Nearest, Bilinear, Bicubic, Gaussian
   ```

2. Modify `plot_canvas.py` functions:
   ```python
   def plot_intensity_map(..., interpolation='none'):
       im = self.axes.imshow(data, 
                             interpolation=interpolation,  # Add this
                             ...)
   ```

3. Add to all intensity map functions:
   - `plot_intensity_map()`
   - `plot_rgb_intensity_map()`
   - In `bfield_intensity_map_window.py`

4. Store preference in config.json

**Files to Modify**:
- `pl_analyzer/gui/widgets/plot_canvas.py` - Add interpolation parameter
- `pl_analyzer/gui/panels/right_panel.py` - Add dropdown in Intensity Map Options
- `pl_analyzer/gui/windows/bfield_intensity_map_window.py` - Add dropdown
- `pl_analyzer/utils/config.py` - Save/load interpolation preference

**matplotlib Reference**:
```python
interpolation options: 'none', 'nearest', 'bilinear', 'bicubic', 
                       'spline16', 'spline36', 'hanning', 'hamming', 
                       'hermite', 'kaiser', 'quadric', 'catrom', 
                       'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
```

---

### 5. Spike Removal UI: Missing Parameter Input Fields
**Status**: 🐛 Bug - UI Broken  
**Priority**: 🔴 High
**Reported**: October 25, 2025

**Issue**: 
The Spike Removal tab is missing input text boxes for parameters. Labels are visible but entry widgets are not rendering, making the feature completely unusable.

**Screenshot Evidence**:
User reported seeing labels like "Median Window (pts)", "Sigma Threshold", "Max Width (pts)", "Min Prominence", "Replace Method", "Neighbor N" but no corresponding input fields.

**What's Wrong**:
- Parameter labels are visible
- Input fields (CTkEntry widgets) are NOT showing
- Makes spike removal feature unusable
- Buttons exist but parameters can't be set
- Likely happened during GUI refactoring or panel reorganization

**Symptoms**:
- Labels present: ✓
- Input boxes present: ✗
- Buttons present: ✓
- Functionality: Broken

**Requirements for Fix**:
- Restore all parameter input fields in Spike Removal tab
- Verify all CTkEntry widgets are properly created and packed/gridded
- Ensure proper layout (labels aligned with inputs)
- Test that parameter values can be entered and saved
- Verify "Save Spike Settings" and other buttons work correctly
- Test "Load Spike Settings" restores values to input fields

**Investigation Needed**:
1. Check where Spike Removal UI is defined:
   - Likely in `pl_analyzer/gui/panels/right_panel.py`
   - Or possibly in a separate view file

2. Look for missing widget creation:
   ```python
   # Should have something like:
   self.median_window_entry = ctk.CTkEntry(...)
   self.sigma_threshold_entry = ctk.CTkEntry(...)
   # etc.
   ```

3. Check layout code:
   ```python
   # Verify grid/pack calls:
   label.grid(row=0, column=0)
   entry.grid(row=0, column=1)  # Is this missing?
   ```

4. Compare with other tabs:
   - Power Analysis tab (working example)
   - Figure Options tab (working example)

**Files to Check**:
- `pl_analyzer/gui/panels/right_panel.py` - Main suspect
- `pl_analyzer/core/spike_removal.py` - Verify functions exist
- Git history to see what changed recently

**Suggested Fix**:
1. Locate Spike Removal tab creation code
2. Verify all CTkEntry widgets are created:
   ```python
   # Example of what should be there:
   self.spike_median_window = ctk.CTkEntry(spike_tab, width=100)
   self.spike_sigma_threshold = ctk.CTkEntry(spike_tab, width=100)
   # ... etc for all parameters
   ```
3. Ensure proper grid layout:
   ```python
   row = 0
   ctk.CTkLabel(spike_tab, text="Median Window (pts):").grid(row=row, column=0)
   self.spike_median_window.grid(row=row, column=1)
   row += 1
   # ... continue for all params
   ```
4. Test manually entering values
5. Test save/load settings

**Testing Checklist**:
- [ ] All input fields visible
- [ ] Can type values in each field
- [ ] "Detect Spikes" uses entered parameters
- [ ] "Save Spike Settings" preserves values to config
- [ ] "Load Spike Settings" restores values from config
- [ ] Other buttons (Clear Markers, etc.) still work

---

## Medium Priority Enhancements

(No items yet - add here as they come up)

---

## Low Priority / Nice-to-Have

(No items yet - add here as they come up)

---

## Completed Items

### [Date] - [Issue Name]
**Issue**: Description of what was broken  
**Fix**: Description of how it was fixed  
**Commit**: [commit hash]

(Move completed items from above sections here with completion date)

---

## How to Use This File

### Adding New Issues
1. Choose appropriate section (High/Medium/Low priority)
2. Use the template format (Status, Priority, Issue, Requirements, Suggested Approach)
3. Include file paths and code examples where helpful
4. Add screenshots or error messages if relevant

### Updating Status
- ❌ Not Working - Feature broken/doesn't work
- ⚠️ Needs Improvement - Works but could be better
- 📝 Enhancement Request - New feature idea
- 🐛 Bug - Broken functionality
- 🚧 In Progress - Someone is working on it
- ✅ Fixed - Completed, moved to "Completed Items"

### Priority Levels
- 🔴 High - Broken core functionality, blocks users
- 🟡 Medium - Important but has workarounds
- 🟢 Low - Nice-to-have, cosmetic, future ideas

---

**Note**: This file is meant to be living documentation. Update it as issues are discovered, fixed, or priorities change.

