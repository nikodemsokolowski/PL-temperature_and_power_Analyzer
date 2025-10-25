import logging
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np

from pl_analyzer.core import processing
from pl_analyzer.core import spike_removal
from pl_analyzer.utils import config

logger = logging.getLogger(__name__)

def _get_polarization_mode(app):
    """Get current polarization mode from UI."""
    if not hasattr(app, 'polarization_mode_var'):
        return 'all'
    display_value = app.polarization_mode_var.get()
    # Map display name to internal value
    mode_map = {
        "All Data": "all",
        "σ+ Only": "sigma+",
        "σ- Only": "sigma-",
        "Both (σ+&σ-)": "both",
        "Sum (σ++σ-)": "sum"
    }
    return mode_map.get(display_value, 'all')

def _filter_by_polarization(metadata_df, file_ids, polarization_mode):
    """
    Filter file IDs based on polarization mode.
    
    Args:
        metadata_df: DataFrame with metadata including 'polarization' column
        file_ids: List of file IDs to filter
        polarization_mode: 'all', 'sigma+', 'sigma-', 'both', or 'sum'
    
    Returns:
        Filtered list of file IDs
    """
    if polarization_mode == 'all':
        return file_ids
    
    if polarization_mode in ['sigma+', 'sigma-']:
        # Filter for specific polarization
        filtered_ids = []
        for fid in file_ids:
            row = metadata_df[metadata_df['file_id'] == fid]
            if not row.empty:
                pol = row.iloc[0].get('polarization', None)
                if pol == polarization_mode:
                    filtered_ids.append(fid)
        return filtered_ids
    
    elif polarization_mode == 'both':
        # Include both sigma+ and sigma-
        filtered_ids = []
        for fid in file_ids:
            row = metadata_df[metadata_df['file_id'] == fid]
            if not row.empty:
                pol = row.iloc[0].get('polarization', None)
                if pol in ['sigma+', 'sigma-']:
                    filtered_ids.append(fid)
        return filtered_ids
    
    # 'sum' mode is handled separately in each plot function
    return file_ids


def _safe_meta_value(value):
    """Convert NaN-like values to None for stable comparisons."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        return value
    return value


def _format_number(value, precision=3):
    """Format numeric metadata for labels; fall back to string if needed."""
    if value is None:
        return None
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def _format_sum_display(temp, power, bfield):
    """Build a human-friendly identifier for sum mode messages."""
    parts = []
    if temp is not None:
        temp_str = _format_number(temp)
        parts.append(f"T={temp_str}K")
    if power is not None:
        power_str = _format_number(power)
        parts.append(f"P={power_str}uW")
    if bfield is not None:
        bfield_str = _format_number(bfield)
        parts.append(f"B={bfield_str}T")
    return ", ".join(parts) if parts else "Unspecified conditions"


def _sort_key_for_meta(value):
    """Utility for stable sorting of metadata values, handling None gracefully."""
    if value is None:
        return (1, 0)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (0, str(value))


def _gather_common_style_options(app):
    """Collect common matplotlib style options from the UI controls."""
    options = {}
    preset_var = getattr(app, 'style_preset_var', None)
    options['preset'] = preset_var.get() if preset_var else 'Default'
    options['ticks_inside'] = getattr(app, 'ticks_inside_var', None) and app.ticks_inside_var.get() == "1"
    options['minor_ticks'] = getattr(app, 'minor_ticks_var', None) and app.minor_ticks_var.get() == "1"

    def _float_from_entry(entry_name, default):
        entry = getattr(app, entry_name, None)
        if entry and entry.get():
            try:
                return float(entry.get())
            except Exception:
                return default
        return default

    options['maj_len'] = _float_from_entry('maj_tick_len_entry', 4.0)
    options['min_len'] = _float_from_entry('min_tick_len_entry', 2.0)
    options['axes_linewidth'] = _float_from_entry('axes_lw_entry', 1.0)

    show_grid_var = getattr(app, 'show_grid_var', None)
    default_grid = True
    try:
        default_grid = show_grid_var.get() == "1" if show_grid_var is not None else True
    except Exception:
        default_grid = True
    options['show_grid'] = default_grid
    return options


def _get_line_width(app, default=1.2):
    entry = getattr(app, 'line_width_entry', None)
    if entry and entry.get():
        try:
            return float(entry.get())
        except Exception:
            return default
    return default


def _get_stack_norm_range(app):
    """Return normalization range tuple or None for stacked plots."""
    try:
        min_entry = getattr(app, 'norm_min_entry', None)
        max_entry = getattr(app, 'norm_max_entry', None)
        if min_entry and max_entry and min_entry.get() and max_entry.get():
            mn = float(min_entry.get())
            mx = float(max_entry.get())
            if mn < mx:
                return (mn, mx)
    except Exception:
        pass
    return None


def _get_stack_offset(app):
    entry = getattr(app, 'offset_entry', None)
    if entry and entry.get():
        try:
            return float(entry.get())
        except Exception:
            return 1.0
    return 1.0


def _parse_bfield_step(app):
    """Parse user-selected B-field spacing in Tesla."""
    step_var = getattr(app, 'bfield_step_var', None)
    if not step_var:
        return None, "All"
    raw = (step_var.get() or "").strip()
    if not raw:
        return None, "All"
    normalized = raw.lower().strip()
    if normalized in {"all", "every", "any"}:
        return None, raw
    cleaned = (
        normalized.replace("tesla", "")
        .replace("t", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid B-field step interval: '{raw}'") from exc
    if value <= 0:
        raise ValueError("B-field step interval must be greater than zero.")
    return value, raw


def _create_sum_spectra(dataset, metadata_df):
    """
    Create summed spectra (sigma+ + sigma-) for matching temperature/power/b-field conditions.

    Returns:
        sum_entries: list of dicts with keys:
            - energy
            - counts
            - label
            - file_id
            - polarization
            - metadata
            - source_file_ids
        unmatched: list of dicts describing conditions without complete sigma+/sigma- pairs
    """
    base_entries, unmatched = dataset.generate_sum_spectra_pairs(metadata_df)
    if not base_entries:
        return [], unmatched

    sum_entries = []
    for item in base_entries:
        meta = item['metadata']
        temp_val = meta.get('temperature_k')
        power_val = meta.get('power_uw')
        bfield_val = meta.get('bfield_t')

        label_components = []
        temp_fmt = _format_number(temp_val)
        power_fmt = _format_number(power_val)
        bfield_fmt = _format_number(bfield_val)
        if temp_fmt is not None:
            label_components.append(f"T={temp_fmt}K")
        if power_fmt is not None:
            label_components.append(f"P={power_fmt}uW")
        if bfield_fmt is not None:
            label_components.append(f"B={bfield_fmt}T")

        label_body = ", ".join(label_components)
        label = f"Sum ({label_body})" if label_body else "Sum (sigma+ + sigma-)"

        sum_entries.append({
            'energy': item['energy'],
            'counts': item['counts'],
            'label': label,
            'file_id': f"sum::{item['sum_key']}",
            'polarization': 'sum',
            'metadata': meta,
            'source_file_ids': item['source_file_ids']
        })

    sum_entries.sort(key=lambda item: (
        _sort_key_for_meta(item['metadata'].get('temperature_k')),
        _sort_key_for_meta(item['metadata'].get('power_uw')),
        _sort_key_for_meta(item['metadata'].get('bfield_t'))
    ))

    return sum_entries, unmatched


def _aggregate_polarization_channels(common_energy, intensity_grid, y_values, polarizations):
    """
    Aggregate spectra by y-value, separating sigma+/sigma- channels.

    Returns:
        {
            'y_axis': np.ndarray,
            'sigma_plus': np.ndarray,
            'sigma_minus': np.ndarray,
            'unpolarized': np.ndarray,
            'combined': np.ndarray,
            'has_plus': bool,
            'has_minus': bool
        }
    """
    if intensity_grid.size == 0:
        return {
            'y_axis': np.array([], dtype=float),
            'sigma_plus': np.empty((0, common_energy.size)),
            'sigma_minus': np.empty((0, common_energy.size)),
            'unpolarized': np.empty((0, common_energy.size)),
            'combined': np.empty((0, common_energy.size)),
            'has_plus': False,
            'has_minus': False
        }

    y_arr = np.asarray(y_values, dtype=float) if len(y_values) else np.array([], dtype=float)
    pol_list = list(polarizations) if polarizations else [None] * intensity_grid.shape[0]

    def _key_from_value(val, idx):
        if np.isfinite(val):
            return round(float(val), 6)
        return f"row_{idx}"

    rows = []
    key_to_index = {}
    for idx, row_counts in enumerate(intensity_grid):
        pol = pol_list[idx] if idx < len(pol_list) else None
        pol = pol or 'unpolarized'
        y_val = y_arr[idx] if idx < y_arr.size else np.nan
        key = _key_from_value(y_val, idx)
        if key not in key_to_index:
            key_to_index[key] = len(rows)
            rows.append({
                'y': float(y_val) if np.isfinite(y_val) else None,
                'order': len(rows),
                'sigma+': [],
                'sigma-': [],
                'sum': [],
                'other': []
            })
        entry = rows[key_to_index[key]]
        if pol == 'sigma+':
            entry['sigma+'].append(row_counts)
        elif pol == 'sigma-':
            entry['sigma-'].append(row_counts)
        elif pol == 'sum':
            entry['sum'].append(row_counts)
        else:
            entry['other'].append(row_counts)

    if not rows:
        cols = common_energy.size
        return {
            'y_axis': np.array([], dtype=float),
            'sigma_plus': np.empty((0, cols)),
            'sigma_minus': np.empty((0, cols)),
            'unpolarized': np.empty((0, cols)),
            'combined': np.empty((0, cols)),
            'has_plus': False,
            'has_minus': False
        }

    def _combine(vectors):
        if not vectors:
            return None
        if len(vectors) == 1:
            return vectors[0]
        return np.mean(np.stack(vectors, axis=0), axis=0)

    cols = intensity_grid.shape[1]
    plus_rows, minus_rows, other_rows, combined_rows = [], [], [], []
    y_axis = []
    has_plus = False
    has_minus = False

    for order, entry in enumerate(rows):
        plus_vec = _combine(entry['sigma+'])
        minus_vec = _combine(entry['sigma-'])
        sum_vec = _combine(entry['sum'])
        other_vec = _combine(entry['other'])

        if sum_vec is not None:
            if plus_vec is None:
                plus_vec = sum_vec.copy()
            else:
                plus_vec = (plus_vec + sum_vec) / 2.0
            if minus_vec is None:
                minus_vec = sum_vec.copy()
            else:
                minus_vec = (minus_vec + sum_vec) / 2.0

        def _ensure(vec):
            if vec is None:
                return np.zeros(cols, dtype=float)
            return vec

        plus_vec = _ensure(plus_vec)
        minus_vec = _ensure(minus_vec)
        other_vec = _ensure(other_vec)

        available = []
        if np.any(np.isfinite(plus_vec)):
            available.append(plus_vec)
            has_plus = has_plus or np.any(plus_vec > 0)
        if np.any(np.isfinite(minus_vec)):
            available.append(minus_vec)
            has_minus = has_minus or np.any(minus_vec > 0)
        if np.any(np.isfinite(other_vec)):
            available.append(other_vec)

        if not available and sum_vec is not None:
            available.append(sum_vec)

        if available:
            combined_vec = np.mean(np.stack(available, axis=0), axis=0)
        else:
            combined_vec = np.zeros(cols, dtype=float)

        y_val = entry['y']
        if y_val is None or not np.isfinite(y_val):
            y_axis.append(float(order))
        else:
            y_axis.append(float(y_val))

        plus_rows.append(plus_vec)
        minus_rows.append(minus_vec)
        other_rows.append(other_vec)
        combined_rows.append(combined_vec)

    return {
        'y_axis': np.array(y_axis, dtype=float),
        'sigma_plus': np.vstack(plus_rows),
        'sigma_minus': np.vstack(minus_rows),
        'unpolarized': np.vstack(other_rows),
        'combined': np.vstack(combined_rows),
        'has_plus': has_plus,
        'has_minus': has_minus
    }


def _normalize_to_unit(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    result = np.zeros_like(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return result
    min_val = float(np.min(arr[finite_mask]))
    max_val = float(np.max(arr[finite_mask]))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        result[finite_mask] = np.clip(arr[finite_mask], 0.0, np.inf)
        if result.max() > 0:
            result /= result.max()
        return np.clip(result, 0.0, 1.0)
    result[finite_mask] = (arr[finite_mask] - min_val) / (max_val - min_val)
    return np.clip(result, 0.0, 1.0)


def _build_additive_rgb(channels):
    plus_norm = _normalize_to_unit(channels['sigma_plus'])
    minus_norm = _normalize_to_unit(channels['sigma_minus'])
    other_norm = _normalize_to_unit(channels['unpolarized'])

    rgb = np.zeros((*plus_norm.shape, 3), dtype=float)
    rgb[..., 0] = plus_norm  # Red channel
    rgb[..., 2] = minus_norm  # Blue channel
    overlap = np.minimum(plus_norm, minus_norm)
    rgb[..., 1] = np.maximum(other_norm, overlap)  # Green highlights overlap/unpolarized
    return np.clip(rgb, 0.0, 1.0)


def _build_alpha_overlay(channels):
    base = _normalize_to_unit(channels['combined'])
    plus_norm = _normalize_to_unit(channels['sigma_plus'])
    minus_norm = _normalize_to_unit(channels['sigma_minus'])
    other_norm = _normalize_to_unit(channels['unpolarized'])

    rgb = np.repeat(base[..., None], 3, axis=2)
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.6 * plus_norm, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] + 0.6 * minus_norm, 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.6 * other_norm, 0.0, 1.0)
    return rgb

def plot_selected_action(app):
    """Clears the plot and then plots the spectra for the selected rows."""
    logger.info("Plot Selected (clear) button clicked.")
    app._last_plot_data = None  # Clear previous plot data
    add_to_plot_action(app)

def add_to_plot_action(app):
    """Adds the spectra of the selected rows to the current plot."""
    logger.info("Add to Plot button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()

    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select one or more files to plot.")
        return

    # Apply polarization filter
    pol_mode = _get_polarization_mode(app)
    metadata_df = active_dataset.get_metadata()
    filtered_ids = _filter_by_polarization(metadata_df, selected_ids, pol_mode)
    
    if not filtered_ids and pol_mode != 'all':
        messagebox.showinfo("No Data", f"No files match the selected polarization mode: {pol_mode}")
        return

    energies, counts, labels = [], [], []
    file_ids_list = []
    polarizations = []
    sum_pairs = []

    if pol_mode == 'sum':
        metadata_subset = metadata_df[metadata_df['file_id'].isin(filtered_ids)]
        sum_entries, unmatched = _create_sum_spectra(active_dataset, metadata_subset)
        if not sum_entries:
            messagebox.showinfo("No Sum Pairs", "No matching sigma+/sigma- pairs were found for the selected data.")
            return
        if unmatched:
            details = ", ".join(
                _format_sum_display(*entry['conditions']) for entry in unmatched[:3]
            )
            if len(unmatched) > 3:
                details += ", ..."
            messagebox.showwarning(
                "Incomplete Sum Pairs",
                f"Skipped {len(unmatched)} unmatched polarization set(s): {details}"
            )
        for entry in sum_entries:
            energies.append(entry['energy'])
            counts.append(entry['counts'])
            labels.append(entry['label'])
            file_ids_list.append(entry['file_id'])
            polarizations.append(entry['polarization'])
            sum_pairs.append(entry['source_file_ids'])
    else:
        for file_id in filtered_ids:
            spectrum = active_dataset.get_processed_spectrum(file_id)
            metadata = active_dataset.metadata.get(file_id, {})
            if spectrum is not None and not spectrum.empty:
                energies.append(spectrum['energy_ev'].values)
                counts.append(spectrum['counts'].values)
                temp = metadata.get('temperature_k', 'N_A')
                power = metadata.get('power_uw', 'N_A')
                pol = metadata.get('polarization', None)
                polarizations.append(pol)

                # Add polarization to label if applicable
                pol_str = f", {pol}" if pol else ""
                labels.append(f"{metadata.get('filename', file_id)} (T={temp}K, P={power}uW{pol_str})")
                file_ids_list.append(file_id)

    if not energies:
        messagebox.showwarning("Plot Error", "Could not load data for selected files.")
        return

    plot_title = "Selected Spectra"
    if len(filtered_ids) == 1:
        plot_title = labels[0]
    else:
        plot_title = "Selected Spectra"

    # If there's existing data and we're not clearing, append to it
    if app._last_plot_data and app._last_plot_data.get('type') == 'selected':
        energies = app._last_plot_data['energies'] + energies
        counts = app._last_plot_data['counts'] + counts
        labels = app._last_plot_data['labels'] + labels
        file_ids_list = app._last_plot_data.get('file_ids', []) + file_ids_list
        polarizations = app._last_plot_data.get('polarizations', []) + polarizations
        existing_pairs = app._last_plot_data.get('sum_source_pairs', [])
        if sum_pairs or existing_pairs:
            sum_pairs = existing_pairs + sum_pairs
        plot_title = "Selected Spectra" # Reset title for multiple plots
    
    app._last_plot_data = {
        'type': 'selected', 
        'energies': energies, 
        'counts': counts, 
        'labels': labels, 
        'title': plot_title, 
        'file_ids': file_ids_list,
        'polarizations': polarizations
    }
    if sum_pairs:
        app._last_plot_data['sum_source_pairs'] = sum_pairs
    _update_plot_style(app)

def clear_plot_action(app):
    """Clears the plot canvas."""
    logger.debug("Clearing plot via button...")
    app.log_scale_var.set("0")
    app.right_panel.plot_canvas.clear_plot(y_scale='linear')

def _update_plot_style(app):
    """Re-plots the last plotted data with the current style settings."""
    if not app._last_plot_data:
        return

    y_scale = 'log' if app.log_scale_var.get() == "1" else 'linear'
    is_stacked = app.stack_var.get() == "1"
    is_normalized = app.normalize_var.get() == "1"
    is_equalize = getattr(app, 'equalize_var', None) and app.equalize_var.get() == "1"

    energies = [np.array(e, copy=True) for e in app._last_plot_data['energies']]
    counts = [np.array(c, copy=True) for c in app._last_plot_data['counts']]
    labels = app._last_plot_data['labels']
    title = app._last_plot_data['title']
    polarizations = app._last_plot_data.get('polarizations', [])
    plot_mode = app._last_plot_data.get('plot_mode')

    pol_mode = _get_polarization_mode(app)
    custom_colors = None
    if pol_mode == 'both' and polarizations and len(polarizations) == len(counts):
        derived_colors = []
        has_explicit = False
        for pol in polarizations:
            if pol == 'sigma+':
                derived_colors.append('#FF0000')
                has_explicit = True
            elif pol == 'sigma-':
                derived_colors.append('#0000FF')
                has_explicit = True
            else:
                derived_colors.append(None)
        if has_explicit:
            custom_colors = derived_colors

    # Optional plot x-range
    xlim = None
    try:
        if getattr(app, 'plot_min_entry', None) and getattr(app, 'plot_max_entry', None):
            xmin = float(app.plot_min_entry.get()) if app.plot_min_entry.get() else None
            xmax = float(app.plot_max_entry.get()) if app.plot_max_entry.get() else None
            if xmin is not None and xmax is not None:
                xlim = (xmin, xmax)
    except Exception:
        xlim = None

    if plot_mode == 'bfield_norm_stack':
        style_options = _gather_common_style_options(app)
        default_width = app._last_plot_data.get('line_width', 1.6)
        line_width = _get_line_width(app, default=default_width)
        try:
            offset = float(app._last_plot_data.get('offset', 1.0))
        except Exception:
            offset = 1.0
        _render_bfield_norm_stack(
            app,
            energies,
            counts,
            labels,
            title,
            y_scale,
            xlim,
            offset=offset,
            style_options=style_options,
            line_width=line_width
        )
        return

    # Equalize-to-lowest-temperature scaling (temperature series only)
    eq_base_max = None
    if is_equalize:
        last_type = app._last_plot_data.get('type')
        if last_type != 'temp_series':
            messagebox.showinfo("Equalize Not Applicable", "Equalize to lowest T works only for temperature series plots.")
            is_equalize = False  # fall back to standard processing below
        if last_type == 'temp_series':
            try:
                # Prefer dedicated equalize range entries if present, else fall back to normalization range
                eq_min_entry = getattr(app, 'eq_min_entry', None)
                eq_max_entry = getattr(app, 'eq_max_entry', None)
                if eq_min_entry is not None and eq_max_entry is not None:
                    min_e = float(eq_min_entry.get()) if eq_min_entry.get() else None
                    max_e = float(eq_max_entry.get()) if eq_max_entry.get() else None
                else:
                    min_e = float(app.norm_min_entry.get()) if app.norm_min_entry.get() else None
                    max_e = float(app.norm_max_entry.get()) if app.norm_max_entry.get() else None
                eq_range = (min_e, max_e) if min_e is not None and max_e is not None else None
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Input", "Equalize range must be valid numbers.")
                return

            def max_in_range(e_arr, c_arr, erange):
                if erange:
                    mn, mx = erange
                    mask = (e_arr >= mn) & (e_arr <= mx)
                    if not np.any(mask):
                        return np.max(c_arr) if c_arr.size else 0.0
                    return float(np.max(c_arr[mask])) if np.any(c_arr[mask]) else 0.0
                return float(np.max(c_arr)) if c_arr.size else 0.0

            # Baseline is the first spectrum (lowest T) in temp_series
            base_max = max_in_range(energies[0], counts[0], eq_range)
            if base_max <= 0:
                messagebox.showwarning("Equalize Warning", "Baseline (lowest T) intensity is zero in the selected range.")
                base_max = 1.0
            eq_base_max = base_max

            new_counts = []
            new_labels = []
            for i, (e_arr, c_arr, lbl) in enumerate(zip(energies, counts, labels)):
                cur_max = max_in_range(e_arr, c_arr, eq_range)
                # factor = how many times current is higher than baseline
                factor = (cur_max / base_max) if base_max > 0 else 1.0
                scaled = c_arr / (factor if factor > 0 else 1.0)
                new_counts.append(scaled)
                # Annotate scaling factor for non-baseline spectra1
                if i == 0:
                    new_labels.append(lbl)
                else:
                    if factor > 0:
                        mult = 1.0 / factor
                        new_labels.append(f"{lbl} (x {mult:.2f})")
                    else:
                        new_labels.append(lbl)

            counts = new_counts
            labels = new_labels

    # Standard normalization (skip if equalize is active)
    if (is_stacked or is_normalized) and not is_equalize:
        try:
            min_e = float(app.norm_min_entry.get()) if app.norm_min_entry.get() else None
            max_e = float(app.norm_max_entry.get()) if app.norm_max_entry.get() else None
            norm_range = (min_e, max_e) if min_e is not None and max_e is not None else None
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Input", "Normalization range must be valid numbers.")
            return

        processed = [
            processing.normalize_spectrum(pd.DataFrame({'energy_ev': e, 'counts': c}), norm_range)
            for e, c in zip(energies, counts)
        ]
        # Guard against None returns from normalize_spectrum
        counts = [p['counts'].values if p is not None else c for p, c in zip(processed, counts)]

    if is_equalize or is_stacked:
        try:
            offset = float(app.offset_entry.get())
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Input", "Offset must be a valid number.")
            return
        # For equalized plots (not normalized), scale offset by baseline peak so
        # an offset of 1.0 provides similar separation as in normalized stacking.
        effective_offset = offset
        y_label = None
        if is_equalize:
            if eq_base_max is not None and eq_base_max > 0:
                effective_offset = offset * eq_base_max
            y_label = "Intensity (arb. units)"
        # Styling
        color_cycle = getattr(app, 'color_cycle_var', None).get() if getattr(app, 'color_cycle_var', None) else None
        if custom_colors is not None:
            color_cycle = None
        line_width = _get_line_width(app, default=1.2)
        style_options = _gather_common_style_options(app)

        app.right_panel.plot_canvas.plot_stacked_data(
            energies, counts, labels_list=labels, title=title, y_scale=y_scale, offset=effective_offset, y_label=y_label,
            color_cycle=color_cycle, line_width=line_width, xlim=xlim, style_options=style_options, colors=custom_colors
        )
    else:
        color_cycle = getattr(app, 'color_cycle_var', None).get() if getattr(app, 'color_cycle_var', None) else None
        if custom_colors is not None:
            color_cycle = None
        line_width = _get_line_width(app, default=1.2)
        style_options = _gather_common_style_options(app)

        app.right_panel.plot_canvas.plot_data(
            energies, counts, labels_list=labels, title=title, y_scale=y_scale, color_cycle=color_cycle,
            line_width=line_width, xlim=xlim, style_options=style_options,
            colors=custom_colors
        )


def plot_power_series_action(app):
    """Plots all power data for the temperature of the first selected file."""
    logger.info("Plot Power Series button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()

    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select a file from the desired temperature series.")
        return

    metadata_all = active_dataset.get_metadata()
    pol_mode = _get_polarization_mode(app)
    
    try:
        target_temp = metadata_all.loc[metadata_all['file_id'] == selected_ids[0], 'temperature_k'].iloc[0]
        series_df = metadata_all[metadata_all['temperature_k'] == target_temp].sort_values(by='power_uw')
        
        # Apply polarization filter
        series_ids = series_df['file_id'].tolist()
        filtered_ids = _filter_by_polarization(metadata_all, series_ids, pol_mode)
        series_df = series_df[series_df['file_id'].isin(filtered_ids)]
        
        if series_df.empty:
            messagebox.showinfo("No Data", f"No data found for T={target_temp}K with polarization mode: {pol_mode}")
            return
        
        if pol_mode == 'sum':
            sum_entries, unmatched = _create_sum_spectra(active_dataset, series_df)
            if not sum_entries:
                messagebox.showinfo("No Sum Pairs", "No matching sigma+/sigma- pairs were found for this temperature series.")
                return
            if unmatched:
                details = ", ".join(
                    _format_sum_display(*entry['conditions']) for entry in unmatched[:3]
                )
                if len(unmatched) > 3:
                    details += ", ..."
                messagebox.showwarning(
                    "Incomplete Sum Pairs",
                    f"Skipped {len(unmatched)} unmatched polarization set(s): {details}"
                )
            sum_entries.sort(key=lambda item: _sort_key_for_meta(item['metadata'].get('power_uw')))
            energies = [entry['energy'] for entry in sum_entries]
            counts = [entry['counts'] for entry in sum_entries]
            file_ids = [entry['file_id'] for entry in sum_entries]
            polarizations = [entry['polarization'] for entry in sum_entries]
            sum_pairs = [entry['source_file_ids'] for entry in sum_entries]
            y_values = []
            labels = []
            for entry in sum_entries:
                meta = entry['metadata']
                power_val = meta.get('power_uw')
                temp_val = meta.get('temperature_k')
                bfield_val = meta.get('bfield_t')
                power_fmt = _format_number(power_val)
                label = f"P = {power_fmt} uW (sum)" if power_fmt is not None else "P = N/A uW (sum)"
                temp_fmt = _format_number(temp_val)
                bfield_fmt = _format_number(bfield_val)
                extras = []
                if temp_fmt is not None:
                    extras.append(f"T={temp_fmt}K")
                if bfield_fmt is not None:
                    extras.append(f"B={bfield_fmt}T")
                if extras:
                    label = f"{label}, " + ", ".join(extras)
                labels.append(label)
                y_values.append(float(power_val) if power_val is not None else np.nan)
            y_values = np.asarray(y_values, dtype=float)
        else:
            energies, counts, labels = [], [], []
            file_ids = []
            polarizations = []
            sum_pairs = []
            y_values = series_df['power_uw'].values
            for _, row in series_df.iterrows():
                spectrum = active_dataset.get_processed_spectrum(row['file_id'])
                if spectrum is not None and not spectrum.empty:
                    energies.append(spectrum['energy_ev'].values)
                    counts.append(spectrum['counts'].values)
                    pol = row.get('polarization', None)
                    polarizations.append(pol)
                    pol_str = f" ({pol})" if pol else ""
                    labels.append(f"P = {row['power_uw']} uW{pol_str}")
                    file_ids.append(row['file_id'])

        if not energies:
            messagebox.showwarning("Plot Error", f"No data found for T={target_temp}K series.")
            return
        
        for i, c in enumerate(counts):
            logger.debug(f"Power Series - Spectrum {i} raw counts min/max: {np.min(c)}, {np.max(c)}")

        pol_title_str = f" ({pol_mode})" if pol_mode != 'all' else ""
        plot_title = f"Power Series at T = {target_temp:.1f} K{pol_title_str}"
        last_plot = {
            'type': 'power_series', 
            'energies': energies, 
            'counts': counts, 
            'labels': labels, 
            'title': plot_title, 
            'y_values': y_values, 
            'y_label': "Power (uW)", 
            'file_ids': file_ids,
            'polarizations': polarizations
        }
        if pol_mode == 'sum':
            last_plot['sum_source_pairs'] = sum_pairs
        app._last_plot_data = last_plot
        _update_plot_style(app)

    except (IndexError, KeyError):
        messagebox.showerror("Error", "Could not find metadata for the selected file.")

def toggle_intensity_map_action(app):
    """Toggles the intensity map view based on the checkbox state."""
    if app.intensity_map_var.get() == "1":
        logger.info("Intensity map checkbox checked.")
        last_plot = app._last_plot_data
        if not last_plot or last_plot.get('type') not in ['power_series', 'temp_series', 'bfield_series']:
            messagebox.showwarning("No Series Data", "Please plot a power, temperature, or B-field series first.")
            app.intensity_map_var.set("0")
            return

        energies = last_plot['energies']
        counts = last_plot['counts']
        y_values = last_plot['y_values']
        y_label = last_plot['y_label']
        base_title = last_plot['title']
        plot_title = base_title.replace(" Series at ", " Series Intensity Map at ")

        try:
            logger.debug(f"Number of spectra: {len(energies)}")
            
            # Find the intersection of energy ranges
            start_energy = max(e.min() for e in energies)
            end_energy = min(e.max() for e in energies)

            if start_energy >= end_energy:
                messagebox.showerror("Plot Error", "The energy ranges of the selected spectra do not overlap.")
                app.intensity_map_var.set("0")
                return

            logger.debug(f"Common energy range: {start_energy} - {end_energy}")
            common_energy = np.linspace(start_energy, end_energy, 1000)

            resampled_intensities = []
            for e, c in zip(energies, counts):
                # Sort by energy to ensure it's monotonically increasing for interpolation
                sort_indices = np.argsort(e)
                e_sorted = e[sort_indices]
                c_sorted = c[sort_indices]
                if getattr(app, 'remove_background_var', None) and app.remove_background_var.get() == "1":
                    c_sorted = processing.subtract_background(c_sorted)
                resampled_intensities.append(np.interp(common_energy, e_sorted, c_sorted))
            intensity_grid = np.array(resampled_intensities)
            logger.debug(f"Intensity grid shape: {intensity_grid.shape}")
            logger.debug(f"Intensity grid min/max: {intensity_grid.min()} / {intensity_grid.max()}")

            energy_mesh, y_mesh = np.meshgrid(common_energy, y_values)

            xlim = None
            intensity_grid_to_plot = intensity_grid.copy()
            
            try:
                vmin = float(app.im_min_intensity_entry.get()) if app.im_min_intensity_entry.get() else None
                vmax = float(app.im_max_intensity_entry.get()) if app.im_max_intensity_entry.get() else None
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Input", "Intensity range must be valid numbers.")
                return

            if app.im_normalize_var.get() == "1":
                logger.debug("Normalization is enabled.")
                try:
                    min_e = float(app.im_norm_min_entry.get()) if app.im_norm_min_entry.get() else common_energy.min()
                    max_e = float(app.im_norm_max_entry.get()) if app.im_norm_max_entry.get() else common_energy.max()
                    xlim = (min_e, max_e)
                    logger.debug(f"Normalization range: {min_e} - {max_e}")

                    norm_indices = (common_energy >= min_e) & (common_energy <= max_e)
                    if np.any(norm_indices):
                        
                        for i in range(intensity_grid_to_plot.shape[0]):
                            max_val = intensity_grid_to_plot[i, norm_indices].max()
                            if max_val > 0:
                                intensity_grid_to_plot[i, :] = intensity_grid_to_plot[i, :] / max_val
                        logger.debug("Intensity grid normalized.")
                            
                except (ValueError, TypeError):
                    messagebox.showerror("Invalid Input", "Normalization range must be valid numbers.")
                    return

            polarizations = last_plot.get('polarizations', [])
            map_mode_var = getattr(app, 'intensity_map_mode_var', None)
            map_mode = map_mode_var.get() if map_mode_var else "Single (grayscale)"
            style_options = {'show_grid': app.show_grid_var.get() == "1"}

            if map_mode == "Single (grayscale)" or not polarizations:
                if vmin is None:
                    vmin = intensity_grid_to_plot.min()
                    logger.debug(f"vmin calculated: {vmin}")
                if vmax is None:
                    vmax = intensity_grid_to_plot.max()
                    logger.debug(f"vmax calculated: {vmax}")

                app.right_panel.plot_canvas.plot_intensity_map(
                    energy_mesh, y_mesh, intensity_grid_to_plot,
                    xlabel="Energy (eV)",
                    ylabel=y_label,
                    title=plot_title,
                    log_c=app.im_log_c_var.get() == "1",
                    log_y=app.im_log_y_var.get() == "1",
                    cmap=app.im_colormap_var.get(),
                    vmin=vmin,
                    vmax=vmax,
                    xlim=xlim
                )
            else:
                channels = _aggregate_polarization_channels(common_energy, intensity_grid_to_plot, y_values, polarizations)
                if channels['y_axis'].size == 0:
                    messagebox.showwarning("No Data", "Could not assemble data for the selected intensity map mode.")
                    app.intensity_map_var.set("0")
                    return

                y_axis = channels['y_axis']
                log_y = app.im_log_y_var.get() == "1"

                if map_mode.startswith("Additive"):
                    rgb = _build_additive_rgb(channels)
                    app.right_panel.plot_canvas.plot_rgb_intensity_map(
                        common_energy,
                        y_axis,
                        rgb,
                        xlabel="Energy (eV)",
                        ylabel=y_label,
                        title=f"{plot_title} (Additive RGB)",
                        xlim=xlim,
                        log_y=log_y,
                        style_options=style_options
                    )
                elif map_mode.startswith("Alpha"):
                    rgb = _build_alpha_overlay(channels)
                    app.right_panel.plot_canvas.plot_rgb_intensity_map(
                        common_energy,
                        y_axis,
                        rgb,
                        xlabel="Energy (eV)",
                        ylabel=y_label,
                        title=f"{plot_title} (Alpha Overlay)",
                        xlim=xlim,
                        log_y=log_y,
                        style_options=style_options
                    )
                elif map_mode.startswith("Diverging"):
                    if not (channels['has_plus'] and channels['has_minus']):
                        messagebox.showinfo(
                            "Insufficient Data",
                            "Diverging map requires both sigma+ and sigma- spectra. Falling back to grayscale."
                        )
                        if map_mode_var:
                            map_mode_var.set("Single (grayscale)")
                        toggle_intensity_map_action(app)
                        return
                    diff_grid = channels['sigma_plus'] - channels['sigma_minus']
                    energy_mesh_div, y_mesh_div = np.meshgrid(common_energy, y_axis)
                    max_abs = float(np.nanmax(np.abs(diff_grid))) if diff_grid.size else 1.0
                    if not np.isfinite(max_abs) or max_abs <= 0:
                        max_abs = 1.0
                    app.right_panel.plot_canvas.plot_intensity_map(
                        energy_mesh_div,
                        y_mesh_div,
                        diff_grid,
                        xlabel="Energy (eV)",
                        ylabel=y_label,
                        title=f"{plot_title} Delta(sigma+ - sigma-)",
                        log_c=False,
                        log_y=log_y,
                        cmap='coolwarm',
                        vmin=-max_abs,
                        vmax=max_abs,
                        xlim=xlim
                    )
                else:
                    # Unknown mode; fallback to single-channel rendering
                    if map_mode_var:
                        map_mode_var.set("Single (grayscale)")
                    toggle_intensity_map_action(app)
                    return
        except Exception as e:
            logger.error(f"Failed to create or plot intensity map: {e}", exc_info=True)
            messagebox.showerror("Plot Error", f"An error occurred while generating the intensity map: {e}")
            app.intensity_map_var.set("0")
    else:
        logger.info("Intensity map checkbox unchecked.")
        _update_plot_style(app)


def intensity_map_mode_changed(app, *_):
    """Re-render intensity map if the mode changes while active."""
    if getattr(app, 'intensity_map_var', None) and app.intensity_map_var.get() == "1":
        toggle_intensity_map_action(app)


def bfield_stack_toggle_action(app):
    """Ensure standard stacking is disabled when the B-field stacked option is enabled."""
    if not hasattr(app, 'bfield_stack_var'):
        return
    try:
        is_enabled = app.bfield_stack_var.get() == "1"
    except Exception:
        return
    if is_enabled and hasattr(app, 'stack_var'):
        try:
            app.stack_var.set("0")
        except Exception:
            pass


def plot_bfield_series_action(app):
    """Plot spectra as a function of magnetic field for the selected series."""
    logger.info("Plot B-Field Series button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()
    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select a file from the desired B-field series.")
        return

    metadata_all = active_dataset.get_metadata()
    if metadata_all is None or metadata_all.empty:
        messagebox.showwarning("No Data", "No metadata available for the active dataset.")
        return

    pol_mode = _get_polarization_mode(app)

    try:
        target_row = metadata_all.loc[metadata_all['file_id'] == selected_ids[0]].iloc[0]
    except (KeyError, IndexError):
        messagebox.showerror("Error", "Could not find metadata for the selected file.")
        return

    series_df = metadata_all.dropna(subset=['bfield_t']).copy()
    if series_df.empty:
        messagebox.showinfo("No Data", "No spectra with magnetic field information were found.")
        return

    def _filter_value(df, column, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return df
        if column not in df.columns:
            return df
        mask = df[column].notna() & np.isclose(df[column].astype(float), float(value), atol=1e-6)
        return df[mask]

    series_df = _filter_value(series_df, 'temperature_k', target_row.get('temperature_k'))
    series_df = _filter_value(series_df, 'power_uw', target_row.get('power_uw'))
    series_df_all = series_df.copy()

    series_ids = series_df['file_id'].tolist()
    filtered_ids = _filter_by_polarization(metadata_all, series_ids, pol_mode)
    series_df = series_df[series_df['file_id'].isin(filtered_ids)]

    if series_df.empty:
        messagebox.showinfo("No Data", f"No data found for the selected criteria with polarization mode: {pol_mode}")
        return

    series_df = series_df.sort_values(by='bfield_t')

    if hasattr(app, 'right_panel') and hasattr(app.right_panel, 'update_bfield_selection'):
        try:
            available_vals = series_df_all['bfield_t'].dropna().astype(float).tolist()
        except Exception:
            available_vals = []
        app.right_panel.update_bfield_selection(available_vals)

    if getattr(app, 'bfield_stack_var', None) and app.bfield_stack_var.get() == "1":
        handled = plot_bfield_normalized_stacked(app, active_dataset, series_df_all, target_row)
        if handled:
            return

    if pol_mode == 'sum':
        sum_entries, unmatched = _create_sum_spectra(active_dataset, series_df)
        if not sum_entries:
            messagebox.showinfo("No Sum Pairs", "No matching sigma+/sigma- pairs were found for this B-field series.")
            return
        if unmatched:
            details = ", ".join(
                _format_sum_display(*entry['conditions']) for entry in unmatched[:3]
            )
            if len(unmatched) > 3:
                details += ", ..."
            messagebox.showwarning(
                "Incomplete Sum Pairs",
                f"Skipped {len(unmatched)} unmatched polarization set(s): {details}"
            )
        sum_entries.sort(key=lambda item: _sort_key_for_meta(item['metadata'].get('bfield_t')))
        energies = [entry['energy'] for entry in sum_entries]
        counts = [entry['counts'] for entry in sum_entries]
        file_ids = [entry['file_id'] for entry in sum_entries]
        polarizations = [entry['polarization'] for entry in sum_entries]
        sum_pairs = [entry['source_file_ids'] for entry in sum_entries]
        labels = []
        y_values = []
        for entry in sum_entries:
            meta = entry['metadata']
            bfield_val = meta.get('bfield_t')
            y_values.append(float(bfield_val) if bfield_val is not None else np.nan)
            b_str = _format_number(bfield_val)
            label = f"B = {b_str} T (sum)" if b_str is not None else "B = N/A T (sum)"
            temp_fmt = _format_number(meta.get('temperature_k'))
            power_fmt = _format_number(meta.get('power_uw'))
            extras = []
            if temp_fmt is not None:
                extras.append(f"T={temp_fmt}K")
            if power_fmt is not None:
                extras.append(f"P={power_fmt}uW")
            if extras:
                label = f"{label}, " + ", ".join(extras)
            labels.append(label)
        y_values = np.asarray(y_values, dtype=float)
    else:
        energies, counts, labels, file_ids, polarizations, y_list = [], [], [], [], [], []
        for _, row in series_df.iterrows():
            spectrum = active_dataset.get_processed_spectrum(row['file_id'])
            if spectrum is None or spectrum.empty:
                continue
            energies.append(spectrum['energy_ev'].values)
            counts.append(spectrum['counts'].values)
            pol = row.get('polarization', None)
            polarizations.append(pol)
            bfield_val = row.get('bfield_t')
            y_list.append(float(bfield_val) if bfield_val is not None else np.nan)
            b_str = _format_number(bfield_val)
            pol_str = f" ({pol})" if pol else ""
            label = f"B = {b_str} T{pol_str}" if b_str is not None else f"B = N/A T{pol_str}"
            labels.append(label)
            file_ids.append(row['file_id'])
        if not energies:
            messagebox.showwarning("Plot Error", "Could not load spectra for the selected B-field series.")
            return
        y_values = np.asarray(y_list, dtype=float)
        sum_pairs = []

    title_parts = []
    temp_val = target_row.get('temperature_k')
    power_val = target_row.get('power_uw')
    if temp_val is not None and not (isinstance(temp_val, float) and np.isnan(temp_val)):
        title_parts.append(f"T = {_format_number(temp_val)} K")
    if power_val is not None and not (isinstance(power_val, float) and np.isnan(power_val)):
        title_parts.append(f"P = {_format_number(power_val)} uW")
    plot_title = "B-Field Series"
    if title_parts:
        plot_title += " at " + ", ".join(title_parts)
    if pol_mode != 'all':
        plot_title += f" ({pol_mode})"

    last_plot = {
        'type': 'bfield_series',
        'energies': energies,
        'counts': counts,
        'labels': labels,
        'title': plot_title,
        'y_values': y_values,
        'y_label': "Magnetic Field (T)",
        'file_ids': file_ids,
        'polarizations': polarizations
    }
    if pol_mode == 'sum' and sum_pairs:
        last_plot['sum_source_pairs'] = sum_pairs

    app._last_plot_data = last_plot
    _update_plot_style(app)


def plot_bfield_normalized_stacked(app, dataset, series_df_all, anchor_row):
    """Create a normalized stacked B-field plot with sigma+/sigma- overlay."""
    logger.info("Preparing B-field normalized stacked plot.")
    if series_df_all is None or series_df_all.empty:
        logger.info("No B-field data available to build the stacked plot.")
        return False

    pair_lookup = {}
    for _, row in series_df_all.iterrows():
        bfield_val = row.get('bfield_t')
        pol = row.get('polarization')
        if pd.isna(bfield_val) or pol not in ('sigma+', 'sigma-'):
            continue
        try:
            bfield_float = float(bfield_val)
        except Exception:
            continue
        key = round(bfield_float, 6)
        entry = pair_lookup.setdefault(key, {'bfield': bfield_float, 'sigma+': None, 'sigma-': None})
        file_id = row.get('file_id')
        if pol == 'sigma+' and entry['sigma+'] is None:
            entry['sigma+'] = file_id
        elif pol == 'sigma-' and entry['sigma-'] is None:
            entry['sigma-'] = file_id

    valid_pairs = [entry for entry in pair_lookup.values() if entry['sigma+'] and entry['sigma-']]
    if not valid_pairs:
        messagebox.showinfo(
            "No Polarization Pairs",
            "No matching sigma+/sigma- pairs were found for this B-field series."
        )
        return False

    valid_pairs.sort(key=lambda item: item['bfield'])

    selected_pairs = valid_pairs.copy()
    selected_values = getattr(app, '_bfield_stack_selected', None)
    if selected_values is not None:
        if not selected_values:
            messagebox.showinfo(
                "No B-Field Selected",
                "Select at least one B-field value to plot."
            )
            return True
        selection_set = {round(val, 6) for val in selected_values}
        if selection_set and len(selection_set) < len(valid_pairs):
            filtered_pairs = [entry for entry in valid_pairs if round(entry['bfield'], 6) in selection_set]
            if filtered_pairs:
                selected_pairs = filtered_pairs
            else:
                messagebox.showinfo(
                    "No B-Field Selected",
                    "Selected B-field values are not available for this series."
                )
                return True

    if not selected_pairs:
        messagebox.showinfo(
            "No B-Field Selected",
            "Select at least one B-field value in the stacked plot options."
        )
        return True

    step_display_str = None
    try:
        step_value, step_label = _parse_bfield_step(app)
    except ValueError as exc:
        messagebox.showerror("Step Interval", str(exc))
        try:
            app.bfield_step_var.set("All")
        except Exception:
            pass
        return True

    if step_value:
        filtered_pairs = []
        last_bfield = None
        for entry in selected_pairs:
            if last_bfield is None or (entry['bfield'] - last_bfield) >= (step_value - 1e-9):
                filtered_pairs.append(entry)
                last_bfield = entry['bfield']
        if not filtered_pairs:
            display = step_label or f"{step_value:g}"
            if display and not display.lower().endswith("t"):
                display = f"{display}T"
            messagebox.showinfo(
                "No B-Field Remaining",
                f"No B-field values remain after applying the {display} interval."
            )
            return True
        selected_pairs = filtered_pairs

    norm_range = _get_stack_norm_range(app)

    energies, counts, labels = [], [], []
    colors, alphas, stack_positions = [], [], []
    file_ids, polarizations = [], []
    source_pairs = []
    skipped_pairs = []
    selected_bfields = []
    stack_index = 0

    plus_color = '#d62728'
    minus_color = '#1f77b4'
    alpha_value = 0.7

    for entry in selected_pairs:
        plus_id = entry['sigma+']
        minus_id = entry['sigma-']
        plus_spec = dataset.get_processed_spectrum(plus_id) if plus_id else None
        minus_spec = dataset.get_processed_spectrum(minus_id) if minus_id else None
        if plus_spec is None or plus_spec.empty or minus_spec is None or minus_spec.empty:
            skipped_pairs.append(entry['bfield'])
            continue

        try:
            plus_df = plus_spec[['energy_ev', 'counts']].copy()
            plus_norm = processing.normalize_spectrum(plus_df, norm_range)
            if plus_norm is not None and not plus_norm.empty:
                energy_plus = plus_norm['energy_ev'].to_numpy(dtype=float)
                counts_plus = plus_norm['counts'].to_numpy(dtype=float)
            else:
                energy_plus = plus_spec['energy_ev'].to_numpy(dtype=float)
                counts_plus = plus_spec['counts'].to_numpy(dtype=float)

            minus_df = minus_spec[['energy_ev', 'counts']].copy()
            minus_norm = processing.normalize_spectrum(minus_df, norm_range)
            if minus_norm is not None and not minus_norm.empty:
                energy_minus = minus_norm['energy_ev'].to_numpy(dtype=float)
                counts_minus = minus_norm['counts'].to_numpy(dtype=float)
            else:
                energy_minus = minus_spec['energy_ev'].to_numpy(dtype=float)
                counts_minus = minus_spec['counts'].to_numpy(dtype=float)
        except Exception as exc:
            logger.error(f"Failed to normalize spectra for B={entry['bfield']}T: {exc}", exc_info=True)
            skipped_pairs.append(entry['bfield'])
            continue

        bfield_label = _format_number(entry['bfield'])
        label_plus = f"B = {bfield_label} T (sigma+)" if bfield_label else "B = ? T (sigma+)"
        label_minus = f"B = {bfield_label} T (sigma-)" if bfield_label else "B = ? T (sigma-)"

        energies.append(energy_plus)
        counts.append(counts_plus)
        labels.append(label_plus)
        colors.append(plus_color)
        alphas.append(alpha_value)
        stack_positions.append(stack_index)
        file_ids.append(plus_id)
        polarizations.append('sigma+')

        energies.append(energy_minus)
        counts.append(counts_minus)
        labels.append(label_minus)
        colors.append(minus_color)
        alphas.append(alpha_value)
        stack_positions.append(stack_index)
        file_ids.append(minus_id)
        polarizations.append('sigma-')

        source_pairs.append({
            'bfield': entry['bfield'],
            'sigma_plus_id': plus_id,
            'sigma_minus_id': minus_id
        })
        selected_bfields.append(entry['bfield'])
        stack_index += 1

    if not energies:
        if skipped_pairs:
            details = ", ".join(f"{_format_number(val)}T" for val in skipped_pairs[:3])
            if len(skipped_pairs) > 3:
                details += ", ..."
            messagebox.showwarning(
                "No Usable Pairs",
                f"All selected B-field pairs were skipped due to missing spectra: {details}"
            )
        return False

    if skipped_pairs:
        details = ", ".join(f"{_format_number(val)}T" for val in skipped_pairs[:3])
        if len(skipped_pairs) > 3:
            details += ", ..."
        messagebox.showwarning(
            "Incomplete Pairs",
            f"Skipped {len(skipped_pairs)} B-field pair(s) missing spectra: {details}"
        )

    offset = _get_stack_offset(app)
    line_width = _get_line_width(app, default=1.6)

    title_segments = ["B-Field Normalized Stacked"]
    if anchor_row is not None:
        temp_val = anchor_row.get('temperature_k')
        power_val = anchor_row.get('power_uw')
        extras = []
        if temp_val is not None and not (isinstance(temp_val, float) and np.isnan(temp_val)):
            extras.append(f"T = {_format_number(temp_val)} K")
        if power_val is not None and not (isinstance(power_val, float) and np.isnan(power_val)):
            extras.append(f"P = {_format_number(power_val)} uW")
        if extras:
            title_segments.append("at " + ", ".join(extras))
    if step_value:
        display = step_label if isinstance(step_label, str) and step_label else f"{step_value:g}"
        display_str = str(display)
        if not display_str.lower().endswith("t"):
            display_str = f"{display_str}T"
        step_display_str = display_str
        title_segments.append(f"(every {display_str})")
    plot_title = " ".join(title_segments)

    app._last_plot_data = {
        'type': 'bfield_norm_stack',
        'plot_mode': 'bfield_norm_stack',
        'energies': energies,
        'counts': counts,
        'labels': labels,
        'title': plot_title,
        'offset': offset,
        'stack_positions': stack_positions,
        'colors': colors,
        'alphas': alphas,
        'y_label': "Normalized Intensity (arb. units)",
        'file_ids': file_ids,
        'polarizations': polarizations,
        'line_width': line_width,
        'bfield_values': selected_bfields,
        'bfield_pairs': source_pairs,
        'norm_range': norm_range,
        'step_interval_t': step_value,
        'step_interval_display': step_display_str
    }

    _update_plot_style(app)
    return True


def _render_bfield_norm_stack(app, energies, counts, labels, title, y_scale, xlim, offset, style_options, line_width):
    """Render the B-field stacked plot using stored overlay metadata."""
    plot_data = app._last_plot_data or {}
    stack_positions = plot_data.get('stack_positions')
    colors = plot_data.get('colors')
    alphas = plot_data.get('alphas')
    y_label = plot_data.get('y_label', "Normalized Intensity (arb. units)")

    color_cycle = None
    if colors is None:
        color_cycle = getattr(app, 'color_cycle_var', None).get() if getattr(app, 'color_cycle_var', None) else None

    try:
        app.right_panel.plot_canvas.plot_stacked_data(
            energies,
            counts,
            labels_list=labels,
            title=title,
            y_scale=y_scale,
            offset=offset,
            y_label=y_label,
            color_cycle=color_cycle,
            line_width=line_width,
            xlim=xlim,
            style_options=style_options,
            colors=colors,
            stack_positions=stack_positions,
            alphas=alphas
        )
    except Exception as exc:
        logger.error(f"Failed to render B-field stacked plot: {exc}", exc_info=True)
        messagebox.showerror("Plot Error", f"Could not render B-field stacked plot.\nDetails: {exc}")


def plot_temp_series_action(app):
    """Plots all temperature data for the power of the first selected file."""
    logger.info("Plot Temp Series button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()

    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select a file from the desired power series.")
        return

    metadata_all = active_dataset.get_metadata()
    pol_mode = _get_polarization_mode(app)
    
    try:
        target_power = metadata_all.loc[metadata_all['file_id'] == selected_ids[0], 'power_uw'].iloc[0]
        series_df = metadata_all[metadata_all['power_uw'] == target_power].sort_values(by='temperature_k')

        # Apply polarization filter
        series_ids = series_df['file_id'].tolist()
        filtered_ids = _filter_by_polarization(metadata_all, series_ids, pol_mode)
        series_df = series_df[series_df['file_id'].isin(filtered_ids)]
        
        if series_df.empty:
            messagebox.showinfo("No Data", f"No data found for P={target_power}uW with polarization mode: {pol_mode}")
            return

        if pol_mode == 'sum':
            sum_entries, unmatched = _create_sum_spectra(active_dataset, series_df)
            if not sum_entries:
                messagebox.showinfo("No Sum Pairs", "No matching sigma+/sigma- pairs were found for this power series.")
                return
            if unmatched:
                details = ", ".join(
                    _format_sum_display(*entry['conditions']) for entry in unmatched[:3]
                )
                if len(unmatched) > 3:
                    details += ", ..."
                messagebox.showwarning(
                    "Incomplete Sum Pairs",
                    f"Skipped {len(unmatched)} unmatched polarization set(s): {details}"
                )
            sum_entries.sort(key=lambda item: _sort_key_for_meta(item['metadata'].get('temperature_k')))
            energies = [entry['energy'] for entry in sum_entries]
            counts = [entry['counts'] for entry in sum_entries]
            file_ids = [entry['file_id'] for entry in sum_entries]
            polarizations = [entry['polarization'] for entry in sum_entries]
            sum_pairs = [entry['source_file_ids'] for entry in sum_entries]
            y_values = []
            labels = []
            for entry in sum_entries:
                meta = entry['metadata']
                temp_val = meta.get('temperature_k')
                power_val = meta.get('power_uw')
                bfield_val = meta.get('bfield_t')
                temp_fmt = _format_number(temp_val)
                label = f"T = {temp_fmt} K (sum)" if temp_fmt is not None else "T = N/A K (sum)"
                power_fmt = _format_number(power_val)
                bfield_fmt = _format_number(bfield_val)
                extras = []
                if power_fmt is not None:
                    extras.append(f"P={power_fmt}uW")
                if bfield_fmt is not None:
                    extras.append(f"B={bfield_fmt}T")
                if extras:
                    label = f"{label}, " + ", ".join(extras)
                labels.append(label)
                y_values.append(float(temp_val) if temp_val is not None else np.nan)
            y_values = np.asarray(y_values, dtype=float)
        else:
            energies, counts, labels = [], [], []
            file_ids = []
            polarizations = []
            sum_pairs = []
            y_values = series_df['temperature_k'].values
            for _, row in series_df.iterrows():
                spectrum = active_dataset.get_processed_spectrum(row['file_id'])
                if spectrum is not None and not spectrum.empty:
                    energies.append(spectrum['energy_ev'].values)
                    counts.append(spectrum['counts'].values)
                    pol = row.get('polarization', None)
                    polarizations.append(pol)
                    pol_str = f" ({pol})" if pol else ""
                    labels.append(f"T = {row['temperature_k']} K{pol_str}")
                    file_ids.append(row['file_id'])

        if not energies:
            messagebox.showwarning("Plot Error", f"No data found for P={target_power}uW series.")
            return

        for i, c in enumerate(counts):
            logger.debug(f"Temp Series - Spectrum {i} raw counts min/max: {np.min(c)}, {np.max(c)}")

        pol_title_str = f" ({pol_mode})" if pol_mode != 'all' else ""
        plot_title = f"Temperature Series at P = {target_power:.1f} uW{pol_title_str}"
        last_plot = {
            'type': 'temp_series', 
            'energies': energies, 
            'counts': counts, 
            'labels': labels, 
            'title': plot_title, 
            'y_values': y_values, 
            'y_label': "Temperature (K)", 
            'file_ids': file_ids,
            'polarizations': polarizations
        }
        if pol_mode == 'sum':
            last_plot['sum_source_pairs'] = sum_pairs
        app._last_plot_data = last_plot
        _update_plot_style(app)

    except (IndexError, KeyError):
        messagebox.showerror("Error", "Could not find metadata for the selected file.")

def export_current_plot_action(app):
    """Exports the currently displayed plot to an image/vector file."""
    try:
        filetypes = [
            ("PNG Image", "*.png"),
            ("PDF Document", "*.pdf"),
            ("SVG Vector", "*.svg"),
            ("All Files", "*.*")
        ]
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes, title="Save Plot As")
        if not path:
            return
        try:
            dpi = int(float(app.fig_dpi_entry.get())) if getattr(app, 'fig_dpi_entry', None) and app.fig_dpi_entry.get() else None
        except Exception:
            dpi = None
        transparent = getattr(app, 'export_transparent_var', None) and app.export_transparent_var.get() == "1"
        ok = app.right_panel.plot_canvas.save_current_figure(path, dpi=dpi, transparent=transparent)
        if ok:
            messagebox.showinfo("Export", f"Plot saved to:\n{path}")
        else:
            messagebox.showerror("Export Error", "Failed to save the current plot.")
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        messagebox.showerror("Export Error", f"An error occurred: {e}")

def apply_theme_preset(app, theme_name: str):
    """Applies a one-click theme by filling figure option controls."""
    try:
        theme = (theme_name or '').lower()
        # Defaults
        presets = {
            'nature (dense)': {
                'font_size': '9', 'tick_font': '8', 'axes_lw': '1.0',
                'maj': '4.0', 'min': '2.0', 'wspace': '0.45', 'hspace': '0.45',
                'legend_mode': 'outside-right', 'legend_font': '8', 'legend_ncol': '1',
                'color': 'viridis', 'columns': '4'
            },
            'aps (two‑col)': {
                'font_size': '8', 'tick_font': '7', 'axes_lw': '1.0',
                'maj': '4.0', 'min': '2.0', 'wspace': '0.40', 'hspace': '0.40',
                'legend_mode': 'last-only', 'legend_font': '7', 'legend_ncol': '1',
                'color': 'tab10', 'columns': '4'
            },
            'acs (single‑col)': {
                'font_size': '10', 'tick_font': '9', 'axes_lw': '1.2',
                'maj': '5.0', 'min': '2.5', 'wspace': '0.50', 'hspace': '0.50',
                'legend_mode': 'outside-right', 'legend_font': '9', 'legend_ncol': '1',
                'color': 'magma', 'columns': '3'
            }
        }
        p = presets.get(theme, None)
        if not p:
            return
        # Fill entries
        app.font_size_entry.delete(0,'end'); app.font_size_entry.insert(0, p['font_size'])
        app.tick_font_size_entry.delete(0,'end'); app.tick_font_size_entry.insert(0, p['tick_font'])
        app.axes_lw_entry.delete(0,'end'); app.axes_lw_entry.insert(0, p['axes_lw'])
        app.maj_tick_len_entry.delete(0,'end'); app.maj_tick_len_entry.insert(0, p['maj'])
        app.min_tick_len_entry.delete(0,'end'); app.min_tick_len_entry.insert(0, p['min'])
        app.grid_wspace_entry.delete(0,'end'); app.grid_wspace_entry.insert(0, p['wspace'])
        app.grid_hspace_entry.delete(0,'end'); app.grid_hspace_entry.insert(0, p['hspace'])
        app.legend_mode_var.set(p['legend_mode'])
        app.legend_font_size_entry.delete(0,'end'); app.legend_font_size_entry.insert(0, p['legend_font'])
        app.legend_ncol_entry.delete(0,'end'); app.legend_ncol_entry.insert(0, p['legend_ncol'])
        app.color_cycle_var.set(p['color'])
        app.fig_cols_entry.delete(0,'end'); app.fig_cols_entry.insert(0, p['columns'])
        # Useful defaults
        app.ticks_inside_var.set('1'); app.minor_ticks_var.set('1')
        app.style_preset_var.set('Compact')
    except Exception as e:
        logger.warning(f"Failed to apply theme: {e}")

def save_figure_settings_action(app):
    """Save current figure options to a JSON file."""
    try:
        import json
        filetypes = [("JSON", "*.json"), ("All Files", "*.*")]
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=filetypes, title="Save Figure Settings")
        if not path:
            return
        d = {
            'fig_cols': app.fig_cols_entry.get(),
            'fig_w': app.fig_width_entry.get(),
            'fig_h': app.fig_height_entry.get(),
            'dpi': app.fig_dpi_entry.get(),
            'font_size': app.font_size_entry.get(),
            'line_width': app.line_width_entry.get(),
            'color_cycle': app.color_cycle_var.get(),
            'legend_mode': app.legend_mode_var.get(),
            'legend_font': app.legend_font_size_entry.get(),
            'legend_ncol': app.legend_ncol_entry.get(),
            'x_label': app.x_label_entry.get(),
            'y_label': app.y_label_entry.get(),
            'suptitle': app.suptitle_entry.get(),
            'wspace': app.grid_wspace_entry.get(),
            'hspace': app.grid_hspace_entry.get(),
            'tick_font': app.tick_font_size_entry.get(),
            'hide_inner': app.hide_inner_labels_var.get(),
            'style_preset': app.style_preset_var.get(),
            'ticks_inside': app.ticks_inside_var.get(),
            'minor_ticks': app.minor_ticks_var.get(),
            'maj_tick': app.maj_tick_len_entry.get(),
            'min_tick': app.min_tick_len_entry.get(),
            'axes_lw': app.axes_lw_entry.get(),
            'subplot_title_template': app.subplot_title_template_entry.get(),
            'subplot_title_mode': app.subplot_title_mode_var.get(),
            'subplot_title_posx': app.subplot_title_posx_entry.get(),
            'subplot_title_posy': app.subplot_title_posy_entry.get(),
            'plot_min': getattr(app,'plot_min_entry',None).get() if getattr(app,'plot_min_entry',None) else '',
            'plot_max': getattr(app,'plot_max_entry',None).get() if getattr(app,'plot_max_entry',None) else '',
            'show_grid': app.show_grid_var.get()
        }
        import io
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
        messagebox.showinfo("Saved", f"Figure settings saved to:\n{path}")
    except Exception as e:
        logger.error(f"Failed to save figure settings: {e}", exc_info=True)
        messagebox.showerror("Error", f"Failed to save settings: {e}")

def load_figure_settings_action(app):
    """Load figure options from a JSON file and populate UI."""
    try:
        import json
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All Files", "*.*")], title="Load Figure Settings")
        if not path:
            return
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        # Set values if present
        def set_entry(e, v): e.delete(0,'end'); e.insert(0, str(v))
        if 'fig_cols' in d: set_entry(app.fig_cols_entry, d['fig_cols'])
        if 'fig_w' in d: set_entry(app.fig_width_entry, d['fig_w'])
        if 'fig_h' in d: set_entry(app.fig_height_entry, d['fig_h'])
        if 'dpi' in d: set_entry(app.fig_dpi_entry, d['dpi'])
        if 'font_size' in d: set_entry(app.font_size_entry, d['font_size'])
        if 'line_width' in d: set_entry(app.line_width_entry, d['line_width'])
        if 'color_cycle' in d: app.color_cycle_var.set(d['color_cycle'])
        if 'legend_mode' in d: app.legend_mode_var.set(d['legend_mode'])
        if 'legend_font' in d: set_entry(app.legend_font_size_entry, d['legend_font'])
        if 'legend_ncol' in d: set_entry(app.legend_ncol_entry, d['legend_ncol'])
        if 'x_label' in d: set_entry(app.x_label_entry, d['x_label'])
        if 'y_label' in d: set_entry(app.y_label_entry, d['y_label'])
        if 'suptitle' in d: set_entry(app.suptitle_entry, d['suptitle'])
        if 'wspace' in d: set_entry(app.grid_wspace_entry, d['wspace'])
        if 'hspace' in d: set_entry(app.grid_hspace_entry, d['hspace'])
        if 'tick_font' in d: set_entry(app.tick_font_size_entry, d['tick_font'])
        if 'hide_inner' in d: app.hide_inner_labels_var.set(str(d['hide_inner']))
        if 'style_preset' in d: app.style_preset_var.set(d['style_preset'])
        if 'ticks_inside' in d: app.ticks_inside_var.set(str(d['ticks_inside']))
        if 'minor_ticks' in d: app.minor_ticks_var.set(str(d['minor_ticks']))
        if 'maj_tick' in d: set_entry(app.maj_tick_len_entry, d['maj_tick'])
        if 'min_tick' in d: set_entry(app.min_tick_len_entry, d['min_tick'])
        if 'axes_lw' in d: set_entry(app.axes_lw_entry, d['axes_lw'])
        if 'subplot_title_template' in d: set_entry(app.subplot_title_template_entry, d['subplot_title_template'])
        if 'subplot_title_mode' in d: app.subplot_title_mode_var.set(d['subplot_title_mode'])
        if 'subplot_title_posx' in d: set_entry(app.subplot_title_posx_entry, d['subplot_title_posx'])
        if 'subplot_title_posy' in d: set_entry(app.subplot_title_posy_entry, d['subplot_title_posy'])
        if 'plot_min' in d and getattr(app,'plot_min_entry',None): set_entry(app.plot_min_entry, d['plot_min'])
        if 'plot_max' in d and getattr(app,'plot_max_entry',None): set_entry(app.plot_max_entry, d['plot_max'])
        if 'show_grid' in d: app.show_grid_var.set(str(d['show_grid']))
        messagebox.showinfo("Loaded", f"Figure settings loaded from:\n{path}")
    except Exception as e:
        logger.error(f"Failed to load figure settings: {e}", exc_info=True)
        messagebox.showerror("Error", f"Failed to load settings: {e}")

def ultimate_plot_action(app):
    """Plots power series at every temperature as a grid of subplots.

    Honors current options:
    - Normalize Spectra / Normalized Stacked with Offset and norm range
    - Equalize to Min T (applies per-temperature to the lowest power)
    - Intensity Map options (normalize, log scales, colormap, intensity range)
    - Figure options (columns, size, dpi, font size, line width, color cycle)
    """
    logger.info("Ultimate Plot button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    metadata_all = active_dataset.get_metadata()
    if metadata_all is None or metadata_all.empty:
        messagebox.showwarning("No Data", "No metadata found in active dataset.")
        return

    try:
        temps = sorted(metadata_all['temperature_k'].dropna().unique())
    except Exception:
        messagebox.showerror("Error", "Temperature metadata missing or invalid.")
        return

    # Figure options with defaults
    def _parse_float(entry, default):
        try:
            return float(entry.get()) if entry and entry.get() else default
        except Exception:
            return default
    def _parse_int(entry, default):
        try:
            return int(float(entry.get())) if entry and entry.get() else default
        except Exception:
            return default

    cols = _parse_int(getattr(app, 'fig_cols_entry', None), 4)
    fig_w = _parse_float(getattr(app, 'fig_width_entry', None), 16.0)
    fig_h = _parse_float(getattr(app, 'fig_height_entry', None), 10.0)
    dpi = _parse_int(getattr(app, 'fig_dpi_entry', None), 150)
    font_size = _parse_int(getattr(app, 'font_size_entry', None), 10)
    line_width = _parse_float(getattr(app, 'line_width_entry', None), 1.0)
    color_cycle = getattr(app, 'color_cycle_var', None).get() if getattr(app, 'color_cycle_var', None) else 'tab10'

    y_scale = 'log' if app.log_scale_var.get() == "1" else 'linear'
    is_stack = app.stack_var.get() == "1"
    is_norm = app.normalize_var.get() == "1"
    is_equalize = getattr(app, 'equalize_var', None) and app.equalize_var.get() == "1"
    is_imap = app.intensity_map_var.get() == "1"

    # Ranges and offset
    norm_range = None
    eq_range = None
    try:
        min_e = float(app.norm_min_entry.get()) if app.norm_min_entry.get() else None
        max_e = float(app.norm_max_entry.get()) if app.norm_max_entry.get() else None
        norm_range = (min_e, max_e) if min_e is not None and max_e is not None else None
    except Exception:
        pass
    try:
        if getattr(app, 'eq_min_entry', None) and getattr(app, 'eq_max_entry', None):
            mn = float(app.eq_min_entry.get()) if app.eq_min_entry.get() else None
            mx = float(app.eq_max_entry.get()) if app.eq_max_entry.get() else None
            eq_range = (mn, mx) if mn is not None and mx is not None else None
    except Exception:
        pass
    try:
        offset = float(app.offset_entry.get()) if app.offset_entry.get() else 1.0
    except Exception:
        offset = 1.0

    # Plot x-range
    xlim = None
    try:
        if getattr(app, 'plot_min_entry', None) and getattr(app, 'plot_max_entry', None):
            xmin = float(app.plot_min_entry.get()) if app.plot_min_entry.get() else None
            xmax = float(app.plot_max_entry.get()) if app.plot_max_entry.get() else None
            if xmin is not None and xmax is not None:
                xlim = (xmin, xmax)
    except Exception:
        xlim = None

    # Build per-temperature series
    grid_titles = []
    line_subplots = []
    map_subplots = []
    map_global_xlim = None

    for T in temps:
        df = metadata_all[metadata_all['temperature_k'] == T].sort_values(by='power_uw')
        if df.empty:
            continue
        energies_list, counts_list, labels = [], [], []
        y_values = df['power_uw'].values
        for _, row in df.iterrows():
            spectrum = active_dataset.get_processed_spectrum(row['file_id'])
            if spectrum is None or spectrum.empty:
                continue
            energies_list.append(spectrum['energy_ev'].values)
            counts_list.append(spectrum['counts'].values)
            labels.append(f"P = {row['power_uw']} uW")

        if not energies_list:
            continue

        # Subplot title formatting
        title_template = getattr(app, 'subplot_title_template_entry', None).get() if getattr(app, 'subplot_title_template_entry', None) and app.subplot_title_template_entry.get() else "{T:.0f} K"
        try:
            grid_titles.append(title_template.format(T=T))
        except Exception:
            grid_titles.append(f"{T:.0f} K")

        if is_imap:
            # Prepare intensity map per temperature
            try:
                # Common energy axis
                start_energy = max(e.min() for e in energies_list)
                end_energy = min(e.max() for e in energies_list)
                if start_energy >= end_energy:
                    continue
                import numpy as np
                common_energy = np.linspace(start_energy, end_energy, 1000)
                resampled = []
                for e, c in zip(energies_list, counts_list):
                    idx = np.argsort(e)
                    e_sorted = e[idx]
                    c_sorted = c[idx]
                    resampled.append(np.interp(common_energy, e_sorted, c_sorted))
                intensity_grid = np.array(resampled)

                # Optional per-row normalization in range
                if app.im_normalize_var.get() == "1":
                    try:
                        nmin = float(app.im_norm_min_entry.get()) if app.im_norm_min_entry.get() else common_energy.min()
                        nmax = float(app.im_norm_max_entry.get()) if app.im_norm_max_entry.get() else common_energy.max()
                        sel = (common_energy >= nmin) & (common_energy <= nmax)
                        if np.any(sel):
                            for i in range(intensity_grid.shape[0]):
                                mx = intensity_grid[i, sel].max()
                                if mx > 0:
                                    intensity_grid[i, :] /= mx
                        if map_global_xlim is None:
                            map_global_xlim = (nmin, nmax)
                    except Exception:
                        pass

                map_subplots.append({'x': common_energy, 'y': y_values, 'z': intensity_grid})
            except Exception as e:
                logger.error(f"Failed to prepare intensity map for T={T}: {e}")
        else:
            # Line plots: normalize/equalize/stack as requested
            processed_counts = counts_list
            processed_labels = labels.copy()

            if is_equalize:
                # Baseline = lowest power in this temperature's series
                def max_in_range(e_arr, c_arr, erange):
                    import numpy as np
                    if erange:
                        mn, mx = erange
                        mask = (e_arr >= mn) & (e_arr <= mx)
                        if not np.any(mask):
                            return float(np.max(c_arr)) if c_arr.size else 0.0
                        return float(np.max(c_arr[mask])) if np.any(c_arr[mask]) else 0.0
                    return float(np.max(c_arr)) if c_arr.size else 0.0

                base_max = max_in_range(energies_list[0], counts_list[0], eq_range or norm_range)
                if base_max <= 0:
                    base_max = 1.0

                new_counts, new_labels = [], []
                for i, (e_arr, c_arr, lbl) in enumerate(zip(energies_list, counts_list, labels)):
                    cur_max = max_in_range(e_arr, c_arr, eq_range or norm_range)
                    factor = (cur_max / base_max) if base_max > 0 else 1.0
                    scaled = c_arr / (factor if factor > 0 else 1.0)
                    new_counts.append(scaled)
                    if i == 0:
                        new_labels.append(lbl)
                    else:
                        mult = 1.0 / factor if factor > 0 else 1.0
                        new_labels.append(f"{lbl} (x {mult:.2f})")
                processed_counts = new_counts
                processed_labels = new_labels

                # Apply stacking with effective offset scaled by base max
                eff_offset = offset * base_max
                stacked_counts = []
                for i, c in enumerate(processed_counts):
                    stacked_counts.append(c + i * eff_offset)
                processed_counts = stacked_counts
            else:
                # Optional normalization
                if is_norm or is_stack:
                    import pandas as pd
                    def _norm_counts(e, c):
                        df = processing.normalize_spectrum(pd.DataFrame({'energy_ev': e, 'counts': c}), norm_range)
                        if df is None or df.empty or 'counts' not in df:
                            return c
                        return df['counts'].values
                    processed_counts = [
                        _norm_counts(e, c)
                        for e, c in zip(energies_list, counts_list)
                    ]
                # Optional stacking with plain offset
                if is_stack:
                    processed_counts = [c + i * offset for i, c in enumerate(processed_counts)]

            line_subplots.append({'energies': energies_list, 'counts': processed_counts, 'labels': processed_labels})

    # Render
    if is_imap and map_subplots:
        try:
            # Color limits
            try:
                vmin = float(app.im_min_intensity_entry.get()) if app.im_min_intensity_entry.get() else None
                vmax = float(app.im_max_intensity_entry.get()) if app.im_max_intensity_entry.get() else None
            except Exception:
                vmin = None; vmax = None
            # Spacing & label options
            try:
                wspace = float(app.grid_wspace_entry.get()) if getattr(app, 'grid_wspace_entry', None) and app.grid_wspace_entry.get() else 0.3
            except Exception:
                wspace = 0.3
            try:
                hspace = float(app.grid_hspace_entry.get()) if getattr(app, 'grid_hspace_entry', None) and app.grid_hspace_entry.get() else 0.3
            except Exception:
                hspace = 0.3
            hide_inner = getattr(app, 'hide_inner_labels_var', None) and app.hide_inner_labels_var.get() == "1"
            try:
                tick_fs = int(float(app.tick_font_size_entry.get())) if getattr(app, 'tick_font_size_entry', None) and app.tick_font_size_entry.get() else None
            except Exception:
                tick_fs = None

            app.right_panel.plot_canvas.plot_intensity_map_grid(
                map_subplots, grid_titles, cols=cols, figsize=(fig_w, fig_h), dpi=dpi, font_size=font_size,
                cmap=app.im_colormap_var.get(), log_c=app.im_log_c_var.get() == "1", log_y=app.im_log_y_var.get() == "1",
                vmin=vmin, vmax=vmax, x_label="Energy (eV)", y_label="Power (uW)", suptitle=(getattr(app,'suptitle_entry',None).get() if getattr(app,'suptitle_entry',None) and app.suptitle_entry.get() else "Power Series at All Temperatures"),
                xlim=(xlim or map_global_xlim), wspace=wspace, hspace=hspace, hide_inner_labels=hide_inner, tick_font_size=tick_fs
            )
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot intensity map grid: {e}")
    elif line_subplots:
        ylab = getattr(app, 'y_label_entry', None).get() if getattr(app, 'y_label_entry', None) and app.y_label_entry.get() else "Intensity (arb. units)"
        if is_equalize or is_norm or is_stack:
            if not getattr(app, 'y_label_entry', None) or not app.y_label_entry.get():
                ylab = "Normalized Intensity (arb. units)" if not is_equalize else "Intensity (arb. units)"
        xlab = getattr(app, 'x_label_entry', None).get() if getattr(app, 'x_label_entry', None) and app.x_label_entry.get() else "Energy (eV)"
        # Legend controls
        legend_mode = getattr(app, 'legend_mode_var', None).get() if getattr(app, 'legend_mode_var', None) else 'per-axes'
        try:
            legend_font = int(float(app.legend_font_size_entry.get())) if getattr(app, 'legend_font_size_entry', None) and app.legend_font_size_entry.get() else None
        except Exception:
            legend_font = None
        try:
            legend_ncol = int(float(app.legend_ncol_entry.get())) if getattr(app, 'legend_ncol_entry', None) and app.legend_ncol_entry.get() else 1
        except Exception:
            legend_ncol = 1
        # Spacing & tick labels
        try:
            wspace = float(app.grid_wspace_entry.get()) if getattr(app, 'grid_wspace_entry', None) and app.grid_wspace_entry.get() else 0.3
        except Exception:
            wspace = 0.3
        try:
            hspace = float(app.grid_hspace_entry.get()) if getattr(app, 'grid_hspace_entry', None) and app.grid_hspace_entry.get() else 0.3
        except Exception:
            hspace = 0.3
        hide_inner = getattr(app, 'hide_inner_labels_var', None) and app.hide_inner_labels_var.get() == "1"
        try:
            tick_fs = int(float(app.tick_font_size_entry.get())) if getattr(app, 'tick_font_size_entry', None) and app.tick_font_size_entry.get() else None
        except Exception:
            tick_fs = None
        # Style options
        style_options = {}
        style_options['preset'] = getattr(app, 'style_preset_var', None).get() if getattr(app, 'style_preset_var', None) else 'Default'
        style_options['ticks_inside'] = getattr(app, 'ticks_inside_var', None) and app.ticks_inside_var.get() == "1"
        style_options['minor_ticks'] = getattr(app, 'minor_ticks_var', None) and app.minor_ticks_var.get() == "1"
        try:
            style_options['maj_len'] = float(app.maj_tick_len_entry.get()) if getattr(app, 'maj_tick_len_entry', None) and app.maj_tick_len_entry.get() else 4.0
        except Exception:
            style_options['maj_len'] = 4.0
        try:
            style_options['min_len'] = float(app.min_tick_len_entry.get()) if getattr(app, 'min_tick_len_entry', None) and app.min_tick_len_entry.get() else 2.0
        except Exception:
            style_options['min_len'] = 2.0
        try:
            style_options['axes_linewidth'] = float(app.axes_lw_entry.get()) if getattr(app, 'axes_lw_entry', None) and app.axes_lw_entry.get() else 1.0
        except Exception:
            style_options['axes_linewidth'] = 1.0

        # Title mode and position
        title_mode = getattr(app, 'subplot_title_mode_var', None).get() if getattr(app, 'subplot_title_mode_var', None) else 'axes'
        try:
            posx = float(app.subplot_title_posx_entry.get()) if getattr(app, 'subplot_title_posx_entry', None) and app.subplot_title_posx_entry.get() else 0.02
            posy = float(app.subplot_title_posy_entry.get()) if getattr(app, 'subplot_title_posy_entry', None) and app.subplot_title_posy_entry.get() else 0.95
        except Exception:
            posx, posy = 0.02, 0.95

        app.right_panel.plot_canvas.plot_line_grid(
            line_subplots, grid_titles, cols=cols, y_scale=y_scale, figsize=(fig_w, fig_h), dpi=dpi,
            font_size=font_size, line_width=line_width, color_cycle=color_cycle,
            x_label=xlab, y_label=ylab, suptitle=(getattr(app,'suptitle_entry',None).get() if getattr(app,'suptitle_entry',None) and app.suptitle_entry.get() else "Power Series at All Temperatures"), xlim=xlim,
            legend_mode=legend_mode, legend_font=legend_font, legend_ncol=legend_ncol,
            wspace=wspace, hspace=hspace, hide_inner_labels=hide_inner, tick_font_size=tick_fs,
            style_options={**style_options, 'show_grid': app.show_grid_var.get() == "1"}, title_mode=title_mode, in_axes_pos=(posx, posy)
        )
    else:
        messagebox.showwarning("Plot", "No data to plot.")

# ----------------- Spike Removal Tools -----------------

def _get_spike_params(app):
    try:
        window = int(float(app.spike_window_entry.get())) if app.spike_window_entry.get() else 7
    except Exception:
        window = 7
    try:
        sigma = float(app.spike_sigma_entry.get()) if app.spike_sigma_entry.get() else 6.0
    except Exception:
        sigma = 6.0
    try:
        maxw = int(float(app.spike_maxw_entry.get())) if app.spike_maxw_entry.get() else 2
    except Exception:
        maxw = 2
    try:
        prom = float(app.spike_prom_entry.get()) if app.spike_prom_entry.get() else 0.0
    except Exception:
        prom = 0.0
    method = app.spike_method_var.get() if getattr(app, 'spike_method_var', None) else 'interp'
    try:
        neighbor = int(float(app.spike_neighbor_entry.get())) if getattr(app, 'spike_neighbor_entry', None) and app.spike_neighbor_entry.get() else 1
    except Exception:
        neighbor = 1
    return window, sigma, maxw, prom, method, neighbor

def detect_spikes_action(app):
    last = app._last_plot_data
    if not last:
        messagebox.showinfo("No Plot", "Please plot spectra first.")
        return
    energies = last['energies']
    counts = last['counts']
    window, sigma, maxw, prom, _, _ = _get_spike_params(app)
    adaptive = getattr(app, 'spike_adaptive_var', None) and app.spike_adaptive_var.get() == "1"
    hybrid = getattr(app, 'spike_hybrid_var', None) and app.spike_hybrid_var.get() == "1"
    try:
        adapt_w = int(float(app.spike_adapt_window_entry.get())) if getattr(app, 'spike_adapt_window_entry', None) and app.spike_adapt_window_entry.get() else max(51, window)
    except Exception:
        adapt_w = max(51, window)
    # Hybrid params
    try:
        broad_w = int(float(app.spike_hybrid_broad_entry.get())) if getattr(app, 'spike_hybrid_broad_entry', None) and app.spike_hybrid_broad_entry.get() else 151
    except Exception:
        broad_w = 151
    try:
        avoid_w = int(float(app.spike_hybrid_avoidw_entry.get())) if getattr(app, 'spike_hybrid_avoidw_entry', None) and app.spike_hybrid_avoidw_entry.get() else 9
    except Exception:
        avoid_w = 9
    try:
        avoid_p = float(app.spike_hybrid_avoidp_entry.get()) if getattr(app, 'spike_hybrid_avoidp_entry', None) and app.spike_hybrid_avoidp_entry.get() else 0.0
    except Exception:
        avoid_p = 0.0
    # Optional detect range filter
    try:
        dmin = float(app.spike_detect_min_entry.get()) if getattr(app, 'spike_detect_min_entry', None) and app.spike_detect_min_entry.get() else None
        dmax = float(app.spike_detect_max_entry.get()) if getattr(app, 'spike_detect_max_entry', None) and app.spike_detect_max_entry.get() else None
    except Exception:
        dmin = dmax = None
    spikes_list = []
    for e, c in zip(energies, counts):
        if dmin is not None and dmax is not None:
            mask = (np.asarray(e) >= dmin) & (np.asarray(e) <= dmax)
        else:
            mask = None
        if hybrid:
            spikes = spike_removal.detect_spikes_hybrid(e, c, base_window=window, mad_window=adapt_w, sigma=sigma, max_width=maxw, min_prominence=prom, broad_window=broad_w, avoid_width=avoid_w, avoid_prominence=avoid_p)
        elif adaptive:
            spikes = spike_removal.detect_spikes_adaptive(e, c, base_window=window, mad_window=adapt_w, sigma=sigma, max_width=maxw, min_prominence=prom)
        else:
            spikes = spike_removal.detect_spikes(e, c, window=window, sigma=sigma, max_width=maxw, min_prominence=prom)
        if mask is not None and len(spikes):
            # keep spikes only in detect window
            x = np.asarray(e)
            spikes = [k for k in spikes if 0 <= k < len(x) and mask[k]]
        spikes_list.append(spikes)
    app._spike_candidates = {
        'indices': spikes_list,
        'energies': energies,
        'counts': counts,
        'file_ids': last.get('file_ids', [])
    }
    # Overlay markers (selected by default)
    app.right_panel.plot_canvas.overlay_spikes(energies, counts, spikes_list)

def toggle_manual_spike_add_action(app):
    # The variable is hosted on RightPanel
    enable = False
    try:
        if getattr(app.right_panel, 'spike_manual_add_var', None) and app.right_panel.spike_manual_add_var.get() == "1":
            enable = True
    except Exception:
        pass
    app.right_panel.plot_canvas.enable_manual_spike_add(enable)

def clear_spike_markers_action(app):
    app.right_panel.plot_canvas.clear_spike_overlay()
    app._spike_candidates = None

def apply_spike_removal_action(app):
    last = app._last_plot_data
    active_dataset = app.data_handler.get_active_dataset()
    if not (last and active_dataset):
        messagebox.showwarning("No Data", "Please plot spectra and ensure a dataset is active.")
        return
    if not getattr(app.right_panel.plot_canvas, 'get_selected_spikes', None):
        messagebox.showwarning("No Spikes", "Please detect spikes first.")
        return
    selected = app.right_panel.plot_canvas.get_selected_spikes()
    if not any(len(s) for s in selected):
        messagebox.showinfo("No Selection", "No spikes selected for removal.")
        return
    window, sigma, maxw, prom, method, neighbor = _get_spike_params(app)
    energies = last['energies']
    counts = last['counts']
    file_ids = last.get('file_ids', [])
    # Apply removal on in-memory arrays and update dataset spectra when file_ids are available
    new_counts = []
    # Save undo snapshot
    if not hasattr(app, '_spike_undo_stack'):
        app._spike_undo_stack = []
    snapshot = {}
    for i, fid in enumerate(file_ids):
        spec_df = active_dataset.get_processed_spectrum(fid)
        if spec_df is not None and not spec_df.empty:
            snapshot[fid] = spec_df['counts'].values.copy()
    app._spike_undo_stack.append(snapshot)

    # Ensure dataset stores original baseline before first cleaning
    if not hasattr(active_dataset, '_spike_original_counts'):
        active_dataset._spike_original_counts = {}

    # Expand manual selections by radius if provided
    try:
        manual_radius = int(float(app.spike_manual_radius_entry.get())) if getattr(app, 'spike_manual_radius_entry', None) and app.spike_manual_radius_entry.get() else 0
    except Exception:
        manual_radius = 0

    for i, (e, c, spikes) in enumerate(zip(energies, counts, selected)):
        if manual_radius > 0 and spikes:
            expanded = set()
            for k in spikes:
                for t in range(k-manual_radius, k+manual_radius+1):
                    if 0 <= t < len(c):
                        expanded.add(t)
            spikes = sorted(expanded)
        cleaned = spike_removal.remove_spikes(e, c, spikes, method=method, neighbor_n=neighbor)
        new_counts.append(cleaned)
        if i < len(file_ids):
            fid = file_ids[i]
            spec_df = active_dataset.get_processed_spectrum(fid)
            if spec_df is not None and not spec_df.empty:
                if fid not in active_dataset._spike_original_counts:
                    active_dataset._spike_original_counts[fid] = spec_df['counts'].values.copy()
                # align energies by nearest indices if lengths match; simple replace
                spec_df['counts'] = cleaned[:len(spec_df)]
                active_dataset.processed_spectra[fid] = spec_df
    # Update last plot data to reflect cleaned counts
    last['counts'] = new_counts
    # Replot with same style
    _update_plot_style(app)
    _spike_log(app, f"Removed spikes from {len(new_counts)} spectra using method='{method}'.")

def toggle_show_original_action(app, show: bool):
    last = app._last_plot_data
    ds = app.data_handler.get_active_dataset()
    if not (last and ds and hasattr(ds, '_spike_original_counts')):
        _spike_log(app, "Original data not available.")
        return
    energies = last['energies']
    file_ids = last.get('file_ids', [])
    if show:
        # Render current cleaned data
        _update_plot_style(app)
        # Gather originals and overlay in dashed orange
        orig_counts = []
        for i, e in enumerate(energies):
            if i < len(file_ids) and file_ids[i] in ds._spike_original_counts:
                arr = ds._spike_original_counts[file_ids[i]]
                orig_counts.append(arr[:len(e)])
            else:
                spec_df = ds.get_processed_spectrum(file_ids[i]) if i < len(file_ids) else None
                orig_counts.append(spec_df['counts'].values[:len(e)] if spec_df is not None else last['counts'][i])
        labels = [f"{lbl} (original)" for lbl in last.get('labels', ['']*len(energies))]
        app.right_panel.plot_canvas.overlay_lines(energies, orig_counts, labels_list=labels, color='orange', alpha=0.8, linestyle='--')
    else:
        _update_plot_style(app)

def revert_curve_to_original_action(app):
    ds = app.data_handler.get_active_dataset()
    last = app._last_plot_data
    if not (ds and last and hasattr(ds, '_spike_original_counts')):
        _spike_log(app, "Nothing to revert.")
        return
    try:
        idx = int(float(app.spike_revert_curve_entry.get())) - 1
    except Exception:
        idx = 0
    if idx < 0 or idx >= len(last['energies']):
        _spike_log(app, "Invalid curve index.")
        return
    if idx >= len(last.get('file_ids', [])):
        _spike_log(app, "No file id for selected curve.")
        return
    fid = last['file_ids'][idx]
    if fid not in ds._spike_original_counts:
        _spike_log(app, "No original snapshot for this curve.")
        return
    arr = ds._spike_original_counts[fid]
    spec_df = ds.get_processed_spectrum(fid)
    if spec_df is not None and not spec_df.empty:
        spec_df['counts'] = arr[:len(spec_df)]
        ds.processed_spectra[fid] = spec_df
        _spike_log(app, f"Reverted curve #{idx+1} to original.")
        # refresh last plot counts for this curve only
        energies = last['energies']
        last['counts'][idx] = arr[:len(energies[idx])]
        _update_plot_style(app)

def apply_prominence_threshold_action(app):
    last = app._last_plot_data
    ds = app.data_handler.get_active_dataset()
    if not (last and ds):
        _spike_log(app, "Apply Prominence: plot something first.")
        return
    try:
        px = float(app.spike_prom_x_entry.get()) if app.spike_prom_x_entry.get() else 0.0
    except Exception:
        px = 0.0
    window, _, _, _, method, neighbor = _get_spike_params(app)
    energies = last['energies']
    counts = last['counts']
    file_ids = last.get('file_ids', [])
    # snapshot for undo
    if not hasattr(app, '_spike_undo_stack'):
        app._spike_undo_stack = []
    snapshot = {}
    for i, fid in enumerate(file_ids):
        spec_df = ds.get_processed_spectrum(fid)
        if spec_df is not None and not spec_df.empty:
            snapshot[fid] = spec_df['counts'].values.copy()
    app._spike_undo_stack.append(snapshot)

    new_counts = []
    for i, (e, y) in enumerate(zip(energies, counts)):
        base = spike_removal.rolling_median(np.asarray(y, dtype=float), max(3, window))
        resid = np.asarray(y, dtype=float) - base
        idxs = list(np.where(resid >= px)[0])
        cleaned = spike_removal.remove_spikes(e, y, idxs, method=method, neighbor_n=neighbor)
        new_counts.append(cleaned)
        if i < len(file_ids):
            fid = file_ids[i]
            spec_df = ds.get_processed_spectrum(fid)
            if spec_df is not None and not spec_df.empty:
                if not hasattr(ds, '_spike_original_counts'):
                    ds._spike_original_counts = {}
                if fid not in ds._spike_original_counts:
                    ds._spike_original_counts[fid] = spec_df['counts'].values.copy()
                spec_df['counts'] = cleaned[:len(spec_df)]
                ds.processed_spectra[fid] = spec_df
    last['counts'] = new_counts
    _update_plot_style(app)
    _spike_log(app, f"Applied prominence threshold X={px} to {len(new_counts)} spectra.")

def undo_spike_removal_action(app):
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset or not getattr(app, '_spike_undo_stack', None):
        _spike_log(app, "Undo: nothing to revert.")
        return
    snapshot = app._spike_undo_stack.pop()
    for fid, arr in snapshot.items():
        spec_df = active_dataset.get_processed_spectrum(fid)
        if spec_df is not None and not spec_df.empty:
            spec_df['counts'] = arr[:len(spec_df)]
            active_dataset.processed_spectra[fid] = spec_df
    _spike_log(app, f"Undo: restored {len(snapshot)} spectra.")
    # Replot if last plot matches these ids
    if app._last_plot_data and 'file_ids' in app._last_plot_data:
        energies = app._last_plot_data['energies']
        file_ids = app._last_plot_data['file_ids']
        new_counts = []
        for fid, e in zip(file_ids, energies):
            spec_df = active_dataset.get_processed_spectrum(fid)
            if spec_df is not None and not spec_df.empty:
                new_counts.append(spec_df['counts'].values[:len(e)])
            else:
                new_counts.append(app._last_plot_data['counts'][len(new_counts)])
        app._last_plot_data['counts'] = new_counts
        _update_plot_style(app)

def create_cleaned_dataset_action(app):
    """Clone the active dataset into a new one with current processed spectra (post-cleaning)."""
    from pl_analyzer.core.data_handler import Dataset
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        _spike_log(app, "No active dataset to clone.")
        return
    # Create new empty dataset
    new_name = app.data_handler.create_empty_dataset()
    new_ds = app.data_handler.get_active_dataset()
    # Copy metadata and raw
    new_ds.metadata = {k: v.copy() for k, v in active_dataset.metadata.items()}
    new_ds.raw_spectra = {k: v.copy() for k, v in active_dataset.raw_spectra.items()}
    new_ds.processed_spectra = {k: v.copy() for k, v in active_dataset.processed_spectra.items()}
    new_ds._update_main_dataframe()
    # Add a tab for the new dataset and make it active in UI
    try:
        app.left_panel.add_dataset_tab(new_name)
        app.update_file_table()
    except Exception:
        pass
    _spike_log(app, f"Created new cleaned dataset: {new_name}")

def sensitivity_sweep_action(app):
    last = app._last_plot_data
    if not last:
        _spike_log(app, "Sweep: plot something first.")
        return
    try:
        smin = float(app.spike_sweep_min_entry.get()) if app.spike_sweep_min_entry.get() else 3.0
        smax = float(app.spike_sweep_max_entry.get()) if app.spike_sweep_max_entry.get() else 10.0
        steps = int(float(app.spike_sweep_steps_entry.get())) if app.spike_sweep_steps_entry.get() else 8
    except Exception:
        smin, smax, steps = 3.0, 10.0, 8
    energies = last['energies']
    counts = last['counts']
    window, _, maxw, prom, _, _ = _get_spike_params(app)
    adaptive = getattr(app, 'spike_adaptive_var', None) and app.spike_adaptive_var.get() == "1"
    hybrid = getattr(app, 'spike_hybrid_var', None) and app.spike_hybrid_var.get() == "1"
    try:
        adapt_w = int(float(app.spike_adapt_window_entry.get())) if getattr(app, 'spike_adapt_window_entry', None) and app.spike_adapt_window_entry.get() else max(51, window)
    except Exception:
        adapt_w = max(51, window)
    try:
        broad_w = int(float(app.spike_hybrid_broad_entry.get())) if getattr(app, 'spike_hybrid_broad_entry', None) and app.spike_hybrid_broad_entry.get() else 151
    except Exception:
        broad_w = 151
    try:
        avoid_w = int(float(app.spike_hybrid_avoidw_entry.get())) if getattr(app, 'spike_hybrid_avoidw_entry', None) and app.spike_hybrid_avoidw_entry.get() else 9
    except Exception:
        avoid_w = 9
    try:
        avoid_p = float(app.spike_hybrid_avoidp_entry.get()) if getattr(app, 'spike_hybrid_avoidp_entry', None) and app.spike_hybrid_avoidp_entry.get() else 0.0
    except Exception:
        avoid_p = 0.0
    sigmas = np.linspace(smin, smax, max(2, steps))
    lines = ["Sensitivity sweep (sigma -> total spikes):"]
    for s in sigmas:
        total = 0
        for e, c in zip(energies, counts):
            if hybrid:
                total += len(spike_removal.detect_spikes_hybrid(e, c, base_window=window, mad_window=adapt_w, sigma=float(s), max_width=maxw, min_prominence=prom, broad_window=broad_w, avoid_width=avoid_w, avoid_prominence=avoid_p))
            elif adaptive:
                total += len(spike_removal.detect_spikes_adaptive(e, c, base_window=window, mad_window=adapt_w, sigma=float(s), max_width=maxw, min_prominence=prom))
            else:
                total += len(spike_removal.detect_spikes(e, c, window=window, sigma=float(s), max_width=maxw, min_prominence=prom))
        lines.append(f"  {s:.2f} -> {total}")
    _spike_log(app, "\n".join(lines))

def auto_remove_spikes_action(app):
    last = app._last_plot_data
    ds = app.data_handler.get_active_dataset()
    if not (last and ds):
        _spike_log(app, "Auto Remove: plot something first.")
        return
    window, sigma, maxw, prom, method, neighbor = _get_spike_params(app)
    adaptive = getattr(app, 'spike_adaptive_var', None) and app.spike_adaptive_var.get() == "1"
    hybrid = getattr(app, 'spike_hybrid_var', None) and app.spike_hybrid_var.get() == "1"
    try:
        adapt_w = int(float(app.spike_adapt_window_entry.get())) if getattr(app, 'spike_adapt_window_entry', None) and app.spike_adapt_window_entry.get() else max(51, window)
    except Exception:
        adapt_w = max(51, window)
    try:
        broad_w = int(float(app.spike_hybrid_broad_entry.get())) if getattr(app, 'spike_hybrid_broad_entry', None) and app.spike_hybrid_broad_entry.get() else 151
    except Exception:
        broad_w = 151
    try:
        avoid_w = int(float(app.spike_hybrid_avoidw_entry.get())) if getattr(app, 'spike_hybrid_avoidw_entry', None) and app.spike_hybrid_avoidw_entry.get() else 9
    except Exception:
        avoid_w = 9
    try:
        avoid_p = float(app.spike_hybrid_avoidp_entry.get()) if getattr(app, 'spike_hybrid_avoidp_entry', None) and app.spike_hybrid_avoidp_entry.get() else 0.0
    except Exception:
        avoid_p = 0.0
    energies = last['energies']
    counts = last['counts']
    file_ids = last.get('file_ids', [])
    # snapshot for undo
    if not hasattr(app, '_spike_undo_stack'):
        app._spike_undo_stack = []
    snapshot = {}
    for fid in file_ids:
        spec_df = ds.get_processed_spectrum(fid)
        if spec_df is not None and not spec_df.empty:
            snapshot[fid] = spec_df['counts'].values.copy()
    app._spike_undo_stack.append(snapshot)

    new_counts = []
    for i, (e, y) in enumerate(zip(energies, counts)):
        if hybrid:
            idxs = spike_removal.detect_spikes_hybrid(e, y, base_window=window, mad_window=adapt_w, sigma=sigma, max_width=maxw, min_prominence=prom, broad_window=broad_w, avoid_width=avoid_w, avoid_prominence=avoid_p)
        elif adaptive:
            idxs = spike_removal.detect_spikes_adaptive(e, y, base_window=window, mad_window=adapt_w, sigma=sigma, max_width=maxw, min_prominence=prom)
        else:
            idxs = spike_removal.detect_spikes(e, y, window=window, sigma=sigma, max_width=maxw, min_prominence=prom)
        cleaned = spike_removal.remove_spikes(e, y, idxs, method=method, neighbor_n=neighbor)
        new_counts.append(cleaned)
        if i < len(file_ids):
            fid = file_ids[i]
            spec_df = ds.get_processed_spectrum(fid)
            if spec_df is not None and not spec_df.empty:
                if not hasattr(ds, '_spike_original_counts'):
                    ds._spike_original_counts = {}
                if fid not in ds._spike_original_counts:
                    ds._spike_original_counts[fid] = spec_df['counts'].values.copy()
                spec_df['counts'] = cleaned[:len(spec_df)]
                ds.processed_spectra[fid] = spec_df
    last['counts'] = new_counts
    _update_plot_style(app)
    _spike_log(app, f"Auto Remove applied to {len(new_counts)} spectra (adaptive={adaptive}).")

def _refresh_counts_from_dataset(app):
    last = app._last_plot_data
    ds = app.data_handler.get_active_dataset()
    if not (last and ds):
        return
    energies = last['energies']
    file_ids = last.get('file_ids', [])
    new_counts = []
    for fid, e in zip(file_ids, energies):
        spec_df = ds.get_processed_spectrum(fid)
        if spec_df is not None and not spec_df.empty:
            new_counts.append(spec_df['counts'].values[:len(e)])
        else:
            new_counts.append(app._last_plot_data['counts'][len(new_counts)])
    last['counts'] = new_counts

def save_spike_settings_action(app):
    import json
    from tkinter import filedialog
    d = {}
    try:
        d = {
            'window': app.spike_window_entry.get(),
            'sigma': app.spike_sigma_entry.get(),
            'max_width': app.spike_maxw_entry.get(),
            'min_prom': app.spike_prom_entry.get(),
            'method': app.spike_method_var.get(),
            'neighbor': app.spike_neighbor_entry.get(),
            'adaptive': app.spike_adaptive_var.get(),
            'adapt_window': app.spike_adapt_window_entry.get(),
            'manual_radius': app.spike_manual_radius_entry.get(),
            'detect_min': app.spike_detect_min_entry.get(),
            'detect_max': app.spike_detect_max_entry.get(),
            'skip_min': app.spike_skip_min_entry.get(),
            'skip_max': app.spike_skip_max_entry.get()
        }
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')], title='Save Spike Settings')
        if not path:
            return
        with open(path,'w',encoding='utf-8') as f:
            json.dump(d,f,indent=2)
        _spike_log(app, f"Saved spike settings to {path}")
    except Exception as e:
        _spike_log(app, f"Save settings failed: {e}")

def load_spike_settings_action(app):
    import json
    from tkinter import filedialog
    try:
        path = filedialog.askopenfilename(filetypes=[('JSON','*.json')], title='Load Spike Settings')
        if not path:
            return
        with open(path,'r',encoding='utf-8') as f:
            d = json.load(f)
        def set_entry(e, val): e.delete(0,'end'); e.insert(0,str(val))
        if 'window' in d: set_entry(app.spike_window_entry, d['window'])
        if 'sigma' in d: set_entry(app.spike_sigma_entry, d['sigma'])
        if 'max_width' in d: set_entry(app.spike_maxw_entry, d['max_width'])
        if 'min_prom' in d: set_entry(app.spike_prom_entry, d['min_prom'])
        if 'method' in d: app.spike_method_var.set(d['method'])
        if 'neighbor' in d: set_entry(app.spike_neighbor_entry, d['neighbor'])
        if 'adaptive' in d: app.spike_adaptive_var.set(str(d['adaptive']))
        if 'adapt_window' in d: set_entry(app.spike_adapt_window_entry, d['adapt_window'])
        if 'manual_radius' in d: set_entry(app.spike_manual_radius_entry, d['manual_radius'])
        if 'detect_min' in d: set_entry(app.spike_detect_min_entry, d['detect_min'])
        if 'detect_max' in d: set_entry(app.spike_detect_max_entry, d['detect_max'])
        if 'skip_min' in d: set_entry(app.spike_skip_min_entry, d['skip_min'])
        if 'skip_max' in d: set_entry(app.spike_skip_max_entry, d['skip_max'])
        _spike_log(app, f"Loaded spike settings from {path}")
    except Exception as e:
        _spike_log(app, f"Load settings failed: {e}")

def review_next_action(app):
    _review_shift(app, +1)

def review_prev_action(app):
    _review_shift(app, -1)

def review_exit_action(app):
    if hasattr(app, '_spike_review_backup') and app._spike_review_backup:
        app._last_plot_data = app._spike_review_backup
        app._spike_review_backup = None
        _update_plot_style(app)

def _review_shift(app, delta):
    ds = app.data_handler.get_active_dataset()
    last = app._last_plot_data
    if not (ds and last and 'file_ids' in last and last['file_ids']):
        _spike_log(app, "Review: plot a series or selection with file ids.")
        return
    if not hasattr(app, '_spike_review_backup') or app._spike_review_backup is None:
        app._spike_review_backup = last
        app._spike_review_index = 0
    else:
        app._spike_review_index = getattr(app, '_spike_review_index', 0)
    app._spike_review_index = (app._spike_review_index + delta) % len(app._spike_review_backup['file_ids'])
    idx = app._spike_review_index
    fid = app._spike_review_backup['file_ids'][idx]
    spec_df = ds.get_processed_spectrum(fid)
    if spec_df is None or spec_df.empty:
        _spike_log(app, "Review: missing spectrum.")
        return
    e = spec_df['energy_ev'].values
    y = spec_df['counts'].values
    label = app._spike_review_backup['labels'][idx] if idx < len(app._spike_review_backup['labels']) else f"{fid}"
    app._last_plot_data = {'type': 'selected', 'energies': [e], 'counts': [y], 'labels': [label], 'title': label, 'file_ids': [fid]}
    _update_plot_style(app)

def clean_current_curve_action(app):
    """Detect+remove spikes on the currently displayed single curve (review mode)."""
    last = app._last_plot_data
    ds = app.data_handler.get_active_dataset()
    if not (last and ds and len(last.get('energies', [])) == 1 and len(last.get('file_ids', [])) == 1):
        _spike_log(app, "Clean Current: enter review mode (Prev/Next) to show a single curve.")
        return
    e = last['energies'][0]
    y = last['counts'][0]
    fid = last['file_ids'][0]
    window, sigma, maxw, prom, method, neighbor = _get_spike_params(app)
    adaptive = getattr(app, 'spike_adaptive_var', None) and app.spike_adaptive_var.get() == "1"
    hybrid = getattr(app, 'spike_hybrid_var', None) and app.spike_hybrid_var.get() == "1"
    try:
        adapt_w = int(float(app.spike_adapt_window_entry.get())) if getattr(app, 'spike_adapt_window_entry', None) and app.spike_adapt_window_entry.get() else max(51, window)
    except Exception:
        adapt_w = max(51, window)
    try:
        broad_w = int(float(app.spike_hybrid_broad_entry.get())) if getattr(app, 'spike_hybrid_broad_entry', None) and app.spike_hybrid_broad_entry.get() else 151
    except Exception:
        broad_w = 151
    try:
        avoid_w = int(float(app.spike_hybrid_avoidw_entry.get())) if getattr(app, 'spike_hybrid_avoidw_entry', None) and app.spike_hybrid_avoidw_entry.get() else 9
    except Exception:
        avoid_w = 9
    try:
        avoid_p = float(app.spike_hybrid_avoidp_entry.get()) if getattr(app, 'spike_hybrid_avoidp_entry', None) and app.spike_hybrid_avoidp_entry.get() else 0.0
    except Exception:
        avoid_p = 0.0
    if hybrid:
        idxs = spike_removal.detect_spikes_hybrid(e, y, base_window=window, mad_window=adapt_w, sigma=sigma, max_width=maxw, min_prominence=prom, broad_window=broad_w, avoid_width=avoid_w, avoid_prominence=avoid_p)
    elif adaptive:
        idxs = spike_removal.detect_spikes_adaptive(e, y, base_window=window, mad_window=adapt_w, sigma=sigma, max_width=maxw, min_prominence=prom)
    else:
        idxs = spike_removal.detect_spikes(e, y, window=window, sigma=sigma, max_width=maxw, min_prominence=prom)
    cleaned = spike_removal.remove_spikes(e, y, idxs, method=method, neighbor_n=neighbor)
    spec_df = ds.get_processed_spectrum(fid)
    if spec_df is not None and not spec_df.empty:
        # stash original once
        if not hasattr(ds, '_spike_original_counts'):
            ds._spike_original_counts = {}
        if fid not in ds._spike_original_counts:
            ds._spike_original_counts[fid] = spec_df['counts'].values.copy()
        spec_df['counts'] = cleaned[:len(spec_df)]
        ds.processed_spectra[fid] = spec_df
        last['counts'] = [cleaned]
        _update_plot_style(app)
        _spike_log(app, f"Cleaned current curve with {len(idxs)} spikes removed.")

def clean_and_next_action(app):
    clean_current_curve_action(app)
    review_next_action(app)

def _spike_log(app, msg: str):
    try:
        if getattr(app, 'spike_log', None):
            app.spike_log.insert('end', msg + "\n")
            app.spike_log.see('end')
            return
    except Exception:
        pass
    # Fallback to console
    logger.info(msg)


def resolution_changed_action(app, resolution):
    """Handle resolution change from the dropdown."""
    if resolution == 'Custom':
        messagebox.showinfo("Custom Resolution", "Custom resolution not yet implemented. Please select a preset.")
        return
    
    try:
        config.save_window_resolution(resolution)
        messagebox.showinfo("Resolution Saved", f"Resolution {resolution} will be applied on next startup.")
    except Exception as e:
        logger.error(f"Error saving resolution: {e}", exc_info=True)
        messagebox.showerror("Error", f"Failed to save resolution: {e}")


def ui_scale_changed_action(app, scale_str):
    """Handle UI scale change from the dropdown."""
    try:
        # Convert percentage string to float (e.g., "85%" -> 0.85)
        scale_value = float(scale_str.rstrip('%')) / 100.0
        config.save_ui_scale(scale_value)
        messagebox.showinfo("UI Scale Saved", f"UI scale {scale_str} will be applied on next startup.")
    except Exception as e:
        logger.error(f"Error saving UI scale: {e}", exc_info=True)
        messagebox.showerror("Error", f"Failed to save UI scale: {e}")

