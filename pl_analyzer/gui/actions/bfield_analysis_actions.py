import logging
from tkinter import TclError, messagebox
from typing import Optional

import numpy as np

from pl_analyzer.core import analysis
from pl_analyzer.gui.windows.bfield_integration_window import EnhancedBFieldIntegrateWindow
from pl_analyzer.gui.windows.gfactor_analysis_window import GFactorAnalysisWindow
from pl_analyzer.gui.windows.bfield_intensity_map_window import BFieldIntensityMapWindow

logger = logging.getLogger(__name__)


def _get_bfield_pol_mode(app) -> str:
    display_value = getattr(app, 'bfield_pol_mode_var', None).get() if getattr(app, 'bfield_pol_mode_var', None) else "Both (σ+&σ-)"
    mapping = getattr(app, '_bfield_pol_display_to_value', {})
    return mapping.get(display_value, 'both')


def _ensure_bfield_integration_window(app) -> EnhancedBFieldIntegrateWindow:
    window = getattr(app, 'bfield_integration_window', None)
    exists = False
    if window is not None:
        try:
            exists = bool(window.winfo_exists())
        except TclError:
            exists = False
    if not exists:
        window = EnhancedBFieldIntegrateWindow(master=app)
        app.bfield_integration_window = window
    return window


def _ensure_gfactor_window(app) -> GFactorAnalysisWindow:
    window = getattr(app, 'gfactor_analysis_window', None)
    if window is None or not window.winfo_exists():
        window = GFactorAnalysisWindow(master=app)
        app.gfactor_analysis_window = window
    return window


def _ensure_bfield_intensity_window(app) -> BFieldIntensityMapWindow:
    window = getattr(app, 'bfield_intensity_window', None)
    exists = False
    if window is not None:
        try:
            exists = bool(window.winfo_exists())
        except TclError:
            exists = False
    if not exists:
        window = BFieldIntensityMapWindow(master=app)
        app.bfield_intensity_window = window
    return window


def _parse_energy_window(app) -> Optional[tuple]:
    min_entry = getattr(app, 'bfield_min_entry', None)
    max_entry = getattr(app, 'bfield_max_entry', None)
    min_text = min_entry.get().strip() if min_entry else ""
    max_text = max_entry.get().strip() if max_entry else ""

    if not min_text and not max_text:
        messagebox.showerror("Integration Range", "Please provide both minimum and maximum energy (eV) values.")
        return None
    if not min_text or not max_text:
        messagebox.showerror("Integration Range", "Both minimum and maximum energy (eV) are required.")
        return None
    try:
        min_e = float(min_text)
        max_e = float(max_text)
    except ValueError:
        messagebox.showerror("Integration Range", "Energy limits must be valid numbers.")
        return None
    if min_e >= max_e:
        messagebox.showerror("Integration Range", "Minimum energy must be less than maximum energy.")
        return None
    return (min_e, max_e)


def integrate_vs_bfield_action(app) -> None:
    logger.info("B-Field Analysis: Integrate vs B-Field.")
    dataset = app.data_handler.get_active_dataset()
    if not dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    integration_range = _parse_energy_window(app)
    if integration_range is None:
        return

    pol_mode = _get_bfield_pol_mode(app)
    try:
        # Always fetch all polarizations; filtering is handled in the integration window.
        result_df = analysis.bfield_intensity_analysis(dataset, integration_range, 'all')
    except Exception as exc:
        logger.error(f"B-field intensity analysis failed: {exc}", exc_info=True)
        messagebox.showerror("Analysis Error", f"Could not compute intensity vs B-field.\nDetails: {exc}")
        return

    if result_df.empty:
        messagebox.showinfo("No Data", "No spectra matched the requested polarization and B-field filters.")
        return

    result_df = result_df.sort_values('bfield_t').reset_index(drop=True)
    result_df['intensity_sum'] = result_df[['intensity_sigma_plus', 'intensity_sigma_minus']].sum(axis=1, min_count=2)
    try:
        result_df['dcp'] = analysis.calculate_dcp(
            result_df['intensity_sigma_plus'].values,
            result_df['intensity_sigma_minus'].values
        )
    except Exception as exc:
        logger.error(f"Failed to calculate DCP: {exc}", exc_info=True)
        result_df['dcp'] = np.nan

    window = _ensure_bfield_integration_window(app)
    window.set_data(
        result_df,
        integration_range=integration_range,
        dataset_name=getattr(dataset, 'name', 'Active Dataset'),
        default_mode=pol_mode
    )
    window.deiconify()
    window.lift()
    window.focus_force()

    series_counts = []
    if result_df['intensity_sigma_plus'].notna().any():
        series_counts.append(f"σ+: {result_df['intensity_sigma_plus'].notna().sum()} pts")
    if result_df['intensity_sigma_minus'].notna().any():
        series_counts.append(f"σ-: {result_df['intensity_sigma_minus'].notna().sum()} pts")
    if result_df['intensity_sum'].notna().any():
        series_counts.append(f"Sum: {result_df['intensity_sum'].notna().sum()} pts")
    summary = " | ".join(series_counts) if series_counts else "No valid intensity data."
    if getattr(app, 'bfield_results_label', None):
        app.bfield_results_label.configure(text=f"Intensity vs B-field: {summary}")


def calculate_gfactor_action(app) -> None:
    logger.info("B-Field Analysis: Calculate g-factor.")
    dataset = app.data_handler.get_active_dataset()
    if not dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    energy_window = _parse_energy_window(app)
    if energy_window is None:
        return

    pol_mode = _get_bfield_pol_mode(app)
    if pol_mode in ('sigma+', 'sigma-'):
        messagebox.showinfo(
            "Polarization Notice",
            "Advanced g-factor analysis requires both sigma+ and sigma- spectra. "
            "The analysis window will use any available pairs automatically."
        )

    window = _ensure_gfactor_window(app)
    try:
        window.configure_dataset(dataset, energy_range=energy_window, default_method='centroid')
    except Exception as exc:
        logger.error(f"Failed to open g-factor analysis window: {exc}", exc_info=True)
        messagebox.showerror("g-Factor Analysis", f"Unable to open analysis window.\nDetails: {exc}")
        return

    if getattr(app, 'bfield_results_label', None):
        summary = window.results_label.cget("text")
        label_text = f"g-Factor: {summary}" if summary else "g-Factor analysis window opened."
        app.bfield_results_label.configure(text=label_text)


def open_bfield_intensity_map_action(app) -> None:
    logger.info("B-Field Analysis: Open intensity map window.")
    dataset = app.data_handler.get_active_dataset()
    if not dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    last_plot = getattr(app, "_last_plot_data", None)
    if not last_plot or last_plot.get("type") != "bfield_series":
        messagebox.showinfo(
            "B-Field Intensity Map",
            "Please plot a B-field series first using 'Plot B-Field Series' to prepare the data."
        )
        return

    try:
        energies = last_plot.get("energies", [])
        counts = last_plot.get("counts", [])
        y_values = last_plot.get("y_values", [])
        polarizations = last_plot.get("polarizations", [])
    except Exception as exc:
        logger.error(f"Invalid B-field series data for intensity map: {exc}", exc_info=True)
        messagebox.showerror("B-Field Intensity Map", "Could not read the current B-field series data.")
        return

    if not energies or not counts:
        messagebox.showinfo(
            "B-Field Intensity Map",
            "No spectra available in the current B-field series selection."
        )
        return

    window = _ensure_bfield_intensity_window(app)
    try:
        window.set_source_data(
            energies=energies,
            counts=counts,
            bfield_values=y_values,
            polarizations=polarizations,
            dataset_name=getattr(dataset, "name", "Active Dataset"),
            base_title=last_plot.get("title", "B-Field Series")
        )
    except Exception as exc:
        logger.error(f"Failed to initialise B-field intensity map window: {exc}", exc_info=True)
        messagebox.showerror("B-Field Intensity Map", f"Unable to initialise the window.\nDetails: {exc}")
        return

    window.deiconify()
    window.lift()
    window.focus_force()
