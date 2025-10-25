import logging
from tkinter import messagebox
import numpy as np
import pandas as pd

from pl_analyzer.core import analysis
from pl_analyzer.gui.analysis_view import AnalysisView
from pl_analyzer.gui.dialogs import IntegrationRangeDialog
from pl_analyzer.gui.temp_analysis_view import TempAnalysisView

logger = logging.getLogger(__name__)

def integrate_selected_action(app):
    """Performs integration on selected spectra within the specified range."""
    logger.info("Integrate Selected button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()

    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select one or more files to integrate.")
        return

    try:
        min_e = float(app.integration_min_entry.get())
        max_e = float(app.integration_max_entry.get())
        integration_range = (min_e, max_e)
    except ValueError:
        messagebox.showerror("Invalid Range", "Invalid integration range.")
        return

    results = {}
    for file_id in selected_ids:
        spectrum_df = active_dataset.get_processed_spectrum(file_id)
        if spectrum_df is not None and not spectrum_df.empty:
            integral = analysis.integrate_spectrum(
                spectrum_df['energy_ev'].values,
                spectrum_df['counts'].values,
                integration_range
            )
            if integral is not None:
                results[file_id] = integral

    if results:
        filename = active_dataset.metadata.get(list(results.keys())[0], {}).get('filename', 'file')
        result_str = f"Integral for {filename}: {list(results.values())[0]:.4g}"
        app.analysis_results_label.configure(text=result_str)
        messagebox.showinfo("Integration Complete", f"Calculated integrals for {len(results)} files.")
    else:
        messagebox.showwarning("Integration Failed", "Could not calculate integral for any selected files.")

def power_dependence_analysis_action(app):
    """Opens the AnalysisView window with integrated data for a power series."""
    logger.info("Power Dependence Analysis button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()

    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select a file from a temperature series.")
        return

    try:
        min_e = float(app.integration_min_entry.get())
        max_e = float(app.integration_max_entry.get())
        integration_range = (min_e, max_e)
    except ValueError:
        messagebox.showerror("Invalid Range", "Invalid integration range.")
        return

    metadata_all = active_dataset.get_metadata()
    try:
        target_temp = metadata_all.loc[metadata_all['file_id'] == selected_ids[0], 'temperature_k'].iloc[0]
        series_df = metadata_all[metadata_all['temperature_k'] == target_temp].sort_values(by='power_uw')

        powers, intensities = [], []
        for _, row in series_df.iterrows():
            spectrum_df = active_dataset.get_processed_spectrum(row['file_id'])
            if spectrum_df is not None and not spectrum_df.empty:
                integral = analysis.integrate_spectrum(
                    spectrum_df['energy_ev'].values,
                    spectrum_df['counts'].values,
                    integration_range
                )
                if integral is not None and integral > 0:
                    powers.append(row['power_uw'])
                    intensities.append(integral)

        if len(powers) < 2:
            messagebox.showwarning("Not Enough Data", "Need at least 2 valid data points for analysis.")
            return

        if app.analysis_window is None or not app.analysis_window.winfo_exists():
            app.analysis_window = AnalysisView(master=app)
        app.analysis_window.load_series_data(np.array(powers), np.array(intensities), target_temp)
        app.analysis_window.lift()

    except (IndexError, KeyError):
        messagebox.showerror("Error", "Could not find metadata for the selected file.")

def temp_dependence_analysis_action(app):
    """Opens the TempAnalysisView window with integrated data for a temperature series."""
    logger.info("Temperature Dependence Analysis button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    active_table = app.left_panel.get_active_file_table()
    if not active_table:
        return
    selected_ids = active_table.get_selected_file_ids()

    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select a file from a power series.")
        return

    try:
        min_e = float(app.integration_min_entry.get())
        max_e = float(app.integration_max_entry.get())
        integration_range = (min_e, max_e)
    except ValueError:
        messagebox.showerror("Invalid Range", "Invalid integration range.")
        return

    metadata_all = active_dataset.get_metadata()
    try:
        target_power = metadata_all.loc[metadata_all['file_id'] == selected_ids[0], 'power_uw'].iloc[0]
        series_df = metadata_all[metadata_all['power_uw'] == target_power].sort_values(by='temperature_k')

        records = series_df[['file_id', 'temperature_k']].to_dict('records')
        if len(records) < 2:
            messagebox.showwarning("Not Enough Data", "Need at least 2 files at the selected power to analyse.", parent=app)
            return

        if app.temp_analysis_window is None or not app.temp_analysis_window.winfo_exists():
            app.temp_analysis_window = TempAnalysisView(master=app)
        app.temp_analysis_window.load_series_data(active_dataset, records, float(target_power), integration_range)
        app.temp_analysis_window.lift()

    except (IndexError, KeyError):
        messagebox.showerror("Error", "Could not find metadata for the selected file.")
