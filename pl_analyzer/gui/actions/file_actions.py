import logging
from tkinter import messagebox, filedialog
import customtkinter as ctk
import pandas as pd
import numpy as np
from pl_analyzer.core import export, file_parser
from pl_analyzer.gui.dialogs import ExportFormatDialog, MagneticSweepSettingsDialog, PolarizationSettingsDialog
from pl_analyzer.utils.config import (
    load_magnetic_sweep_config,
    save_magnetic_sweep_config,
    load_polarization_config,
    save_polarization_config,
)
from pl_analyzer.core.file_parser import set_polarization_config, set_magnetic_sweep_config

logger = logging.getLogger(__name__)


def load_dataset_action(app):
    """Action to create a new dataset from selected files."""
    logger.info("Load New Dataset action initiated.")
    try:
        filepaths = filedialog.askopenfilenames(
            title="Select PL Data Files for New Dataset",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepaths:
            logger.info("No files selected for new dataset.")
            return

        dataset_name = app.data_handler.create_new_dataset(filepaths)
        if dataset_name:
            messagebox.showinfo("Load Successful", f"Successfully created '{dataset_name}' with {len(filepaths)} files.")
            app.left_panel.add_dataset_tab(dataset_name)
            app.update_file_table()
            app._initialize_processing_state()
            app._update_processing_button_states()
        else:
            messagebox.showwarning("Load Failed", "No files were successfully loaded into the new dataset.")
    except Exception as e:
        logger.error(f"Error creating new dataset: {e}", exc_info=True)
        messagebox.showerror("Load Error", f"An error occurred while creating the dataset:\n{e}")


def add_files_to_dataset_action(app):
    """Action to add files to the currently active dataset."""
    logger.info("Add Files to Dataset action initiated.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Active Dataset", "Please create a dataset first.")
        return

    filepaths = filedialog.askopenfilenames(
        title=f"Add Files to '{active_dataset.name}'",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not filepaths:
        logger.info("No files selected to add.")
        return

    num_loaded = active_dataset.load_files(list(filepaths))
    if num_loaded > 0:
        messagebox.showinfo("Add Successful", f"Successfully added {num_loaded} files to '{active_dataset.name}'.")
        app.update_file_table()
    else:
        messagebox.showwarning("Add Failed", "No new files were successfully loaded.")


def create_sum_dataset_action(app):
    """Create or refresh a virtual dataset containing sigma+ + sigma- sums."""
    logger.info("Create Sum Dataset action initiated.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Active Dataset", "Please select a dataset first.")
        return

    metadata_df = active_dataset.get_metadata()
    if metadata_df is None or metadata_df.empty or 'polarization' not in metadata_df.columns:
        messagebox.showinfo("Unavailable", "Active dataset does not contain polarization metadata.")
        return

    try:
        dataset_name, unmatched = app.data_handler.create_sum_dataset(active_dataset.name)
    except (ValueError, KeyError) as exc:
        logger.error(f"Failed to create sum dataset: {exc}", exc_info=True)
        messagebox.showerror("Sum Dataset Error", f"Unable to create sum dataset:\n{exc}")
        return

    if not dataset_name:
        messagebox.showinfo("No Matching Pairs", "No matching sigma+/sigma- pairs were found to sum.")
        return

    app.left_panel.add_dataset_tab(dataset_name)
    app.update_file_table()
    app._initialize_processing_state()
    app._update_processing_button_states()

    new_dataset = app.data_handler.get_active_dataset()
    num_pairs = len(new_dataset.get_metadata()) if new_dataset and new_dataset.get_metadata() is not None else 0
    messagebox.showinfo("Sum Dataset Created", f"Created '{dataset_name}' with {num_pairs} summed spectra.")

    if unmatched:
        messagebox.showwarning(
            "Incomplete Pairs",
            f"Skipped {len(unmatched)} unmatched sigma+/sigma- set(s)."
        )


def import_spectra_action(app):
    """Action to import spectra from a previously exported file."""
    logger.info("Import Spectra action initiated.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        dataset_name = app.data_handler.create_empty_dataset()
        app.left_panel.add_dataset_tab(dataset_name)
        active_dataset = app.data_handler.get_active_dataset()

    filepath = filedialog.askopenfilename(
        title="Select Spectra File to Import",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not filepath:
        logger.info("No file selected for import.")
        return

    try:
        spectra_df = file_parser.parse_exported_spectra(filepath)
        if spectra_df is not None and not spectra_df.empty:
            active_dataset.add_spectra_from_dataframe(spectra_df)
            messagebox.showinfo("Import Successful", f"Successfully imported {len(spectra_df)} spectra into '{active_dataset.name}'.")
            app.update_file_table()
        else:
            messagebox.showwarning("Import Failed", "Could not parse any spectra from the selected file.")
    except Exception as e:
        logger.error(f"Error importing spectra: {e}", exc_info=True)
        messagebox.showerror("Import Error", f"An error occurred while importing spectra:\n{e}")


def clear_all_data_action(app):
    """Callback for the Clear All Data button."""
    logger.info("Clear All Data button clicked.")
    if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all datasets and loaded data?"):
        app.data_handler.clear_all_data()
        logger.info("All data cleared by user.")
        app.left_panel.remove_all_dataset_tabs()
        app.clear_plot_action()
        app._update_processing_button_states()
        messagebox.showinfo("Data Cleared", "All datasets have been cleared.")


def clear_dataset_action(app):
    """Callback for the Clear Dataset button."""
    logger.info("Clear Dataset button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Active Dataset", "Please select a dataset to clear.")
        return

    if messagebox.askyesno("Confirm Clear", f"Are you sure you want to clear the '{active_dataset.name}' dataset?"):
        app.data_handler.clear_dataset(active_dataset.name)
        logger.info(f"Dataset '{active_dataset.name}' cleared by user.")
        app.left_panel.remove_dataset_tab(active_dataset.name)
        app.clear_plot_action()
        app._update_processing_button_states()
        messagebox.showinfo("Dataset Cleared", f"The '{active_dataset.name}' dataset has been cleared.")


def clear_selected_action(app):
    """Callback for the Clear Selected Data button."""
    logger.info("Clear Selected Data button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Active Dataset", "Please select a dataset first.")
        return

    selected_ids = app.left_panel.get_active_file_table().get_selected_file_ids()
    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select one or more files to clear.")
        return

    if messagebox.askyesno("Confirm Clear", f"Are you sure you want to clear the {len(selected_ids)} selected files?"):
        active_dataset.clear_files(selected_ids)
        logger.info(f"{len(selected_ids)} files cleared from '{active_dataset.name}' by user.")
        app.update_file_table()
        app.clear_plot_action()
        app._update_processing_button_states()
        messagebox.showinfo("Data Cleared", f"{len(selected_ids)} files have been cleared.")


def export_integrated_action(app):
    """Exports integrated intensity vs power for the selected temperature series."""
    logger.info("Export Integrated Data button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset or active_dataset.data is None or active_dataset.data.empty:
        messagebox.showwarning("No Data", "No data in the active dataset to export.")
        return

    selected_ids = app.left_panel.get_active_file_table().get_selected_file_ids()
    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select a file from the desired temperature series to export.")
        return

    first_selected_id = selected_ids[0]
    metadata_all = active_dataset.get_metadata()
    try:
        target_temp = metadata_all.loc[metadata_all['file_id'] == first_selected_id, 'temperature_k'].iloc[0]
        if pd.isna(target_temp):
            messagebox.showwarning("Missing Info", "Temperature not found for the selected file.")
            return
    except (IndexError, KeyError):
        messagebox.showerror("Error", f"Could not find metadata for selected file ID: {first_selected_id}")
        return

    try:
        min_e = float(app.integration_min_entry.get())
        max_e = float(app.integration_max_entry.get())
        integration_range = (min_e, max_e)
    except ValueError:
        messagebox.showerror("Invalid Range", "Invalid integration range for export.")
        return

    default_filename = f"{active_dataset.name}_integrated_T{target_temp:.1f}K.txt"
    save_path = filedialog.asksaveasfilename(initialfile=default_filename, defaultextension=".txt")
    if not save_path:
        return

    try:
        # This export function might need adjustment to accept a Dataset object
        # For now, we pass the necessary parts of the dataset.
        export.export_integrated_data(active_dataset, target_temp, integration_range, save_path)
        messagebox.showinfo("Export Successful", f"Integrated data exported to:\n{save_path}")
    except Exception as e:
        logger.error(f"Error exporting integrated data: {e}", exc_info=True)
        messagebox.showerror("Export Error", f"Failed to export data:\n{e}")


def export_spectra_action(app):
    """Exports the processed spectra for the selected files."""
    logger.info("Export Selected Spectra button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "No active dataset to export from.")
        return

    selected_ids = app.left_panel.get_active_file_table().get_selected_file_ids()
    if not selected_ids:
        messagebox.showinfo("No Selection", "Please select one or more files to export.")
        return

    default_filename = f"{active_dataset.name}_selected_spectra.csv"
    save_path = filedialog.asksaveasfilename(initialfile=default_filename, defaultextension=".csv")
    if not save_path:
        return

    try:
        # This export function might need adjustment as well
        export.export_processed_spectra(active_dataset, selected_ids, save_path)
        messagebox.showinfo("Export Successful", f"Selected spectra exported to:\n{save_path}")
    except Exception as e:
        logger.error(f"Error exporting selected spectra: {e}", exc_info=True)
        messagebox.showerror("Export Error", f"Failed to export spectra:\n{e}")


def export_all_spectra_action(app):
    """Exports all processed spectra from the active dataset."""
    logger.info("Export All Spectra button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "No active dataset to export from.")
        return

    default_filename = f"{active_dataset.name}_all_spectra.csv"
    save_path = filedialog.asksaveasfilename(initialfile=default_filename, defaultextension=".csv")
    if not save_path:
        return

    try:
        export.export_all_processed_spectra(active_dataset, save_path)
        messagebox.showinfo("Export Successful", f"All spectra exported to:\n{save_path}")
    except Exception as e:
        logger.error(f"Error exporting all spectra: {e}", exc_info=True)
        messagebox.showerror("Export Error", f"Failed to export spectra:\n{e}")


def export_integrated_all_action(app):
    """Exports integrated intensity vs power for all data."""
    logger.info("Export All Integrated Data button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset or active_dataset.data is None or active_dataset.data.empty:
        messagebox.showwarning("No Data", "No data in the active dataset to export.")
        return

    try:
        min_e = float(app.integration_min_entry.get())
        max_e = float(app.integration_max_entry.get())
        integration_range = (min_e, max_e)
    except ValueError:
        messagebox.showerror("Invalid Range", "Invalid integration range for export.")
        return

    # Ask user for format
    dialog = ExportFormatDialog(app, title="Export Format")
    format_choice = dialog.get_format()

    if not format_choice:
        logger.info("Export format selection cancelled.")
        return

    default_filename = f"{active_dataset.name}_integrated_all.txt"
    save_path = filedialog.asksaveasfilename(initialfile=default_filename, defaultextension=".txt")
    if not save_path:
        return

    try:
        export.export_integrated_data_all(active_dataset, integration_range, save_path, format_choice)
        messagebox.showinfo("Export Successful", f"Integrated data exported to:\n{save_path}")
    except Exception as e:
        logger.error(f"Error exporting integrated data: {e}", exc_info=True)
        messagebox.showerror("Export Error", f"Failed to export data:\n{e}")


def polarization_settings_action(app):
    """Action to open Polarization Settings dialog and update configuration."""
    logger.info("Polarization Settings action initiated.")
    from pl_analyzer.gui.dialogs import PolarizationSettingsDialog

    try:
        # Load current configuration
        current_config = load_polarization_config()
        
        # Show dialog
        dialog = PolarizationSettingsDialog(app, current_config)
        new_config = dialog.get_config()
        
        if new_config:
            # Save to config file
            save_polarization_config(new_config['sigma_plus'], new_config['sigma_minus'])
            
            # Update file parser configuration
            set_polarization_config(new_config['sigma_plus'], new_config['sigma_minus'])
            
            # Ask if user wants to re-parse existing datasets
            if app.data_handler.datasets:
                response = messagebox.askyesno(
                    "Re-parse Datasets?",
                    "Would you like to re-parse all loaded datasets with the new polarization settings?\n\n"
                    "This will update the polarization detection for all files.",
                    parent=app
                )
                
                if response:
                    # Re-parse all datasets
                    for dataset_name, dataset in app.data_handler.datasets.items():
                        for file_id, metadata in dataset.metadata.items():
                            filename = metadata.get('filename', '')
                            new_polarization = file_parser._detect_polarization(filename)
                            metadata['polarization'] = new_polarization
                        dataset._update_main_dataframe()
                    
                    # Refresh file table
                    app.update_file_table()
                    messagebox.showinfo("Success", "Polarization settings updated and datasets re-parsed.")
                else:
                    messagebox.showinfo("Success", "Polarization settings updated.\nNew files will use the updated configuration.")
        else:
            messagebox.showinfo("Success", "Polarization settings saved.\nThe new configuration will be used for future file loading.")
                
    except Exception as e:
        logger.error(f"Error in polarization settings: {e}", exc_info=True)
        messagebox.showerror("Settings Error", f"An error occurred:\n{e}")


def magnetic_sweep_settings_action(app):
    """Open settings dialog for magnetic sweep filename parsing defaults."""
    logger.info("Magnetic sweep settings action initiated.")
    try:
        current_config = load_magnetic_sweep_config()
        dialog = MagneticSweepSettingsDialog(app, current_config)
        new_config = dialog.get_config()
        if not new_config:
            return

        save_magnetic_sweep_config(new_config)
        set_magnetic_sweep_config(new_config)

        if app.data_handler.datasets:
            response = messagebox.askyesno(
                "Re-parse Datasets?",
                "Apply new magnetic sweep settings to all currently loaded datasets?",
                parent=app
            )
            if response:
                for dataset in app.data_handler.datasets.values():
                    for file_id, metadata in dataset.metadata.items():
                        filename = metadata.get('filename', '')
                        parsed = file_parser.parse_filename(filename)
                        if parsed:
                            for key in ('temperature_k', 'power_uw', 'time_s', 'bfield_t', 'polarization'):
                                value = parsed.get(key, metadata.get(key))
                                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                    metadata[key] = value
                    dataset._update_main_dataframe()
                app.update_file_table()
                messagebox.showinfo("Success", "Magnetic sweep settings applied to all datasets.")
            else:
                messagebox.showinfo("Success", "Magnetic sweep settings saved. New files will use the updated configuration.")
        else:
            messagebox.showinfo("Success", "Magnetic sweep settings saved. New files will use the updated configuration.")
    except Exception as e:
        logger.error(f"Error in magnetic sweep settings: {e}", exc_info=True)
        messagebox.showerror("Settings Error", f"An error occurred:\n{e}")
