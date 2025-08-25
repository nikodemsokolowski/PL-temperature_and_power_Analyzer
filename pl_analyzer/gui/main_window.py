import customtkinter as ctk
import logging
from tkinter import messagebox, filedialog # Added filedialog
import pandas as pd # Added pandas

import numpy as np # Added numpy import

# Import the DataHandler
from ..core.data_handler import DataHandler
# Import processing functions
from ..core import processing
# Import analysis functions
from ..core import analysis
# Import export functions
from ..core import export
# Import dialogs
from .dialogs import GfFactorDialog
# Import other GUI components as they are created
from .widgets.file_table import FileTable
from .widgets.plot_canvas import PlotCanvas
from .analysis_view import AnalysisView

logger = logging.getLogger(__name__)

class MainWindow(ctk.CTkFrame):
    """
    The main window frame for the PL Analyzer application.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master # Reference to the root CTk window
        self.data_handler = DataHandler()
        self._initialize_processing_state() # Initialize flags

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1) # Plot area column expands
        self.grid_rowconfigure(0, weight=1)    # Main content row expands

        # --- Create main frames ---
        self.left_panel = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_rowconfigure(1, weight=1) # File list area expands

        self.right_panel = ctk.CTkFrame(self, corner_radius=0)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.right_panel.grid_rowconfigure(0, weight=1) # Plot area expands
        self.right_panel.grid_columnconfigure(0, weight=1)

        # --- Left Panel Widgets ---
        self.controls_frame = ctk.CTkFrame(self.left_panel)
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.load_button = ctk.CTkButton(self.controls_frame, text="Load Files", command=self.load_files_action)
        self.load_button.pack(pady=5, padx=10, fill="x")

        self.clear_button = ctk.CTkButton(self.controls_frame, text="Clear Data", command=self.clear_data_action, fg_color="red")
        self.clear_button.pack(pady=5, padx=10, fill="x")

        # Export Controls Frame (within left_panel controls)
        self.export_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.export_frame.pack(pady=(10,5), padx=5, fill="x")
        self.export_frame.grid_columnconfigure((0, 1), weight=1)

        self.export_label = ctk.CTkLabel(self.export_frame, text="Export Data:", font=ctk.CTkFont(weight="bold"))
        self.export_label.grid(row=0, column=0, columnspan=2, padx=5, pady=(0,2), sticky="w")

        self.export_integrated_button = ctk.CTkButton(self.export_frame, text="Integrated Data", command=self.export_integrated_action)
        self.export_integrated_button.grid(row=1, column=0, padx=5, pady=2, sticky="ew")

        self.export_spectra_button = ctk.CTkButton(self.export_frame, text="Selected Spectra", command=self.export_spectra_action)
        self.export_spectra_button.grid(row=1, column=1, padx=5, pady=2, sticky="ew")


        # File List/Table Frame
        self.file_list_frame = ctk.CTkFrame(self.left_panel)
        self.file_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.file_list_frame.grid_rowconfigure(0, weight=1)    # Make table expand vertically
        self.file_list_frame.grid_columnconfigure(0, weight=1) # Make table expand horizontally

        # Instantiate FileTable widget
        self.file_table = FileTable(master=self.file_list_frame)
        self.file_table.grid(row=0, column=0, sticky="nsew")

        # Processing Controls Frame
        self.processing_frame = ctk.CTkFrame(self.left_panel)
        self.processing_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.processing_frame.grid_columnconfigure(0, weight=1) # Make buttons expand

        self.processing_label = ctk.CTkLabel(self.processing_frame, text="Processing Steps", font=ctk.CTkFont(weight="bold"))
        self.processing_label.grid(row=0, column=0, padx=10, pady=(5, 2), sticky="w")

        self.normalize_time_button = ctk.CTkButton(self.processing_frame, text="Normalize by Time", command=self.normalize_time_action)
        self.normalize_time_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.rescale_gf_button = ctk.CTkButton(self.processing_frame, text="Rescale by GF Factor", command=self.rescale_gf_action)
        self.rescale_gf_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.correct_response_button = ctk.CTkButton(self.processing_frame, text="Correct Spectrometer Response", command=self.correct_response_action)
        self.correct_response_button.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # --- Right Panel Widgets ---
        # Plot Area Frame (holds canvas and plot controls)
        self.plot_area_frame = ctk.CTkFrame(self.right_panel)
        self.plot_area_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_area_frame.grid_rowconfigure(1, weight=1) # Canvas row expands
        self.plot_area_frame.grid_columnconfigure(0, weight=1) # Canvas col expands

        # Plot Controls Frame
        self.plot_controls_frame = ctk.CTkFrame(self.plot_area_frame, fg_color="transparent")
        self.plot_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        # Add plot controls
        self.plot_selected_button = ctk.CTkButton(self.plot_controls_frame, text="Plot Selected", command=self.plot_selected_action)
        self.plot_selected_button.pack(side="left", padx=(5, 2))

        self.plot_power_series_button = ctk.CTkButton(self.plot_controls_frame, text="Plot Power Series", command=self.plot_power_series_action)
        self.plot_power_series_button.pack(side="left", padx=2)

        self.plot_temp_series_button = ctk.CTkButton(self.plot_controls_frame, text="Plot Temp Series", command=self.plot_temp_series_action)
        self.plot_temp_series_button.pack(side="left", padx=2)

        self.clear_plot_button = ctk.CTkButton(self.plot_controls_frame, text="Clear Plot", command=self.clear_plot_action)
        self.clear_plot_button.pack(side="left", padx=(2, 10))

        # Log scale checkbox - place it to the right
        self.log_scale_var = ctk.StringVar(value="0")
        self.log_scale_checkbox = ctk.CTkCheckBox(self.plot_controls_frame, text="Log Y-Axis", variable=self.log_scale_var, command=self._update_plot_scale)
        self.log_scale_checkbox.pack(side="right", padx=(10, 5))


        # Instantiate PlotCanvas widget
        self.plot_canvas = PlotCanvas(master=self.plot_area_frame)
        self.plot_canvas.grid(row=1, column=0, sticky="nsew")

        # Analysis Controls/Results Frame
        self.analysis_frame = ctk.CTkFrame(self.right_panel, height=100) # Reduced height slightly
        self.analysis_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0,5))
        self.analysis_frame.grid_columnconfigure((1, 3), weight=1) # Allow entry fields to expand a bit

        # Integration Controls
        self.integration_label = ctk.CTkLabel(self.analysis_frame, text="Integration Range (eV):", font=ctk.CTkFont(weight="bold"))
        self.integration_label.grid(row=0, column=0, padx=(10, 2), pady=5, sticky="w")

        self.integration_min_entry = ctk.CTkEntry(self.analysis_frame, placeholder_text="Min E (eV)", width=100)
        self.integration_min_entry.grid(row=0, column=1, padx=2, pady=5, sticky="ew")

        self.integration_max_entry = ctk.CTkEntry(self.analysis_frame, placeholder_text="Max E (eV)", width=100)
        self.integration_max_entry.grid(row=0, column=2, padx=2, pady=5, sticky="ew")

        self.integrate_button = ctk.CTkButton(self.analysis_frame, text="Integrate Selected", command=self.integrate_selected_action)
        self.integrate_button.grid(row=0, column=3, padx=2, pady=5)

        self.power_analysis_button = ctk.CTkButton(self.analysis_frame, text="Power Dependence Analysis", command=self.power_dependence_analysis_action)
        self.power_analysis_button.grid(row=0, column=4, padx=(2, 10), pady=5, sticky="e")


        # Placeholder for results display (e.g., status bar or text box)
        self.analysis_results_label = ctk.CTkLabel(self.analysis_frame, text="", anchor="w")
        self.analysis_results_label.grid(row=1, column=0, columnspan=5, padx=10, pady=5, sticky="ew")

        # Keep track of the analysis window
        self.analysis_window: Optional[AnalysisView] = None

    def _initialize_processing_state(self):
        """Initializes or resets the processing state flags."""
        self._time_normalized = False
        self._gf_rescaled = False
        self._response_corrected = False
        logger.debug("Processing state flags reset.")

    def _update_processing_button_states(self):
        """Enables/disables processing buttons based on state flags."""
        self.normalize_time_button.configure(state="disabled" if self._time_normalized else "normal")
        self.rescale_gf_button.configure(state="disabled" if self._gf_rescaled else "normal")
        self.correct_response_button.configure(state="disabled" if self._response_corrected else "normal")
        logger.debug(f"Processing button states updated: Time={self._time_normalized}, GF={self._gf_rescaled}, Resp={self._response_corrected}")


    def load_files_action(self):
        """Callback for the Load Files button."""
        logger.info("Load Files button clicked.")
        try:
            num_loaded = self.data_handler.load_files() # Opens file dialog
            if num_loaded > 0:
                messagebox.showinfo("Load Successful", f"Successfully loaded {num_loaded} files.")
                logger.info(f"Successfully loaded {num_loaded} files via dialog.")
                # Reset processing state for new data
                self._initialize_processing_state()
                self._update_processing_button_states()
                # Update the file table widget
                self.update_file_table()
            else:
                logger.info("No files were loaded.")
                # messagebox.showinfo("Load Info", "No new files were loaded.") # Optional: can be annoying
        except Exception as e:
            logger.error(f"Error during file loading: {e}", exc_info=True)
            messagebox.showerror("Load Error", f"An error occurred while loading files:\n{e}")

    def clear_data_action(self):
        """Callback for the Clear Data button."""
        logger.info("Clear Data button clicked.")
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all loaded data?"):
            self.data_handler.clear_data()
            logger.info("Data cleared by user.")
            # Update file table and clear plot
            self.update_file_table()
            self.clear_plot_action() # Clear the plot as well
            # Reset processing state
            self._initialize_processing_state()
            self._update_processing_button_states()
            messagebox.showinfo("Data Cleared", "All loaded data has been cleared.")

    # --- Processing Actions ---
    def normalize_time_action(self):
        """Callback for Normalize by Time button."""
        logger.info("Normalize by Time button clicked.")
        if self._time_normalized:
            messagebox.showinfo("Already Applied", "Time normalization has already been applied to this dataset.")
            return
        if self.data_handler.data is None or self.data_handler.data.empty:
            messagebox.showwarning("No Data", "Please load data before processing.")
            return
        try:
            processing.normalize_by_acquisition_time(self.data_handler)
            self._time_normalized = True # Set flag
            self._update_processing_button_states() # Update buttons
            messagebox.showinfo("Processing Complete", "Acquisition time normalization applied.")
            # TODO: Update plot if needed (e.g., replot currently selected)
            # self.plot_selected_action() # Example: replot selected after processing
        except Exception as e:
            logger.error(f"Error during time normalization: {e}", exc_info=True)
            messagebox.showerror("Processing Error", f"An error occurred during time normalization:\n{e}")

    def rescale_gf_action(self):
        """Callback for Rescale by GF Factor button."""
        logger.info("Rescale by GF Factor button clicked.")
        if self._gf_rescaled:
            messagebox.showinfo("Already Applied", "Grey Filter rescaling has already been applied to this dataset.")
            return
        if self.data_handler.data is None or self.data_handler.data.empty:
            messagebox.showwarning("No Data", "Please load data before processing.")
            return

        # Check if any loaded files actually have GF marked
        if not self.data_handler.metadata or not any(m.get('gf_present', False) for m in self.data_handler.metadata.values()):
            messagebox.showinfo("No GF Data", "No loaded files are marked as using a Grey Filter.")
            return

        dialog = GfFactorDialog(text="Enter GF Transmission Factor (e.g., 0.052):", title="GF Factor")
        factor = dialog.get_input()

        if factor is not None:
            try:
                processing.rescale_by_grey_filter(self.data_handler, factor)
                self._gf_rescaled = True # Set flag
                self._update_processing_button_states() # Update buttons
                messagebox.showinfo("Processing Complete", f"Grey Filter rescaling applied with factor {factor}.")
                # TODO: Update plot if needed
                # self.update_plot()
            except ValueError as ve: # Catch specific error from processing function
                logger.error(f"Invalid GF factor provided: {ve}")
                messagebox.showerror("Input Error", str(ve))
            except Exception as e:
                logger.error(f"Error during GF rescaling: {e}", exc_info=True)
                messagebox.showerror("Processing Error", f"An error occurred during GF rescaling:\n{e}")
        else:
            logger.info("GF rescaling cancelled by user or invalid input.")

    def correct_response_action(self):
        """Callback for Correct Spectrometer Response button."""
        logger.info("Correct Spectrometer Response button clicked.")
        if self._response_corrected:
            messagebox.showinfo("Already Applied", "Spectrometer response correction has already been applied to this dataset.")
            return
        if self.data_handler.data is None or self.data_handler.data.empty:
            messagebox.showwarning("No Data", "Please load data before processing.")
            return
        try:
            processing.correct_spectrometer_response(self.data_handler)
            self._response_corrected = True # Set flag
            self._update_processing_button_states() # Update buttons
            messagebox.showinfo("Processing Complete", "Spectrometer response correction applied (where applicable).")
            # TODO: Update plot if needed
            # self.plot_selected_action() # Example: replot selected after processing
        except Exception as e:
            logger.error(f"Error during spectrometer response correction: {e}", exc_info=True)
            messagebox.showerror("Processing Error", f"An error occurred during response correction:\n{e}")


    # --- Update methods ---
    def update_file_table(self):
        """Updates the file table with the current data from DataHandler."""
        logger.debug("Updating file table...")
        metadata = self.data_handler.get_metadata()
        # Pass metadata to the FileTable widget's update method
        self.file_table.update_data(metadata)

    # --- Plotting Actions ---
    def plot_selected_action(self):
        """Plots the spectra corresponding to the selected rows in the FileTable."""
        logger.info("Plot Selected button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select one or more files from the table to plot.")
            return

        energies = []
        counts = []
        labels = []
        plot_title = "Selected Spectra"

        for file_id in selected_ids:
            spectrum = self.data_handler.get_processed_spectrum(file_id)
            metadata = self.data_handler.metadata.get(file_id, {})
            if spectrum is not None and not spectrum.empty:
                energies.append(spectrum['energy_ev'].values)
                counts.append(spectrum['counts'].values)
                # Create a concise label from parameters
                temp = metadata.get('temperature_k', 'N/A')
                power = metadata.get('power_uw', 'N/A')
                label = f"{metadata.get('filename', file_id)} (T={temp}K, P={power}uW)"
                labels.append(label)
            else:
                logger.warning(f"Could not retrieve processed spectrum for selected file_id: {file_id}")

        if not energies:
            messagebox.showwarning("Plot Error", "Could not load data for any selected files.")
            self.plot_canvas.clear_plot()
            return

        if len(selected_ids) == 1: # Use filename if only one spectrum plotted
            plot_title = labels[0]

        # Get current scale state
        y_scale = 'log' if self.log_scale_var.get() == "1" else 'linear'
        self.plot_canvas.plot_data(energies, counts, labels_list=labels, title=plot_title, y_scale=y_scale)

    def clear_plot_action(self):
        """Clears the plot canvas."""
        logger.debug("Clearing plot via button...")
        # Reset scale to linear when clearing
        self.log_scale_var.set("0")
        self.plot_canvas.clear_plot(y_scale='linear')

    def _update_plot_scale(self):
        """Updates the plot scale based on the checkbox and replots current data if possible."""
        y_scale = 'log' if self.log_scale_var.get() == "1" else 'linear'
        logger.debug(f"Updating plot Y-axis scale to: {y_scale}")
        # Attempt to replot whatever was last plotted (selected, power series, or temp series)
        # This requires storing the last plotted data or re-running the last plot action.
        # Simple approach: just update the scale on the current axes content.
        # This might not replot data if axes are cleared, but works for toggling scale on existing plot.
        self.plot_canvas.set_yscale(y_scale)


    def plot_power_series_action(self):
        """Plots all power data for the temperature of the first selected file."""
        logger.info("Plot Power Series button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select a file from the desired temperature series.")
            return

        first_selected_id = selected_ids[0]
        metadata_all = self.data_handler.get_metadata()
        if metadata_all is None or metadata_all.empty:
            messagebox.showwarning("No Data", "No data loaded.")
            return

        try:
            target_temp = metadata_all.loc[metadata_all['file_id'] == first_selected_id, 'temperature_k'].iloc[0]
            if pd.isna(target_temp):
                messagebox.showwarning("Missing Info", f"Temperature not found for selected file: {first_selected_id}")
                return

            # Filter DataFrame for the target temperature
            series_df = metadata_all[metadata_all['temperature_k'] == target_temp].sort_values(by='power_uw')
            if series_df.empty:
                messagebox.showinfo("No Series Data", f"No other files found for Temperature = {target_temp} K.")
                return

            series_file_ids = series_df['file_id'].tolist()
            logger.debug(f"Found {len(series_file_ids)} files for T={target_temp}K power series.")

            energies = []
            counts = []
            labels = []
            plot_title = f"Power Series at T = {target_temp:.1f} K"

            for file_id in series_file_ids:
                spectrum = self.data_handler.get_processed_spectrum(file_id)
                metadata = self.data_handler.metadata.get(file_id, {})
                if spectrum is not None and not spectrum.empty:
                    energies.append(spectrum['energy_ev'].values)
                    counts.append(spectrum['counts'].values)
                    power = metadata.get('power_uw', 'N/A')
                    labels.append(f"P = {power} uW")
                else:
                    logger.warning(f"Could not retrieve spectrum for file_id: {file_id} in power series.")

            if not energies:
                messagebox.showwarning("Plot Error", f"Could not load data for any files in the T={target_temp}K series.")
                self.plot_canvas.clear_plot()
                return

            y_scale = 'log' if self.log_scale_var.get() == "1" else 'linear'
            self.plot_canvas.plot_data(energies, counts, labels_list=labels, title=plot_title, y_scale=y_scale)

        except IndexError:
            messagebox.showerror("Error", f"Could not find metadata for selected file ID: {first_selected_id}")
        except Exception as e:
            logger.error(f"Error plotting power series: {e}", exc_info=True)
            messagebox.showerror("Plot Error", f"An unexpected error occurred while plotting the power series:\n{e}")


    def plot_temp_series_action(self):
        """Plots all temperature data for the power of the first selected file."""
        logger.info("Plot Temp Series button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select a file from the desired power series.")
            return

        first_selected_id = selected_ids[0]
        metadata_all = self.data_handler.get_metadata()
        if metadata_all is None or metadata_all.empty:
            messagebox.showwarning("No Data", "No data loaded.")
            return

        try:
            target_power = metadata_all.loc[metadata_all['file_id'] == first_selected_id, 'power_uw'].iloc[0]
            if pd.isna(target_power):
                messagebox.showwarning("Missing Info", f"Power not found for selected file: {first_selected_id}")
                return

            # Filter DataFrame for the target power
            series_df = metadata_all[metadata_all['power_uw'] == target_power].sort_values(by='temperature_k')
            if series_df.empty:
                messagebox.showinfo("No Series Data", f"No other files found for Power = {target_power} uW.")
                return

            series_file_ids = series_df['file_id'].tolist()
            logger.debug(f"Found {len(series_file_ids)} files for P={target_power}uW temperature series.")

            energies = []
            counts = []
            labels = []
            plot_title = f"Temperature Series at P = {target_power:.1f} uW"

            for file_id in series_file_ids:
                spectrum = self.data_handler.get_processed_spectrum(file_id)
                metadata = self.data_handler.metadata.get(file_id, {})
                if spectrum is not None and not spectrum.empty:
                    energies.append(spectrum['energy_ev'].values)
                    counts.append(spectrum['counts'].values)
                    temp = metadata.get('temperature_k', 'N/A')
                    labels.append(f"T = {temp} K")
                else:
                    logger.warning(f"Could not retrieve spectrum for file_id: {file_id} in temperature series.")

            if not energies:
                messagebox.showwarning("Plot Error", f"Could not load data for any files in the P={target_power}uW series.")
                self.plot_canvas.clear_plot()
                return

            y_scale = 'log' if self.log_scale_var.get() == "1" else 'linear'
            self.plot_canvas.plot_data(energies, counts, labels_list=labels, title=plot_title, y_scale=y_scale)

        except IndexError:
            messagebox.showerror("Error", f"Could not find metadata for selected file ID: {first_selected_id}")
        except Exception as e:
            logger.error(f"Error plotting temperature series: {e}", exc_info=True)
            messagebox.showerror("Plot Error", f"An unexpected error occurred while plotting the temperature series:\n{e}")

    # --- Analysis Actions ---
    def integrate_selected_action(self):
        """Performs integration on selected spectra within the specified range."""
        logger.info("Integrate Selected button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select one or more files from the table to integrate.")
            return

        # Get and validate integration range
        try:
            min_e_str = self.integration_min_entry.get()
            max_e_str = self.integration_max_entry.get()
            if not min_e_str or not max_e_str:
                raise ValueError("Min and Max energy values cannot be empty.")
            min_e = float(min_e_str)
            max_e = float(max_e_str)
            if min_e >= max_e:
                raise ValueError("Min energy must be less than Max energy.")
            integration_range = (min_e, max_e)
            logger.debug(f"Integration range set to: {integration_range} eV")
        except ValueError as e:
            messagebox.showerror("Invalid Range", f"Invalid integration range: {e}")
            logger.warning(f"Invalid integration range input: {e}")
            return

        results = {}
        skipped_count = 0
        first_result_str = "No results calculated."

        for i, file_id in enumerate(selected_ids):
            spectrum_df = self.data_handler.get_processed_spectrum(file_id)
            if spectrum_df is None or spectrum_df.empty:
                logger.warning(f"No processed spectrum data found for file_id {file_id}. Skipping integration.")
                skipped_count += 1
                continue

            energy = spectrum_df['energy_ev'].values
            counts = spectrum_df['counts'].values

            integral = analysis.integrate_spectrum(energy, counts, integration_range)

            if integral is not None:
                results[file_id] = integral
                filename = self.data_handler.metadata.get(file_id, {}).get('filename', file_id)
                result_str = f"Integral for {filename}: {integral:.4g}"
                logger.info(result_str)
                if i == 0: # Store first result for display
                    first_result_str = result_str
            else:
                logger.warning(f"Integration failed or returned None for file_id {file_id}.")
                skipped_count += 1

        # Display results (simple approach: show first result, log others)
        num_calculated = len(results)
        if num_calculated > 0:
            self.analysis_results_label.configure(text=first_result_str + (f" (+{num_calculated-1} more)" if num_calculated > 1 else ""))
            info_message = f"Integration complete for {num_calculated} file(s)."
            if skipped_count > 0:
                info_message += f"\nSkipped {skipped_count} file(s) due to errors or lack of data in range."
            info_message += "\nSee log file for detailed results."
            messagebox.showinfo("Integration Complete", info_message)
            # TODO: Implement proper results display/export later
        else:
            self.analysis_results_label.configure(text="Integration failed for all selected files.")
            messagebox.showwarning("Integration Failed", f"Could not calculate integral for any selected files (Skipped: {skipped_count}). Check range and data.")
        # Return the results dictionary for potential use elsewhere (like power analysis)
        return results

    def power_dependence_analysis_action(self):
        """Opens the AnalysisView window with integrated data for a power series."""
        logger.info("Power Dependence Analysis button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select a file from the desired temperature series to analyze.")
            return

        # --- Get Integration Range (same as integrate_selected_action) ---
        try:
            min_e_str = self.integration_min_entry.get()
            max_e_str = self.integration_max_entry.get()
            if not min_e_str or not max_e_str:
                raise ValueError("Min and Max energy values cannot be empty.")
            min_e = float(min_e_str)
            max_e = float(max_e_str)
            if min_e >= max_e:
                raise ValueError("Min energy must be less than Max energy.")
            integration_range = (min_e, max_e)
        except ValueError as e:
            messagebox.showerror("Invalid Range", f"Invalid integration range for analysis: {e}")
            return

        # --- Get Power Series Data (similar to plot_power_series_action) ---
        first_selected_id = selected_ids[0]
        metadata_all = self.data_handler.get_metadata()
        if metadata_all is None or metadata_all.empty:
            messagebox.showwarning("No Data", "No data loaded.")
            return

        try:
            target_temp = metadata_all.loc[metadata_all['file_id'] == first_selected_id, 'temperature_k'].iloc[0]
            if pd.isna(target_temp):
                messagebox.showwarning("Missing Info", f"Temperature not found for selected file: {first_selected_id}")
                return

            series_df = metadata_all[metadata_all['temperature_k'] == target_temp].sort_values(by='power_uw')
            if series_df.empty:
                messagebox.showinfo("No Series Data", f"No files found for Temperature = {target_temp} K.")
                return

            series_file_ids = series_df['file_id'].tolist()

            # --- Integrate each spectrum in the series ---
            powers = []
            integrated_intensities = []
            integration_failed_count = 0

            for file_id in series_file_ids:
                spectrum_df = self.data_handler.get_processed_spectrum(file_id)
                power_val = self.data_handler.metadata.get(file_id, {}).get('power_uw')

                if spectrum_df is None or spectrum_df.empty or power_val is None or pd.isna(power_val):
                    logger.warning(f"Skipping file {file_id} in power series analysis due to missing data/power.")
                    integration_failed_count += 1
                    continue

                integral = analysis.integrate_spectrum(
                    spectrum_df['energy_ev'].values,
                    spectrum_df['counts'].values,
                    integration_range
                )

                if integral is not None and integral > 0: # Ensure positive intensity for log plot/fit
                    powers.append(power_val)
                    integrated_intensities.append(integral)
                else:
                    logger.warning(f"Integration failed or result non-positive for file {file_id}. Skipping.")
                    integration_failed_count += 1

            # Filter out non-positive intensity values before passing to analysis view/fit
            powers_np = np.array(powers)
            intensities_np = np.array(integrated_intensities)
            positive_mask = intensities_np > 0
            powers_filtered = powers_np[positive_mask]
            intensities_filtered = intensities_np[positive_mask]

            num_filtered_out = len(powers_np) - len(powers_filtered)
            if num_filtered_out > 0:
                logger.warning(f"Filtered out {num_filtered_out} non-positive intensity points for power analysis.")
                messagebox.showwarning("Data Filtered", f"Filtered out {num_filtered_out} non-positive integrated intensity points before analysis.")


            if len(powers_filtered) < 2:
                messagebox.showwarning("Not Enough Data", f"Need at least 2 valid positive data points for power dependence analysis. Found {len(powers_filtered)} after filtering.")
                return

            # --- Open Analysis Window ---
            # Check if window already exists
            if self.analysis_window is None or not self.analysis_window.winfo_exists():
                logger.info("Creating new AnalysisView window.")
                self.analysis_window = AnalysisView(master=self) # Pass main window as master
                # Pass the filtered data
                self.analysis_window.load_series_data(powers_filtered, intensities_filtered, target_temp)
            else:
                logger.info("AnalysisView window already open. Loading new data.")
                # Pass the filtered data
                self.analysis_window.load_series_data(powers_filtered, intensities_filtered, target_temp)
                self.analysis_window.lift() # Bring window to front
                self.analysis_window.focus()

            if integration_failed_count > 0:
                messagebox.showwarning("Integration Issues", f"Skipped {integration_failed_count} files during integration for the analysis. Check logs.")


        except IndexError:
            messagebox.showerror("Error", f"Could not find metadata for selected file ID: {first_selected_id}")
        except Exception as e:
            logger.error(f"Error during power dependence analysis setup: {e}", exc_info=True)
            messagebox.showerror("Analysis Error", f"An unexpected error occurred:\n{e}")

    # --- Export Actions ---
    def export_integrated_action(self):
        """Exports integrated intensity vs power for the selected temperature series."""
        logger.info("Export Integrated Data button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select a file from the desired temperature series to export.")
            return
        if self.data_handler.data is None or self.data_handler.data.empty:
            messagebox.showwarning("No Data", "No data loaded.")
            return

        # Get target temperature from the first selected file
        first_selected_id = selected_ids[0]
        metadata_all = self.data_handler.get_metadata()
        try:
            target_temp = metadata_all.loc[metadata_all['file_id'] == first_selected_id, 'temperature_k'].iloc[0]
            if pd.isna(target_temp):
                messagebox.showwarning("Missing Info", f"Temperature not found for selected file: {first_selected_id}. Cannot determine series.")
                return
        except IndexError:
            messagebox.showerror("Error", f"Could not find metadata for selected file ID: {first_selected_id}")
            return
        except Exception as e:
            logger.error(f"Error getting target temperature for export: {e}", exc_info=True)
            messagebox.showerror("Error", f"An unexpected error occurred getting series info:\n{e}")
            return

        # Get integration range (reuse logic from integrate_selected)
        try:
            min_e_str = self.integration_min_entry.get()
            max_e_str = self.integration_max_entry.get()
            if not min_e_str or not max_e_str: raise ValueError("Min/Max energy required.")
            min_e = float(min_e_str)
            max_e = float(max_e_str)
            if min_e >= max_e: raise ValueError("Min energy < Max energy required.")
            integration_range = (min_e, max_e)
        except ValueError as e:
            messagebox.showerror("Invalid Range", f"Invalid integration range for export: {e}")
            return

        # Get save path - suggest filename with temperature
        default_filename = f"integrated_data_T{target_temp:.1f}K.txt"
        save_path = filedialog.asksaveasfilename(
            title="Save Integrated Data As",
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=[("Text files (Tab separated)", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not save_path:
            logger.info("Export integrated data cancelled by user.")
            return

        # Perform export for the specific temperature
        try:
            export.export_integrated_data(self.data_handler, target_temp, integration_range, save_path)
            messagebox.showinfo("Export Successful", f"Integrated data for T={target_temp:.1f}K exported successfully to:\n{save_path}")
        except (ValueError, IOError) as e:
            logger.error(f"Error exporting integrated data: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to export integrated data:\n{e}")
        except Exception as e:
            logger.critical(f"Unexpected error exporting integrated data: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n{e}")


    def export_spectra_action(self):
        """Exports the processed spectra for the selected files."""
        logger.info("Export Selected Spectra button clicked.")
        selected_ids = self.file_table.get_selected_file_ids()

        if not selected_ids:
            messagebox.showinfo("No Selection", "Please select one or more files from the table to export.")
            return

        # Get save path
        save_path = filedialog.asksaveasfilename(
            title="Save Selected Spectra As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not save_path:
            logger.info("Export selected spectra cancelled by user.")
            return

        # Perform export
        try:
            export.export_processed_spectra(self.data_handler, selected_ids, save_path)
            messagebox.showinfo("Export Successful", f"Selected spectra exported successfully to:\n{save_path}")
        except (ValueError, IOError) as e:
            logger.error(f"Error exporting selected spectra: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to export selected spectra:\n{e}")
        except Exception as e:
            logger.critical(f"Unexpected error exporting selected spectra: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n{e}")


if __name__ == '__main__':
    # This allows running main_window.py directly for testing/preview
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running main_window.py directly for testing...")

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Main Window Test")
    root.geometry("1000x700")

    main_frame = MainWindow(master=root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    root.mainloop()
