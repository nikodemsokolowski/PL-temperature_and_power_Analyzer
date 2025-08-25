import customtkinter as ctk
import logging
import numpy as np
import pandas as pd
from tkinter import messagebox
from typing import Optional, Dict, List, Tuple

from .widgets.plot_canvas import PlotCanvas # Re-use the plot canvas
from ..core import analysis # Import fitting function

logger = logging.getLogger(__name__)

class AnalysisView(ctk.CTkToplevel):
    """
    A top-level window for displaying and analyzing power dependence data.
    (Integrated Intensity vs. Power with power law fitting).
    Attempts log-log plot by default, falls back to linear if data is not suitable.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Power Dependence Analysis")
        self.geometry("800x650")
        # self.grab_set() # Optional modal behavior

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Plot area expands

        # Data storage
        self.power_data: Optional[np.ndarray] = None
        self.intensity_data: Optional[np.ndarray] = None
        self.current_temperature: Optional[float] = None

        # --- Controls Frame ---
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((1, 3, 5), weight=1) # Allow spacing

        self.temp_label = ctk.CTkLabel(self.controls_frame, text="Temperature (K): N/A", font=ctk.CTkFont(weight="bold"))
        self.temp_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Fit Controls
        self.fix_a_var = ctk.StringVar(value="0") # Use StringVar for checkbox state
        self.fix_a_checkbox = ctk.CTkCheckBox(self.controls_frame, text="Fix 'a':", variable=self.fix_a_var, command=self._toggle_fix_entry)
        self.fix_a_checkbox.grid(row=0, column=1, padx=(10,0), pady=5, sticky="e")
        self.fix_a_entry = ctk.CTkEntry(self.controls_frame, width=80, state="disabled")
        self.fix_a_entry.grid(row=0, column=2, padx=(0,10), pady=5, sticky="w")

        self.fix_k_var = ctk.StringVar(value="0")
        self.fix_k_checkbox = ctk.CTkCheckBox(self.controls_frame, text="Fix 'k':", variable=self.fix_k_var, command=self._toggle_fix_entry)
        self.fix_k_checkbox.grid(row=0, column=3, padx=(10,0), pady=5, sticky="e")
        self.fix_k_entry = ctk.CTkEntry(self.controls_frame, width=80, state="disabled")
        self.fix_k_entry.grid(row=0, column=4, padx=(0,10), pady=5, sticky="w")

        self.fit_button = ctk.CTkButton(self.controls_frame, text="Fit Power Law", command=self.perform_fit)
        self.fit_button.grid(row=0, column=5, padx=10, pady=5, sticky="e")

        # --- Plot Frame ---
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.plot_canvas = PlotCanvas(self.plot_frame) # Embed plot canvas
        self.plot_canvas.pack(fill="both", expand=True)

        # --- Plan: __init__ Method ---
        # Configure plot with correct labels and default linear scale initially
        self.plot_canvas.axes.set_xlabel("Laser Power (uW)")
        self.plot_canvas.axes.set_ylabel("Integrated Intensity (arb. units)")
        self.plot_canvas.axes.set_title("Power Dependence")
        # Set initial scale to linear (update_plot will attempt log scale later)
        self.plot_canvas.axes.set_xscale('linear')
        self.plot_canvas.axes.set_yscale('linear')
        self.plot_canvas.axes.grid(True, which='both', linestyle='--', alpha=0.6)
        self.plot_canvas.canvas.draw()

        # --- Results Frame ---
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.results_label = ctk.CTkLabel(self.results_frame, text="Fit Results: ", anchor="w")
        self.results_label.pack(fill="x", padx=5, pady=2)

    def _toggle_fix_entry(self):
        """Enable/disable entry fields based on checkbox state."""
        self.fix_a_entry.configure(state="normal" if self.fix_a_var.get() == "1" else "disabled")
        self.fix_k_entry.configure(state="normal" if self.fix_k_var.get() == "1" else "disabled")
        if self.fix_a_var.get() == "1" and self.fix_k_var.get() == "1":
            messagebox.showwarning("Invalid Selection", "Cannot fix both 'a' and 'k' simultaneously. Please uncheck one.")

    def load_series_data(self, power_uw: np.ndarray, intensity: np.ndarray, temperature: float):
        """
        Loads the power series data into the view and plots it.

        Args:
            power_uw: Numpy array of power values.
            intensity: Numpy array of corresponding integrated intensities.
            temperature: The temperature (K) for this series.
        """
        logger.info(f"Loading power series data for T={temperature:.1f}K into AnalysisView.")
        self.power_data = power_uw
        self.intensity_data = intensity
        self.current_temperature = temperature

        self.temp_label.configure(text=f"Temperature (K): {temperature:.1f}")
        self.update_plot() # Plot the raw data
        self.results_label.configure(text="Fit Results: ") # Clear previous results

    def update_plot(self, fit_params: Optional[Dict] = None):
        """Updates the plot with data points and optionally the fitted curve."""
        # --- Plan: update_plot Method ---
        # Log received data
        temp_str = f"{self.current_temperature:.1f}" if self.current_temperature is not None else "N/A"
        if self.power_data is not None and self.intensity_data is not None:
            logger.debug(f"update_plot called for T={temp_str}K. "
                        f"Power shape: {self.power_data.shape}, Intensity shape: {self.intensity_data.shape}")
        else:
            logger.warning(f"update_plot called for T={temp_str}K but data is None.")
            # Handle case where data is not loaded yet or was cleared
            self.plot_canvas.axes.clear()
            self.plot_canvas.axes.set_xlabel("Laser Power (uW)")
            self.plot_canvas.axes.set_ylabel("Integrated Intensity (arb. units)")
            self.plot_canvas.axes.set_title(f"Power Dependence at T = {temp_str} K (No Data)")
            self.plot_canvas.axes.set_xscale('linear') # Ensure linear scale for no data plot
            self.plot_canvas.axes.set_yscale('linear')
            self.plot_canvas.axes.grid(True, which='both', linestyle='--', alpha=0.6)
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.canvas.draw()
            return

        # --- Plan: Always clear axes and set correct labels/title ---
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.set_xlabel("Laser Power (uW)")
        self.plot_canvas.axes.set_ylabel("Integrated Intensity (arb. units)")
        self.plot_canvas.axes.set_title(f"Power Dependence at T = {temp_str} K")

        # --- Plan: Perform check for valid positive data ---
        # Filter data: only positive power and positive intensity for log plotting and fitting
        valid_data_mask = (self.power_data > 0) & (self.intensity_data > 0)
        power_to_plot = self.power_data[valid_data_mask]
        intensity_to_plot = self.intensity_data[valid_data_mask]
        logger.debug(f"Found {len(power_to_plot)} data points with positive power and intensity.")

        # --- Plan: If no valid positive data exists ---
        if len(power_to_plot) == 0:
            logger.warning(f"No positive data points found for log plot at T={temp_str}K. Plotting all data on linear scale.")
            # Plot original data on linear scale as fallback
            power_to_plot = self.power_data # Reassign to plot all data
            intensity_to_plot = self.intensity_data
            if len(power_to_plot) > 0:
                self.plot_canvas.axes.plot(power_to_plot, intensity_to_plot, 'o', label='Data (contains non-positive)')
            else:
                logger.warning(f"No data points at all found for T={temp_str}K.")
            # --- Plan: Ensure axes are set to 'linear' ---
            self.plot_canvas.axes.set_xscale('linear')
            self.plot_canvas.axes.set_yscale('linear')
            # Add title note about scale
            self.plot_canvas.axes.set_title(f"Power Dependence at T = {temp_str} K (Linear Scale)")
            self.plot_canvas.axes.grid(True, which='both', linestyle='--', alpha=0.6)
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.canvas.draw()
            return # Exit after handling no positive data case

        # --- Plan: If valid positive data exists ---
        logger.debug(f"Proceeding with plotting {len(power_to_plot)} valid positive data points for T={temp_str}K.")
        # Plot valid data points
        self.plot_canvas.axes.plot(power_to_plot, intensity_to_plot, 'o', label='Data')

        # Plot fit curve if available (based on the valid positive data range)
        if fit_params and 'a' in fit_params and 'k' in fit_params:
            try:
                # Generate smooth curve for plotting - logspace is better for log scale
                min_power_log = np.log10(power_to_plot.min())
                max_power_log = np.log10(power_to_plot.max())
                # Handle case where min/max are the same or invalid
                if np.isfinite(min_power_log) and np.isfinite(max_power_log) and min_power_log <= max_power_log:
                    power_fit = np.logspace(min_power_log, max_power_log, 100)
                    intensity_fit = analysis.power_law(power_fit, fit_params['a'], fit_params['k'])
                    label_fit = f"Fit: I = {fit_params['a']:.3g} * P^{fit_params['k']:.3f}"
                    self.plot_canvas.axes.plot(power_fit, intensity_fit, '-', label=label_fit)
                    self.plot_canvas.axes.legend()
                else:
                    logger.warning("Cannot generate fit line: Invalid range for logspace.")
            except Exception as e:
                logger.error(f"Error generating fit line: {e}", exc_info=True)


        # --- Plan: Attempt to set log scale, fallback to linear ---
        try:
            self.plot_canvas.axes.set_xscale('log')
            self.plot_canvas.axes.set_yscale('log')
            logger.debug("Successfully set log scale.")
        except ValueError as e:
            logger.warning(f"Log scale failed (likely due to data range): {e}. Falling back to linear scale.")
            self.plot_canvas.axes.set_xscale('linear')
            self.plot_canvas.axes.set_yscale('linear')
            # Update title if fallback occurred
            self.plot_canvas.axes.set_title(f"Power Dependence at T = {temp_str} K (Linear Scale Fallback)")


        # --- Plan: Final draw steps ---
        self.plot_canvas.axes.grid(True, which='both', linestyle='--', alpha=0.6)
        self.plot_canvas.figure.tight_layout() # Adjust layout
        self.plot_canvas.canvas.draw()

    def perform_fit(self):
        """Callback for the Fit button."""
        logger.info("Perform Fit button clicked.")
        # Use only positive data for fitting power law
        if self.power_data is None or self.intensity_data is None:
            messagebox.showerror("Error", "No data loaded to fit.")
            return

        # --- Filter data for fitting ---
        valid_data_mask = (self.power_data > 0) & (self.intensity_data > 0)
        power_to_fit = self.power_data[valid_data_mask]
        intensity_to_fit = self.intensity_data[valid_data_mask]

        if len(power_to_fit) < 2: # Need at least 2 points for a fit
            messagebox.showerror("Fit Error", "Need at least two data points with positive power and intensity to perform a fit.")
            logger.error(f"Fit aborted: Only {len(power_to_fit)} valid positive data points.")
            return
        # --- End Filter ---

        fixed_params = {}
        try:
            if self.fix_a_var.get() == "1":
                fixed_a = float(self.fix_a_entry.get())
                if fixed_a <= 0: raise ValueError("'a' must be positive")
                fixed_params['a'] = fixed_a
            if self.fix_k_var.get() == "1":
                fixed_k = float(self.fix_k_entry.get())
                fixed_params['k'] = fixed_k
            if len(fixed_params) == 2:
                messagebox.showerror("Error", "Cannot fix both 'a' and 'k'.")
                return
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid value for fixed parameter: {e}")
            return

        logger.debug(f"Attempting power law fit on {len(power_to_fit)} points with fixed params: {fixed_params}")
        try:
            fitted_params, param_errors = analysis.fit_power_law(
                power_to_fit, # Use filtered data
                intensity_to_fit, # Use filtered data
                fixed_params=fixed_params if fixed_params else None
            )
        except Exception as e:
            logger.error(f"Error during curve_fit: {e}", exc_info=True)
            fitted_params, param_errors = None, None


        if fitted_params and param_errors:
            logger.info(f"Fit successful: {fitted_params}, Errors: {param_errors}")
            self.update_plot(fit_params=fitted_params) # Update plot with fit curve
            # Display results
            a_val = fitted_params['a']
            k_val = fitted_params['k']
            a_err = param_errors.get('a', np.nan) # Use .get for safety
            k_err = param_errors.get('k', np.nan)
            results_text = f"Fit Results: a = {a_val:.4g} ± {a_err:.2g},   k = {k_val:.4f} ± {k_err:.2f}"
            self.results_label.configure(text=results_text)
            # Optional: showinfo could be annoying if fitting often
            # messagebox.showinfo("Fit Successful", f"Power law fit completed.\n{results_text}")
        else:
            logger.error("Power law fit failed.")
            self.update_plot() # Re-plot just the data
            self.results_label.configure(text="Fit Results: Fit Failed")
            messagebox.showerror("Fit Error", "Could not fit the data to the power law. Check data and console log.")


# Example usage (for testing purposes)
if __name__ == '__main__':
    # --- Setup Logging ---
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    # Set root logger level (adjust as needed)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Show DEBUG messages and above
    root_logger.addHandler(console_handler)
    # Optional: Set specific logger levels if needed
    logging.getLogger('matplotlib').setLevel(logging.INFO) # Quieten matplotlib logs

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    # Need a dummy root window
    root = ctk.CTk()
    root.withdraw() # Hide the dummy root window

    # --- Create Test Data ---
    # Case 1: Good data for log-log
    test_power_good = np.array([1, 5, 10, 20, 50, 100], dtype=float)
    true_a = 5.0
    true_k = 1.2
    noise_good = np.random.normal(0, 0.1 * analysis.power_law(test_power_good, true_a, true_k), size=len(test_power_good))
    test_intensity_good = analysis.power_law(test_power_good, true_a, true_k) + noise_good
    test_intensity_good = np.maximum(test_intensity_good, 1e-5) # Ensure positive

    # Case 2: Data with zeros or negatives (will force linear plot)
    test_power_bad = np.array([-5, 0, 10, 20, 50, 100], dtype=float) # Includes zero and negative power
    noise_bad = np.random.normal(0, 5, size=len(test_power_bad))
    test_intensity_bad = analysis.power_law(np.maximum(test_power_bad, 1e-9), true_a, true_k) + noise_bad # Avoid log(0) in generation


    logger.info("Creating AnalysisView window (Test 1: Good Data)...")
    analysis_window_1 = AnalysisView(master=None) # No master for standalone test
    analysis_window_1.load_series_data(test_power_good, test_intensity_good, temperature=10.0)
    analysis_window_1.title("Analysis View - Good Data (Log Attempt)")


    logger.info("Creating AnalysisView window (Test 2: Bad Data)...")
    analysis_window_2 = AnalysisView(master=None) # No master for standalone test
    analysis_window_2.load_series_data(test_power_bad, test_intensity_bad, temperature=20.0)
    analysis_window_2.title("Analysis View - Bad Data (Linear Fallback)")

    # Keep the windows open (Need root's mainloop for Toplevels)
    root.mainloop() # Use the hidden root's mainloop

    logger.info("AnalysisView closed.")