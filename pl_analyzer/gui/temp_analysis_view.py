import customtkinter as ctk
import logging
import numpy as np
from tkinter import colorchooser, filedialog, messagebox
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ..core import analysis

logger = logging.getLogger(__name__)


class TempAnalysisView(ctk.CTkToplevel):
    """Dual-plot temperature analysis window with integrated styling and export controls."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Temperature Dependence Analysis")
        self.geometry("980x720")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Data containers
        self.dataset = None
        self.series_records: List[Dict[str, float]] = []
        self.integration_range: Optional[Tuple[float, float]] = None
        self.temp_data: np.ndarray = np.array([])
        self.intensity_data: np.ndarray = np.array([])
        self.current_power: Optional[float] = None
        self._current_fit: Optional[Dict[str, object]] = None

        # Plot styling
        self.plot_color = "#1f77b4"
        self.marker_options = {
            "Circle (o)": "o",
            "Square (s)": "s",
            "Triangle (^)": "^",
            "Diamond (D)": "D",
            "No Markers": "None",
        }

        self._build_controls()
        self._build_plot_area()

    def _build_controls(self) -> None:
        """Create the top control panel."""
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.power_label = ctk.CTkLabel(
            self.controls_frame,
            text="Power (uW): N/A",
            font=ctk.CTkFont(weight="bold"),
        )
        self.power_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.show_arrhenius_var = ctk.StringVar(value="on")
        self.show_arrhenius_check = ctk.CTkCheckBox(
            self.controls_frame,
            text="Show Arrhenius Plot",
            variable=self.show_arrhenius_var,
            onvalue="on",
            offvalue="off",
            command=self._update_plots,
        )
        self.show_arrhenius_check.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.show_linear_var = ctk.StringVar(value="on")
        self.show_linear_check = ctk.CTkCheckBox(
            self.controls_frame,
            text="Show Intensity Plot",
            variable=self.show_linear_var,
            onvalue="on",
            offvalue="off",
            command=self._update_plots,
        )
        self.show_linear_check.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.equation_label = ctk.CTkLabel(
            self.controls_frame,
            text="I(T) = I0 / (1 + C1·exp(-Ea1/kT) + C2·exp(-Ea2/kT))",
            font=ctk.CTkFont(size=13),
        )
        self.equation_label.grid(row=1, column=0, columnspan=4, padx=10, pady=(0, 5), sticky="w")

        self.arrhenius_axis_label = ctk.CTkLabel(self.controls_frame, text="Arrhenius X-Axis:")
        self.arrhenius_axis_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        axis_options = ["1000/T (1/K)", "T (K)"]
        self.arrhenius_axis_var = ctk.StringVar(value=axis_options[0])
        self.arrhenius_axis_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=axis_options,
            variable=self.arrhenius_axis_var,
            command=lambda _: self._update_plots(),
        )
        self.arrhenius_axis_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        self.color_button = ctk.CTkButton(
            self.controls_frame,
            text=self.plot_color.upper(),
            command=self._choose_color,
        )
        self.color_button.grid(row=2, column=2, padx=10, pady=5, sticky="ew")

        self.marker_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=list(self.marker_options.keys()),
            command=lambda _: self._update_plots(),
        )
        self.marker_menu.set("Circle (o)")
        self.marker_menu.grid(row=2, column=3, padx=10, pady=5, sticky="ew")

        marker_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        marker_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=(0, 5), sticky="ew")
        marker_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(marker_frame, text="Marker Size").grid(row=0, column=0, sticky="w")
        self.marker_size_slider = ctk.CTkSlider(
            marker_frame,
            from_=3.0,
            to=12.0,
            number_of_steps=18,
            command=lambda value: self._on_marker_size_change(value),
        )
        self.marker_size_slider.grid(row=1, column=0, sticky="ew")
        self.marker_size_slider.set(6.0)
        self.marker_size_value = ctk.CTkLabel(marker_frame, text="6.0")
        self.marker_size_value.grid(row=1, column=1, padx=(6, 0), sticky="w")

        linewidth_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        linewidth_frame.grid(row=3, column=2, columnspan=2, padx=5, pady=(0, 5), sticky="ew")
        linewidth_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(linewidth_frame, text="Line Width").grid(row=0, column=0, sticky="w")
        self.line_width_slider = ctk.CTkSlider(
            linewidth_frame,
            from_=0.5,
            to=5.0,
            number_of_steps=18,
            command=lambda value: self._on_line_width_change(value),
        )
        self.line_width_slider.grid(row=1, column=0, sticky="ew")
        self.line_width_slider.set(1.8)
        self.line_width_value = ctk.CTkLabel(linewidth_frame, text="1.8")
        self.line_width_value.grid(row=1, column=1, padx=(6, 0), sticky="w")

        self.integration_label = ctk.CTkLabel(
            self.controls_frame,
            text="Integration Range (eV):",
            font=ctk.CTkFont(weight="bold"),
        )
        self.integration_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        self.integration_min_entry = ctk.CTkEntry(self.controls_frame, placeholder_text="Min E (eV)")
        self.integration_min_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.integration_max_entry = ctk.CTkEntry(self.controls_frame, placeholder_text="Max E (eV)")
        self.integration_max_entry.grid(row=4, column=2, padx=5, pady=5, sticky="ew")
        self.apply_integration_button = ctk.CTkButton(
            self.controls_frame,
            text="Apply Range",
            command=self._apply_integration_range,
        )
        self.apply_integration_button.grid(row=4, column=3, padx=10, pady=5, sticky="ew")

        self.fit_button_1_exp = ctk.CTkButton(
            self.controls_frame,
            text="Fit Arrhenius (1 Exp)",
            command=self.perform_fit_1_exp,
        )
        self.fit_button_1_exp.grid(row=5, column=0, padx=10, pady=5, sticky="ew")

        self.fit_button_2_exp = ctk.CTkButton(
            self.controls_frame,
            text="Fit Arrhenius (2 Exp)",
            command=self.perform_fit_2_exp,
        )
        self.fit_button_2_exp.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

        self.export_button = ctk.CTkButton(
            self.controls_frame,
            text="Export Figure",
            command=self._export_figure,
        )
        self.export_button.grid(row=5, column=3, padx=10, pady=5, sticky="ew")

        self.params_frame = ctk.CTkFrame(self.controls_frame)
        self.params_frame.grid(row=6, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        self.param_entries: Dict[str, ctk.CTkEntry] = {}
        self.param_fix_vars: Dict[str, ctk.StringVar] = {}
        params = ["I0", "C1", "Ea1", "C2", "Ea2"]
        for i, param in enumerate(params):
            label = ctk.CTkLabel(self.params_frame, text=f"{param}:")
            label.grid(row=0, column=i * 2, padx=(10, 2), pady=5)

            entry = ctk.CTkEntry(self.params_frame, width=80)
            entry.grid(row=0, column=i * 2 + 1, padx=(0, 5), pady=5)
            self.param_entries[param] = entry

            fix_var = ctk.StringVar(value="off")
            fix_check = ctk.CTkCheckBox(
                self.params_frame,
                text="Fix",
                variable=fix_var,
                onvalue="on",
                offvalue="off",
            )
            fix_check.grid(row=1, column=i * 2, columnspan=2, padx=5, pady=(0, 5))
            self.param_fix_vars[param] = fix_var

        self.results_label = ctk.CTkLabel(self.controls_frame, text="Fit Results: ", anchor="w")
        self.results_label.grid(row=7, column=0, columnspan=4, padx=10, pady=(5, 0), sticky="ew")

    def _build_plot_area(self) -> None:
        """Create the Matplotlib canvas and toolbar."""
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9.5, 5.5), dpi=110)
        self.figure.subplots_adjust(wspace=0.28)
        self.ax_arrhenius = self.figure.add_subplot(1, 2, 1)
        self.ax_linear = self.figure.add_subplot(1, 2, 2)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar_frame = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def _choose_color(self) -> None:
        """Open a color chooser dialog for the plot color."""
        rgb_hex = colorchooser.askcolor(color=self.plot_color, parent=self)
        if rgb_hex and rgb_hex[1]:
            self.plot_color = rgb_hex[1]
            self.color_button.configure(text=self.plot_color.upper())
            self._update_plots()

    def _on_marker_size_change(self, value: float) -> None:
        """Update marker size label and redraw plots."""
        self.marker_size_value.configure(text=f"{float(value):.1f}")
        self._update_plots()

    def _on_line_width_change(self, value: float) -> None:
        """Update line width label and redraw plots."""
        self.line_width_value.configure(text=f"{float(value):.2f}")
        self._update_plots()

    def _resolve_marker(self) -> Optional[str]:
        """Return the matplotlib marker string for the current selection."""
        marker_key = self.marker_menu.get()
        marker = self.marker_options.get(marker_key, "o")
        return None if marker == "None" else marker

    def load_series_data(
        self,
        dataset,
        series_records: List[Dict[str, float]],
        power: float,
        integration_range: Tuple[float, float],
    ) -> None:
        """Load dataset metadata and redraw plots for the selected temperature series."""
        logger.info("Loading temperature series data into TempAnalysisView.")
        self.dataset = dataset
        self.series_records = sorted(
            [record for record in series_records if record.get("file_id")],
            key=lambda rec: rec.get("temperature_k", 0.0),
        )
        self.current_power = power
        self.integration_range = integration_range

        self.power_label.configure(text=f"Power (uW): {power:.2f}")
        self.integration_min_entry.delete(0, "end")
        self.integration_max_entry.delete(0, "end")
        self.integration_min_entry.insert(0, f"{integration_range[0]:g}")
        self.integration_max_entry.insert(0, f"{integration_range[1]:g}")

        self._recalculate_series(show_messages=False)

    def _recalculate_series(self, show_messages: bool = True) -> None:
        """Recompute integrated intensities for the current integration range."""
        temps, intensities = self._compute_series_data()
        self.temp_data = temps
        self.intensity_data = intensities
        self._clear_fit(update_label=False)

        if temps.size == 0:
            msg = "No valid spectra within the selected integration range."
            self.results_label.configure(text=f"Fit Results: {msg}")
            if show_messages:
                messagebox.showwarning("No Data", msg, parent=self)
        elif temps.size < 2:
            msg = "Need at least two temperature points for fitting."
            self.results_label.configure(text=f"Fit Results: {msg}")
            if show_messages:
                messagebox.showinfo("Insufficient Data", msg, parent=self)
        else:
            self.results_label.configure(text="Fit Results: ")

        self._update_plots()

    def _compute_series_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate spectra for each record and return sorted temperature/intensity arrays."""
        if self.dataset is None or not self.integration_range:
            return np.array([]), np.array([])

        min_e, max_e = self.integration_range
        temps: List[float] = []
        intensities: List[float] = []

        for record in self.series_records:
            file_id = record.get("file_id")
            temperature = record.get("temperature_k")
            if file_id is None or temperature is None:
                continue

            spectrum_df = self.dataset.get_processed_spectrum(file_id)
            if spectrum_df is None or spectrum_df.empty:
                logger.debug("Skipping %s (missing or empty spectrum).", file_id)
                continue

            try:
                integral = analysis.integrate_spectrum(
                    spectrum_df["energy_ev"].values,
                    spectrum_df["counts"].values,
                    (min_e, max_e),
                )
            except Exception as exc:
                logger.error("Failed to integrate spectrum for %s: %s", file_id, exc, exc_info=True)
                continue

            if integral is None or integral <= 0:
                logger.debug("Skipping %s (non-positive integral).", file_id)
                continue

            temps.append(float(temperature))
            intensities.append(float(integral))

        if not temps:
            return np.array([]), np.array([])

        temps_array = np.asarray(temps, dtype=float)
        intensities_array = np.asarray(intensities, dtype=float)
        order = np.argsort(temps_array)
        return temps_array[order], intensities_array[order]

    def _update_plots(self) -> None:
        """Refresh both subplots according to current settings."""
        if not hasattr(self, "ax_arrhenius"):
            return

        show_arrhenius = self.show_arrhenius_var.get() == "on"
        show_linear = self.show_linear_var.get() == "on"
        marker = self._resolve_marker()
        marker_size = float(self.marker_size_slider.get())
        line_width = float(self.line_width_slider.get())

        temps = self.temp_data
        intensities = self.intensity_data

        self.ax_arrhenius.clear()
        self.ax_linear.clear()

        if show_arrhenius and temps.size > 0:
            if self.arrhenius_axis_var.get() == "T (K)":
                x_data = temps
                x_label = "Temperature (K)"
            else:
                x_data = 1000.0 / temps
                x_label = "1000 / T (1/K)"
            self.ax_arrhenius.set_xlabel(x_label)
            self.ax_arrhenius.set_ylabel("Integrated Intensity (arb. units)")
            self.ax_arrhenius.grid(True, which="both", linestyle="--", alpha=0.6)
            self.ax_arrhenius.plot(
                x_data,
                intensities,
                linestyle="-",
                marker=marker,
                markersize=marker_size if marker else None,
                linewidth=line_width,
                color=self.plot_color,
                label="Data",
            )

            if self._current_fit:
                fit_params = self._current_fit["params"]
                fit_type = self._current_fit["type"]
                temp_linspace = np.linspace(temps.min(), temps.max(), 200)
                if fit_type == "1_exp":
                    fit_vals = analysis.arrhenius_1_exp(temp_linspace, *fit_params)
                    legend_label = f"Fit (Ea = {fit_params[1] * 1000:.1f} meV)"
                else:
                    fit_vals = analysis.arrhenius_2_exp(temp_linspace, *fit_params)
                    legend_label = "Fit (2 Exp)"
                x_fit = temp_linspace if self.arrhenius_axis_var.get() == "T (K)" else 1000.0 / temp_linspace
                self.ax_arrhenius.plot(
                    x_fit,
                    fit_vals,
                    linestyle="--",
                    linewidth=line_width,
                    color=self.plot_color,
                    label=legend_label,
                )
            self.ax_arrhenius.legend()
        else:
            self.ax_arrhenius.set_visible(False)

        if show_linear and temps.size > 0:
            self.ax_linear.set_visible(True)
            self.ax_linear.set_xlabel("Temperature (K)")
            self.ax_linear.set_ylabel("Integrated Intensity (arb. units)")
            self.ax_linear.grid(True, which="both", linestyle="--", alpha=0.6)
            self.ax_linear.plot(
                temps,
                intensities,
                linestyle="-",
                marker=marker,
                markersize=marker_size if marker else None,
                linewidth=line_width,
                color=self.plot_color,
            )
        else:
            self.ax_linear.set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _clear_fit(self, update_label: bool = True) -> None:
        """Reset cached fit information."""
        self._current_fit = None
        if update_label:
            self.results_label.configure(text="Fit Results: ")

    def _apply_integration_range(self) -> None:
        """Validate and apply the integration range entered in the UI."""
        range_values = self._validate_integration_inputs()
        if not range_values:
            return
        self.integration_range = range_values
        self._recalculate_series(show_messages=True)

    def _validate_integration_inputs(self) -> Optional[Tuple[float, float]]:
        """Validate integration range entries."""
        try:
            min_e = float(self.integration_min_entry.get())
            max_e = float(self.integration_max_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Integration range must be numeric.", parent=self)
            return None
        if max_e <= min_e:
            messagebox.showerror("Invalid Input", "Max energy must be greater than min energy.", parent=self)
            return None
        return min_e, max_e

    def perform_fit_1_exp(self) -> None:
        """Fit the Arrhenius 1-exponential model to the current data."""
        if self.temp_data.size < 2:
            messagebox.showinfo("Not Enough Data", "Need at least two valid points to perform a fit.", parent=self)
            return

        popt, _ = analysis.arrhenius_fit_1_exp(self.temp_data, self.intensity_data)
        if popt is None:
            self.results_label.configure(text="Fit Results: Fit failed (1 Exp).")
            messagebox.showerror("Fit Error", "Could not fit data to the 1-exponential model.", parent=self)
            return

        self._current_fit = {"type": "1_exp", "params": popt}
        ea_mev = popt[1] * 1000
        results_text = f"Fit Results (1 Exp): I0 = {popt[0]:.4g}, Ea = {ea_mev:.2f} meV"
        self.results_label.configure(text=results_text)
        self._update_plots()

    def perform_fit_2_exp(self) -> None:
        """Fit the Arrhenius 2-exponential model with optional parameter constraints."""
        if self.temp_data.size < 3:
            messagebox.showinfo("Not Enough Data", "Need at least three valid points to perform a 2-exp fit.", parent=self)
            return

        initial_guesses: Dict[str, float] = {}
        fixed_params: Dict[str, float] = {}
        for param, entry in self.param_entries.items():
            value_str = entry.get().strip()
            if value_str:
                try:
                    value = float(value_str)
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid number for {param}: {value_str}", parent=self)
                    return
                if self.param_fix_vars[param].get() == "on":
                    fixed_params[param] = value
                else:
                    initial_guesses[param] = value

        popt, _ = analysis.arrhenius_fit_2_exp(
            self.temp_data,
            self.intensity_data,
            initial_guesses,
            fixed_params,
        )

        if popt is None:
            self.results_label.configure(text="Fit Results: Fit failed (2 Exp).")
            messagebox.showerror("Fit Error", "Could not fit data to the 2-exponential model.", parent=self)
            return

        self._current_fit = {"type": "2_exp", "params": popt}
        ea1_mev = popt[2] * 1000
        ea2_mev = popt[4] * 1000
        results_text = (
            f"Fit Results (2 Exp): I0 = {popt[0]:.4g}, C1 = {popt[1]:.2g}, Ea1 = {ea1_mev:.2f} meV, "
            f"C2 = {popt[3]:.2g}, Ea2 = {ea2_mev:.2f} meV"
        )
        self.results_label.configure(text=results_text)
        self._update_plots()

    def _export_figure(self) -> None:
        """Export the current figure at high DPI suitable for publication."""
        if self.temp_data.size == 0:
            messagebox.showinfo("Export Figure", "No data available to export.", parent=self)
            return

        filepath = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
            ],
            initialfile="temperature_analysis.png",
        )
        if not filepath:
            return

        try:
            self.figure.savefig(filepath, dpi=600, bbox_inches="tight")
        except Exception as exc:
            logger.error("Failed to export temperature analysis figure: %s", exc, exc_info=True)
            messagebox.showerror("Export Error", f"Could not export figure.\nDetails: {exc}", parent=self)
            return

        messagebox.showinfo("Export Complete", f"Figure saved to {filepath}", parent=self)
