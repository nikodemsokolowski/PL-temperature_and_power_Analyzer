import customtkinter as ctk
import numpy as np
from typing import List, Dict, Optional

from .widgets.plot_canvas import PlotCanvas


class BFieldAnalysisView(ctk.CTkToplevel):
    """Lightweight window for visualising B-field analysis outputs."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("B-Field Analysis")
        self.geometry("820x620")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        self.plot_canvas = PlotCanvas(self.plot_frame)
        self.plot_canvas.grid(row=0, column=0, sticky="nsew")

        self.results_label = ctk.CTkLabel(self, text="", anchor="w", wraplength=780)
        self.results_label.grid(row=1, column=0, padx=12, pady=(6, 10), sticky="ew")

    def show_intensity_vs_bfield(self, series: List[Dict[str, Optional[object]]]) -> None:
        """
        Display integrated intensity vs B-field curves.

        Args:
            series: List of dicts with keys:
                - 'bfield': sequence of B values
                - 'intensity': sequence of intensities
                - 'label': legend entry
                - 'color': optional matplotlib-compatible colour string
        """
        energies = []
        counts = []
        labels = []
        colors = []
        summaries = []

        for entry in series:
            b_vals = np.asarray(entry.get('bfield', []), dtype=float)
            intensities = np.asarray(entry.get('intensity', []), dtype=float)
            if b_vals.size == 0 or intensities.size == 0:
                continue

            mask = (~np.isnan(b_vals)) & (~np.isnan(intensities))
            if mask.sum() == 0:
                continue

            b_vals = b_vals[mask]
            intensities = intensities[mask]
            order = np.argsort(b_vals)
            b_vals = b_vals[order]
            intensities = intensities[order]

            energies.append(b_vals)
            counts.append(intensities)
            labels.append(entry.get('label', 'Series'))
            colors.append(entry.get('color'))
            summaries.append(f"{labels[-1]}: {b_vals.size} pts")

        if energies:
            self.plot_canvas.plot_data(
                energies,
                counts,
                labels_list=labels,
                title="Integrated Intensity vs B-Field",
                x_label="B-Field (T)",
                y_label="Integrated Intensity (arb. units)",
                colors=colors,
                style_options={'show_grid': True}
            )
        else:
            # Clear plot gracefully
            self.plot_canvas.plot_data(
                [], [], title="Integrated Intensity vs B-Field",
                x_label="B-Field (T)",
                y_label="Integrated Intensity (arb. units)",
                style_options={'show_grid': True}
            )

        summary_text = " | ".join(summaries) if summaries else "No valid intensity data available."
        self._update_results(summary_text)

    def show_zeeman_fit(self, fit_result: Optional[Dict[str, float]]) -> None:
        """
        Display Zeeman splitting data with linear fit and report g-factor.

        Args:
            fit_result: Dictionary returned by analysis.calculate_g_factor.
        """
        if not fit_result:
            self._update_results("g-factor fit failed.")
            return

        bfield = np.asarray(fit_result.get('bfield_t', []), dtype=float)
        delta_mev = np.asarray(fit_result.get('delta_mev', []), dtype=float)
        fit_delta = np.asarray(fit_result.get('fit_delta_mev', []), dtype=float)

        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)

        ax.scatter(bfield, delta_mev, color='tab:blue', marker='o', label='ΔE data')
        ax.plot(bfield, fit_delta, color='tab:red', linewidth=2, label='Linear fit')
        ax.set_xlabel("B-Field (T)")
        ax.set_ylabel("ΔE (meV)")
        ax.set_title("Zeeman Splitting vs B-Field")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        self.plot_canvas.canvas.draw_idle()

        g = fit_result.get('g_factor', float('nan'))
        g_unc = fit_result.get('g_uncertainty', float('nan'))
        r_sq = fit_result.get('r_squared', float('nan'))
        intercept = fit_result.get('intercept_mev', float('nan'))
        intercept_unc = fit_result.get('intercept_uncertainty', float('nan'))

        text_parts = []
        if not np.isnan(g):
            if not np.isnan(g_unc):
                text_parts.append(f"g = {g:.3f} ± {g_unc:.3f}")
            else:
                text_parts.append(f"g = {g:.3f}")
        if not np.isnan(r_sq):
            text_parts.append(f"R² = {r_sq:.3f}")
        if not np.isnan(intercept):
            if not np.isnan(intercept_unc):
                text_parts.append(f"ΔE₀ = {intercept:.3f} ± {intercept_unc:.3f} meV")
            else:
                text_parts.append(f"ΔE₀ = {intercept:.3f} meV")

        summary_text = " | ".join(text_parts) if text_parts else "Fit complete."
        self._update_results(summary_text)

    def _update_results(self, text: str) -> None:
        self.results_label.configure(text=text or "")
