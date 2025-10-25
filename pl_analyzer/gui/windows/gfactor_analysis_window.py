import csv
import logging
from pathlib import Path
from typing import Optional, Tuple

import customtkinter as ctk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox

from pl_analyzer.core import analysis

logger = logging.getLogger(__name__)


class GFactorAnalysisWindow(ctk.CTkToplevel):
    """Advanced analysis window for g-factor extraction and Zeeman splitting plots."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Advanced g-Factor Analysis")
        self.geometry("1100x780")
        self.minsize(980, 720)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # State
        self._dataset = None
        self._energy_range: Optional[Tuple[float, float]] = None
        self._method = "centroid"
        self._smoothing_points = 0
        self._manual_g: Optional[float] = None
        self._manual_intercept: float = 0.0
        self._peaks_df = pd.DataFrame()
        self._zeeman_df = pd.DataFrame()
        self._fit_df = pd.DataFrame()
        self._fit_result: Optional[dict] = None

        self._build_controls()
        self._build_plot_area()

    def _build_controls(self) -> None:
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="new")
        for col in range(0, 8):
            self.controls_frame.grid_columnconfigure(col, weight=0)
        self.controls_frame.grid_columnconfigure(7, weight=1)

        # Row 0: Energy range
        energy_label = ctk.CTkLabel(self.controls_frame, text="Energy range (eV):")
        energy_label.grid(row=0, column=0, padx=(8, 4), pady=4, sticky="w")
        self.energy_min_entry = ctk.CTkEntry(self.controls_frame, width=100, placeholder_text="Min")
        self.energy_min_entry.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        self.energy_max_entry = ctk.CTkEntry(self.controls_frame, width=100, placeholder_text="Max")
        self.energy_max_entry.grid(row=0, column=2, padx=4, pady=4, sticky="w")

        method_label = ctk.CTkLabel(self.controls_frame, text="Peak method:")
        method_label.grid(row=0, column=3, padx=(16, 4), pady=4, sticky="e")
        self.method_var = ctk.StringVar(value="Centroid")
        self.method_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            variable=self.method_var,
            values=["Centroid", "Maximum", "Gaussian"],
            command=lambda _: self.recalculate_analysis()
        )
        self.method_menu.grid(row=0, column=4, padx=4, pady=4, sticky="w")

        smoothing_label = ctk.CTkLabel(self.controls_frame, text="Smoothing (pts):")
        smoothing_label.grid(row=0, column=5, padx=(16, 4), pady=4, sticky="e")
        self.smoothing_entry = ctk.CTkEntry(self.controls_frame, width=80, placeholder_text="0")
        self.smoothing_entry.insert(0, "0")
        self.smoothing_entry.grid(row=0, column=6, padx=4, pady=4, sticky="w")

        # Row 1: Fit range and manual overlay
        fit_label = ctk.CTkLabel(self.controls_frame, text="Fit B range (T):")
        fit_label.grid(row=1, column=0, padx=(8, 4), pady=4, sticky="w")
        self.fit_min_entry = ctk.CTkEntry(self.controls_frame, width=100, placeholder_text="Min B")
        self.fit_min_entry.grid(row=1, column=1, padx=4, pady=4, sticky="w")
        self.fit_max_entry = ctk.CTkEntry(self.controls_frame, width=100, placeholder_text="Max B")
        self.fit_max_entry.grid(row=1, column=2, padx=4, pady=4, sticky="w")

        manual_g_label = ctk.CTkLabel(self.controls_frame, text="Manual g-factor:")
        manual_g_label.grid(row=1, column=3, padx=(16, 4), pady=4, sticky="e")
        self.manual_g_entry = ctk.CTkEntry(self.controls_frame, width=90, placeholder_text="e.g. 2.0")
        self.manual_g_entry.grid(row=1, column=4, padx=4, pady=4, sticky="w")

        manual_int_label = ctk.CTkLabel(self.controls_frame, text="Manual intercept (meV):")
        manual_int_label.grid(row=1, column=5, padx=(16, 4), pady=4, sticky="e")
        self.manual_intercept_entry = ctk.CTkEntry(self.controls_frame, width=90, placeholder_text="0.0")
        self.manual_intercept_entry.insert(0, "0.0")
        self.manual_intercept_entry.grid(row=1, column=6, padx=4, pady=4, sticky="w")

        self.manual_apply_button = ctk.CTkButton(
            self.controls_frame,
            text="Apply Manual g",
            width=150,
            command=self._apply_manual_overlay
        )
        self.manual_apply_button.grid(row=1, column=7, padx=(8, 8), pady=4, sticky="e")

        # Row 2: Action buttons
        self.recalc_button = ctk.CTkButton(
            self.controls_frame,
            text="Recalculate",
            width=130,
            command=self.recalculate_analysis
        )
        self.recalc_button.grid(row=2, column=0, padx=(8, 4), pady=(6, 4), sticky="w")

        self.export_data_button = ctk.CTkButton(
            self.controls_frame,
            text="Export Data",
            width=130,
            command=self._export_data
        )
        self.export_data_button.grid(row=2, column=1, padx=4, pady=(6, 4), sticky="w")

        self.export_plot_button = ctk.CTkButton(
            self.controls_frame,
            text="Export Figure",
            width=130,
            command=self._export_figure
        )
        self.export_plot_button.grid(row=2, column=2, padx=4, pady=(6, 4), sticky="w")

        self.reset_manual_button = ctk.CTkButton(
            self.controls_frame,
            text="Clear Manual g",
            width=130,
            command=self._clear_manual_overlay
        )
        self.reset_manual_button.grid(row=2, column=3, padx=(16, 4), pady=(6, 4), sticky="w")

        self.results_label = ctk.CTkLabel(self.controls_frame, text="", anchor="w")
        self.results_label.grid(row=3, column=0, columnspan=8, padx=8, pady=(4, 2), sticky="ew")

        self.summary_label = ctk.CTkLabel(self.controls_frame, text="", anchor="w")
        self.summary_label.grid(row=4, column=0, columnspan=8, padx=8, pady=(0, 2), sticky="ew")

    def _build_plot_area(self) -> None:
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(8.8, 6.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar_frame = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def configure_dataset(
        self,
        dataset,
        energy_range: Optional[Tuple[float, float]] = None,
        default_method: str = "centroid"
    ) -> None:
        """Initialise window with a dataset and optional defaults."""
        self._dataset = dataset
        self._energy_range = energy_range
        self._method = default_method or "centroid"
        self._manual_g = None
        self._manual_intercept = 0.0

        if energy_range:
            self.energy_min_entry.delete(0, "end")
            self.energy_min_entry.insert(0, f"{energy_range[0]:.4f}")
            self.energy_max_entry.delete(0, "end")
            self.energy_max_entry.insert(0, f"{energy_range[1]:.4f}")
        else:
            self.energy_min_entry.delete(0, "end")
            self.energy_max_entry.delete(0, "end")

        self.method_var.set(self._method.title())
        self.smoothing_entry.delete(0, "end")
        self.smoothing_entry.insert(0, "0")
        self.fit_min_entry.delete(0, "end")
        self.fit_max_entry.delete(0, "end")
        self.manual_g_entry.delete(0, "end")
        self.manual_intercept_entry.delete(0, "end")
        self.manual_intercept_entry.insert(0, "0.0")

        self.recalculate_analysis()
        self.deiconify()
        self.lift()
        self.focus_force()

    def recalculate_analysis(self) -> None:
        """Recompute peak energies, Zeeman splitting, and g-factor."""
        if self._dataset is None:
            return

        energy_range = self._parse_energy_range()
        method = self.method_var.get().lower()
        smoothing = self._parse_int(self.smoothing_entry.get(), allow_zero=True)
        if smoothing is None:
            messagebox.showerror("Invalid Smoothing", "Smoothing points must be a non-negative integer.")
            return

        try:
            peaks_df = analysis.extract_bfield_peak_energies(
                self._dataset,
                energy_range=energy_range,
                method=method,
                smoothing_points=smoothing
            )
        except ValueError as exc:
            messagebox.showerror("Peak Extraction Error", str(exc))
            return

        self._peaks_df = peaks_df
        if peaks_df.empty:
            self._zeeman_df = pd.DataFrame()
            self._fit_df = pd.DataFrame()
            self._fit_result = None
            self._update_results_label("No peaks could be extracted for the selected parameters.")
            self._update_summary()
            self._draw_empty_plots("No data available.")
            return

        zeeman_df = analysis.prepare_zeeman_dataframe(peaks_df)
        self._zeeman_df = zeeman_df
        if zeeman_df.empty:
            self._fit_df = pd.DataFrame()
            self._fit_result = None
            self._update_results_label("Unable to pair sigma+/sigma- spectra at matching B-fields.")
            self._update_summary()
            self._draw_empty_plots("Insufficient paired data.")
            return

        fit_mask = self._build_fit_mask(zeeman_df)
        fit_df = zeeman_df[fit_mask].copy()
        if fit_df.empty:
            self._fit_df = pd.DataFrame()
            self._fit_result = None
            self._update_results_label("Fit range does not include any valid points.")
            self._update_summary()
            self._draw_empty_plots("No points within selected fit range.")
            return

        fit_result = analysis.calculate_g_factor(
            fit_df['bfield_t'].to_numpy(dtype=float),
            fit_df['sigma_plus_ev'].to_numpy(dtype=float),
            fit_df['sigma_minus_ev'].to_numpy(dtype=float)
        )
        if not fit_result:
            self._fit_df = fit_df
            self._fit_result = None
            self._update_results_label("Linear fit failed. Try adjusting the energy window or fit range.")
            self._update_summary()
            self._draw_empty_plots("g-factor fit failed.")
            return

        self._fit_df = fit_df
        self._fit_result = fit_result
        self._update_results_label(self._format_fit_summary(fit_result))
        self._update_summary()
        self._plot_results()

    def _parse_energy_range(self) -> Optional[Tuple[float, float]]:
        min_text = self.energy_min_entry.get().strip()
        max_text = self.energy_max_entry.get().strip()

        if not min_text and not max_text:
            return self._energy_range

        if not min_text or not max_text:
            messagebox.showerror("Energy Range", "Please provide both minimum and maximum energy.")
            return self._energy_range

        try:
            min_e = float(min_text)
            max_e = float(max_text)
        except ValueError:
            messagebox.showerror("Energy Range", "Energy limits must be numeric.")
            return self._energy_range

        if min_e >= max_e:
            messagebox.showerror("Energy Range", "Minimum energy must be less than maximum energy.")
            return self._energy_range

        return (min_e, max_e)

    def _parse_int(self, value: str, allow_zero: bool = False) -> Optional[int]:
        value = (value or "").strip()
        if not value:
            return 0 if allow_zero else None
        try:
            parsed = int(float(value))
        except ValueError:
            return None
        if parsed < 0:
            return None
        if parsed == 0 and allow_zero:
            return 0
        return max(parsed, 1)

    def _build_fit_mask(self, zeeman_df: pd.DataFrame) -> np.ndarray:
        fit_min_text = self.fit_min_entry.get().strip()
        fit_max_text = self.fit_max_entry.get().strip()

        mask = np.ones(len(zeeman_df), dtype=bool)
        try:
            if fit_min_text:
                min_b = float(fit_min_text)
                mask &= zeeman_df['bfield_t'] >= min_b
            if fit_max_text:
                max_b = float(fit_max_text)
                mask &= zeeman_df['bfield_t'] <= max_b
        except ValueError:
            messagebox.showerror("Fit Range", "B-field limits must be numeric.")
            return np.ones(len(zeeman_df), dtype=bool)
        return mask

    def _plot_results(self) -> None:
        if self._zeeman_df.empty:
            self._draw_empty_plots("No paired data to display.")
            return

        self.figure.clear()
        grid = self.figure.add_gridspec(3, 1, height_ratios=[3, 3, 2], hspace=0.14)
        ax_energy = self.figure.add_subplot(grid[0])
        ax_delta = self.figure.add_subplot(grid[1], sharex=ax_energy)
        ax_resid = self.figure.add_subplot(grid[2], sharex=ax_energy)

        # Energy vs B-field
        ax_energy.scatter(
            self._zeeman_df['bfield_t'],
            self._zeeman_df['sigma_plus_ev'],
            color="#d62728",
            marker="o",
            label="sigma+"
        )
        ax_energy.scatter(
            self._zeeman_df['bfield_t'],
            self._zeeman_df['sigma_minus_ev'],
            color="#1f77b4",
            marker="s",
            label="sigma-"
        )
        ax_energy.set_ylabel("Energy (eV)")
        ax_energy.grid(True, linestyle="--", alpha=0.4)
        ax_energy.legend(loc="best")
        ax_energy.set_title("Peak Energies vs B-Field")

        # Delta E plot
        ax_delta.scatter(
            self._zeeman_df['bfield_t'],
            self._zeeman_df['delta_mev'],
            color="#2ca02c",
            marker="o",
            label="ΔE data"
        )
        if self._fit_result:
            sorted_idx = np.argsort(self._fit_result['bfield_t'])
            ax_delta.plot(
                self._fit_result['bfield_t'][sorted_idx],
                self._fit_result['fit_delta_mev'][sorted_idx],
                color="#ff7f0e",
                linewidth=2.0,
                label="Linear fit"
            )
        if self._manual_g is not None:
            self._plot_manual_overlay(ax_delta)

        ax_delta.set_ylabel("ΔE (meV)")
        ax_delta.grid(True, linestyle="--", alpha=0.4)
        ax_delta.legend(loc="best")

        # Residuals for fitted points
        ax_resid.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        if self._fit_result:
            ax_resid.scatter(
                self._fit_result['bfield_t'],
                self._fit_result['residuals_mev'],
                color="#8c564b",
                marker="o"
            )
        ax_resid.set_ylabel("Residual (meV)")
        ax_resid.set_xlabel("B-Field (T)")
        ax_resid.grid(True, linestyle=":", alpha=0.4)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_empty_plots(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_manual_overlay(self, axis) -> None:
        try:
            mu_b = 0.05788  # meV/T
            b_vals = np.linspace(
                float(self._zeeman_df['bfield_t'].min()),
                float(self._zeeman_df['bfield_t'].max()),
                num=100
            )
            delta_manual = self._manual_g * mu_b * b_vals + self._manual_intercept
            axis.plot(
                b_vals,
                delta_manual,
                color="#9467bd",
                linestyle="--",
                linewidth=1.5,
                label=f"Manual g={self._manual_g:.3f}"
            )
        except Exception as exc:
            logger.debug(f"Failed to draw manual overlay: {exc}")

    def _apply_manual_overlay(self) -> None:
        g_text = self.manual_g_entry.get().strip()
        if not g_text:
            self._manual_g = None
            self._manual_intercept = 0.0
            self.update()
            self._plot_results()
            return

        try:
            self._manual_g = float(g_text)
        except ValueError:
            messagebox.showerror("Manual g-factor", "Manual g-factor must be numeric.")
            return

        intercept_text = self.manual_intercept_entry.get().strip()
        if intercept_text:
            try:
                self._manual_intercept = float(intercept_text)
            except ValueError:
                messagebox.showerror("Manual Intercept", "Manual intercept must be numeric.")
                return
        else:
            self._manual_intercept = 0.0

        self._plot_results()

    def _clear_manual_overlay(self) -> None:
        self._manual_g = None
        self._manual_intercept = 0.0
        self.manual_g_entry.delete(0, "end")
        self.manual_intercept_entry.delete(0, "end")
        self.manual_intercept_entry.insert(0, "0.0")
        self._plot_results()

    def _update_results_label(self, text: str) -> None:
        self.results_label.configure(text=text or "")

    def _format_fit_summary(self, fit_result: dict) -> str:
        g = fit_result.get('g_factor')
        g_unc = fit_result.get('g_uncertainty')
        intercept = fit_result.get('intercept_mev')
        intercept_unc = fit_result.get('intercept_uncertainty')
        r_sq = fit_result.get('r_squared')
        parts = []
        if g is not None and np.isfinite(g):
            if g_unc is not None and np.isfinite(g_unc):
                parts.append(f"g = {g:.4f} ± {g_unc:.4f}")
            else:
                parts.append(f"g = {g:.4f}")
        if intercept is not None and np.isfinite(intercept):
            if intercept_unc is not None and np.isfinite(intercept_unc):
                parts.append(f"ΔE₀ = {intercept:.3f} ± {intercept_unc:.3f} meV")
            else:
                parts.append(f"ΔE₀ = {intercept:.3f} meV")
        if r_sq is not None and np.isfinite(r_sq):
            parts.append(f"R² = {r_sq:.4f}")
        return " | ".join(parts) if parts else "Fit complete."

    def _update_summary(self) -> None:
        if self._peaks_df.empty:
            self.summary_label.configure(text="No peaks extracted.")
            return
        method = self.method_var.get()
        smoothing = self.smoothing_entry.get().strip() or "0"
        n_plus = int((self._peaks_df['polarization'] == 'sigma+').sum())
        n_minus = int((self._peaks_df['polarization'] == 'sigma-').sum())
        n_pairs = int(len(self._zeeman_df))
        summary = (
            f"Method: {method} | Smoothing: {smoothing} pts | "
            f"sigma+: {n_plus} pts | sigma-: {n_minus} pts | Paired: {n_pairs}"
        )
        self.summary_label.configure(text=summary)

    def _export_data(self) -> None:
        if self._zeeman_df.empty:
            messagebox.showinfo("Export Data", "No analysis data available for export.")
            return

        initial_name = "gfactor_analysis.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=initial_name
        )
        if not filepath:
            return

        try:
            export_path = Path(filepath)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            export_df = self._zeeman_df.copy()
            if self._fit_result:
                mapping = {float(b): (float(f), float(r)) for b, f, r in zip(
                    self._fit_result['bfield_t'],
                    self._fit_result['fit_delta_mev'],
                    self._fit_result['residuals_mev']
                )}
                export_df['fit_delta_mev'] = export_df['bfield_t'].map(lambda b: mapping.get(float(b), (np.nan, np.nan))[0])
                export_df['residual_mev'] = export_df['bfield_t'].map(lambda b: mapping.get(float(b), (np.nan, np.nan))[1])
            else:
                export_df['fit_delta_mev'] = np.nan
                export_df['residual_mev'] = np.nan

            export_df['method'] = self.method_var.get().lower()
            export_df['smoothing_points'] = self.smoothing_entry.get().strip() or "0"

            header_lines = [
                ["Advanced g-factor analysis export"],
                [f"Peak method: {self.method_var.get()}"],
                [f"Smoothing: {self.smoothing_entry.get().strip() or '0'} points"],
            ]
            energy_range = self._parse_energy_range()
            if energy_range:
                header_lines.append([f"Energy window (eV): {energy_range[0]:.6f} - {energy_range[1]:.6f}"])
            if self._fit_result:
                header_lines.append([self._format_fit_summary(self._fit_result)])
            if self._manual_g is not None:
                header_lines.append([f"Manual overlay: g={self._manual_g:.4f}, intercept={self._manual_intercept:.4f} meV"])
            header_lines.append([])

            with export_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                for line in header_lines:
                    writer.writerow(line)
                columns = export_df.columns.tolist()
                writer.writerow(columns)
                for _, row in export_df.iterrows():
                    formatted_row = []
                    for col in columns:
                        value = row[col]
                        if isinstance(value, str):
                            formatted_row.append(value)
                        else:
                            formatted_row.append(self._format_float(value))
                    writer.writerow(formatted_row)
        except Exception as exc:
            logger.error(f"Failed to export g-factor data: {exc}", exc_info=True)
            messagebox.showerror("Export Error", f"Could not export data.\nDetails: {exc}")
            return

        messagebox.showinfo("Export Complete", f"Exported analysis data to {filepath}")

    def _export_figure(self) -> None:
        if self._zeeman_df.empty:
            messagebox.showinfo("Export Figure", "No figure available to export.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            initialfile="gfactor_analysis.png"
        )
        if not filepath:
            return
        try:
            self.figure.savefig(filepath, dpi=300)
        except Exception as exc:
            logger.error(f"Failed to export figure: {exc}", exc_info=True)
            messagebox.showerror("Export Error", f"Could not export figure.\nDetails: {exc}")
            return
        messagebox.showinfo("Export Complete", f"Figure saved to {filepath}")

    @staticmethod
    def _format_float(value) -> str:
        if isinstance(value, (float, int)) and np.isfinite(value):
            return f"{float(value):.6g}"
        return ""
