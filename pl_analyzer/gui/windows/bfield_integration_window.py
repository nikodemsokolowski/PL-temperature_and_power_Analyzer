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


class EnhancedBFieldIntegrateWindow(ctk.CTkToplevel):
    """Interactive window for visualising integrated intensity vs B-field plots."""

    _STYLE_MAP = {
        "Lines + Markers": {"linestyle": "-", "marker": "o"},
        "Lines Only": {"linestyle": "-", "marker": None},
        "Markers Only": {"linestyle": "", "marker": "o"},
    }

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Integrated Intensity vs B-Field")
        self.geometry("960x720")
        self.minsize(820, 640)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Destroy>", self._on_destroy)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._data: Optional[pd.DataFrame] = None
        self._integration_range: Optional[Tuple[float, float]] = None
        self._dataset_name: str = ""
        self._default_mode: str = "all"
        self._norm_factor: Optional[float] = None

        self._build_controls()
        self._build_plot()

    def _handle_close(self) -> None:
        """Destroy window and clear app reference when user closes it."""
        try:
            if hasattr(self.master, "bfield_integration_window") and self.master.bfield_integration_window is self:
                self.master.bfield_integration_window = None
        finally:
            self.destroy()

    def _on_destroy(self, event) -> None:
        """Ensure master reference is cleared when window is actually destroyed."""
        if event.widget is not self:
            return
        if hasattr(self.master, "bfield_integration_window") and self.master.bfield_integration_window is self:
            self.master.bfield_integration_window = None

    def _build_controls(self) -> None:
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="new")
        for col in range(0, 8):
            self.control_frame.grid_columnconfigure(col, weight=0)
        self.control_frame.grid_columnconfigure(7, weight=1)

        self.show_sigma_plus_var = ctk.BooleanVar(value=True)
        self.show_sigma_minus_var = ctk.BooleanVar(value=True)
        self.show_sum_var = ctk.BooleanVar(value=True)
        self.show_dcp_var = ctk.BooleanVar(value=False)
        self.normalize_var = ctk.BooleanVar(value=False)
        self.plot_style_var = ctk.StringVar(value="Lines + Markers")

        self.plus_checkbox = ctk.CTkCheckBox(
            self.control_frame,
            text="Plot sigma+ intensity",
            variable=self.show_sigma_plus_var,
            command=self.update_plot
        )
        self.plus_checkbox.grid(row=0, column=0, padx=(8, 6), pady=4, sticky="w")

        self.minus_checkbox = ctk.CTkCheckBox(
            self.control_frame,
            text="Plot sigma- intensity",
            variable=self.show_sigma_minus_var,
            command=self.update_plot
        )
        self.minus_checkbox.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        self.sum_checkbox = ctk.CTkCheckBox(
            self.control_frame,
            text="Plot sum intensity",
            variable=self.show_sum_var,
            command=self.update_plot
        )
        self.sum_checkbox.grid(row=0, column=2, padx=6, pady=4, sticky="w")

        self.dcp_checkbox = ctk.CTkCheckBox(
            self.control_frame,
            text="Show DCP subplot",
            variable=self.show_dcp_var,
            command=self.update_plot
        )
        self.dcp_checkbox.grid(row=0, column=3, padx=6, pady=4, sticky="w")

        self.normalize_checkbox = ctk.CTkCheckBox(
            self.control_frame,
            text="Normalize intensities",
            variable=self.normalize_var,
            command=self.update_plot
        )
        self.normalize_checkbox.grid(row=0, column=4, padx=6, pady=4, sticky="w")

        style_label = ctk.CTkLabel(self.control_frame, text="Plot style:")
        style_label.grid(row=0, column=5, padx=(12, 4), pady=4, sticky="e")
        self.style_menu = ctk.CTkOptionMenu(
            self.control_frame,
            variable=self.plot_style_var,
            values=list(self._STYLE_MAP.keys()),
            command=lambda _: self.update_plot()
        )
        self.style_menu.grid(row=0, column=6, padx=4, pady=4, sticky="ew")

        self.export_button = ctk.CTkButton(
            self.control_frame,
            text="Export CSV",
            width=120,
            command=self._export_csv
        )
        self.export_button.grid(row=0, column=7, padx=(6, 8), pady=4, sticky="e")

        self.info_label = ctk.CTkLabel(self.control_frame, text="", anchor="w")
        self.info_label.grid(row=1, column=0, columnspan=8, padx=8, pady=(2, 0), sticky="ew")

    def _build_plot(self) -> None:
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(7.5, 5.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar_frame = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.summary_label = ctk.CTkLabel(self, text="", anchor="w", wraplength=900)
        self.summary_label.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="ew")

    def set_data(
        self,
        data: pd.DataFrame,
        integration_range: Tuple[float, float],
        dataset_name: str,
        default_mode: Optional[str] = None
    ) -> None:
        """Load new analysis results into the window."""
        if data is None or data.empty:
            self._data = pd.DataFrame(columns=['bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum', 'dcp'])
        else:
            required_cols = {'bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum'}
            missing = required_cols.difference(data.columns)
            if missing:
                raise ValueError(f"Missing required columns for integration window: {missing}")
            df = data.copy()
            if 'dcp' not in df.columns:
                try:
                    df['dcp'] = analysis.calculate_dcp(
                        df['intensity_sigma_plus'].values,
                        df['intensity_sigma_minus'].values
                    )
                except Exception as exc:
                    logger.error(f"Failed to calculate DCP in window: {exc}")
                    df['dcp'] = np.nan
            self._data = df

        self._integration_range = integration_range
        self._dataset_name = dataset_name or "Dataset"
        self._default_mode = default_mode or "all"

        self._configure_default_checks()
        self._update_info_label()
        self._update_summary()
        self.update_plot()

    def _configure_default_checks(self) -> None:
        """Set checkbox defaults based on available data and user selection."""
        if self._data is None or self._data.empty:
            self.show_sigma_plus_var.set(False)
            self.show_sigma_minus_var.set(False)
            self.show_sum_var.set(False)
            self.show_dcp_var.set(False)
            return

        has_plus = self._data['intensity_sigma_plus'].notna().any()
        has_minus = self._data['intensity_sigma_minus'].notna().any()
        has_sum = self._data['intensity_sum'].notna().any()
        has_dcp = np.isfinite(self._data['dcp']).any()

        mode = (self._default_mode or "all").lower()
        if mode == "sigma+":
            self.show_sigma_plus_var.set(has_plus)
            self.show_sigma_minus_var.set(False)
            self.show_sum_var.set(False)
        elif mode == "sigma-":
            self.show_sigma_plus_var.set(False)
            self.show_sigma_minus_var.set(has_minus)
            self.show_sum_var.set(False)
        elif mode == "sum":
            self.show_sigma_plus_var.set(False)
            self.show_sigma_minus_var.set(False)
            self.show_sum_var.set(has_sum)
        else:
            self.show_sigma_plus_var.set(has_plus)
            self.show_sigma_minus_var.set(has_minus)
            self.show_sum_var.set(has_sum)

        if not any([self.show_sigma_plus_var.get(), self.show_sigma_minus_var.get(), self.show_sum_var.get()]):
            # Ensure at least one trace is visible if data exists.
            if has_plus:
                self.show_sigma_plus_var.set(True)
            elif has_minus:
                self.show_sigma_minus_var.set(True)
            elif has_sum:
                self.show_sum_var.set(True)

        # DCP is useful only if both polarizations exist.
        self.show_dcp_var.set(False)
        self.dcp_checkbox.configure(state="normal" if has_dcp else "disabled")

    def _update_info_label(self) -> None:
        if not self._integration_range:
            text = f"Dataset: {self._dataset_name}"
        else:
            min_e, max_e = self._integration_range
            text = f"Dataset: {self._dataset_name} | Integration range: {min_e:.4f} to {max_e:.4f} eV"
        self.info_label.configure(text=text)

    def update_plot(self) -> None:
        """Redraw the plot using the current toggle states."""
        if self._data is None or self._data.empty:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self._update_summary()
            return

        show_dcp = self.show_dcp_var.get() and np.isfinite(self._data['dcp']).any()

        self.figure.clear()
        if show_dcp:
            grid = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax_main = self.figure.add_subplot(grid[0])
            ax_dcp = self.figure.add_subplot(grid[1], sharex=ax_main)
        else:
            ax_main = self.figure.add_subplot(111)
            ax_dcp = None

        x_values = self._data['bfield_t'].to_numpy(dtype=float)
        self._norm_factor = self._calculate_normalization_factor()
        style = self._STYLE_MAP.get(self.plot_style_var.get(), self._STYLE_MAP["Lines + Markers"])

        plotted = 0
        if self.show_sigma_plus_var.get():
            plotted += self._plot_series(ax_main, x_values, self._data['intensity_sigma_plus'].to_numpy(dtype=float), "sigma+", "#d62728", style)
        if self.show_sigma_minus_var.get():
            plotted += self._plot_series(ax_main, x_values, self._data['intensity_sigma_minus'].to_numpy(dtype=float), "sigma-", "#1f77b4", style)
        if self.show_sum_var.get():
            plotted += self._plot_series(ax_main, x_values, self._data['intensity_sum'].to_numpy(dtype=float), "sum", "#2f2f2f", style)

        if plotted == 0:
            ax_main.text(0.5, 0.5, "No intensity series selected", transform=ax_main.transAxes, ha="center", va="center")
        else:
            ax_main.legend(loc="best")

        ax_main.set_ylabel("Normalized Intensity (arb. units)" if self._use_normalized_values() else "Integrated Intensity (arb. units)")
        if show_dcp:
            ax_main.set_xlabel("")
            ax_main.tick_params(labelbottom=False)
        else:
            ax_main.set_xlabel("B-Field (T)")
        ax_main.set_title(f"{self._dataset_name} — Integrated Intensity vs B-Field")
        ax_main.grid(True, linestyle="--", alpha=0.4)

        if show_dcp and ax_dcp is not None:
            dcp_vals = self._data['dcp'].to_numpy(dtype=float)
            mask = np.isfinite(x_values) & np.isfinite(dcp_vals)
            if np.any(mask):
                ax_dcp.plot(x_values[mask], dcp_vals[mask], color="#9467bd", marker="o", linestyle="-", label="DCP")
                ax_dcp.set_ylim(-1.05, 1.05)
                ax_dcp.set_ylabel("DCP")
                ax_dcp.grid(True, linestyle=":", alpha=0.4)
                ax_dcp.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
            else:
                ax_dcp.text(0.5, 0.5, "DCP not available", transform=ax_dcp.transAxes, ha="center", va="center")
                ax_dcp.set_ylabel("DCP")
            ax_dcp.set_xlabel("B-Field (T)")

        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._update_summary()

    def _plot_series(self, axis, x_vals, y_vals, label, color, style_config) -> int:
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(mask):
            return 0

        y_plot = y_vals[mask]
        if self._use_normalized_values() and self._norm_factor and self._norm_factor > 0:
            y_plot = y_plot / self._norm_factor

        axis.plot(
            x_vals[mask],
            y_plot,
            label=label,
            color=color,
            linestyle=style_config["linestyle"],
            marker=style_config["marker"],
            linewidth=1.6
        )
        return 1

    def _calculate_normalization_factor(self) -> Optional[float]:
        if not self._use_normalized_values() or self._data is None or self._data.empty:
            return None
        values = []
        if self._data['intensity_sigma_plus'].notna().any():
            values.append(self._data['intensity_sigma_plus'].to_numpy(dtype=float))
        if self._data['intensity_sigma_minus'].notna().any():
            values.append(self._data['intensity_sigma_minus'].to_numpy(dtype=float))
        if self._data['intensity_sum'].notna().any():
            values.append(self._data['intensity_sum'].to_numpy(dtype=float))
        if not values:
            return None
        stacked = np.concatenate(values)
        max_val = np.nanmax(np.abs(stacked))
        if not np.isfinite(max_val) or max_val <= 0:
            return None
        return float(max_val)

    def _use_normalized_values(self) -> bool:
        return bool(self.normalize_var.get())

    def _update_summary(self) -> None:
        if self._data is None or self._data.empty:
            self.summary_label.configure(text="No integrated intensity data to display.")
            return

        counts = []
        counts.append(f"sigma+: {int(self._data['intensity_sigma_plus'].notna().sum())}")
        counts.append(f"sigma-: {int(self._data['intensity_sigma_minus'].notna().sum())}")
        counts.append(f"sum: {int(self._data['intensity_sum'].notna().sum())}")
        counts.append(f"DCP: {int(np.isfinite(self._data['dcp']).sum())}")

        norm_text = "on" if self._use_normalized_values() and self._norm_factor else "off"
        summary = f"Series counts — {', '.join(counts)} | Normalization: {norm_text}"
        self.summary_label.configure(text=summary)

    def _export_csv(self) -> None:
        if self._data is None or self._data.empty:
            messagebox.showinfo("Export Data", "No data available for export.")
            return

        initial_name = f"{self._dataset_name}_integrated_vs_bfield.csv".replace(" ", "_")
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

            norm_active = self._use_normalized_values() and self._norm_factor

            columns = ["bfield_t", "intensity_sigma_plus", "intensity_sigma_minus", "intensity_sum", "dcp"]
            header = ["B-Field (T)", "I_sigma_plus", "I_sigma_minus", "I_sum", "DCP"]
            if norm_active:
                header.extend(["I_sigma_plus_norm", "I_sigma_minus_norm", "I_sum_norm"])

            with export_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([f"Dataset: {self._dataset_name}"])
                if self._integration_range:
                    min_e, max_e = self._integration_range
                    writer.writerow([f"Integration range (eV): {min_e:.6f} - {max_e:.6f}"])
                writer.writerow([f"Normalization applied: {'Yes' if norm_active else 'No'}"])
                writer.writerow([f"Plot style: {self.plot_style_var.get()}"])
                writer.writerow([])
                writer.writerow(header)

                for _, row in self._data[columns].iterrows():
                    base_values = [
                        self._format_float(row["bfield_t"]),
                        self._format_float(row["intensity_sigma_plus"]),
                        self._format_float(row["intensity_sigma_minus"]),
                        self._format_float(row["intensity_sum"]),
                        self._format_float(row["dcp"], precision=6)
                    ]
                    if norm_active:
                        base_values.extend([
                            self._format_normalized(row["intensity_sigma_plus"]),
                            self._format_normalized(row["intensity_sigma_minus"]),
                            self._format_normalized(row["intensity_sum"])
                        ])
                    writer.writerow(base_values)
        except Exception as exc:
            logger.error(f"Failed to export integrated B-field data: {exc}", exc_info=True)
            messagebox.showerror("Export Error", f"Could not export data.\nDetails: {exc}")
            return

        messagebox.showinfo("Export Complete", f"Exported data to {filepath}")

    def _format_normalized(self, value: float) -> str:
        if not self._norm_factor or self._norm_factor <= 0 or value is None:
            return ""
        if not np.isfinite(value):
            return ""
        return self._format_float(float(value) / self._norm_factor)

    @staticmethod
    def _format_float(value: float, precision: int = 6) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)) and np.isfinite(value):
            return f"{value:.{precision}g}"
        return ""
