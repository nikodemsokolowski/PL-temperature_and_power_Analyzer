import logging
from pathlib import Path  # noqa: F401  # kept for possible future export enhancements
from typing import Iterable, List, Optional, Sequence, Tuple

import customtkinter as ctk
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib import colors as mcolors
from tkinter import filedialog, messagebox

from pl_analyzer.gui.widgets.plot_canvas import PlotCanvas
from pl_analyzer.gui.actions import plot_actions

logger = logging.getLogger(__name__)


class BFieldIntensityMapWindow(ctk.CTkToplevel):
    """Dedicated window for configuring and rendering B-field intensity maps."""

    _DEFAULT_COLORMAPS_PLUS = ["Reds", "OrRd", "PuRd"]
    _DEFAULT_COLORMAPS_MINUS = ["Blues", "PuBu", "GnBu"]

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("B-Field Intensity Map")
        self.geometry("1180x760")
        self.minsize(1020, 660)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Destroy>", self._on_destroy)

        # Stored data
        self._energies: List[np.ndarray] = []
        self._counts: List[np.ndarray] = []
        self._y_values: List[float] = []
        self._polarizations: List[Optional[str]] = []
        self._dataset_name: str = ""
        self._base_title: str = ""
        self._auto_global_range: Optional[Tuple[float, float]] = None
        self._auto_sigma_plus_range: Optional[Tuple[float, float]] = None
        self._auto_sigma_minus_range: Optional[Tuple[float, float]] = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.controls_frame = ctk.CTkFrame(self, corner_radius=8)
        self.controls_frame.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsw")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        self.canvas_frame = ctk.CTkFrame(self, corner_radius=8)
        self.canvas_frame.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = PlotCanvas(self.canvas_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self._build_controls()

    # ------------------------------------------------------------------ UI Construction
    def _build_controls(self) -> None:
        self.summary_label = ctk.CTkLabel(
            self.controls_frame,
            text="No B-field series loaded.",
            anchor="w",
            font=ctk.CTkFont(weight="bold")
        )
        self.summary_label.grid(row=0, column=0, padx=12, pady=(12, 8), sticky="ew")

        # Energy range
        energy_frame = ctk.CTkFrame(self.controls_frame)
        energy_frame.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        energy_frame.grid_columnconfigure((1, 2), weight=1)

        ctk.CTkLabel(energy_frame, text="Energy Range (eV):").grid(
            row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="w"
        )
        self.energy_min_entry = ctk.CTkEntry(energy_frame, placeholder_text="Min")
        self.energy_min_entry.grid(row=0, column=1, padx=4, pady=(8, 4), sticky="ew")
        self.energy_max_entry = ctk.CTkEntry(energy_frame, placeholder_text="Max")
        self.energy_max_entry.grid(row=0, column=2, padx=(4, 8), pady=(8, 4), sticky="ew")

        # Global intensity range
        intensity_frame = ctk.CTkFrame(self.controls_frame)
        intensity_frame.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        intensity_frame.grid_columnconfigure((1, 2), weight=1)

        ctk.CTkLabel(intensity_frame, text="Intensity Range (Single mode):").grid(
            row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="w"
        )
        self.intensity_min_entry = ctk.CTkEntry(intensity_frame, placeholder_text="Auto")
        self.intensity_min_entry.grid(row=0, column=1, padx=4, pady=(8, 4), sticky="ew")
        self.intensity_max_entry = ctk.CTkEntry(intensity_frame, placeholder_text="Auto")
        self.intensity_max_entry.grid(row=0, column=2, padx=(4, 8), pady=(8, 4), sticky="ew")

        # Colormap and log options
        cmap_frame = ctk.CTkFrame(self.controls_frame)
        cmap_frame.grid(row=3, column=0, padx=12, pady=6, sticky="ew")
        cmap_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cmap_frame, text="Colormap (Single mode):").grid(
            row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="w"
        )
        self.cmap_var = ctk.StringVar(value="viridis")
        self.cmap_menu = ctk.CTkOptionMenu(
            cmap_frame,
            variable=self.cmap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "gray", "turbo"]
        )
        self.cmap_menu.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="ew")

        self.log_color_var = ctk.BooleanVar(value=False)
        self.log_y_var = ctk.BooleanVar(value=False)
        self.log_color_checkbox = ctk.CTkCheckBox(
            cmap_frame,
            text="Log Color Scale",
            variable=self.log_color_var
        )
        self.log_color_checkbox.grid(row=1, column=0, padx=(8, 4), pady=4, sticky="w")
        self.log_y_checkbox = ctk.CTkCheckBox(
            cmap_frame,
            text="Log Y-Axis (B-field)",
            variable=self.log_y_var
        )
        self.log_y_checkbox.grid(row=1, column=1, padx=(4, 8), pady=4, sticky="w")

        # Map mode
        mode_frame = ctk.CTkFrame(self.controls_frame)
        mode_frame.grid(row=4, column=0, padx=12, pady=6, sticky="ew")
        mode_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(mode_frame, text="Map Mode:").grid(
            row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="w"
        )
        self.map_mode_var = ctk.StringVar(value="Single (grayscale)")
        self.map_mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            variable=self.map_mode_var,
            values=[
                "Single (grayscale)",
                "Additive RGB (sigma+/sigma-)",
                "Alpha Overlay",
                "Diverging (sigma+-sigma-)"
            ],
            command=lambda _: self._update_mode_controls()
        )
        self.map_mode_menu.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="ew")

        # RGB specific controls
        self.rgb_frame = ctk.CTkFrame(self.controls_frame)
        self.rgb_frame.grid(row=5, column=0, padx=12, pady=6, sticky="ew")
        self.rgb_frame.grid_columnconfigure((1, 2), weight=1)

        ctk.CTkLabel(self.rgb_frame, text="RGB Options").grid(
            row=0, column=0, columnspan=3, padx=8, pady=(8, 4), sticky="w"
        )

        self.show_sigma_plus_var = ctk.BooleanVar(value=True)
        self.show_sigma_minus_var = ctk.BooleanVar(value=True)
        self.show_info_box_var = ctk.BooleanVar(value=True)
        self.hide_colorbars_var = ctk.BooleanVar(value=False)

        self.show_sigma_plus_checkbox = ctk.CTkCheckBox(
            self.rgb_frame,
            text="Show σ+ scale bar",
            variable=self.show_sigma_plus_var,
            command=self._sync_colorbar_visibility
        )
        self.show_sigma_plus_checkbox.grid(row=1, column=0, columnspan=3, padx=(8, 4), pady=2, sticky="w")

        self.show_sigma_minus_checkbox = ctk.CTkCheckBox(
            self.rgb_frame,
            text="Show σ- scale bar",
            variable=self.show_sigma_minus_var,
            command=self._sync_colorbar_visibility
        )
        self.show_sigma_minus_checkbox.grid(row=2, column=0, columnspan=3, padx=(8, 4), pady=2, sticky="w")

        self.hide_all_checkbox = ctk.CTkCheckBox(
            self.rgb_frame,
            text="Hide all scale bars",
            variable=self.hide_colorbars_var,
            command=self._toggle_hide_colorbars
        )
        self.hide_all_checkbox.grid(row=3, column=0, columnspan=3, padx=(8, 4), pady=2, sticky="w")

        self.normalized_rgb_var = ctk.BooleanVar(value=False)
        self.normalized_rgb_checkbox = ctk.CTkCheckBox(
            self.rgb_frame,
            text="Normalize spectra before RGB map",
            variable=self.normalized_rgb_var
        )
        self.normalized_rgb_checkbox.grid(row=4, column=0, columnspan=3, padx=(8, 4), pady=(4, 8), sticky="w")

        self.show_info_box_checkbox = ctk.CTkCheckBox(
            self.rgb_frame,
            text="Show intensity info box",
            variable=self.show_info_box_var,
            command=self._on_show_info_toggle
        )
        self.show_info_box_checkbox.grid(row=5, column=0, columnspan=3, padx=(8, 4), pady=2, sticky="w")

        ctk.CTkLabel(self.rgb_frame, text="Info box position:").grid(
            row=6, column=0, padx=(8, 4), pady=(4, 4), sticky="w"
        )
        self.info_position_var = ctk.StringVar(value="Top Left (inside)")
        self.info_position_menu = ctk.CTkOptionMenu(
            self.rgb_frame,
            variable=self.info_position_var,
            values=["Top Left (inside)", "Top Right (inside)", "Outside Right"]
        )
        self.info_position_menu.grid(row=6, column=1, columnspan=2, padx=(4, 8), pady=(4, 4), sticky="ew")

        ctk.CTkLabel(self.rgb_frame, text="σ+ colormap:").grid(
            row=7, column=0, padx=(8, 4), pady=(8, 4), sticky="w"
        )
        self.sigma_plus_cmap_var = ctk.StringVar(value=self._DEFAULT_COLORMAPS_PLUS[0])
        self.sigma_plus_cmap_menu = ctk.CTkOptionMenu(
            self.rgb_frame,
            variable=self.sigma_plus_cmap_var,
            values=self._DEFAULT_COLORMAPS_PLUS
        )
        self.sigma_plus_cmap_menu.grid(row=7, column=1, columnspan=2, padx=(4, 8), pady=(8, 4), sticky="ew")

        ctk.CTkLabel(self.rgb_frame, text="σ- colormap:").grid(
            row=8, column=0, padx=(8, 4), pady=(8, 4), sticky="w"
        )
        self.sigma_minus_cmap_var = ctk.StringVar(value=self._DEFAULT_COLORMAPS_MINUS[0])
        self.sigma_minus_cmap_menu = ctk.CTkOptionMenu(
            self.rgb_frame,
            variable=self.sigma_minus_cmap_var,
            values=self._DEFAULT_COLORMAPS_MINUS
        )
        self.sigma_minus_cmap_menu.grid(row=8, column=1, columnspan=2, padx=(4, 8), pady=(8, 4), sticky="ew")

        # σ+ / σ- ranges
        self._build_sigma_range_rows()

        # Additional RGB rendering preferences
        prefs_frame = ctk.CTkFrame(self.rgb_frame)
        prefs_frame.grid(row=10, column=0, columnspan=3, padx=8, pady=(6, 4), sticky="ew")
        prefs_frame.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(prefs_frame, text="Below-floor color:").grid(
            row=0, column=0, padx=(4, 4), pady=2, sticky="w"
        )
        self.floor_under_color_var = ctk.StringVar(value="white")
        self.floor_under_color_menu = ctk.CTkOptionMenu(
            prefs_frame,
            variable=self.floor_under_color_var,
            values=["white", "black"]
        )
        self.floor_under_color_menu.grid(row=0, column=1, padx=4, pady=2, sticky="ew")

        ctk.CTkLabel(prefs_frame, text="Alpha background:").grid(
            row=0, column=2, padx=(8, 4), pady=2, sticky="w"
        )
        self.alpha_bg_mode_var = ctk.StringVar(value="Gray (data)")
        self.alpha_bg_mode_menu = ctk.CTkOptionMenu(
            prefs_frame,
            variable=self.alpha_bg_mode_var,
            values=["Gray (data)", "White", "Black"]
        )
        self.alpha_bg_mode_menu.grid(row=0, column=3, padx=4, pady=2, sticky="ew")

        ctk.CTkLabel(prefs_frame, text="σ+ cmap offset (0-0.5):").grid(
            row=1, column=0, padx=(4, 4), pady=2, sticky="w"
        )
        self.sigma_plus_offset_entry = ctk.CTkEntry(prefs_frame, placeholder_text="0.0")
        self.sigma_plus_offset_entry.grid(row=1, column=1, padx=4, pady=2, sticky="ew")

        ctk.CTkLabel(prefs_frame, text="σ- cmap offset (0-0.5):").grid(
            row=1, column=2, padx=(8, 4), pady=2, sticky="w"
        )
        self.sigma_minus_offset_entry = ctk.CTkEntry(prefs_frame, placeholder_text="0.0")
        self.sigma_minus_offset_entry.grid(row=1, column=3, padx=4, pady=2, sticky="ew")

        # Export options
        export_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        export_frame.grid(row=8, column=0, padx=12, pady=(8, 0), sticky="ew")
        export_frame.grid_columnconfigure((1, 2), weight=1)

        ctk.CTkLabel(export_frame, text="Export format:").grid(row=0, column=0, padx=(4, 4), pady=4, sticky="w")
        self.export_format_var = ctk.StringVar(value="PNG")
        self.export_format_menu = ctk.CTkOptionMenu(
            export_frame, variable=self.export_format_var, values=["PNG", "PDF", "SVG", "JPEG"]
        )
        self.export_format_menu.grid(row=0, column=1, padx=(4, 4), pady=4, sticky="ew")

        ctk.CTkLabel(export_frame, text="DPI:").grid(row=0, column=2, padx=(4, 4), pady=4, sticky="w")
        self.export_dpi_entry = ctk.CTkEntry(export_frame, width=60)
        self.export_dpi_entry.insert(0, "300")
        self.export_dpi_entry.grid(row=0, column=3, padx=(4, 4), pady=4, sticky="ew")

        self.export_transparent_var = ctk.BooleanVar(value=False)
        self.export_transparent_checkbox = ctk.CTkCheckBox(
            export_frame, text="Transparent background", variable=self.export_transparent_var
        )
        self.export_transparent_checkbox.grid(row=1, column=0, columnspan=4, padx=(4, 4), pady=(0, 6), sticky="w")

        # Action buttons
        button_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        button_frame.grid(row=10, column=0, padx=12, pady=(12, 8), sticky="ew")
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.generate_button = ctk.CTkButton(button_frame, text="Generate Map", command=self._generate_map)
        self.generate_button.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.export_button = ctk.CTkButton(button_frame, text="Export Image", command=self._export_figure)
        self.export_button.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.close_button = ctk.CTkButton(button_frame, text="Close", command=self._handle_close)
        self.close_button.grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.controls_frame,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=11, slant="italic")
        )
        self.status_label.grid(row=11, column=0, padx=12, pady=(0, 8), sticky="ew")

        self._update_mode_controls()
        self._on_show_info_toggle()

    def _build_sigma_range_rows(self) -> None:
        """Construct σ+ and σ- range input entries."""
        self.sigma_range_frame = ctk.CTkFrame(self.rgb_frame)
        self.sigma_range_frame.grid(row=9, column=0, columnspan=3, padx=8, pady=(6, 4), sticky="ew")
        self.sigma_range_frame.grid_columnconfigure((1, 2), weight=1)

        ctk.CTkLabel(self.sigma_range_frame, text="σ+ Range:").grid(
            row=0, column=0, padx=(4, 4), pady=2, sticky="w"
        )
        self.sigma_plus_min_entry = ctk.CTkEntry(self.sigma_range_frame, placeholder_text="Auto")
        self.sigma_plus_min_entry.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        self.sigma_plus_max_entry = ctk.CTkEntry(self.sigma_range_frame, placeholder_text="Auto")
        self.sigma_plus_max_entry.grid(row=0, column=2, padx=4, pady=2, sticky="ew")

        ctk.CTkLabel(self.sigma_range_frame, text="σ- Range:").grid(
            row=1, column=0, padx=(4, 4), pady=2, sticky="w"
        )
        self.sigma_minus_min_entry = ctk.CTkEntry(self.sigma_range_frame, placeholder_text="Auto")
        self.sigma_minus_min_entry.grid(row=1, column=1, padx=4, pady=2, sticky="ew")
        self.sigma_minus_max_entry = ctk.CTkEntry(self.sigma_range_frame, placeholder_text="Auto")
        self.sigma_minus_max_entry.grid(row=1, column=2, padx=4, pady=2, sticky="ew")

        ctk.CTkLabel(self.sigma_range_frame, text="σ+ floor (0-0.95):").grid(
            row=2, column=0, padx=(4, 4), pady=2, sticky="w"
        )
        self.sigma_plus_floor_entry = ctk.CTkEntry(self.sigma_range_frame, placeholder_text="0.0")
        self.sigma_plus_floor_entry.grid(row=2, column=1, columnspan=2, padx=4, pady=2, sticky="ew")

        ctk.CTkLabel(self.sigma_range_frame, text="σ- floor (0-0.95):").grid(
            row=3, column=0, padx=(4, 4), pady=2, sticky="w"
        )
        self.sigma_minus_floor_entry = ctk.CTkEntry(self.sigma_range_frame, placeholder_text="0.0")
        self.sigma_minus_floor_entry.grid(row=3, column=1, columnspan=2, padx=4, pady=2, sticky="ew")

    # ------------------------------------------------------------------ Window lifecycle
    def _handle_close(self) -> None:
        try:
            if hasattr(self.master, "bfield_intensity_window") and self.master.bfield_intensity_window is self:
                self.master.bfield_intensity_window = None
        finally:
            self.destroy()

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        if hasattr(self.master, "bfield_intensity_window") and self.master.bfield_intensity_window is self:
            self.master.bfield_intensity_window = None

    # ------------------------------------------------------------------ Data setup
    def set_source_data(
        self,
        *,
        energies: Sequence[np.ndarray],
        counts: Sequence[np.ndarray],
        bfield_values: Sequence[float],
        polarizations: Sequence[Optional[str]],
        dataset_name: str,
        base_title: str
    ) -> None:
        """Provide the spectra required for generating the intensity map."""
        try:
            self._energies = [np.asarray(e, dtype=float) for e in energies]
            self._counts = [np.asarray(c, dtype=float) for c in counts]
            self._y_values = [float(v) if v is not None else np.nan for v in bfield_values]
            self._polarizations = list(polarizations) if polarizations else [None] * len(self._energies)
            self._dataset_name = dataset_name
            self._base_title = base_title
        except Exception as exc:
            logger.error(f"Failed to prepare intensity map data: {exc}", exc_info=True)
            messagebox.showerror("Intensity Map", f"Unable to load B-field spectra.\nDetails: {exc}")
            return

        total = len(self._energies)
        bfield_array = np.asarray(self._y_values, dtype=float)
        finite_mask = np.isfinite(bfield_array)
        unique_fields = np.unique(np.round(bfield_array[finite_mask], 3)) if finite_mask.any() else []
        if total == 0 or not unique_fields.size:
            self.summary_label.configure(text="No usable B-field spectra available.")
        else:
            self.summary_label.configure(
                text=f"{dataset_name}: {len(unique_fields)} B-field values ({unique_fields.min():.3f}–{unique_fields.max():.3f} T)"
            )

        # Populate default ranges if not already set
        if self._energies:
            min_energy = min(float(e.min()) for e in self._energies if e.size)
            max_energy = max(float(e.max()) for e in self._energies if e.size)
            self._set_entry_if_empty(self.energy_min_entry, f"{min_energy:.6f}")
            self._set_entry_if_empty(self.energy_max_entry, f"{max_energy:.6f}")

        # Reset status
        self.status_label.configure(text="Ready. Adjust settings and click Generate Map.")

    # ------------------------------------------------------------------ Helpers
    @staticmethod
    def _set_entry_if_empty(entry: ctk.CTkEntry, value: str) -> None:
        if not entry.get():
            entry.insert(0, value)

    def _update_mode_controls(self) -> None:
        mode = self.map_mode_var.get()
        is_rgb_mode = mode.startswith("Additive") or mode.startswith("Alpha")
        is_diverging = mode.startswith("Diverging")

        state_rgb = "normal" if is_rgb_mode else "disabled"
        for widget in [
            self.show_sigma_plus_checkbox,
            self.show_sigma_minus_checkbox,
            self.show_info_box_checkbox,
            self.hide_all_checkbox,
            self.normalized_rgb_checkbox,
            self.sigma_plus_cmap_menu,
            self.sigma_minus_cmap_menu,
            self.sigma_plus_min_entry,
            self.sigma_plus_max_entry,
            self.sigma_minus_min_entry,
            self.sigma_minus_max_entry,
            self.sigma_plus_floor_entry,
            self.sigma_minus_floor_entry,
            self.info_position_menu,
            getattr(self, 'floor_under_color_menu', None),
            getattr(self, 'alpha_bg_mode_menu', None),
            getattr(self, 'sigma_plus_offset_entry', None),
            getattr(self, 'sigma_minus_offset_entry', None),
        ]:
            if widget is not None:
                widget.configure(state=state_rgb)

        self.cmap_menu.configure(state="disabled" if is_rgb_mode else "normal")
        self.intensity_min_entry.configure(state="disabled" if is_rgb_mode else "normal")
        self.intensity_max_entry.configure(state="disabled" if is_rgb_mode else "normal")
        self.log_color_checkbox.configure(state="disabled" if is_diverging else "normal")

        if is_rgb_mode:
            self.normalized_rgb_checkbox.configure(state="normal")
            self.info_position_menu.configure(state="normal" if self.show_info_box_var.get() else "disabled")
        else:
            self.normalized_rgb_var.set(False)
            self.normalized_rgb_checkbox.configure(state="disabled")
            self.info_position_menu.configure(state="disabled")

        if is_diverging:
            self.show_info_box_var.set(False)
            self.hide_colorbars_var.set(False)
            self.show_sigma_plus_var.set(True)
            self.show_sigma_minus_var.set(True)


        self._on_show_info_toggle()

    def _toggle_hide_colorbars(self) -> None:
        hide = self.hide_colorbars_var.get()
        if hide:
            self.show_sigma_plus_var.set(False)
            self.show_sigma_minus_var.set(False)
        else:
            if not self.show_sigma_plus_var.get() and not self.show_sigma_minus_var.get():
                self.show_sigma_plus_var.set(True)
                self.show_sigma_minus_var.set(True)

    def _sync_colorbar_visibility(self) -> None:
        if self.show_sigma_plus_var.get() or self.show_sigma_minus_var.get():
            self.hide_colorbars_var.set(False)
        else:
            self.hide_colorbars_var.set(True)

    def _on_show_info_toggle(self) -> None:
        mode = self.map_mode_var.get()
        is_rgb_mode = mode.startswith("Additive") or mode.startswith("Alpha")
        if not is_rgb_mode:
            self.info_position_menu.configure(state="disabled")
            return
        state = "normal" if self.show_info_box_var.get() else "disabled"
        self.info_position_menu.configure(state=state)

    def _normalize_channel(
        self,
        data: np.ndarray,
        manual_range: Optional[Tuple[float, float]],
        use_log: bool,
        label: str,
        allow_empty: bool = False
    ) -> Tuple[np.ndarray, Normalize, Tuple[float, float]]:
        array = np.asarray(data, dtype=float)
        if array.size == 0:
            default_range = manual_range if manual_range else (0.0, 1.0)
            norm = LogNorm(vmin=max(default_range[0], 1e-6), vmax=max(default_range[1], 1.0)) if use_log else Normalize(
                vmin=default_range[0], vmax=default_range[1]
            )
            return np.zeros_like(array), norm, default_range

        working = np.asarray(array, dtype=float)
        finite_mask = np.isfinite(working)

        if use_log:
            positive_mask = (working > 0) & finite_mask
            if not positive_mask.any():
                if allow_empty:
                    default_range = manual_range if manual_range else (1e-6, 1.0)
                    norm = LogNorm(vmin=default_range[0], vmax=default_range[1])
                    return np.zeros_like(array), norm, default_range
                raise ValueError(f"{label}: no positive intensity values available for logarithmic scaling.")
        else:
            if not finite_mask.any():
                if allow_empty:
                    default_range = manual_range if manual_range else (0.0, 1.0)
                    norm = Normalize(vmin=default_range[0], vmax=default_range[1])
                    return np.zeros_like(array), norm, default_range
                raise ValueError(f"{label}: no finite intensity values available.")

        if manual_range is None:
            if use_log:
                vmin = float(np.nanmin(working[positive_mask]))
                vmax = float(np.nanmax(working[positive_mask]))
            else:
                vmin = float(np.nanmin(working[finite_mask]))
                vmax = float(np.nanmax(working[finite_mask]))
        else:
            vmin, vmax = manual_range

        if use_log:
            if vmin <= 0:
                vmin = max(float(np.nanmin(working[working > 0])), 1e-6)
            if vmax <= vmin:
                vmax = vmin * 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmax <= vmin:
                vmax = vmin + max(abs(vmin), 1.0) * 1e-6
            norm = Normalize(vmin=vmin, vmax=vmax)

        normalized = norm(array, clip=True)
        if np.ma.isMaskedArray(normalized):
            normalized = normalized.filled(0.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized, norm, (vmin, vmax)

    @staticmethod
    def _parse_float(value: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        return float(value)

    def _parse_entry_pair(self, min_entry: ctk.CTkEntry, max_entry: ctk.CTkEntry, label: str) -> Optional[Tuple[float, float]]:
        text_min = min_entry.get().strip()
        text_max = max_entry.get().strip()
        if not text_min and not text_max:
            return None
        if not text_min or not text_max:
            raise ValueError(f"{label}: specify both minimum and maximum values.")
        min_val = float(text_min)
        max_val = float(text_max)
        if min_val >= max_val:
            raise ValueError(f"{label}: minimum must be less than maximum.")
        return (min_val, max_val)

    @staticmethod
    def _parse_floor_entry(entry: ctk.CTkEntry) -> float:
        text = entry.get().strip()
        if not text:
            return 0.0
        try:
            value = float(text)
        except Exception:
            return 0.0
        return float(min(max(value, 0.0), 0.95))

    @staticmethod
    def _parse_offset_entry(entry: ctk.CTkEntry) -> float:
        text = entry.get().strip()
        if not text:
            return 0.0
        try:
            value = float(text)
        except Exception:
            return 0.0
        return float(min(max(value, 0.0), 0.5))

    @staticmethod
    def _apply_floor(values: np.ndarray, floor_value: float) -> np.ndarray:
        floor = float(min(max(floor_value, 0.0), 0.95))
        if floor <= 0:
            return np.clip(values, 0.0, 1.0)
        scale = 1.0 - floor
        if scale <= 0:
            return np.zeros_like(values)
        return np.clip((values - floor) / scale, 0.0, 1.0)

    @staticmethod
    def _truncated_cmap(name: str, offset: float, start_color: Optional[str] = None) -> mcolors.Colormap:
        base = cm.get_cmap(name)
        off = float(min(max(offset, 0.0), 0.5))
        colors = base(np.linspace(off, 1.0, 256))
        if start_color:
            sc = start_color.lower()
            if sc.startswith('black'):
                colors[0, :3] = (0.0, 0.0, 0.0)
            elif sc.startswith('white'):
                colors[0, :3] = (1.0, 1.0, 1.0)
        return mcolors.LinearSegmentedColormap.from_list(f"{name}_trunc_{off:.2f}", colors)

    @staticmethod
    def _normalize_row(row: np.ndarray) -> np.ndarray:
        arr = np.asarray(row, dtype=float)
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, nan=0.0)
        min_val = float(np.nanmin(arr))
        max_val = float(np.nanmax(arr))
        if not np.isfinite(max_val) or max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _build_common_energy_grid(self, energy_range: Tuple[float, float]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """Resample spectra onto a common energy axis within the requested range."""
        emin, emax = energy_range
        if emin >= emax:
            raise ValueError("Energy range minimum must be less than maximum.")
        common_energy = np.linspace(emin, emax, 1200)

        resampled: List[np.ndarray] = []
        y_values: List[float] = []
        pols: List[str] = []

        for idx, energy in enumerate(self._energies):
            counts = self._counts[idx]
            if energy.size == 0 or counts.size == 0:
                continue
            order = np.argsort(energy)
            energy_sorted = energy[order]
            counts_sorted = counts[order]
            try:
                resampled_counts = np.interp(common_energy, energy_sorted, counts_sorted)
            except Exception:
                continue
            resampled.append(resampled_counts)
            y_values.append(self._y_values[idx] if idx < len(self._y_values) else np.nan)
            pols.append(self._polarizations[idx] if idx < len(self._polarizations) else None)

        if not resampled:
            raise ValueError("No spectra available within the requested energy range.")
        return common_energy, resampled, pols

    # ------------------------------------------------------------------ Map generation
    def _generate_map(self) -> None:
        if not self._energies or not self._counts:
            messagebox.showinfo("Intensity Map", "No B-field series data available. Plot a B-field series first.")
            return

        try:
            # Energy range
            default_energy_min = min(float(e.min()) for e in self._energies if e.size)
            default_energy_max = max(float(e.max()) for e in self._energies if e.size)
            energy_range = (
                self._parse_float(self.energy_min_entry.get()) or default_energy_min,
                self._parse_float(self.energy_max_entry.get()) or default_energy_max
            )
            common_energy, resampled, polarizations = self._build_common_energy_grid(energy_range)

            mode = self.map_mode_var.get()
            log_y = self.log_y_var.get()

            normalize_rows = bool(self.normalized_rgb_var.get()) and not mode.startswith("Single")
            if normalize_rows:
                normed = []
                for arr in resampled:
                    normed.append(self._normalize_row(arr))
                resampled = normed

            intensity_grid = np.vstack(resampled)
            y_values = np.array(self._y_values[:len(resampled)], dtype=float)
            if y_values.size != intensity_grid.shape[0]:
                y_values = np.linspace(0, intensity_grid.shape[0] - 1, intensity_grid.shape[0])

            # Sorting by B-field for clarity
            sort_idx = np.argsort(y_values)
            y_sorted = y_values[sort_idx]
            intensity_sorted = intensity_grid[sort_idx]
            polarizations_sorted = [polarizations[i] if i < len(polarizations) else None for i in sort_idx]

            self._auto_global_range = (
                float(np.nanmin(intensity_sorted)),
                float(np.nanmax(intensity_sorted))
            )

            # mode/log_y already computed above
            if mode.startswith("Single"):
                vmin_vmax = self._parse_entry_pair(self.intensity_min_entry, self.intensity_max_entry, "Intensity range")
                if vmin_vmax is None:
                    vmin_vmax = self._auto_global_range
                log_c = self.log_color_var.get()
                if log_c and vmin_vmax[0] <= 0:
                    messagebox.showwarning(
                        "Log Scale",
                        "Log color scale requires positive intensities. Adjust the intensity range."
                    )
                    return

                energy_mesh, field_mesh = np.meshgrid(common_energy, y_sorted)
                self.canvas.plot_intensity_map(
                    energy_mesh,
                    field_mesh,
                    intensity_sorted,
                    xlabel="Energy (eV)",
                    ylabel="Magnetic Field (T)",
                    title=f"{self._base_title} - Intensity Map",
                    log_c=log_c,
                    log_y=log_y,
                    cmap=self.cmap_var.get(),
                    vmin=vmin_vmax[0],
                    vmax=vmin_vmax[1],
                    xlim=energy_range,
                    colorbar_label="Intensity (arb. units)"
                )
                info_summary = f"Rendered single-channel map ({intensity_sorted.shape[0]} spectra)."

            else:
                        channels = plot_actions._aggregate_polarization_channels(common_energy, intensity_sorted, y_sorted, polarizations_sorted)
                        if channels['y_axis'].size == 0:
                            raise ValueError("Unable to assemble polarization channels for the selected data.")

                        sigma_plus = channels['sigma_plus']
                        sigma_minus = channels['sigma_minus']
                        combined = channels['combined']
                        other = channels['unpolarized']
                        y_axis_channels = channels['y_axis']

                        plus_manual = self._parse_entry_pair(
                            self.sigma_plus_min_entry, self.sigma_plus_max_entry, "σ+ intensity range"
                        )
                        minus_manual = self._parse_entry_pair(
                            self.sigma_minus_min_entry, self.sigma_minus_max_entry, "σ- intensity range"
                        )

                        plus_norm_vals, plus_norm_obj, plus_range = self._normalize_channel(
                            sigma_plus, plus_manual, self.log_color_var.get(), "σ+ intensity"
                        )
                        minus_norm_vals, minus_norm_obj, minus_range = self._normalize_channel(
                            sigma_minus, minus_manual, self.log_color_var.get(), "σ- intensity"
                        )
                        combined_norm_vals, _, _ = self._normalize_channel(
                            combined, None, self.log_color_var.get(), "Combined intensity", allow_empty=True
                        )
                        other_norm_vals, _, _ = self._normalize_channel(
                            other, None, self.log_color_var.get(), "Unpolarized intensity", allow_empty=True
                        )

                        self._auto_sigma_plus_range = plus_range
                        self._auto_sigma_minus_range = minus_range

                        order = np.argsort(y_axis_channels)
                        y_axis_sorted = y_axis_channels[order]
                        plus_norm_vals = plus_norm_vals[order]
                        minus_norm_vals = minus_norm_vals[order]
                        combined_norm_vals = combined_norm_vals[order]
                        other_norm_vals = other_norm_vals[order]

                        plus_floor = self._parse_floor_entry(self.sigma_plus_floor_entry)
                        minus_floor = self._parse_floor_entry(self.sigma_minus_floor_entry)
                        combined_floor = min(plus_floor, minus_floor)

                        plus_norm_vals = self._apply_floor(plus_norm_vals, plus_floor)
                        minus_norm_vals = self._apply_floor(minus_norm_vals, minus_floor)
                        combined_norm_vals = self._apply_floor(combined_norm_vals, combined_floor)
                        other_norm_vals = self._apply_floor(other_norm_vals, combined_floor)

                        show_plus = self.show_sigma_plus_var.get()
                        show_minus = self.show_sigma_minus_var.get()
                        hide_all = self.hide_colorbars_var.get()

                        if mode.startswith("Diverging"):
                            if not (channels['has_plus'] and channels['has_minus']):
                                messagebox.showinfo(
                                    "Insufficient Data",
                                    "Diverging map requires both σ+ and σ- spectra."
                                )
                                return
                            diff_grid = sigma_plus - sigma_minus
                            diff_grid = diff_grid[order]
                            vmax = float(np.nanmax(np.abs(diff_grid)))
                            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                            energy_mesh, field_mesh = np.meshgrid(common_energy, y_axis_sorted)
                            info_lines: List[str] = []
                            if self.show_info_box_var.get():
                                info_lines.append(f"σ+: [{plus_range[0]:.3g}, {plus_range[1]:.3g}]")
                                info_lines.append(f"σ-: [{minus_range[0]:.3g}, {minus_range[1]:.3g}]")
                                if plus_floor > 0 or minus_floor > 0:
                                    info_lines.append(f"Floors: σ+/σ- = {plus_floor:.2f}/{minus_floor:.2f}")
                                if self.log_color_var.get():
                                    info_lines.append("Log scale (base 10)")
                                if normalize_rows:
                                    info_lines.append("Spectra normalized")
                            info_text = "\n".join(info_lines) if info_lines else None
                            self.canvas.plot_intensity_map(
                                energy_mesh,
                                field_mesh,
                                diff_grid,
                                xlabel="Energy (eV)",
                                ylabel="Magnetic Field (T)",
                                title=f"{self._base_title} Δ(σ+ - σ-)",
                                log_c=False,
                                log_y=log_y,
                                cmap="coolwarm",
                                vmin=-vmax,
                                vmax=vmax,
                                xlim=energy_range,
                                colorbar_label="Δ Intensity (arb. units)",
                                info_text=info_text
                            )
                            info_summary = f"Diverging map using {diff_grid.shape[0]} B-field values."
                            if normalize_rows:
                                info_summary += " (spectra normalized)"
                        else:
                            plus_cmap_name = self.sigma_plus_cmap_var.get()
                            minus_cmap_name = self.sigma_minus_cmap_var.get()

                            # Apply user-defined offsets and below-floor colour
                            start_color = self.floor_under_color_var.get() if hasattr(self, 'floor_under_color_var') else 'white'
                            plus_off = self._parse_offset_entry(self.sigma_plus_offset_entry) if hasattr(self, 'sigma_plus_offset_entry') else 0.0
                            minus_off = self._parse_offset_entry(self.sigma_minus_offset_entry) if hasattr(self, 'sigma_minus_offset_entry') else 0.0

                            plus_cmap_obj = self._truncated_cmap(plus_cmap_name, plus_off, start_color)
                            minus_cmap_obj = self._truncated_cmap(minus_cmap_name, minus_off, start_color)

                            plus_colors = plus_cmap_obj(plus_norm_vals)[..., :3]
                            minus_colors = minus_cmap_obj(minus_norm_vals)[..., :3]
                            combined_colors = np.repeat(combined_norm_vals[..., None], 3, axis=2)
                            other_colors = np.repeat(other_norm_vals[..., None], 3, axis=2)

                            if mode.startswith("Additive"):
                                rgb = np.clip(
                                    combined_colors * 0.25
                                    + other_colors * 0.15
                                    + plus_colors * 0.75
                                    + minus_colors * 0.75,
                                    0.0,
                                    1.0
                                )
                            else:
                                # Alpha overlay background mode
                                bg_mode = getattr(self, 'alpha_bg_mode_var', None).get() if hasattr(self, 'alpha_bg_mode_var') else 'Gray (data)'
                                if isinstance(bg_mode, str) and bg_mode.lower().startswith('white'):
                                    base = np.ones_like(combined_colors)
                                elif isinstance(bg_mode, str) and bg_mode.lower().startswith('black'):
                                    base = np.zeros_like(combined_colors)
                                else:
                                    base = np.clip(combined_colors * 0.6 + other_colors * 0.2, 0.0, 1.0)
                                rgb = np.clip(base + plus_colors * 0.3 + minus_colors * 0.3, 0.0, 1.0)

                            info_lines: List[str] = []
                            if self.show_info_box_var.get():
                                info_lines.append(f"σ+: [{plus_range[0]:.3g}, {plus_range[1]:.3g}]")
                                info_lines.append(f"σ-: [{minus_range[0]:.3g}, {minus_range[1]:.3g}]")
                                if plus_floor > 0 or minus_floor > 0:
                                    info_lines.append(f"Floors: σ+/σ- = {plus_floor:.2f}/{minus_floor:.2f}")
                                if self.log_color_var.get():
                                    info_lines.append("Log scale (base 10)")
                                if normalize_rows:
                                    info_lines.append("Spectra normalized")
                            info_text = "\n".join(info_lines) if info_lines else None

                            self.canvas.plot_rgb_intensity_map(
                                common_energy,
                                y_axis_sorted,
                                rgb,
                                xlabel="Energy (eV)",
                                ylabel="Magnetic Field (T)",
                                title=f"{self._base_title} ({mode})",
                                xlim=energy_range,
                                log_y=log_y,
                                style_options={'show_grid': False},
                                show_sigma_plus_bar=show_plus and not hide_all,
                                show_sigma_minus_bar=show_minus and not hide_all,
                                sigma_plus_norm=plus_norm_obj if show_plus and not hide_all else None,
                                sigma_minus_norm=minus_norm_obj if show_minus and not hide_all else None,
                                sigma_plus_cmap=plus_cmap_obj,
                                sigma_minus_cmap=minus_cmap_obj,
                                show_info_box=bool(info_text),
                                info_text=info_text,
                                info_position=self.info_position_var.get()
                            )
                            info_summary = f"{mode} map rendered with {rgb.shape[0]} B-field slices."
                            if normalize_rows:
                                info_summary += " (spectra normalized)"

            self.status_label.configure(text=info_summary)
        except ValueError as exc:
            messagebox.showerror("Intensity Map", str(exc))
        except Exception as exc:
            logger.error(f"Failed to generate B-field intensity map: {exc}", exc_info=True)
            messagebox.showerror("Intensity Map", f"An unexpected error occurred.\nDetails: {exc}")

    # ------------------------------------------------------------------ Export
    def _export_figure(self) -> None:
        format_choice = (self.export_format_var.get() or "PNG").lower()
        format_map = {
            "png": ("PNG Image", "*.png"),
            "pdf": ("PDF Document", "*.pdf"),
            "svg": ("SVG Vector", "*.svg"),
            "jpeg": ("JPEG Image", "*.jpg")
        }
        label, pattern = format_map.get(format_choice, format_map["png"])
        default_ext = pattern.replace("*", "")
        filepath = filedialog.asksaveasfilename(
            title="Export Intensity Map",
            defaultextension=default_ext,
            filetypes=[format_map["png"], format_map["pdf"], format_map["svg"], format_map["jpeg"]]
        )
        if not filepath:
            return
        if not filepath.lower().endswith(default_ext):
            filepath = f"{filepath}{default_ext}"
        try:
            dpi = int(float(self.export_dpi_entry.get())) if self.export_dpi_entry.get() else 300
        except Exception:
            dpi = 300
        transparent = bool(self.export_transparent_var.get())
        try:
            saved = self.canvas.save_current_figure(filepath, dpi=dpi, transparent=transparent)
            if saved:
                messagebox.showinfo("Export Complete", f"Saved intensity map to:\n{filepath}")
        except Exception as exc:
            logger.error(f"Failed to export intensity map: {exc}", exc_info=True)
            messagebox.showerror("Export Error", f"Could not export intensity map.\nDetails: {exc}")
