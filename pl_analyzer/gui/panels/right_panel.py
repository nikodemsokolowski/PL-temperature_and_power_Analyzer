import customtkinter as ctk

from ..widgets.plot_canvas import PlotCanvas
from ..actions import plot_actions, analysis_actions, bfield_analysis_actions
from ..windows.bfield_selection_window import BFieldSelectionWindow

class RightPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.app = master

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Plot Area Frame
        self.plot_area_frame = ctk.CTkFrame(self)
        self.plot_area_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_area_frame.grid_rowconfigure(1, weight=1)
        self.plot_area_frame.grid_columnconfigure(0, weight=1)

        # Plot Controls
        self.plot_controls_frame = ctk.CTkFrame(self.plot_area_frame, fg_color="transparent")
        self.plot_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.plot_controls_frame.grid_columnconfigure(7, weight=1) # Add weight to push checkboxes

        # --- Row 0: Main plot buttons ---
        self.plot_selected_button = ctk.CTkButton(self.plot_controls_frame, text="Plot Selected", command=lambda: plot_actions.plot_selected_action(self.app))
        self.plot_selected_button.grid(row=0, column=0, padx=(5, 2), pady=(0, 2))

        self.add_to_plot_button = ctk.CTkButton(self.plot_controls_frame, text="Add to Plot", command=lambda: plot_actions.add_to_plot_action(self.app))
        self.add_to_plot_button.grid(row=0, column=1, padx=2, pady=(0, 2))

        self.plot_power_series_button = ctk.CTkButton(self.plot_controls_frame, text="Plot Power Series", command=lambda: plot_actions.plot_power_series_action(self.app))
        self.plot_power_series_button.grid(row=0, column=2, padx=2, pady=(0, 2))

        self.plot_temp_series_button = ctk.CTkButton(self.plot_controls_frame, text="Plot Temp Series", command=lambda: plot_actions.plot_temp_series_action(self.app))
        self.plot_temp_series_button.grid(row=0, column=3, padx=2, pady=(0, 2))

        self.clear_plot_button = ctk.CTkButton(self.plot_controls_frame, text="Clear Plot", command=lambda: plot_actions.clear_plot_action(self.app))
        self.clear_plot_button.grid(row=0, column=4, padx=(2, 2), pady=(0, 2))

        self.ultimate_plot_button = ctk.CTkButton(self.plot_controls_frame, text="Ultimate Plot Grid", command=lambda: plot_actions.ultimate_plot_action(self.app))
        self.ultimate_plot_button.grid(row=0, column=5, padx=(2, 2), pady=(0, 2))

        self.export_plot_button = ctk.CTkButton(self.plot_controls_frame, text="Export Plot", command=lambda: plot_actions.export_current_plot_action(self.app))
        self.export_plot_button.grid(row=0, column=6, padx=(2, 10), pady=(0, 2))

        # --- Row 1: Checkboxes ---
        self.normalize_checkbox = ctk.CTkCheckBox(self.plot_controls_frame, text="Normalize Spectra", variable=self.app.normalize_var, command=lambda: plot_actions._update_plot_style(self.app))
        self.normalize_checkbox.grid(row=1, column=0, padx=(5, 10), pady=(2, 0), sticky="w")

        self.log_scale_checkbox = ctk.CTkCheckBox(self.plot_controls_frame, text="Log Y-Axis", variable=self.app.log_scale_var, command=lambda: plot_actions._update_plot_style(self.app))
        self.log_scale_checkbox.grid(row=1, column=1, padx=10, pady=(2, 0), sticky="w")

        self.stack_checkbox = ctk.CTkCheckBox(self.plot_controls_frame, text="Normalized Stacked", variable=self.app.stack_var, command=lambda: plot_actions._update_plot_style(self.app))
        self.stack_checkbox.grid(row=1, column=2, padx=10, pady=(2, 0), sticky="w")

        self.intensity_map_checkbox = ctk.CTkCheckBox(self.plot_controls_frame, text="Plot Intensity Map", variable=self.app.intensity_map_var, command=lambda: plot_actions.toggle_intensity_map_action(self.app))
        self.intensity_map_checkbox.grid(row=1, column=3, padx=10, pady=(2, 0), sticky="w")

        # Equalize to lowest temperature (applies to Temperature Series)
        self.equalize_checkbox = ctk.CTkCheckBox(
            self.plot_controls_frame,
            text="Equalize to Min T",
            variable=self.app.equalize_var,
            command=lambda: plot_actions._update_plot_style(self.app)
        )
        self.equalize_checkbox.grid(row=1, column=4, padx=10, pady=(2, 0), sticky="w")

        # --- Row 2: Polarization Controls ---
        pol_label = ctk.CTkLabel(self.plot_controls_frame, text="Polarization:")
        pol_label.grid(row=2, column=0, padx=(5, 2), pady=(2, 0), sticky="w")

        self.app.polarization_mode_var = ctk.StringVar(value="all")
        pol_modes = ["All Data", "σ+ Only", "σ- Only", "Both (σ+&σ-)", "Sum (σ++σ-)"]
        pol_values = ["all", "sigma+", "sigma-", "both", "sum"]
        self.pol_menu = ctk.CTkOptionMenu(
            self.plot_controls_frame,
            variable=self.app.polarization_mode_var,
            values=pol_modes,
            command=lambda _: plot_actions._update_plot_style(self.app)
        )
        # Map display names to internal values
        self.app._pol_display_to_value = dict(zip(pol_modes, pol_values))
        self.app._pol_value_to_display = dict(zip(pol_values, pol_modes))
        self.pol_menu.grid(row=2, column=1, columnspan=2, padx=2, pady=(2, 0), sticky="w")

        self.plot_bfield_series_button = ctk.CTkButton(self.plot_controls_frame, text="Plot B-Field Series", command=lambda: plot_actions.plot_bfield_series_action(self.app))
        self.plot_bfield_series_button.grid(row=2, column=3, padx=(2, 2), pady=(2, 0))

        self.bfield_stack_checkbox = ctk.CTkCheckBox(
            self.plot_controls_frame,
            text="Normalized Stacked (B-Field)",
            variable=self.app.bfield_stack_var,
            command=lambda: plot_actions.bfield_stack_toggle_action(self.app)
        )
        self.bfield_stack_checkbox.grid(row=2, column=4, padx=(10, 0), pady=(2, 0), sticky="w")

        # Plot Canvas
        self.plot_canvas = PlotCanvas(master=self.plot_area_frame)
        self.plot_canvas.grid(row=1, column=0, sticky="nsew")

        # Analysis Tabs
        self.analysis_tabs = ctk.CTkTabview(self, height=200)
        self.analysis_tabs.grid(row=1, column=0, sticky="ew", padx=5, pady=(0,5))

        self.power_analysis_tab = self.analysis_tabs.add("Power Analysis")
        self.temp_analysis_tab = self.analysis_tabs.add("Temperature Analysis")
        self.stacked_plot_tab = self.analysis_tabs.add("Stacked Plot Options")
        self.intensity_map_tab = self.analysis_tabs.add("Intensity Map Options")
        self.bfield_analysis_tab = self.analysis_tabs.add("B-Field Analysis")
        self.figure_options_tab = self.analysis_tabs.add("Figure Options")
        self.general_options_tab = self.analysis_tabs.add("General Options")
        self.spike_tab = self.analysis_tabs.add("Spike Removal")


        self._create_power_analysis_tab()
        self._create_temp_analysis_tab()
        self._create_stacked_plot_tab()
        self._create_intensity_map_tab()
        self._create_bfield_analysis_tab()
        self._create_figure_options_tab()
        self._create_general_options_tab()
        self._create_spike_tab()

    def _initialize_processing_state(self):
        """Resets the UI elements in the right panel to their default state."""
        self.app.log_scale_var.set("0")
        self.app.stack_var.set("0")
        self.app.normalize_var.set("0")
        self.app.equalize_var.set("0")
        self.app.integration_min_entry.delete(0, 'end')
        self.app.integration_max_entry.delete(0, 'end')
        self.app.norm_min_entry.delete(0, 'end')
        self.app.norm_max_entry.delete(0, 'end')
        self.app.offset_entry.delete(0, 'end')
        self.app.offset_entry.insert(0, "1.0")
        if hasattr(self.app, 'eq_min_entry'):
            self.app.eq_min_entry.delete(0, 'end')
        if hasattr(self.app, 'eq_max_entry'):
            self.app.eq_max_entry.delete(0, 'end')
        if hasattr(self.app, 'plot_min_entry'):
            self.app.plot_min_entry.delete(0, 'end')
        if hasattr(self.app, 'plot_max_entry'):
            self.app.plot_max_entry.delete(0, 'end')
        if hasattr(self.app, 'analysis_results_label'):
            self.app.analysis_results_label.configure(text="")
        if hasattr(self.app, 'intensity_map_mode_var'):
            self.app.intensity_map_mode_var.set("Single (grayscale)")
            if hasattr(self, 'im_mode_menu'):
                self.im_mode_menu.set("Single (grayscale)")
        if hasattr(self.app, 'bfield_min_entry'):
            self.app.bfield_min_entry.delete(0, 'end')
        if hasattr(self.app, 'bfield_max_entry'):
            self.app.bfield_max_entry.delete(0, 'end')
        if hasattr(self, 'bfield_pol_menu'):
            self.app.bfield_pol_mode_var.set("Both (σ+&σ-)")
            self.bfield_pol_menu.set("Both (σ+&σ-)")
        if hasattr(self.app, 'bfield_results_label'):
            self.app.bfield_results_label.configure(text="")

    def _create_power_analysis_tab(self):
        self.power_analysis_tab.grid_columnconfigure((1, 2), weight=1)

        self.integration_label = ctk.CTkLabel(self.power_analysis_tab, text="Integration Range (eV):", font=ctk.CTkFont(weight="bold"))
        self.integration_label.grid(row=0, column=0, padx=(10, 2), pady=5, sticky="w")

        self.app.integration_min_entry = ctk.CTkEntry(self.power_analysis_tab, placeholder_text="Min E (eV)")
        self.app.integration_min_entry.grid(row=0, column=1, padx=2, pady=5, sticky="ew")

        self.app.integration_max_entry = ctk.CTkEntry(self.power_analysis_tab, placeholder_text="Max E (eV)")
        self.app.integration_max_entry.grid(row=0, column=2, padx=2, pady=5, sticky="ew")

        self.integrate_button = ctk.CTkButton(self.power_analysis_tab, text="Integrate Selected", command=lambda: analysis_actions.integrate_selected_action(self.app))
        self.integrate_button.grid(row=0, column=3, padx=5, pady=5)

        self.power_analysis_button = ctk.CTkButton(self.power_analysis_tab, text="Power Dependence Analysis", command=lambda: analysis_actions.power_dependence_analysis_action(self.app))
        self.power_analysis_button.grid(row=0, column=4, padx=5, pady=5)

        self.app.analysis_results_label = ctk.CTkLabel(self.power_analysis_tab, text="", anchor="w")
        self.app.analysis_results_label.grid(row=1, column=0, columnspan=5, padx=10, pady=5, sticky="ew")

    def _create_temp_analysis_tab(self):
        self.temp_analysis_tab.grid_columnconfigure(0, weight=1)
        self.temp_analysis_tab.grid_rowconfigure(0, weight=1)

        # The button now uses the integration range from the "Power Analysis" tab.
        # A label is added to inform the user.
        info_label = ctk.CTkLabel(self.temp_analysis_tab, 
                                  text="Uses the 'Integration Range' from the 'Power Analysis' tab.",
                                  font=ctk.CTkFont(size=12, slant="italic"))
        info_label.pack(pady=(10, 5))

        self.temp_analysis_button = ctk.CTkButton(self.temp_analysis_tab, text="Temperature Dependence Analysis", command=lambda: analysis_actions.temp_dependence_analysis_action(self.app))
        self.temp_analysis_button.pack(pady=5, padx=10)

    def _create_stacked_plot_tab(self):
        self.stacked_plot_tab.grid_columnconfigure((1, 2, 4), weight=1)

        self.norm_label = ctk.CTkLabel(self.stacked_plot_tab, text="Normalization Range (eV):", font=ctk.CTkFont(weight="bold"))
        self.norm_label.grid(row=0, column=0, padx=(10, 2), pady=5, sticky="w")

        self.app.norm_min_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Min E (eV)")
        self.app.norm_min_entry.grid(row=0, column=1, padx=2, pady=5, sticky="ew")

        self.app.norm_max_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Max E (eV)")
        self.app.norm_max_entry.grid(row=0, column=2, padx=2, pady=5, sticky="ew")

        self.offset_label = ctk.CTkLabel(self.stacked_plot_tab, text="Offset:", font=ctk.CTkFont(weight="bold"))
        self.offset_label.grid(row=0, column=3, padx=(10, 2), pady=5, sticky="w")

        self.app.offset_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Offset")
        self.app.offset_entry.grid(row=0, column=4, padx=2, pady=5, sticky="ew")
        self.app.offset_entry.insert(0, "1.0")

        # Row 1: Equalize range controls
        self.eq_label = ctk.CTkLabel(self.stacked_plot_tab, text="Equalize Range (eV):", font=ctk.CTkFont(weight="bold"))
        self.eq_label.grid(row=1, column=0, padx=(10, 2), pady=5, sticky="w")

        self.app.eq_min_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Min E (eV)")
        self.app.eq_min_entry.grid(row=1, column=1, padx=2, pady=5, sticky="ew")

        self.app.eq_max_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Max E (eV)")
        self.app.eq_max_entry.grid(row=1, column=2, padx=2, pady=5, sticky="ew")

        # Row 2: Plot range controls
        self.plot_range_label = ctk.CTkLabel(self.stacked_plot_tab, text="Plot Range (eV):", font=ctk.CTkFont(weight="bold"))
        self.plot_range_label.grid(row=2, column=0, padx=(10, 2), pady=5, sticky="w")

        self.app.plot_min_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Min E (eV)")
        self.app.plot_min_entry.grid(row=2, column=1, padx=2, pady=5, sticky="ew")

        self.app.plot_max_entry = ctk.CTkEntry(self.stacked_plot_tab, placeholder_text="Max E (eV)")
        self.app.plot_max_entry.grid(row=2, column=2, padx=2, pady=5, sticky="ew")

        # Row 3+: B-field selection controls
        bfield_header = ctk.CTkLabel(
            self.stacked_plot_tab,
            text="B-Field Selection",
            font=ctk.CTkFont(weight="bold")
        )
        bfield_header.grid(row=3, column=0, padx=(10, 2), pady=(12, 4), sticky="w")

        self.bfield_selection_summary = ctk.CTkLabel(
            self.stacked_plot_tab,
            text="B-Fields: none",
            anchor="w"
        )
        self.bfield_selection_summary.grid(row=3, column=1, columnspan=2, padx=2, pady=(12, 4), sticky="ew")

        self.bfield_selection_button = ctk.CTkButton(
            self.stacked_plot_tab,
            text="Select B-Fields...",
            command=self._open_bfield_selection_window
        )
        self.bfield_selection_button.grid(row=3, column=3, padx=4, pady=(12, 4), sticky="ew")

        step_label = ctk.CTkLabel(self.stacked_plot_tab, text="Step Interval (T):", font=ctk.CTkFont(weight="bold"))
        step_label.grid(row=4, column=0, padx=(10, 2), pady=(0, 8), sticky="w")
        self.bfield_step_combo = ctk.CTkComboBox(
            self.stacked_plot_tab,
            variable=self.app.bfield_step_var,
            values=["All", "0.5", "1", "1.5", "2", "3", "4", "5"],
            state="normal",
            width=120
        )
        self.bfield_step_combo.grid(row=4, column=1, padx=(2, 12), pady=(0, 8), sticky="w")

        self.bfield_step_hint = ctk.CTkLabel(
            self.stacked_plot_tab,
            text="Enter Tesla spacing (e.g., 0.75) or 'All'",
            font=ctk.CTkFont(size=11, slant="italic"),
            anchor="w"
        )
        self.bfield_step_hint.grid(row=4, column=2, columnspan=3, padx=(0, 10), pady=(0, 8), sticky="w")

    def _open_bfield_selection_window(self):
        values = getattr(self.app, '_bfield_stack_values', [])
        window = getattr(self.app, 'bfield_selection_window', None)
        if window is None or not window.winfo_exists():
            window = BFieldSelectionWindow(
                master=self.app,
                on_selection_change=self._on_bfield_selection_changed
            )
            self.app.bfield_selection_window = window
        selected = getattr(self.app, '_bfield_stack_selected', []) or []
        window.set_values(values, selected)
        window.deiconify()
        window.lift()
        window.focus_force()

    def _on_bfield_selection_changed(self, selected_values):
        self.app._bfield_stack_selected = selected_values
        self._refresh_bfield_summary()

    def _refresh_bfield_summary(self):
        values = getattr(self.app, '_bfield_stack_values', [])
        selected = getattr(self.app, '_bfield_stack_selected', [])
        if not values:
            text = "B-Fields: none available"
        else:
            if not selected or len(selected) == len(values):
                text = f"B-Fields: All ({len(values)})"
            else:
                preview = ", ".join(f"{val:.2f}" for val in selected[:4])
                if len(selected) > 4:
                    preview += ", ..."
                text = f"B-Fields: {preview} ({len(selected)} of {len(values)})"
        self.bfield_selection_summary.configure(text=text)

    def update_bfield_selection(self, bfield_values):
        """Update stored B-field values and refresh selection summary/window."""
        try:
            sorted_values = sorted({float(val) for val in bfield_values})
        except Exception:
            sorted_values = []

        existing_selected = getattr(self.app, '_bfield_stack_selected', None)
        if not sorted_values:
            self.app._bfield_stack_values = []
            self.app._bfield_stack_selected = []
            self._refresh_bfield_summary()
            window = getattr(self.app, 'bfield_selection_window', None)
            if window and window.winfo_exists():
                window.set_values([], [])
            return

        if existing_selected is None:
            selected = sorted_values.copy()
        else:
            selected = []
            for val in existing_selected:
                for candidate in sorted_values:
                    if abs(candidate - val) <= 1e-6:
                        selected.append(candidate)
                        break
            if not selected:
                selected = sorted_values.copy()

        self.app._bfield_stack_values = sorted_values
        self.app._bfield_stack_selected = selected
        self._refresh_bfield_summary()

        window = getattr(self.app, 'bfield_selection_window', None)
        if window and window.winfo_exists():
            window.set_values(sorted_values, selected)

    def _create_intensity_map_tab(self):
        self.intensity_map_tab.grid_columnconfigure((1, 2, 4), weight=1)

        # Row 0: Normalization
        self.im_normalize_checkbox = ctk.CTkCheckBox(self.intensity_map_tab, text="Normalize", variable=self.app.im_normalize_var, command=lambda: plot_actions.toggle_intensity_map_action(self.app))
        self.im_normalize_checkbox.grid(row=0, column=0, padx=(10, 2), pady=5, sticky="w")
        self.app.im_norm_min_entry = ctk.CTkEntry(self.intensity_map_tab, placeholder_text="Min E (eV)")
        self.app.im_norm_min_entry.grid(row=0, column=1, padx=2, pady=5, sticky="ew")
        self.app.im_norm_max_entry = ctk.CTkEntry(self.intensity_map_tab, placeholder_text="Max E (eV)")
        self.app.im_norm_max_entry.grid(row=0, column=2, padx=2, pady=5, sticky="ew")

        # Row 1: Log Scales
        self.im_log_c_checkbox = ctk.CTkCheckBox(self.intensity_map_tab, text="Log Color Scale", variable=self.app.im_log_c_var, command=lambda: plot_actions.toggle_intensity_map_action(self.app))
        self.im_log_c_checkbox.grid(row=1, column=0, padx=(10, 2), pady=5, sticky="w")
        self.im_log_y_checkbox = ctk.CTkCheckBox(self.intensity_map_tab, text="Log Y-Axis", variable=self.app.im_log_y_var, command=lambda: plot_actions.toggle_intensity_map_action(self.app))
        self.im_log_y_checkbox.grid(row=1, column=1, padx=(10, 2), pady=5, sticky="w")

        # Row 2: Intensity Range and Colormap
        self.im_range_label = ctk.CTkLabel(self.intensity_map_tab, text="Intensity Range:")
        self.im_range_label.grid(row=2, column=0, padx=(10, 2), pady=5, sticky="w")
        self.app.im_min_intensity_entry = ctk.CTkEntry(self.intensity_map_tab, placeholder_text="Min")
        self.app.im_min_intensity_entry.grid(row=2, column=1, padx=2, pady=5, sticky="ew")
        self.app.im_max_intensity_entry = ctk.CTkEntry(self.intensity_map_tab, placeholder_text="Max")
        self.app.im_max_intensity_entry.grid(row=2, column=2, padx=2, pady=5, sticky="ew")

        self.im_colormap_label = ctk.CTkLabel(self.intensity_map_tab, text="Colormap:")
        self.im_colormap_label.grid(row=2, column=3, padx=(10, 2), pady=5, sticky="w")
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray', 'jet']
        self.im_colormap_menu = ctk.CTkOptionMenu(self.intensity_map_tab, variable=self.app.im_colormap_var, values=colormaps, command=lambda _: plot_actions.toggle_intensity_map_action(self.app))
        self.im_colormap_menu.grid(row=2, column=4, padx=2, pady=5, sticky="ew")

        self.im_mode_label = ctk.CTkLabel(self.intensity_map_tab, text="Map Mode:")
        self.im_mode_label.grid(row=3, column=0, padx=(10, 2), pady=5, sticky="w")

        modes = [
            "Single (grayscale)",
            "Additive RGB (sigma+/sigma-)",
            "Alpha Overlay",
            "Diverging (sigma+-sigma-)"
        ]
        self.app.intensity_map_mode_var = ctk.StringVar(value=modes[0])
        self.im_mode_menu = ctk.CTkOptionMenu(
            self.intensity_map_tab,
            variable=self.app.intensity_map_mode_var,
            values=modes,
            command=lambda _: plot_actions.intensity_map_mode_changed(self.app)
        )
        self.im_mode_menu.grid(row=3, column=1, columnspan=3, padx=2, pady=5, sticky="ew")

    def _create_bfield_analysis_tab(self):
        self.bfield_analysis_tab.grid_columnconfigure((0, 1, 2), weight=1)

        range_label = ctk.CTkLabel(self.bfield_analysis_tab, text="Integration Range (eV):", font=ctk.CTkFont(weight="bold"))
        range_label.grid(row=0, column=0, padx=(10, 2), pady=6, sticky="w")

        self.app.bfield_min_entry = ctk.CTkEntry(self.bfield_analysis_tab, placeholder_text="Min E (eV)")
        self.app.bfield_min_entry.grid(row=0, column=1, padx=2, pady=6, sticky="ew")

        self.app.bfield_max_entry = ctk.CTkEntry(self.bfield_analysis_tab, placeholder_text="Max E (eV)")
        self.app.bfield_max_entry.grid(row=0, column=2, padx=(2, 10), pady=6, sticky="ew")

        pol_label = ctk.CTkLabel(self.bfield_analysis_tab, text="Polarization Mode:", font=ctk.CTkFont(weight="bold"))
        pol_label.grid(row=1, column=0, padx=(10, 2), pady=4, sticky="w")

        pol_display = ["σ+ Only", "σ- Only", "Both (σ+&σ-)", "Sum (σ++σ-)", "All Polarizations"]
        pol_values = ["sigma+", "sigma-", "both", "sum", "all"]
        self.app.bfield_pol_mode_var = ctk.StringVar(value="Both (σ+&σ-)")
        self.bfield_pol_menu = ctk.CTkOptionMenu(
            self.bfield_analysis_tab,
            variable=self.app.bfield_pol_mode_var,
            values=pol_display
        )
        self.app._bfield_pol_display_to_value = dict(zip(pol_display, pol_values))
        self.app._bfield_pol_value_to_display = dict(zip(pol_values, pol_display))
        self.bfield_pol_menu.grid(row=1, column=1, columnspan=2, padx=(2, 10), pady=4, sticky="ew")
        self.bfield_pol_menu.set("Both (σ+&σ-)")

        self.bfield_integrate_button = ctk.CTkButton(
            self.bfield_analysis_tab,
            text="Integrate vs B-Field",
            command=lambda: bfield_analysis_actions.integrate_vs_bfield_action(self.app)
        )
        self.bfield_integrate_button.grid(row=2, column=0, padx=(10, 5), pady=(8, 4), sticky="ew")

        self.bfield_gfactor_button = ctk.CTkButton(
            self.bfield_analysis_tab,
            text="Calculate g-Factor",
            command=lambda: bfield_analysis_actions.calculate_gfactor_action(self.app)
        )
        self.bfield_gfactor_button.grid(row=2, column=1, padx=5, pady=(8, 4), sticky="ew")

        self.bfield_intensity_button = ctk.CTkButton(
            self.bfield_analysis_tab,
            text="B-Field Intensity Map",
            command=lambda: bfield_analysis_actions.open_bfield_intensity_map_action(self.app)
        )
        self.bfield_intensity_button.grid(row=2, column=2, padx=(5, 10), pady=(8, 4), sticky="ew")

        self.app.bfield_results_label = ctk.CTkLabel(self.bfield_analysis_tab, text="", anchor="w", wraplength=420)
        self.app.bfield_results_label.grid(row=3, column=0, columnspan=3, padx=10, pady=(4, 8), sticky="ew")

    def _create_figure_options_tab(self):
        self.figure_options_tab.grid_columnconfigure((1, 3, 5), weight=1)

        # Columns
        cols_label = ctk.CTkLabel(self.figure_options_tab, text="Grid Columns:")
        cols_label.grid(row=0, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.fig_cols_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 4")
        self.app.fig_cols_entry.grid(row=0, column=1, padx=2, pady=5, sticky="ew")
        self.app.fig_cols_entry.insert(0, "4")

        # Figure size
        size_label = ctk.CTkLabel(self.figure_options_tab, text="Figure Size (in):")
        size_label.grid(row=0, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.fig_width_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="Width")
        self.app.fig_width_entry.grid(row=0, column=3, padx=2, pady=5, sticky="ew")
        self.app.fig_width_entry.insert(0, "16")
        self.app.fig_height_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="Height")
        self.app.fig_height_entry.grid(row=0, column=4, padx=2, pady=5, sticky="ew")
        self.app.fig_height_entry.insert(0, "10")

        # DPI
        dpi_label = ctk.CTkLabel(self.figure_options_tab, text="DPI:")
        dpi_label.grid(row=1, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.fig_dpi_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 150")
        self.app.fig_dpi_entry.grid(row=1, column=1, padx=2, pady=5, sticky="ew")
        self.app.fig_dpi_entry.insert(0, "150")

        # Font size
        fs_label = ctk.CTkLabel(self.figure_options_tab, text="Font Size:")
        fs_label.grid(row=1, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.font_size_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 10")
        self.app.font_size_entry.grid(row=1, column=3, padx=2, pady=5, sticky="ew")
        self.app.font_size_entry.insert(0, "10")

        # Line width
        lw_label = ctk.CTkLabel(self.figure_options_tab, text="Line Width:")
        lw_label.grid(row=1, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.line_width_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 1.0")
        self.app.line_width_entry.grid(row=1, column=5, padx=2, pady=5, sticky="ew")
        self.app.line_width_entry.insert(0, "1.2")

        # Color cycle
        cc_label = ctk.CTkLabel(self.figure_options_tab, text="Color Cycle:")
        cc_label.grid(row=2, column=0, padx=(10,2), pady=5, sticky="w")
        cycles = [
            'tab10','tab20','Set1','Set2','Dark2','Accent',
            'viridis','plasma','inferno','magma','cividis',
            'twilight','twilight_shifted',
            'Spectral','coolwarm','bwr','seismic','RdYlBu','RdBu',
            'Blues','PuBu','GnBu','BuGn','Greens','Oranges','Reds'
        ]
        self.app.color_cycle_var = ctk.StringVar(value='tab10')
        self.color_cycle_menu = ctk.CTkOptionMenu(self.figure_options_tab, variable=self.app.color_cycle_var, values=cycles)
        self.color_cycle_menu.grid(row=2, column=1, padx=2, pady=5, sticky="ew")

        # Legend options
        legend_mode_label = ctk.CTkLabel(self.figure_options_tab, text="Legend:")
        legend_mode_label.grid(row=2, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.legend_mode_var = ctk.StringVar(value='per-axes')
        legend_modes = ['per-axes','outside-right','last-only','none']
        self.legend_mode_menu = ctk.CTkOptionMenu(self.figure_options_tab, variable=self.app.legend_mode_var, values=legend_modes)
        self.legend_mode_menu.grid(row=2, column=3, padx=2, pady=5, sticky="ew")

        legend_fs_label = ctk.CTkLabel(self.figure_options_tab, text="Legend Font:")
        legend_fs_label.grid(row=3, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.legend_font_size_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 8")
        self.app.legend_font_size_entry.grid(row=3, column=1, padx=2, pady=5, sticky="ew")
        self.app.legend_font_size_entry.insert(0, "8")

        legend_ncol_label = ctk.CTkLabel(self.figure_options_tab, text="Legend Columns:")
        legend_ncol_label.grid(row=3, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.legend_ncol_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 1")
        self.app.legend_ncol_entry.grid(row=3, column=3, padx=2, pady=5, sticky="ew")
        self.app.legend_ncol_entry.insert(0, "1")

        # Show/Hide grid
        self.grid_show_checkbox = ctk.CTkCheckBox(self.figure_options_tab, text="Show Grid", variable=self.app.show_grid_var)
        self.grid_show_checkbox.grid(row=3, column=4, padx=(10,2), pady=5, sticky="w")

        # Style preset and tick options
        style_label = ctk.CTkLabel(self.figure_options_tab, text="Style Preset:")
        style_label.grid(row=3, column=0, padx=(10,2), pady=5, sticky="w")
        styles = ['Default','Compact','Nature','APS','Science']
        self.app.style_preset_var = ctk.StringVar(value='Default')
        self.style_preset_menu = ctk.CTkOptionMenu(self.figure_options_tab, variable=self.app.style_preset_var, values=styles)
        self.style_preset_menu.grid(row=3, column=1, padx=2, pady=5, sticky="ew")

        self.app.ticks_inside_var = ctk.StringVar(value="1")
        self.ticks_inside_checkbox = ctk.CTkCheckBox(self.figure_options_tab, text="Ticks Inside", variable=self.app.ticks_inside_var)
        self.ticks_inside_checkbox.grid(row=3, column=2, padx=(10,2), pady=5, sticky="w")

        self.app.minor_ticks_var = ctk.StringVar(value="0")
        self.minor_ticks_checkbox = ctk.CTkCheckBox(self.figure_options_tab, text="Show Minor Ticks", variable=self.app.minor_ticks_var)
        self.minor_ticks_checkbox.grid(row=3, column=3, padx=(10,2), pady=5, sticky="w")

        majlen_label = ctk.CTkLabel(self.figure_options_tab, text="Major Tick:")
        majlen_label.grid(row=3, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.maj_tick_len_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="4.0")
        self.app.maj_tick_len_entry.grid(row=3, column=5, padx=2, pady=5, sticky="ew")
        self.app.maj_tick_len_entry.insert(0, "4.0")

        minlen_label = ctk.CTkLabel(self.figure_options_tab, text="Minor Tick:")
        minlen_label.grid(row=4, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.min_tick_len_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="2.0")
        self.app.min_tick_len_entry.grid(row=4, column=1, padx=2, pady=5, sticky="ew")
        self.app.min_tick_len_entry.insert(0, "2.0")

        axlw_label = ctk.CTkLabel(self.figure_options_tab, text="Axes LW:")
        axlw_label.grid(row=4, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.axes_lw_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="1.0")
        self.app.axes_lw_entry.grid(row=4, column=3, padx=2, pady=5, sticky="ew")
        self.app.axes_lw_entry.insert(0, "1.0")

        # Subplot title formatting
        st_label = ctk.CTkLabel(self.figure_options_tab, text="Subplot Title:")
        st_label.grid(row=4, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.subplot_title_template_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="{T:.0f} K")
        self.app.subplot_title_template_entry.grid(row=4, column=5, padx=2, pady=5, sticky="ew")

        stmode_label = ctk.CTkLabel(self.figure_options_tab, text="Title Mode:")
        stmode_label.grid(row=5, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.subplot_title_mode_var = ctk.StringVar(value='axes')
        st_modes = ['axes','in-axes','none']
        self.subplot_title_mode_menu = ctk.CTkOptionMenu(self.figure_options_tab, variable=self.app.subplot_title_mode_var, values=st_modes)
        self.subplot_title_mode_menu.grid(row=5, column=1, padx=2, pady=5, sticky="ew")

        stposx_label = ctk.CTkLabel(self.figure_options_tab, text="Title X:")
        stposx_label.grid(row=5, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.subplot_title_posx_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="0.02")
        self.app.subplot_title_posx_entry.grid(row=5, column=3, padx=2, pady=5, sticky="ew")
        self.app.subplot_title_posx_entry.insert(0, "0.02")

        stposy_label = ctk.CTkLabel(self.figure_options_tab, text="Title Y:")
        stposy_label.grid(row=5, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.subplot_title_posy_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="0.95")
        self.app.subplot_title_posy_entry.grid(row=5, column=5, padx=2, pady=5, sticky="ew")
        self.app.subplot_title_posy_entry.insert(0, "0.95")

        # Theme (one‑click)
        theme_label = ctk.CTkLabel(self.figure_options_tab, text="Theme:")
        theme_label.grid(row=6, column=0, padx=(10,2), pady=5, sticky="w")
        themes = ['None','Nature (dense)','APS (two‑col)','ACS (single‑col)']
        self.app.theme_var = ctk.StringVar(value='None')
        self.theme_menu = ctk.CTkOptionMenu(self.figure_options_tab, variable=self.app.theme_var, values=themes, command=lambda t: plot_actions.apply_theme_preset(self.app, t))
        self.theme_menu.grid(row=6, column=1, padx=2, pady=5, sticky="ew")

        # Save / Load settings
        self.save_fig_settings_button = ctk.CTkButton(self.figure_options_tab, text="Save Settings", command=lambda: plot_actions.save_figure_settings_action(self.app))
        self.save_fig_settings_button.grid(row=6, column=2, padx=2, pady=5, sticky="ew")
        self.load_fig_settings_button = ctk.CTkButton(self.figure_options_tab, text="Load Settings", command=lambda: plot_actions.load_figure_settings_action(self.app))
        self.load_fig_settings_button.grid(row=6, column=3, padx=2, pady=5, sticky="ew")

        # Axis labels and spacing
        xl_label = ctk.CTkLabel(self.figure_options_tab, text="X Label:")
        xl_label.grid(row=3, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.x_label_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="Energy (eV)")
        self.app.x_label_entry.grid(row=3, column=5, padx=2, pady=5, sticky="ew")

        yl_label = ctk.CTkLabel(self.figure_options_tab, text="Y Label:")
        yl_label.grid(row=4, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.y_label_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="Intensity (arb. units)")
        self.app.y_label_entry.grid(row=4, column=5, padx=2, pady=5, sticky="ew")

        suptitle_label = ctk.CTkLabel(self.figure_options_tab, text="Suptitle:")
        suptitle_label.grid(row=4, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.suptitle_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="Power Series at All Temperatures")
        self.app.suptitle_entry.grid(row=4, column=3, padx=2, pady=5, sticky="ew")

        wspace_label = ctk.CTkLabel(self.figure_options_tab, text="W-Space:")
        wspace_label.grid(row=5, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.grid_wspace_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="0.3")
        self.app.grid_wspace_entry.grid(row=5, column=1, padx=2, pady=5, sticky="ew")
        self.app.grid_wspace_entry.insert(0, "0.3")

        hspace_label = ctk.CTkLabel(self.figure_options_tab, text="H-Space:")
        hspace_label.grid(row=5, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.grid_hspace_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="0.3")
        self.app.grid_hspace_entry.grid(row=5, column=3, padx=2, pady=5, sticky="ew")
        self.app.grid_hspace_entry.insert(0, "0.3")

        tick_fs_label = ctk.CTkLabel(self.figure_options_tab, text="Tick Font:")
        tick_fs_label.grid(row=5, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.tick_font_size_entry = ctk.CTkEntry(self.figure_options_tab, placeholder_text="e.g. 8")
        self.app.tick_font_size_entry.grid(row=5, column=5, padx=2, pady=5, sticky="ew")

        self.app.hide_inner_labels_var = ctk.StringVar(value="1")
        self.hide_inner_checkbox = ctk.CTkCheckBox(self.figure_options_tab, text="Hide Inner Axes Labels", variable=self.app.hide_inner_labels_var)
        self.hide_inner_checkbox.grid(row=6, column=0, columnspan=2, padx=(10,2), pady=5, sticky="w")

        # Transparent background option for export
        self.app.export_transparent_var = ctk.StringVar(value="0")
        self.transparent_checkbox = ctk.CTkCheckBox(self.figure_options_tab, text="Transparent Background (export)", variable=self.app.export_transparent_var)
        self.transparent_checkbox.grid(row=6, column=2, columnspan=2, padx=(10, 2), pady=5, sticky="w")

    def _create_spike_tab(self):
        """Spike removal tools UI."""
        self.spike_tab.grid_columnconfigure((1, 3, 5), weight=1)

        # Parameters
        w_label = ctk.CTkLabel(self.spike_tab, text="Median Window (pts):")
        w_label.grid(row=0, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.spike_window_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="7")
        self.app.spike_window_entry.grid(row=0, column=1, padx=2, pady=5, sticky="ew")
        self.app.spike_window_entry.insert(0, "7")

        s_label = ctk.CTkLabel(self.spike_tab, text="Sigma Threshold:")
        s_label.grid(row=0, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.spike_sigma_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="6.0")
        self.app.spike_sigma_entry.grid(row=0, column=3, padx=2, pady=5, sticky="ew")
        self.app.spike_sigma_entry.insert(0, "6.0")

        wmax_label = ctk.CTkLabel(self.spike_tab, text="Max Width (pts):")
        wmax_label.grid(row=0, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.spike_maxw_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="2")
        self.app.spike_maxw_entry.grid(row=0, column=5, padx=2, pady=5, sticky="ew")
        self.app.spike_maxw_entry.insert(0, "2")

        prom_label = ctk.CTkLabel(self.spike_tab, text="Min Prominence:")
        prom_label.grid(row=1, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.spike_prom_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="0.0")
        self.app.spike_prom_entry.grid(row=1, column=1, padx=2, pady=5, sticky="ew")
        self.app.spike_prom_entry.insert(0, "0.0")

        method_label = ctk.CTkLabel(self.spike_tab, text="Replace Method:")
        method_label.grid(row=1, column=2, padx=(10,2), pady=5, sticky="w")
        self.app.spike_method_var = ctk.StringVar(value='interp')
        self.spike_method_menu = ctk.CTkOptionMenu(self.spike_tab, variable=self.app.spike_method_var, values=['interp','median','nan','neighbor'])
        self.spike_method_menu.grid(row=1, column=3, padx=2, pady=5, sticky="ew")

        neigh_label = ctk.CTkLabel(self.spike_tab, text="Neighbor N:")
        neigh_label.grid(row=1, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.spike_neighbor_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="1")
        self.app.spike_neighbor_entry.grid(row=1, column=5, padx=2, pady=5, sticky="ew")
        self.app.spike_neighbor_entry.insert(0, "1")

        # Adaptive detection and manual radius
        self.app.spike_adaptive_var = ctk.StringVar(value="1")
        self.spike_adaptive_checkbox = ctk.CTkCheckBox(self.spike_tab, text="Adaptive Detector", variable=self.app.spike_adaptive_var)
        self.spike_adaptive_checkbox.grid(row=1, column=6, padx=(10,2), pady=5, sticky="w")

        # Hybrid mode (Adaptive + Prominence peak-top avoidance)
        self.app.spike_hybrid_var = ctk.StringVar(value="0")
        self.spike_hybrid_checkbox = ctk.CTkCheckBox(self.spike_tab, text="Hybrid Mode", variable=self.app.spike_hybrid_var)
        self.spike_hybrid_checkbox.grid(row=1, column=7, padx=(10,2), pady=5, sticky="w")

        aw_label = ctk.CTkLabel(self.spike_tab, text="Adaptive Window:")
        aw_label.grid(row=1, column=8, padx=(10,2), pady=5, sticky="w")
        self.app.spike_adapt_window_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="51")
        self.app.spike_adapt_window_entry.grid(row=1, column=9, padx=2, pady=5, sticky="ew")
        self.app.spike_adapt_window_entry.insert(0, "51")

        mr_label = ctk.CTkLabel(self.spike_tab, text="Manual Radius:")
        mr_label.grid(row=1, column=10, padx=(10,2), pady=5, sticky="w")
        self.app.spike_manual_radius_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="0")
        self.app.spike_manual_radius_entry.grid(row=1, column=11, padx=2, pady=5, sticky="ew")
        self.app.spike_manual_radius_entry.insert(0, "0")

        # Hybrid params
        hb_label = ctk.CTkLabel(self.spike_tab, text="Hybrid BroadWin:")
        hb_label.grid(row=1, column=12, padx=(10,2), pady=5, sticky="w")
        self.app.spike_hybrid_broad_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="151")
        self.app.spike_hybrid_broad_entry.grid(row=1, column=13, padx=2, pady=5, sticky="ew")
        self.app.spike_hybrid_broad_entry.insert(0, "151")
        havw_label = ctk.CTkLabel(self.spike_tab, text="Avoid Width:")
        havw_label.grid(row=1, column=14, padx=(10,2), pady=5, sticky="w")
        self.app.spike_hybrid_avoidw_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="9")
        self.app.spike_hybrid_avoidw_entry.grid(row=1, column=15, padx=2, pady=5, sticky="ew")
        self.app.spike_hybrid_avoidw_entry.insert(0, "9")
        havp_label = ctk.CTkLabel(self.spike_tab, text="Avoid Prom:")
        havp_label.grid(row=1, column=16, padx=(10,2), pady=5, sticky="w")
        self.app.spike_hybrid_avoidp_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="0")
        self.app.spike_hybrid_avoidp_entry.grid(row=1, column=17, padx=2, pady=5, sticky="ew")
        self.app.spike_hybrid_avoidp_entry.insert(0, "0")

        # Buttons
        self.spike_detect_button = ctk.CTkButton(self.spike_tab, text="Detect Spikes", command=lambda: plot_actions.detect_spikes_action(self.app))
        self.spike_detect_button.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.spike_manual_add_var = ctk.StringVar(value="0")
        self.spike_manual_add_checkbox = ctk.CTkCheckBox(self.spike_tab, text="Manual Add Mode (click plot)", variable=self.spike_manual_add_var, command=lambda: plot_actions.toggle_manual_spike_add_action(self.app))
        self.spike_manual_add_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.spike_apply_button = ctk.CTkButton(self.spike_tab, text="Apply Removal", fg_color="#d9534f", command=lambda: plot_actions.apply_spike_removal_action(self.app))
        self.spike_apply_button.grid(row=2, column=3, padx=5, pady=5)

        self.spike_clear_button = ctk.CTkButton(self.spike_tab, text="Clear Markers", command=lambda: plot_actions.clear_spike_markers_action(self.app))
        self.spike_clear_button.grid(row=2, column=4, padx=5, pady=5)

        # Undo and clone dataset
        self.spike_undo_button = ctk.CTkButton(self.spike_tab, text="Undo", command=lambda: plot_actions.undo_spike_removal_action(self.app))
        self.spike_undo_button.grid(row=2, column=5, padx=5, pady=5)

        self.spike_clone_button = ctk.CTkButton(self.spike_tab, text="Create Cleaned Dataset", command=lambda: plot_actions.create_cleaned_dataset_action(self.app))
        self.spike_clone_button.grid(row=2, column=6, padx=5, pady=5)

        self.spike_auto_apply_button = ctk.CTkButton(self.spike_tab, text="Remove Spikes (Auto)", command=lambda: plot_actions.auto_remove_spikes_action(self.app))
        self.spike_auto_apply_button.grid(row=2, column=7, padx=5, pady=5)

        # Sensitivity sweep
        sweep_label = ctk.CTkLabel(self.spike_tab, text="Sweep σ min/max/steps:")
        sweep_label.grid(row=3, column=0, padx=(10,2), pady=5, sticky="w")
        self.app.spike_sweep_min_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="3")
        self.app.spike_sweep_min_entry.grid(row=3, column=1, padx=2, pady=5, sticky="ew")
        self.app.spike_sweep_min_entry.insert(0, "3")
        self.app.spike_sweep_max_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="10")
        self.app.spike_sweep_max_entry.grid(row=3, column=2, padx=2, pady=5, sticky="ew")
        self.app.spike_sweep_max_entry.insert(0, "10")
        self.app.spike_sweep_steps_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="8")
        self.app.spike_sweep_steps_entry.grid(row=3, column=3, padx=2, pady=5, sticky="ew")
        self.app.spike_sweep_steps_entry.insert(0, "8")
        self.spike_sweep_button = ctk.CTkButton(self.spike_tab, text="Sensitivity Sweep", command=lambda: plot_actions.sensitivity_sweep_action(self.app))
        self.spike_sweep_button.grid(row=3, column=4, padx=5, pady=5)

        # Detection and skip ranges
        det_label = ctk.CTkLabel(self.spike_tab, text="Detect Range (eV):")
        det_label.grid(row=3, column=5, padx=(10,2), pady=5, sticky="w")
        self.app.spike_detect_min_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="Min")
        self.app.spike_detect_min_entry.grid(row=3, column=6, padx=2, pady=5, sticky="ew")
        self.app.spike_detect_max_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="Max")
        self.app.spike_detect_max_entry.grid(row=3, column=7, padx=2, pady=5, sticky="ew")

        skip_label = ctk.CTkLabel(self.spike_tab, text="Skip Removal Range (eV):")
        skip_label.grid(row=3, column=8, padx=(10,2), pady=5, sticky="w")
        self.app.spike_skip_min_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="Min")
        self.app.spike_skip_min_entry.grid(row=3, column=9, padx=2, pady=5, sticky="ew")
        self.app.spike_skip_max_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="Max")
        self.app.spike_skip_max_entry.grid(row=3, column=10, padx=2, pady=5, sticky="ew")

        # Log textbox
        self.app.spike_log = ctk.CTkTextbox(self.spike_tab, height=100)
        self.app.spike_log.grid(row=4, column=0, columnspan=18, padx=10, pady=5, sticky="ew")

        # Original/Per-curve controls
        self.spike_show_orig_var = ctk.StringVar(value="0")
        self.spike_show_orig_checkbox = ctk.CTkCheckBox(self.spike_tab, text="Show Original (toggle)", variable=self.spike_show_orig_var, command=lambda: plot_actions.toggle_show_original_action(self.app, self.spike_show_orig_var.get()=="1"))
        self.spike_show_orig_checkbox.grid(row=5, column=0, padx=5, pady=5, sticky="w")

        revert_label = ctk.CTkLabel(self.spike_tab, text="Revert Curve #:")
        revert_label.grid(row=5, column=1, padx=(10,2), pady=5, sticky="w")
        self.app.spike_revert_curve_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="1")
        self.app.spike_revert_curve_entry.grid(row=5, column=2, padx=2, pady=5, sticky="ew")
        self.spike_revert_button = ctk.CTkButton(self.spike_tab, text="Revert to Original", command=lambda: plot_actions.revert_curve_to_original_action(self.app))
        self.spike_revert_button.grid(row=5, column=3, padx=5, pady=5)

        # Prominence accept
        promx_label = ctk.CTkLabel(self.spike_tab, text="Prominence X:")
        promx_label.grid(row=5, column=4, padx=(10,2), pady=5, sticky="w")
        self.app.spike_prom_x_entry = ctk.CTkEntry(self.spike_tab, placeholder_text="500")
        self.app.spike_prom_x_entry.grid(row=5, column=5, padx=2, pady=5, sticky="ew")
        self.spike_prom_accept_button = ctk.CTkButton(self.spike_tab, text="Apply Prominence > X", command=lambda: plot_actions.apply_prominence_threshold_action(self.app))
        self.spike_prom_accept_button.grid(row=5, column=6, padx=5, pady=5)

        # Review navigation
        self.spike_prev_button = ctk.CTkButton(self.spike_tab, text="Prev", command=lambda: plot_actions.review_prev_action(self.app))
        self.spike_prev_button.grid(row=5, column=7, padx=5, pady=5)
        self.spike_next_button = ctk.CTkButton(self.spike_tab, text="Next", command=lambda: plot_actions.review_next_action(self.app))
        self.spike_next_button.grid(row=5, column=8, padx=5, pady=5)
        self.spike_exit_review_button = ctk.CTkButton(self.spike_tab, text="Exit Review", command=lambda: plot_actions.review_exit_action(self.app))
        self.spike_exit_review_button.grid(row=5, column=9, padx=5, pady=5)

        # Save/Load spike settings
        self.spike_save_button = ctk.CTkButton(self.spike_tab, text="Save Spike Settings", command=lambda: plot_actions.save_spike_settings_action(self.app))
        self.spike_save_button.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.spike_load_button = ctk.CTkButton(self.spike_tab, text="Load Spike Settings", command=lambda: plot_actions.load_spike_settings_action(self.app))
        self.spike_load_button.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # Clean this curve (review)
        self.spike_clean_current_button = ctk.CTkButton(self.spike_tab, text="Clean This Curve", command=lambda: plot_actions.clean_current_curve_action(self.app))
        self.spike_clean_current_button.grid(row=6, column=2, padx=5, pady=5)
        self.spike_clean_next_button = ctk.CTkButton(self.spike_tab, text="Clean & Next", command=lambda: plot_actions.clean_and_next_action(self.app))
        self.spike_clean_next_button.grid(row=6, column=3, padx=5, pady=5)

    def _create_general_options_tab(self):
        """Creates the General Options tab for application-level settings."""
        self.general_options_tab.grid_columnconfigure((1, 3), weight=1)

        # UI Scaling
        scaling_label = ctk.CTkLabel(self.general_options_tab, text="UI Scale:", font=ctk.CTkFont(weight="bold"))
        scaling_label.grid(row=0, column=0, padx=(10, 2), pady=10, sticky="w")

        from pl_analyzer.utils import config as app_config
        current_scale = app_config.load_ui_scale()
        scale_percentage = f"{int(current_scale * 100)}%"
        
        scale_options = ['75%', '85%', '100%', '110%', '125%']
        if scale_percentage not in scale_options:
            scale_percentage = '100%'
        
        self.app.ui_scale_var = ctk.StringVar(value=scale_percentage)
        self.ui_scale_menu = ctk.CTkOptionMenu(
            self.general_options_tab,
            variable=self.app.ui_scale_var,
            values=scale_options,
            command=lambda s: plot_actions.ui_scale_changed_action(self.app, s)
        )
        self.ui_scale_menu.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        scale_info = ctk.CTkLabel(
            self.general_options_tab,
            text="Changes take effect on next restart",
            font=ctk.CTkFont(size=10, slant="italic")
        )
        scale_info.grid(row=1, column=0, columnspan=2, padx=(10, 2), pady=(0, 10), sticky="w")

        # Window Resolution
        resolution_label = ctk.CTkLabel(self.general_options_tab, text="Window Resolution:", font=ctk.CTkFont(weight="bold"))
        resolution_label.grid(row=2, column=0, padx=(10, 2), pady=10, sticky="w")

        resolutions = ['1920x1080', '1600x900', '1280x720', '1200x800', '1024x768']
        current_res = app_config.load_window_resolution()
        self.app.resolution_var = ctk.StringVar(value=current_res if current_res in resolutions else 'Custom')
        self.resolution_menu = ctk.CTkOptionMenu(
            self.general_options_tab,
            variable=self.app.resolution_var,
            values=resolutions + ['Custom'],
            command=lambda r: plot_actions.resolution_changed_action(self.app, r)
        )
        self.resolution_menu.grid(row=2, column=1, padx=5, pady=10, sticky="ew")

        resolution_info = ctk.CTkLabel(
            self.general_options_tab,
            text="Changes take effect on next restart",
            font=ctk.CTkFont(size=10, slant="italic")
        )
        resolution_info.grid(row=3, column=0, columnspan=2, padx=(10, 2), pady=(0, 10), sticky="w")
