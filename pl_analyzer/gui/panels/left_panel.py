import customtkinter as ctk

from ..widgets.file_table import FileTable
from ..actions import file_actions, processing_actions, plot_actions

class LeftPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.app = master
        self.file_tables: dict[str, FileTable] = {}

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Top-level Controls ---
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)

        self.load_dataset_button = ctk.CTkButton(self.controls_frame, text="Load New Dataset", command=lambda: file_actions.load_dataset_action(self.app))
        self.load_dataset_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.add_files_button = ctk.CTkButton(self.controls_frame, text="Add Files to Dataset", command=lambda: file_actions.add_files_to_dataset_action(self.app))
        self.add_files_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.import_spectra_button = ctk.CTkButton(self.controls_frame, text="Import Spectra", command=lambda: file_actions.import_spectra_action(self.app))
        self.import_spectra_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.clear_all_button = ctk.CTkButton(self.controls_frame, text="Clear All Datasets", command=lambda: file_actions.clear_all_data_action(self.app), fg_color="red")
        self.clear_all_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.clear_dataset_button = ctk.CTkButton(self.controls_frame, text="Clear Selected Dataset", command=lambda: file_actions.clear_dataset_action(self.app))
        self.clear_dataset_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.clear_selected_button = ctk.CTkButton(self.controls_frame, text="Clear Selected Data", command=lambda: file_actions.clear_selected_action(self.app))
        self.clear_selected_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.create_sum_button = ctk.CTkButton(
            self.controls_frame,
            text="Create Sum Dataset",
            command=lambda: file_actions.create_sum_dataset_action(self.app)
        )
        self.create_sum_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Magnetic/Polarization checkbox
        from pl_analyzer.utils import config as app_config
        has_mag_pol = app_config.load_has_magnetic_polarization()
        self.app.has_mag_pol_var = ctk.StringVar(value="1" if has_mag_pol else "0")
        self.mag_pol_checkbox = ctk.CTkCheckBox(self.controls_frame, text="Has Magnetic/Polarization Data", 
                                                 variable=self.app.has_mag_pol_var,
                                                 command=lambda: self._toggle_mag_pol_controls())
        self.mag_pol_checkbox.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.settings_button = ctk.CTkButton(self.controls_frame, text="⚙ Polarization Settings", command=lambda: file_actions.polarization_settings_action(self.app))
        self.settings_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.magnetic_settings_button = ctk.CTkButton(self.controls_frame, text="\u2699 Magnetic Sweep Settings", command=lambda: file_actions.magnetic_sweep_settings_action(self.app))
        self.magnetic_settings_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # --- Tab View for Datasets ---
        self.tab_view = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        # Initially, no tabs exist. They are added dynamically.

        # --- Frames that are now below the tab view ---
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.export_frame = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.export_frame.grid(row=0, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.export_frame.grid_columnconfigure((0, 1), weight=1)

        self.export_label = ctk.CTkLabel(self.export_frame, text="Export Data:", font=ctk.CTkFont(weight="bold"))
        self.export_label.grid(row=0, column=0, columnspan=2, padx=5, pady=(0,2), sticky="w")

        self.export_integrated_button = ctk.CTkButton(self.export_frame, text="Integrated Data", command=lambda: file_actions.export_integrated_action(self.app))
        self.export_integrated_button.grid(row=1, column=0, padx=5, pady=2, sticky="ew")

        self.export_spectra_button = ctk.CTkButton(self.export_frame, text="Selected Spectra", command=lambda: file_actions.export_spectra_action(self.app))
        self.export_spectra_button.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.export_all_spectra_button = ctk.CTkButton(self.export_frame, text="All Spectra", command=lambda: file_actions.export_all_spectra_action(self.app))
        self.export_all_spectra_button.grid(row=2, column=0, padx=5, pady=2, sticky="ew")

        self.export_integrated_all_button = ctk.CTkButton(self.export_frame, text="All Integrated Data", command=lambda: file_actions.export_integrated_all_action(self.app))
        self.export_integrated_all_button.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        self.processing_frame = ctk.CTkFrame(self.bottom_frame)
        self.processing_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.processing_frame.grid_columnconfigure(0, weight=1)

        self.processing_label = ctk.CTkLabel(self.processing_frame, text="Processing Steps", font=ctk.CTkFont(weight="bold"))
        self.processing_label.grid(row=0, column=0, padx=10, pady=(5, 2), sticky="w")

        self.normalize_time_button = ctk.CTkButton(self.processing_frame, text="Normalize by Time", command=lambda: processing_actions.normalize_time_action(self.app))
        self.normalize_time_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.rescale_gf_button = ctk.CTkButton(self.processing_frame, text="Rescale by GF Factor", command=lambda: processing_actions.rescale_gf_action(self.app))
        self.rescale_gf_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.correct_response_button = ctk.CTkButton(self.processing_frame, text="Correct Spectrometer Response", command=lambda: processing_actions.correct_response_action(self.app))
        self.correct_response_button.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.subtract_baseline_button = ctk.CTkButton(self.processing_frame, text="Subtract Baseline (min value)", command=lambda: processing_actions.subtract_baseline_action(self.app))
        self.subtract_baseline_button.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        # Initialize magnetic/polarization controls state
        # This needs to be called after right_panel is created, so we'll do it via a callback
        self.app.after(100, self._toggle_mag_pol_controls)

    def add_dataset_tab(self, name: str):
        """Adds a new tab for a dataset."""
        if name in self.tab_view._tab_dict:
            self.tab_view.set(name)
            return

        tab = self.tab_view.add(name)
        file_table = FileTable(master=tab)
        file_table.pack(fill="both", expand=True)
        self.file_tables[name] = file_table
        self.tab_view.set(name) # Make the new tab active

    def remove_all_dataset_tabs(self):
        """Removes all dataset tabs from the view."""
        for name in list(self.file_tables.keys()):
            self.tab_view.delete(name)
        self.file_tables.clear()

    def remove_dataset_tab(self, name: str):
        """Removes a single dataset tab from the view."""
        if name in self.file_tables:
            self.tab_view.delete(name)
            del self.file_tables[name]

    def get_active_file_table(self) -> FileTable | None:
        """Returns the FileTable of the currently active tab."""
        active_tab_name = self.tab_view.get()
        return self.file_tables.get(active_tab_name)

    def on_tab_change(self):
        """Callback for when the active tab changes."""
        active_tab_name = self.tab_view.get()
        if active_tab_name:
            self.app.data_handler.set_active_dataset(active_tab_name)
            self.app.update_file_table() # Update table content for the new active dataset

    def _toggle_mag_pol_controls(self):
        """Enable or disable magnetic/polarization controls based on checkbox state."""
        from pl_analyzer.utils import config as app_config
        has_mag_pol = self.app.has_mag_pol_var.get() == "1"
        
        # Save state to config
        app_config.save_has_magnetic_polarization(has_mag_pol)
        
        # Toggle polarization settings buttons
        state = "normal" if has_mag_pol else "disabled"
        self.settings_button.configure(state=state)
        self.magnetic_settings_button.configure(state=state)
        
        # Toggle polarization dropdown and B-field button in right panel
        if hasattr(self.app, 'right_panel'):
            if hasattr(self.app.right_panel, 'pol_menu'):
                self.app.right_panel.pol_menu.configure(state=state)
            if hasattr(self.app.right_panel, 'plot_bfield_series_button'):
                self.app.right_panel.plot_bfield_series_button.configure(state=state)
            # Toggle B-Field Analysis tab
            if hasattr(self.app.right_panel, 'analysis_tabs'):
                try:
                    if has_mag_pol:
                        # Enable the tab (it's always present, just need to allow selection)
                        pass  # CustomTkinter tabs don't have a direct disable method
                    else:
                        # We can't fully disable a tab in CTkTabview, but we can switch away from it
                        current_tab = self.app.right_panel.analysis_tabs.get()
                        if current_tab == "B-Field Analysis":
                            self.app.right_panel.analysis_tabs.set("Power Analysis")
                except Exception:
                    pass

        if not has_mag_pol:
            if hasattr(self.app, 'polarization_mode_var'):
                self.app.polarization_mode_var.set("all")
                try:
                    plot_actions._update_plot_style(self.app)
                except Exception:
                    pass
            if hasattr(self.app, 'right_panel') and hasattr(self.app.right_panel, 'pol_menu'):
                try:
                    self.app.right_panel.pol_menu.set("All Data")
                except Exception:
                    pass

