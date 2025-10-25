import customtkinter as ctk
import logging
from tkinter import messagebox
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class GfFactorDialog(ctk.CTkInputDialog):
    """
    A simple dialog to get the Grey Filter transmission factor from the user.
    Inherits from CTkInputDialog for convenience.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Grey Filter Factor")
        # self._label.configure(text="Enter GF Transmission Factor (e.g., 0.052):")

    def get_input(self) -> Optional[float]:
        """
        Gets the input from the dialog and attempts to convert it to a float.

        Returns:
            The input value as a float, or None if input is invalid or cancelled.
        """
        value_str = super().get_input() # This waits for user input
        if value_str is None or value_str.strip() == "":
            logger.debug("GF Factor dialog cancelled or input empty.")
            return None
        try:
            value_float = float(value_str)
            if value_float <= 0:
                logger.warning(f"Invalid GF factor entered (non-positive): {value_float}")
                # Optionally show a warning messagebox here
                return None
            logger.debug(f"GF Factor entered: {value_float}")
            return value_float
        except ValueError:
            logger.warning(f"Invalid GF factor entered (not a number): {value_str}")
            # Optionally show a warning messagebox here
            return None

class IntegrationRangeDialog(ctk.CTkToplevel):
    """
    A dialog to get a valid integration range (min and max energy) from the user.
    """
    def __init__(self, master, title="Set Range", **kwargs):
        super().__init__(master, **kwargs)
        self.title(title)
        self.geometry("300x150")
        self.lift()
        self.attributes("-topmost", True)
        self.grab_set() # Make modal

        self.grid_columnconfigure(1, weight=1)

        self.min_label = ctk.CTkLabel(self, text="Min Energy (eV):")
        self.min_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.min_entry = ctk.CTkEntry(self)
        self.min_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.max_label = ctk.CTkLabel(self, text="Max Energy (eV):")
        self.max_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.max_entry = ctk.CTkEntry(self)
        self.max_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.ok_button = ctk.CTkButton(self, text="OK", command=self._on_ok)
        self.ok_button.grid(row=2, column=0, padx=10, pady=10)
        self.cancel_button = ctk.CTkButton(self, text="Cancel", command=self._on_cancel)
        self.cancel_button.grid(row=2, column=1, padx=10, pady=10)

        self.result: Optional[tuple[float, float]] = None
        self.wait_window() # Wait until the window is destroyed

    def _on_ok(self):
        try:
            min_e = float(self.min_entry.get())
            max_e = float(self.max_entry.get())
            if min_e >= max_e:
                messagebox.showerror("Invalid Range", "Min energy must be less than Max energy.", parent=self)
                return
            self.result = (min_e, max_e)
            self.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for the energy range.", parent=self)

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def get_range(self):
        return self.result


class ExportFormatDialog(ctk.CTkToplevel):
    """
    A dialog to get the export format from the user using radio buttons.
    """
    def __init__(self, master, title="Export Format", **kwargs):
        super().__init__(master, **kwargs)
        self.title(title)
        self.geometry("450x250")
        self.lift()
        self.attributes("-topmost", True)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)

        self.main_label = ctk.CTkLabel(self, text="Choose the export format for integrated data:")
        self.main_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.format_var = ctk.StringVar(value="power")
        
        self.power_radio = ctk.CTkRadioButton(self, text="Group by Power", variable=self.format_var, value="power")
        self.power_radio.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        self.power_desc = ctk.CTkLabel(self, text="      (Power as index, Temperature as columns)", text_color="gray")
        self.power_desc.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="w")

        self.temp_radio = ctk.CTkRadioButton(self, text="Group by Temperature", variable=self.format_var, value="temperature")
        self.temp_radio.grid(row=3, column=0, padx=20, pady=5, sticky="w")
        self.temp_desc = ctk.CTkLabel(self, text="      (Temperature as index, Power as columns)", text_color="gray")
        self.temp_desc.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="w")

        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.ok_button = ctk.CTkButton(self.button_frame, text="OK", command=self._on_ok)
        self.ok_button.pack(side="left", padx=10)
        
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side="left", padx=10)

        self.result: Optional[str] = None
        self.wait_window()

    def _on_ok(self):
        self.result = self.format_var.get()
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def get_format(self):
        return self.result


class PolarizationSettingsDialog(ctk.CTkToplevel):
    """
    Dialog for configuring polarization string mappings.
    Allows users to define which strings in filenames indicate σ+ and σ- polarization.
    """
    def __init__(self, master, current_config: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Polarization Settings")
        self.geometry("600x400")
        self.lift()
        self.attributes("-topmost", True)
        self.grab_set()  # Make modal

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header
        header_label = ctk.CTkLabel(
            self, 
            text="Configure Polarization String Mappings", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        header_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        info_label = ctk.CTkLabel(
            self,
            text="Enter comma-separated strings that indicate each polarization in your filenames.\n"
                 "These are case-insensitive. Examples: pol1, ROI1, sig+, sigma+, CP_plus",
            text_color="gray"
        )
        info_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Sigma+ configuration
        sigma_plus_frame = ctk.CTkFrame(self)
        sigma_plus_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        sigma_plus_frame.grid_columnconfigure(1, weight=1)

        sigma_plus_label = ctk.CTkLabel(sigma_plus_frame, text="σ+ Strings:")
        sigma_plus_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.sigma_plus_entry = ctk.CTkEntry(sigma_plus_frame, placeholder_text="e.g., pol1, ROI1, sig+")
        self.sigma_plus_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Load current config
        if current_config and 'sigma_plus' in current_config:
            self.sigma_plus_entry.insert(0, ', '.join(current_config['sigma_plus']))

        # Sigma- configuration
        sigma_minus_frame = ctk.CTkFrame(self)
        sigma_minus_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        sigma_minus_frame.grid_columnconfigure(1, weight=1)

        sigma_minus_label = ctk.CTkLabel(sigma_minus_frame, text="σ- Strings:")
        sigma_minus_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.sigma_minus_entry = ctk.CTkEntry(sigma_minus_frame, placeholder_text="e.g., pol2, ROI2, sig-")
        self.sigma_minus_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Load current config
        if current_config and 'sigma_minus' in current_config:
            self.sigma_minus_entry.insert(0, ', '.join(current_config['sigma_minus']))

        # Preview section
        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        preview_label = ctk.CTkLabel(preview_frame, text="Test Parsing:", font=ctk.CTkFont(weight="bold"))
        preview_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.test_entry = ctk.CTkEntry(preview_frame, placeholder_text="Enter a test filename...")
        self.test_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        preview_frame.grid_columnconfigure(0, weight=1)
        
        test_button = ctk.CTkButton(preview_frame, text="Test", command=self._test_parsing, width=100)
        test_button.grid(row=1, column=1, padx=10, pady=5)
        
        self.test_result_label = ctk.CTkLabel(preview_frame, text="", text_color="gray")
        self.test_result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=5, column=0, pady=20)
        
        self.ok_button = ctk.CTkButton(button_frame, text="Apply", command=self._on_ok, width=100)
        self.ok_button.pack(side="left", padx=10)
        
        self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self._on_cancel, width=100)
        self.cancel_button.pack(side="left", padx=10)

        self.result: Optional[dict] = None
        self.wait_window()

    def _parse_string_list(self, text: str) -> list:
        """Parse comma-separated string into list, strip whitespace."""
        if not text or text.strip() == "":
            return []
        return [s.strip() for s in text.split(',') if s.strip()]

    def _test_parsing(self):
        """Test the current configuration against a filename."""
        test_filename = self.test_entry.get().strip()
        if not test_filename:
            self.test_result_label.configure(text="Please enter a test filename")
            return
        
        # Get current values
        sigma_plus_list = self._parse_string_list(self.sigma_plus_entry.get())
        sigma_minus_list = self._parse_string_list(self.sigma_minus_entry.get())
        
        # Test detection
        filename_lower = test_filename.lower()
        detected = None
        
        for indicator in sigma_plus_list:
            if indicator.lower() in filename_lower:
                detected = "σ+"
                break
        
        if not detected:
            for indicator in sigma_minus_list:
                if indicator.lower() in filename_lower:
                    detected = "σ-"
                    break
        
        if detected:
            self.test_result_label.configure(
                text=f"✓ Detected: {detected}",
                text_color="green"
            )
        else:
            self.test_result_label.configure(
                text="✗ No polarization detected",
                text_color="orange"
            )

    def _on_ok(self):
        sigma_plus_list = self._parse_string_list(self.sigma_plus_entry.get())
        sigma_minus_list = self._parse_string_list(self.sigma_minus_entry.get())
        
        if not sigma_plus_list and not sigma_minus_list:
            messagebox.showwarning(
                "Empty Configuration",
                "At least one polarization must have detection strings.",
                parent=self
            )
            return
        
        self.result = {
            'sigma_plus': sigma_plus_list,
            'sigma_minus': sigma_minus_list
        }
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def get_config(self):
        return self.result


class MagneticSweepSettingsDialog(ctk.CTkToplevel):
    """Dialog to configure magnetic sweep defaults for filename parsing."""

    def __init__(self, master, current_config: Optional[dict] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Magnetic Sweep Settings")
        self.geometry("540x640")
        self.lift()
        self.attributes("-topmost", True)
        self.grab_set()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(6, weight=1)

        self.time_range_rows: List[dict] = []

        def add_entry(label_text, row, placeholder="", value=None):
            label = ctk.CTkLabel(self, text=label_text)
            label.grid(row=row, column=0, padx=15, pady=10, sticky="w")
            entry = ctk.CTkEntry(self, placeholder_text=placeholder)
            entry.grid(row=row, column=1, padx=15, pady=10, sticky="ew")
            if value is not None and value != "":
                entry.insert(0, self._format_value(value))
            return entry

        cfg = current_config or {}

        self.min_b_entry = add_entry("Min B-field (T):", 0, "e.g., 0.0", cfg.get("min_bfield_t", ""))
        self.max_b_entry = add_entry("Max B-field (T):", 1, "optional", cfg.get("max_bfield_t", ""))
        self.step_entry = add_entry("Step (T):", 2, "e.g., 0.5", cfg.get("step_t", ""))
        self.temp_entry = add_entry("Temperature (K):", 3, "optional", cfg.get("temperature_k", ""))
        self.power_entry = add_entry("Power (uW):", 4, "optional", cfg.get("power_uw", ""))

        stored_direction = cfg.get("sweep_direction")
        if stored_direction not in ("low_to_high", "high_to_low"):
            stored_direction = "low_to_high"
        direction_frame = ctk.CTkFrame(self)
        direction_frame.grid(row=5, column=0, columnspan=2, padx=15, pady=(5, 5), sticky="ew")
        direction_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(direction_frame, text="Sweep Direction", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 5), sticky="w"
        )
        self.sweep_direction_var = ctk.StringVar(value=stored_direction)
        radio_frame = ctk.CTkFrame(direction_frame, fg_color="transparent")
        radio_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        min_label = self._format_value(cfg.get("min_bfield_t"))
        max_label = self._format_value(cfg.get("max_bfield_t"))
        forward_text = "Low to High"
        reverse_text = "High to Low"
        if min_label and max_label:
            forward_text = f"Low to High ({min_label}T -> {max_label}T)"
            reverse_text = f"High to Low ({max_label}T -> {min_label}T)"
        elif min_label:
            forward_text = f"Low to High (>= {min_label}T)"
        elif max_label:
            reverse_text = f"High to Low (<= {max_label}T)"
        ctk.CTkRadioButton(
            radio_frame,
            text=forward_text,
            value="low_to_high",
            variable=self.sweep_direction_var,
        ).grid(row=0, column=0, padx=(0, 20), pady=2, sticky="w")
        ctk.CTkRadioButton(
            radio_frame,
            text=reverse_text,
            value="high_to_low",
            variable=self.sweep_direction_var,
        ).grid(row=0, column=1, pady=2, sticky="w")

        time_frame = ctk.CTkFrame(self)
        time_frame.grid(row=6, column=0, columnspan=2, padx=15, pady=(5, 5), sticky="nsew")
        time_frame.grid_columnconfigure(0, weight=1)
        time_frame.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(time_frame, text="Acquisition Time Ranges", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 5), sticky="w"
        )
        header_frame = ctk.CTkFrame(time_frame, fg_color="transparent")
        header_frame.grid(row=1, column=0, sticky="ew", padx=5)
        header_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        header_labels = [
            ("B-field From (T)", 0),
            ("B-field To (T)", 1),
            ("Time (s)", 2),
            ("", 3),
        ]
        for text, column in header_labels:
            ctk.CTkLabel(header_frame, text=text).grid(row=0, column=column, padx=5, pady=(0, 2), sticky="w")

        self.time_rows_container = ctk.CTkScrollableFrame(time_frame, height=170, fg_color="transparent")
        self.time_rows_container.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0, 10))

        self.add_range_button = ctk.CTkButton(
            time_frame,
            text="+ Add Range",
            command=lambda: self._add_time_range_row(),
            width=120,
        )
        self.add_range_button.grid(row=3, column=0, padx=5, pady=(0, 10), sticky="w")

        time_ranges = cfg.get("time_ranges")
        populated = False
        if isinstance(time_ranges, list) and time_ranges:
            for item in time_ranges:
                if not isinstance(item, dict):
                    continue
                self._add_time_range_row(item.get("b_min"), item.get("b_max"), item.get("time_s"))
                populated = True
        if not populated and cfg.get("time_s") is not None:
            self._add_time_range_row(cfg.get("min_bfield_t"), cfg.get("max_bfield_t"), cfg.get("time_s"))
            populated = True
        if not populated:
            self._add_time_range_row()

        roi_frame = ctk.CTkFrame(self)
        roi_frame.grid(row=7, column=0, columnspan=2, padx=15, pady=(10, 5), sticky="ew")
        roi_frame.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(roi_frame, text="ROI Mapping", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=4, padx=10, pady=(10, 5), sticky="w"
        )

        ctk.CTkLabel(roi_frame, text="ROI 1:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(roi_frame, text="ROI 2:").grid(row=1, column=2, padx=10, pady=5, sticky="w")

        roi_map = cfg.get("roi_map", {}) if isinstance(cfg.get("roi_map", {}), dict) else {}
        options = ["sigma+", "sigma-", "none"]

        self.roi1_var = ctk.StringVar(value=roi_map.get("1", "sigma+"))
        self.roi2_var = ctk.StringVar(value=roi_map.get("2", "sigma-"))

        self.roi1_menu = ctk.CTkOptionMenu(roi_frame, values=options, variable=self.roi1_var)
        self.roi1_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.roi2_menu = ctk.CTkOptionMenu(roi_frame, values=options, variable=self.roi2_var)
        self.roi2_menu.grid(row=1, column=3, padx=10, pady=5, sticky="ew")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=8, column=0, columnspan=2, pady=20)

        self.ok_button = ctk.CTkButton(button_frame, text="Apply", command=self._on_ok, width=100)
        self.ok_button.pack(side="left", padx=10)
        self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self._on_cancel, width=100)
        self.cancel_button.pack(side="left", padx=10)

        self.result: Optional[dict] = None
        self.wait_window()

    @staticmethod
    def _parse_optional_float(value: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        try:
            return float(value.replace('p', '.'))
        except ValueError:
            return None

    @staticmethod
    def _format_value(value: Optional[float]) -> str:
        """Format optional numeric value for display in entry fields."""
        if value is None or value == "":
            return ""
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    def _add_time_range_row(self, b_min: Optional[float] = None, b_max: Optional[float] = None, time_s: Optional[float] = None):
        """Create a new time range row."""
        row_frame = ctk.CTkFrame(self.time_rows_container, fg_color="transparent")
        row_frame.grid_columnconfigure((0, 1, 2), weight=1)

        min_entry = ctk.CTkEntry(row_frame, width=90, placeholder_text="e.g., 0.0")
        min_entry.grid(row=0, column=0, padx=(0, 5), pady=3, sticky="ew")
        max_entry = ctk.CTkEntry(row_frame, width=90, placeholder_text="optional")
        max_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        time_entry = ctk.CTkEntry(row_frame, width=90, placeholder_text="e.g., 30")
        time_entry.grid(row=0, column=2, padx=5, pady=3, sticky="ew")

        if b_min not in (None, ""):
            min_entry.insert(0, self._format_value(b_min))
        if b_max not in (None, ""):
            max_entry.insert(0, self._format_value(b_max))
        if time_s not in (None, ""):
            time_entry.insert(0, self._format_value(time_s))

        row_data: Dict[str, object] = {
            "frame": row_frame,
            "min_entry": min_entry,
            "max_entry": max_entry,
            "time_entry": time_entry,
        }

        remove_button = ctk.CTkButton(
            row_frame,
            text="Remove",
            width=70,
            command=lambda data=row_data: self._remove_time_range_row(data),
        )
        remove_button.grid(row=0, column=3, padx=(5, 0), pady=3)

        self.time_range_rows.append(row_data)
        self._refresh_time_range_rows()

    def _remove_time_range_row(self, row_data: dict):
        """Remove a time range row from the UI."""
        try:
            self.time_range_rows.remove(row_data)
        except ValueError:
            return
        frame = row_data.get("frame")
        if frame:
            frame.destroy()
        self._refresh_time_range_rows()

    def _refresh_time_range_rows(self):
        """Re-grid time range rows to maintain order."""
        for index, row in enumerate(self.time_range_rows):
            frame = row.get("frame")
            if frame:
                frame.grid_forget()
                frame.grid(row=index, column=0, sticky="ew")

    def _collect_time_ranges(self) -> Optional[List[Dict[str, Optional[float]]]]:
        """Validate and collect time range entries."""
        collected: List[Dict[str, Optional[float]]] = []
        for row in self.time_range_rows:
            min_entry: ctk.CTkEntry = row["min_entry"]
            max_entry: ctk.CTkEntry = row["max_entry"]
            time_entry: ctk.CTkEntry = row["time_entry"]

            min_str = min_entry.get().strip()
            max_str = max_entry.get().strip()
            time_str = time_entry.get().strip()

            if not min_str and not max_str and not time_str:
                continue

            try:
                b_min = float(min_str.replace('p', '.'))
            except ValueError:
                messagebox.showerror("Invalid Range", "Please enter a valid numeric B-field start for each time range.", parent=self)
                min_entry.focus_set()
                return None

            b_max: Optional[float] = None
            if max_str:
                try:
                    b_max = float(max_str.replace('p', '.'))
                except ValueError:
                    messagebox.showerror("Invalid Range", "Please enter a valid numeric B-field end or leave it blank.", parent=self)
                    max_entry.focus_set()
                    return None
                if b_max < b_min - 1e-9:
                    messagebox.showerror("Invalid Range", "B-field end must be greater than or equal to the start.", parent=self)
                    max_entry.focus_set()
                    return None

            if not time_str:
                messagebox.showerror("Invalid Range", "Please enter an acquisition time (s) for each B-field range.", parent=self)
                time_entry.focus_set()
                return None
            try:
                time_val = float(time_str.replace('p', '.'))
                if time_val <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Range", "Acquisition time must be a positive number.", parent=self)
                time_entry.focus_set()
                return None

            collected.append({'b_min': b_min, 'b_max': b_max, 'time_s': time_val})

        collected.sort(key=lambda item: item['b_min'])
        return collected

    def _on_ok(self):
        try:
            min_b = float(self.min_b_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please provide a numeric minimum B-field.", parent=self)
            return
        try:
            step = float(self.step_entry.get())
            if step <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please provide a positive numeric step.", parent=self)
            return

        max_b = self._parse_optional_float(self.max_b_entry.get())
        if max_b is not None and max_b < min_b:
            messagebox.showerror("Invalid Input", "Maximum B-field must be greater than or equal to the minimum.", parent=self)
            return

        temperature = self._parse_optional_float(self.temp_entry.get())
        power = self._parse_optional_float(self.power_entry.get())
        time_ranges = self._collect_time_ranges()
        if time_ranges is None:
            return

        roi_map: Dict[str, str] = {}
        roi1 = self.roi1_var.get()
        roi2 = self.roi2_var.get()
        if roi1 != "none":
            roi_map['1'] = roi1
        if roi2 != "none":
            roi_map['2'] = roi2

        self.result = {
            'min_bfield_t': min_b,
            'max_bfield_t': max_b,
            'step_t': step,
            'temperature_k': temperature,
            'power_uw': power,
            'time_s': None,
            'time_ranges': time_ranges,
            'sweep_direction': self.sweep_direction_var.get(),
            'roi_map': roi_map
        }
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def get_config(self):
        return self.result


# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    # Need a dummy root window to attach the dialog to
    root = ctk.CTk()
    root.withdraw() # Hide the dummy root window

    print("Showing GF Factor Dialog...")
    dialog = GfFactorDialog(text="Enter GF Transmission Factor (e.g., 0.052):", title="GF Factor")
    factor = dialog.get_input()

    if factor is not None:
        print(f"User entered factor: {factor}")
    else:
        print("User cancelled or entered invalid input.")

    root.destroy() # Clean up the dummy window
