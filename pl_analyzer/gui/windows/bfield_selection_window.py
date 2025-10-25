import logging
from typing import Callable, Iterable, List, Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)


class BFieldSelectionWindow(ctk.CTkToplevel):
    """Popup window for selecting B-field values to include in stacked plots."""

    def __init__(self, master, on_selection_change: Callable[[List[float]], None], **kwargs):
        super().__init__(master, **kwargs)
        self.title("Select B-Field Values")
        self.geometry("360x460")
        self.minsize(320, 360)
        self.resizable(True, True)

        self._on_selection_change = on_selection_change
        self._values: List[float] = []
        self._selected: List[float] = []
        self._checkbox_vars: dict[float, ctk.BooleanVar] = {}

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Destroy>", self._on_destroy)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        info_label = ctk.CTkLabel(
            self,
            text="Toggle the B-field values to include in the normalized stacked plot.",
            wraplength=320,
            anchor="w",
            justify="left"
        )
        info_label.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        self.checkbox_frame = ctk.CTkScrollableFrame(self)
        self.checkbox_frame.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.checkbox_frame.grid_columnconfigure((0, 1, 2), weight=1)

        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.grid(row=2, column=0, padx=12, pady=(6, 4), sticky="ew")
        action_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.select_all_button = ctk.CTkButton(
            action_frame,
            text="Select All",
            command=self._select_all
        )
        self.select_all_button.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.deselect_all_button = ctk.CTkButton(
            action_frame,
            text="Deselect All",
            command=self._deselect_all
        )
        self.deselect_all_button.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.invert_button = ctk.CTkButton(
            action_frame,
            text="Invert",
            command=self._invert_selection
        )
        self.invert_button.grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        self.apply_button = ctk.CTkButton(
            action_frame,
            text="Apply",
            command=self._apply_selection
        )
        self.apply_button.grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        self.summary_label = ctk.CTkLabel(self, text="No B-field values available.", anchor="w")
        self.summary_label.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="ew")

    def set_values(self, values: Iterable[float], selected: Optional[Iterable[float]] = None) -> None:
        """Populate the checkbox list with the provided B-field values."""
        try:
            new_values = sorted({float(val) for val in values})
        except Exception as exc:
            logger.error(f"Failed to parse B-field values for selection window: {exc}", exc_info=True)
            new_values = []

        if selected is None:
            selected_values = new_values.copy()
        else:
            try:
                selected_values = sorted({float(val) for val in selected})
            except Exception:
                selected_values = new_values.copy()

        rebuild = len(new_values) != len(self._values) or any(
            abs(a - b) > 1e-6 for a, b in zip(new_values, self._values)
        )

        self._values = new_values
        self._selected = selected_values

        if rebuild:
            for widget in self.checkbox_frame.winfo_children():
                widget.destroy()
            self._checkbox_vars.clear()
            for idx, value in enumerate(self._values):
                var = ctk.BooleanVar(value=False)
                checkbox = ctk.CTkCheckBox(
                    self.checkbox_frame,
                    text=f"{value:.2f} T",
                    variable=var,
                    command=lambda v=value: self._toggle_value(v)
                )
                row = idx // 3
                col = idx % 3
                checkbox.grid(row=row, column=col, padx=6, pady=4, sticky="w")
                self._checkbox_vars[value] = var

        self._apply_selection_to_vars()
        self._update_summary()

    def _apply_selection_to_vars(self) -> None:
        selected_set = {round(val, 6) for val in self._selected}
        for value, var in self._checkbox_vars.items():
            var.set(round(value, 6) in selected_set)

    def _toggle_value(self, value: float) -> None:
        rounded = round(value, 6)
        var = self._checkbox_vars.get(value)
        state = bool(var.get()) if var is not None else False
        if state:
            if rounded not in {round(v, 6) for v in self._selected}:
                self._selected.append(value)
        else:
            self._selected = [v for v in self._selected if abs(v - rounded) > 1e-6]
        self._selected.sort()
        self._update_summary()
        self._emit_selection()

    def _select_all(self) -> None:
        self._selected = self._values.copy()
        for var in self._checkbox_vars.values():
            var.set(True)
        self._update_summary()
        self._emit_selection()

    def _deselect_all(self) -> None:
        self._selected = []
        for var in self._checkbox_vars.values():
            var.set(False)
        self._update_summary()
        self._emit_selection()

    def _invert_selection(self) -> None:
        selected_set = {round(val, 6) for val in self._selected}
        inverted = []
        for value in self._values:
            include = round(value, 6) not in selected_set
            var = self._checkbox_vars.get(value)
            if var is not None:
                var.set(include)
            if include:
                inverted.append(value)
        self._selected = inverted
        self._update_summary()
        self._emit_selection()

    def _update_summary(self) -> None:
        if not self._values:
            summary = "No B-field values available."
        elif not self._selected:
            summary = "No B-fields selected."
        elif len(self._selected) == len(self._values):
            summary = f"All B-fields selected ({len(self._values)})"
        else:
            preview = ", ".join(f"{val:.2f}" for val in self._selected[:5])
            if len(self._selected) > 5:
                preview += ", ..."
            summary = f"Selected {len(self._selected)} of {len(self._values)}: {preview}"
        self.summary_label.configure(text=summary)

    def _emit_selection(self) -> None:
        if callable(self._on_selection_change):
            try:
                self._on_selection_change(self._selected.copy())
            except Exception as exc:
                logger.error(f"B-field selection callback failed: {exc}", exc_info=True)

    def _apply_selection(self) -> None:
        self._emit_selection()

    def _handle_close(self) -> None:
        self.destroy()

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        master = getattr(self, "master", None)
        if master is not None and hasattr(master, "bfield_selection_window"):
            master.bfield_selection_window = None
