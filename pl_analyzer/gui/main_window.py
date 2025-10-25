import customtkinter as ctk
import logging
from typing import Optional

from ..core.data_handler import DataHandler
from .panels.left_panel import LeftPanel
from .panels.right_panel import RightPanel
from .analysis_view import AnalysisView
from .temp_analysis_view import TempAnalysisView
from .bfield_analysis_view import BFieldAnalysisView

logger = logging.getLogger(__name__)

class MainWindow(ctk.CTkFrame):
    """
    The main window frame for the PL Analyzer application.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.data_handler = DataHandler()
        self._last_plot_data = None
        self.analysis_window: Optional[AnalysisView] = None
        self.temp_analysis_window: Optional[TempAnalysisView] = None
        self.bfield_analysis_window: Optional[BFieldAnalysisView] = None

        # --- Create main frames ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = LeftPanel(self, width=300, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")

        # --- Initialize UI variables ---
        self.log_scale_var = ctk.StringVar(value="0")
        self.stack_var = ctk.StringVar(value="0")
        self.bfield_stack_var = ctk.StringVar(value="0")
        self.normalize_var = ctk.StringVar(value="0")
        self.equalize_var = ctk.StringVar(value="0")
        self.intensity_map_var = ctk.StringVar(value="0")
        self.show_grid_var = ctk.StringVar(value="1")
        self.bfield_step_var = ctk.StringVar(value="All")
        self._bfield_stack_selected = None
        self._bfield_stack_values = []
        self.bfield_selection_window = None
        
        # --- Initialize Intensity Map UI variables ---
        self.im_normalize_var = ctk.StringVar(value="0")
        self.im_log_c_var = ctk.StringVar(value="0")
        self.im_log_y_var = ctk.StringVar(value="0")
        self.im_colormap_var = ctk.StringVar(value="viridis")


        self.right_panel = RightPanel(self, corner_radius=0)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    def clear_plot_action(self):
        self.right_panel.plot_canvas.clear_plot()

    def _initialize_processing_state(self):
        """Initializes the processing state for a new dataset."""
        self.right_panel._initialize_processing_state()

    def _update_processing_button_states(self):
        """Enables/disables processing buttons based on the active dataset's state."""
        active_dataset = self.data_handler.get_active_dataset()
        if active_dataset:
            self.left_panel.normalize_time_button.configure(state="disabled" if active_dataset.time_normalized else "normal")
            self.left_panel.rescale_gf_button.configure(state="disabled" if active_dataset.gf_rescaled else "normal")
            self.left_panel.correct_response_button.configure(state="disabled" if active_dataset.response_corrected else "normal")
            self.left_panel.subtract_baseline_button.configure(state="disabled" if active_dataset.background_subtracted else "normal")
        else:
            self.left_panel.normalize_time_button.configure(state="disabled")
            self.left_panel.rescale_gf_button.configure(state="disabled")
            self.left_panel.correct_response_button.configure(state="disabled")
            self.left_panel.subtract_baseline_button.configure(state="disabled")

    def update_file_table(self):
        """Updates the file table for the active dataset."""
        logger.debug("Updating file table for the active dataset...")
        active_table = self.left_panel.get_active_file_table()
        if active_table:
            active_dataset = self.data_handler.get_active_dataset()
            if active_dataset:
                metadata = active_dataset.get_metadata()
                active_table.update_data(metadata)
            else:
                active_table.update_data(None) # Clear table if no active dataset
        else:
            logger.debug("No active file table to update.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running main_window.py directly for testing...")

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Main Window Test")
    root.geometry("1000x700")

    main_frame = MainWindow(master=root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    root.mainloop()
