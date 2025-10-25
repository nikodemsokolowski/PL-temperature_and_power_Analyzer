# run_app.py

import customtkinter as ctk
import logging
import sys

# --- Use ABSOLUTE import for MainWindow ---
from pl_analyzer.gui.main_window import MainWindow
from pl_analyzer.utils import config
from pl_analyzer.gui.actions import file_actions

def on_closing(app_instance, main_window_instance):
    """Callback for when the application window is closed."""
    datasets = main_window_instance.data_handler.get_all_datasets_filepaths()
    config.save_last_datasets(datasets)
    app_instance.destroy()

if __name__ == '__main__':
    # Basic logging config (copied from pl_analyzer/main.py)
    # TODO: Replace with setup_logging() if you implement it
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("run_app") # Use a specific logger name if desired
    logger.info("Starting PL Analyzer Application via run_app.py...")

    # Setup CustomTkinter appearance (copied from pl_analyzer/main.py)
    ctk.set_appearance_mode("System") # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue") # Themes: "blue" (default), "green", "dark-blue"

    # Apply UI scaling before creating window
    ui_scale = config.load_ui_scale()
    ctk.set_widget_scaling(ui_scale)
    ctk.set_window_scaling(ui_scale)
    logger.info(f"Applied UI scaling: {ui_scale} ({int(ui_scale * 100)}%)")

    # Create the root window (copied from pl_analyzer/main.py)
    app = ctk.CTk()
    app.title("PL Analyzer")
    # Load resolution from config
    resolution = config.load_window_resolution()
    app.geometry(resolution)

    # Instantiate the main window (copied from pl_analyzer/main.py)
    # Uses the absolute import from above
    main_window = MainWindow(master=app)
    main_window.pack(fill="both", expand=True)

    # Load last used datasets
    last_datasets = config.load_last_datasets()
    if last_datasets:
        for name, filepaths in last_datasets.items():
            if filepaths:
                dataset_name = main_window.data_handler.create_new_dataset(filepaths)
                if dataset_name:
                    main_window.left_panel.add_dataset_tab(dataset_name)
        
        # After loading all, update the UI for the last active or first dataset
        if main_window.data_handler.get_dataset_names():
            # Make the first loaded dataset active
            active_name = main_window.data_handler.get_dataset_names()[0]
            main_window.data_handler.set_active_dataset(active_name)
            main_window.left_panel.tab_view.set(active_name)
            main_window.update_file_table()
            main_window._update_processing_button_states()

    # Set the closing protocol
    app.protocol("WM_DELETE_WINDOW", lambda: on_closing(app, main_window))

    # Run the application loop (copied from pl_analyzer/main.py)
    try:
        app.mainloop()
    except Exception as e:
        logger.critical(f"Unhandled exception occurred: {e}", exc_info=True)
        # Optionally show an error dialog to the user here too
        sys.exit(1)
    finally:
        logger.info("PL Analyzer Application closed.")
