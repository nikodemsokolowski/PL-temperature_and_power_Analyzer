# run_app.py

import customtkinter as ctk
import logging
import sys

# --- Use ABSOLUTE import for MainWindow ---
from pl_analyzer.gui.main_window import MainWindow
if __name__ == '__main__':
    # Basic logging config (copied from pl_analyzer/main.py)
    # TODO: Replace with setup_logging() if you implement it
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("run_app") # Use a specific logger name if desired
    logger.info("Starting PL Analyzer Application via run_app.py...")

    # Setup CustomTkinter appearance (copied from pl_analyzer/main.py)
    ctk.set_appearance_mode("System") # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue") # Themes: "blue" (default), "green", "dark-blue"

    # Create the root window (copied from pl_analyzer/main.py)
    app = ctk.CTk()
    app.title("PL Analyzer")
    app.geometry("1200x800") # Initial size

    # Instantiate the main window (copied from pl_analyzer/main.py)
    # Uses the absolute import from above
    main_window = MainWindow(master=app)
    main_window.pack(fill="both", expand=True)

    # Run the application loop (copied from pl_analyzer/main.py)
    try:
        app.mainloop()
    except Exception as e:
        logger.critical(f"Unhandled exception occurred: {e}", exc_info=True)
        # Optionally show an error dialog to the user here too
        sys.exit(1)
    finally:
        logger.info("PL Analyzer Application closed.")