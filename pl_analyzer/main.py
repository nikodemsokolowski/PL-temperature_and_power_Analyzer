import customtkinter as ctk
import logging
import sys

from .gui.main_window import MainWindow
from .utils.config import load_polarization_config, load_magnetic_sweep_config
from .core.file_parser import set_polarization_config, set_magnetic_sweep_config

def main():
    """Main function to run the PL Analyzer application."""
    # TODO: Uncomment and implement logging setup later
    # setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting PL Analyzer Application...")

    # Load and apply configuration defaults
    try:
        pol_config = load_polarization_config()
        set_polarization_config(pol_config['sigma_plus'], pol_config['sigma_minus'])
        logger.info(f"Loaded polarization configuration: σ+={pol_config['sigma_plus']}, σ-={pol_config['sigma_minus']}")
    except Exception as e:
        logger.warning(f"Could not load polarization config, using defaults: {e}")

    try:
        sweep_config = load_magnetic_sweep_config()
        set_magnetic_sweep_config(sweep_config)
        logger.info("Loaded magnetic sweep configuration.")
    except Exception as e:
        logger.warning(f"Could not load magnetic sweep config, using defaults: {e}")

    # Setup CustomTkinter appearance
    ctk.set_appearance_mode("System") # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue") # Themes: "blue" (default), "green", "dark-blue"

    app = ctk.CTk()
    app.title("PL Analyzer")
    app.geometry("1200x800") # Initial size

    # Instantiate the main window
    main_window = MainWindow(master=app)
    main_window.pack(fill="both", expand=True)

    # Placeholder removed

    try:
        app.mainloop()
    except Exception as e:
        logger.critical(f"Unhandled exception occurred: {e}", exc_info=True)
        # Optionally show an error dialog to the user here
        sys.exit(1)
    finally:
        logger.info("PL Analyzer Application closed.")

if __name__ == "__main__":
    # Basic logging config for now, will be replaced by setup_logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()

