import customtkinter as ctk
import logging
from typing import Optional

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
