import logging
from tkinter import messagebox

from pl_analyzer.core import processing
from pl_analyzer.gui.dialogs import GfFactorDialog

logger = logging.getLogger(__name__)

def normalize_time_action(app):
    """Callback for Normalize by Time button."""
    logger.info("Normalize by Time button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return
    if active_dataset.time_normalized:
        messagebox.showinfo("Already Applied", "Time normalization has already been applied.")
        return

    try:
        processing.normalize_by_acquisition_time(active_dataset)
        active_dataset.time_normalized = True
        app._update_processing_button_states()
        messagebox.showinfo("Processing Complete", "Time normalization applied.")
    except Exception as e:
        logger.error(f"Error during time normalization: {e}", exc_info=True)
        messagebox.showerror("Processing Error", f"An error occurred: {e}")

def rescale_gf_action(app):
    """Callback for Rescale by GF Factor button."""
    logger.info("Rescale by GF Factor button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return
    if active_dataset.gf_rescaled:
        messagebox.showinfo("Already Applied", "GF rescaling has already been applied.")
        return

    if not any(m.get('gf_present', False) for m in active_dataset.metadata.values()):
        messagebox.showinfo("No GF Data", "No files with Grey Filter found in this dataset.")
        return

    dialog = GfFactorDialog(text="Enter GF Transmission Factor:", title="GF Factor")
    factor = dialog.get_input()

    if factor is not None:
        try:
            processing.rescale_by_grey_filter(active_dataset, factor)
            active_dataset.gf_rescaled = True
            app._update_processing_button_states()
            messagebox.showinfo("Processing Complete", f"GF rescaling applied with factor {factor}.")
        except Exception as e:
            logger.error(f"Error during GF rescaling: {e}", exc_info=True)
            messagebox.showerror("Processing Error", f"An error occurred: {e}")

def correct_response_action(app):
    """Callback for Correct Spectrometer Response button."""
    logger.info("Correct Spectrometer Response button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return
    if active_dataset.response_corrected:
        messagebox.showinfo("Already Applied", "Response correction has already been applied.")
        return

    try:
        processing.correct_spectrometer_response(active_dataset)
        active_dataset.response_corrected = True
        app._update_processing_button_states()
        messagebox.showinfo("Processing Complete", "Spectrometer response correction applied.")
    except Exception as e:
        logger.error(f"Error during response correction: {e}", exc_info=True)
        messagebox.showerror("Processing Error", f"An error occurred: {e}")

def subtract_baseline_action(app):
    """Callback for Subtract Baseline (min value) button."""
    logger.info("Subtract Baseline button clicked.")
    active_dataset = app.data_handler.get_active_dataset()
    if not active_dataset:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return
    if active_dataset.background_subtracted:
        messagebox.showinfo("Already Applied", "Baseline subtraction has already been applied.")
        return

    try:
        processing.subtract_baseline_from_dataset(active_dataset)
        active_dataset.background_subtracted = True
        app._update_processing_button_states()
        messagebox.showinfo("Processing Complete", "Baseline (minimum value) subtracted from all spectra.")
    except Exception as e:
        logger.error(f"Error during baseline subtraction: {e}", exc_info=True)
        messagebox.showerror("Processing Error", f"An error occurred: {e}")