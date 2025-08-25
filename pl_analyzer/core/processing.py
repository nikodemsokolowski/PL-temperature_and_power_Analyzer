import logging
import pandas as pd
from typing import Dict, Optional

from .data_handler import DataHandler # To access data and metadata

logger = logging.getLogger(__name__)

# Spectrometer response time correction factors (as provided)
# Using original acquisition time as the key
RESPONSE_CORRECTION_FACTORS = {
    0.1: 2.67411052994443,
    0.2: 1.74082446540644,
    0.3: 1.44263778631468,
    0.5: 1.20406605805442,
}

def normalize_by_acquisition_time(data_handler: DataHandler):
    """
    Divides the counts of all loaded spectra by their acquisition time.

    Modifies the 'counts' column in the data_handler.processed_spectra DataFrames.
    """
    logger.info("Applying acquisition time normalization...")
    processed_count = 0
    skipped_count = 0

    for file_id, metadata in data_handler.metadata.items():
        acq_time = metadata.get('time_s')
        spectrum_df = data_handler.get_processed_spectrum(file_id)

        if spectrum_df is None or spectrum_df.empty:
            logger.warning(f"No processed spectrum found for file_id {file_id}. Skipping time normalization.")
            skipped_count += 1
            continue

        if acq_time is None or not isinstance(acq_time, (int, float)) or acq_time <= 0:
            logger.warning(f"Invalid or missing acquisition time ({acq_time}) for file_id {file_id}. Skipping time normalization.")
            skipped_count += 1
            continue

        try:
            # Perform division - modifies the DataFrame in place
            spectrum_df['counts'] = spectrum_df['counts'] / acq_time
            logger.debug(f"Normalized counts for file_id {file_id} by time {acq_time}s.")
            processed_count += 1
        except Exception as e:
            logger.error(f"Error during time normalization for file_id {file_id}: {e}", exc_info=True)
            skipped_count += 1

    logger.info(f"Acquisition time normalization complete. Processed: {processed_count}, Skipped: {skipped_count}.")

def rescale_by_grey_filter(data_handler: DataHandler, gf_transmission_factor: float):
    """
    Divides the counts of spectra marked with 'gf_present' by the provided factor.

    Modifies the 'counts' column in the data_handler.processed_spectra DataFrames.

    Args:
        data_handler: The DataHandler instance containing the data.
        gf_transmission_factor: The user-provided transmission factor (e.g., 0.052).
    """
    logger.info(f"Applying Grey Filter rescaling with factor: {gf_transmission_factor}...")
    if not isinstance(gf_transmission_factor, (int, float)) or gf_transmission_factor <= 0:
        logger.error(f"Invalid Grey Filter transmission factor provided: {gf_transmission_factor}. Aborting rescaling.")
        raise ValueError("Grey Filter transmission factor must be a positive number.")

    processed_count = 0
    skipped_count = 0

    for file_id, metadata in data_handler.metadata.items():
        is_gf_present = metadata.get('gf_present', False)
        spectrum_df = data_handler.get_processed_spectrum(file_id)

        if not is_gf_present:
            # logger.debug(f"No GF present for file_id {file_id}. Skipping rescaling.")
            continue # Only process files marked with GF

        if spectrum_df is None or spectrum_df.empty:
            logger.warning(f"No processed spectrum found for file_id {file_id} (GF present). Skipping rescaling.")
            skipped_count += 1
            continue

        try:
            # Perform division - modifies the DataFrame in place
            spectrum_df['counts'] = spectrum_df['counts'] / gf_transmission_factor
            logger.debug(f"Rescaled counts for file_id {file_id} (GF present) by factor {gf_transmission_factor}.")
            processed_count += 1
        except Exception as e:
            logger.error(f"Error during GF rescaling for file_id {file_id}: {e}", exc_info=True)
            skipped_count += 1

    logger.info(f"Grey Filter rescaling complete. Processed: {processed_count}, Skipped: {skipped_count}.")


def correct_spectrometer_response(data_handler: DataHandler):
    """
    Divides counts by spectrometer response factors based on original acquisition time.

    Uses the factors defined in RESPONSE_CORRECTION_FACTORS.
    Warns and skips if the acquisition time is not in the predefined keys.
    Modifies the 'counts' column in the data_handler.processed_spectra DataFrames.
    """
    logger.info("Applying spectrometer response time correction...")
    processed_count = 0
    skipped_count = 0

    for file_id, metadata in data_handler.metadata.items():
        acq_time = metadata.get('time_s')
        spectrum_df = data_handler.get_processed_spectrum(file_id)

        if spectrum_df is None or spectrum_df.empty:
            logger.warning(f"No processed spectrum found for file_id {file_id}. Skipping response correction.")
            skipped_count += 1
            continue

        if acq_time is None or not isinstance(acq_time, (int, float)):
            logger.warning(f"Invalid or missing acquisition time ({acq_time}) for file_id {file_id}. Skipping response correction.")
            skipped_count += 1
            continue

        # Find the correction factor
        correction_factor = RESPONSE_CORRECTION_FACTORS.get(acq_time)

        if correction_factor is None:
            logger.warning(f"No defined spectrometer response correction factor for acquisition time {acq_time}s (file_id {file_id}). Skipping correction for this file.")
            skipped_count += 1
            continue

        if correction_factor <= 0:
            logger.warning(f"Invalid correction factor ({correction_factor}) found for time {acq_time}s. Skipping correction for file_id {file_id}.")
            skipped_count += 1
            continue

        try:
            # Perform division - modifies the DataFrame in place
            spectrum_df['counts'] = spectrum_df['counts'] / correction_factor
            logger.debug(f"Corrected counts for file_id {file_id} using factor {correction_factor} for time {acq_time}s.")
            processed_count += 1
        except Exception as e:
            logger.error(f"Error during spectrometer response correction for file_id {file_id}: {e}", exc_info=True)
            skipped_count += 1

    logger.info(f"Spectrometer response correction complete. Processed: {processed_count}, Skipped: {skipped_count}.")

# Example usage (requires a DataHandler instance populated with data)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Setup Dummy Data ---
    print("--- Setting up dummy data ---")
    handler = DataHandler()

    # Create dummy files (similar to DataHandler test)
    os.makedirs('temp_process_test', exist_ok=True)
    file1 = 'temp_process_test/Data_10K_100uW_0.1s.csv'
    file2 = 'temp_process_test/Data_10K_50uW_0.5s_GF1.csv'
    file3 = 'temp_process_test/Data_20K_100uW_1.0s.csv' # Time not in correction factors
    file4 = 'temp_process_test/Data_20K_50uW_0.2s_GF2.csv'

    pd.DataFrame({'wl': [600, 601], 'cts': [1000, 1200]}).to_csv(file1, sep='\t', header=False, index=False)
    pd.DataFrame({'wl': [700, 701], 'cts': [2000, 2400]}).to_csv(file2, sep='\t', header=False, index=False)
    pd.DataFrame({'wl': [650, 651], 'cts': [3000, 3600]}).to_csv(file3, sep='\t', header=False, index=False)
    pd.DataFrame({'wl': [750, 751], 'cts': [4000, 4800]}).to_csv(file4, sep='\t', header=False, index=False)

    handler.load_files([file1, file2, file3, file4])
    print("\nInitial Processed Spectra:")
    for fid in handler.get_all_file_ids():
        print(f"\nFile ID: {fid}")
        print(handler.get_metadata().loc[handler.get_metadata()['file_id'] == fid])
        print(handler.get_processed_spectrum(fid))

    # --- Test Time Normalization ---
    print("\n--- Testing Time Normalization ---")
    normalize_by_acquisition_time(handler)
    print("\nProcessed Spectra after Time Normalization:")
    for fid in handler.get_all_file_ids():
        print(f"\nFile ID: {fid}")
        print(handler.get_processed_spectrum(fid))

    # --- Test GF Rescaling ---
    print("\n--- Testing GF Rescaling (Factor=10) ---")
    try:
        rescale_by_grey_filter(handler, 10.0)
        print("\nProcessed Spectra after GF Rescaling:")
        for fid in handler.get_all_file_ids():
            print(f"\nFile ID: {fid}")
            print(handler.get_processed_spectrum(fid))
    except ValueError as e:
        print(f"GF Rescaling Error: {e}")

    # --- Test Spectrometer Correction ---
    print("\n--- Testing Spectrometer Correction ---")
    correct_spectrometer_response(handler)
    print("\nProcessed Spectra after Spectrometer Correction:")
    for fid in handler.get_all_file_ids():
        print(f"\nFile ID: {fid}")
        print(handler.get_processed_spectrum(fid))


    # Clean up
    import shutil
    shutil.rmtree('temp_process_test')
    print("\n--- Test complete ---")
