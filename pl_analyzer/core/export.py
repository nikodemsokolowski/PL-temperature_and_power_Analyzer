import logging
import os
import pandas as pd
import numpy as np
from tkinter import filedialog
from typing import List, Tuple, Optional

from .data_handler import DataHandler
from .analysis import integrate_spectrum # Use the existing integration function

logger = logging.getLogger(__name__)

# Corrected function signature to include 'temperature'
def export_integrated_data(data_handler: DataHandler, temperature: float, integration_range: Tuple[float, float], save_path: str):
    """
    Calculates integrated intensity for a specific power series (defined by temperature)
    and exports the results (Power vs Integrated Intensity) to a tab-separated file.

    Args:
        data_handler: The DataHandler instance containing the data.
        temperature: The target temperature for the power series.
        integration_range: Tuple (min_energy_ev, max_energy_ev) for integration.
        save_path: The full path where the file should be saved.
    """
    logger.info(f"Starting export of integrated data for T={temperature}K to {save_path} with range {integration_range} eV.")
    metadata_df = data_handler.get_metadata()
    if metadata_df is None or metadata_df.empty:
        logger.warning("No metadata found in DataHandler.")
        raise ValueError("No data loaded to export.")

    # Get all files for this temperature
    series_df = metadata_df[metadata_df['temperature_k'] == temperature].sort_values(by='power_uw')
    if series_df.empty:
        logger.warning(f"No files found for T={temperature}K.")
        raise ValueError(f"No files found for the specified temperature: {temperature} K.")

    results = []
    # Integrate each spectrum in the power series
    for _, row in series_df.iterrows():
        file_id = row['file_id']
        power_uw = row['power_uw']
        spectrum_df = data_handler.get_processed_spectrum(file_id)

        if spectrum_df is None or spectrum_df.empty or pd.isna(power_uw):
            logger.warning(f"Skipping file {file_id} (T={temperature}K, P={power_uw}uW) due to missing data/power.")
            continue

        integral = integrate_spectrum(
            spectrum_df['energy_ev'].values,
            spectrum_df['counts'].values,
            integration_range
        )

        if integral is not None:
            results.append({'Power (uW)': power_uw, 'Integrated Intensity': integral})
            logger.debug(f"  Integrated T={temperature}K, P={power_uw}uW -> {integral:.4g}")
        else:
            logger.warning(f"  Integration failed for T={temperature}K, P={power_uw}uW.")

    if not results:
        logger.warning(f"No integrated data was successfully calculated for T={temperature}K.")
        raise ValueError(f"Failed to calculate integrated intensity for T={temperature}K series.")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    try:
        # Export with tab separator
        results_df.to_csv(save_path, sep='\t', index=False, float_format='%.6g')
        logger.info(f"Successfully exported integrated data for T={temperature}K ({len(results_df)} points) to {save_path}")
    except Exception as e:
        logger.error(f"Failed to write integrated data CSV to {save_path}: {e}", exc_info=True)
        raise IOError(f"Failed to save file: {e}")


def export_processed_spectra(data_handler: DataHandler, file_ids: List[str], save_path: str):
    """
    Exports the processed spectra (Energy vs Counts) for the given file IDs
    into a single tab-separated file.

    Args:
        data_handler: The DataHandler instance containing the data.
        file_ids: A list of file_ids to export.
        save_path: The full path where the CSV file should be saved.
    """
    logger.info(f"Starting export of {len(file_ids)} processed spectra to {save_path}.")
    if not file_ids:
        raise ValueError("No file IDs provided for export.")

    all_spectra_dict = {}
    energy_basis = None # Use the energy axis from the first valid spectrum

    # Prepare data for DataFrame creation
    for file_id in file_ids:
        spectrum_df = data_handler.get_processed_spectrum(file_id)
        metadata = data_handler.metadata.get(file_id, {})

        if spectrum_df is None or spectrum_df.empty:
            logger.warning(f"No processed spectrum data found for file_id {file_id}. Skipping export.")
            continue

        # Use simplified T/P for column header
        temp = metadata.get('temperature_k', 'NA')
        power = metadata.get('power_uw', 'NA')
        # Format numbers nicely in the header
        temp_str = f"{temp:.1f}" if isinstance(temp, (int, float)) else str(temp)
        power_str = f"{power:.3g}" if isinstance(power, (int, float)) else str(power)
        col_name = f"T{temp_str}K_P{power_str}uW"

        # Ensure consistent energy axis - interpolate if necessary (simple approach: use first spectrum's axis)
        current_energy = spectrum_df['energy_ev'].values
        current_counts = spectrum_df['counts'].values

        if energy_basis is None:
            energy_basis = current_energy
            all_spectra_dict['Energy (eV)'] = energy_basis
            all_spectra_dict[col_name] = current_counts
            logger.debug(f"Set energy basis from file {file_id} (length {len(energy_basis)}).")
        else:
            # Simple check: if energy axes differ significantly, warn or interpolate
            if not np.allclose(energy_basis, current_energy, atol=1e-6): # Tolerance for floating point comparison
                logger.warning(f"Energy axis for file {file_id} differs from basis. Interpolating.")
                try:
                    # Interpolate counts onto the energy_basis
                    # Ensure energy arrays are sorted for interpolation
                    sort_idx_current = np.argsort(current_energy)
                    interp_counts = np.interp(energy_basis, current_energy[sort_idx_current], current_counts[sort_idx_current], left=0, right=0) # Fill outside with 0
                    all_spectra_dict[col_name] = interp_counts
                except Exception as e:
                    logger.error(f"Interpolation failed for file {file_id}: {e}. Skipping.", exc_info=True)
                    continue
            else:
                # Energy axes match (within tolerance)
                all_spectra_dict[col_name] = current_counts

    if len(all_spectra_dict) <= 1: # Only contains 'Energy (eV)' or failed for all
        logger.error("Failed to prepare any spectra for export.")
        raise ValueError("No valid spectra found for export.")

    # Create DataFrame
    try:
        export_df = pd.DataFrame(all_spectra_dict)
        # Ensure Energy is the first column
        cols = ['Energy (eV)'] + [col for col in export_df.columns if col != 'Energy (eV)']
        export_df = export_df[cols]
    except Exception as e:
        logger.error(f"Failed to create DataFrame for export: {e}", exc_info=True)
        raise ValueError(f"Error creating export DataFrame: {e}")

    # Save to file with tab separator
    try:
        export_df.to_csv(save_path, sep='\t', index=False, float_format='%.6g')
        logger.info(f"Successfully exported {len(export_df.columns) - 1} spectra to {save_path}")
    except Exception as e:
        logger.error(f"Failed to write spectra CSV to {save_path}: {e}", exc_info=True)
        raise IOError(f"Failed to save file: {e}")


# Example Usage (requires a populated DataHandler)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print("--- Testing Export Functions ---")

    # --- Setup Dummy Data ---
    handler = DataHandler()
    os.makedirs('temp_export_test', exist_ok=True)
    file1 = 'temp_export_test/Data_10K_100uW_0.1s.csv'
    file2 = 'temp_export_test/Data_10K_50uW_0.5s_GF1.csv'
    file3 = 'temp_export_test/Data_20K_100uW_1.0s.csv'
    file4 = 'temp_export_test/Data_20K_50uW_0.2s_GF2.csv'
    # Slightly different energy axis for file 4
    pd.DataFrame({'wl': [600, 601], 'cts': [1000, 1200]}).to_csv(file1, sep='\t', header=False, index=False)
    pd.DataFrame({'wl': [600, 601], 'cts': [2000, 2400]}).to_csv(file2, sep='\t', header=False, index=False)
    pd.DataFrame({'wl': [600, 601], 'cts': [3000, 3600]}).to_csv(file3, sep='\t', header=False, index=False)
    pd.DataFrame({'wl': [600.1, 601.1], 'cts': [4000, 4800]}).to_csv(file4, sep='\t', header=False, index=False)
    handler.load_files([file1, file2, file3, file4])

    # --- Test Export Processed Spectra ---
    save_path_spectra = 'temp_export_test/processed_spectra_export.csv'
    print(f"\nAttempting to export processed spectra to: {save_path_spectra}")
    try:
        export_processed_spectra(handler, handler.get_all_file_ids(), save_path_spectra)
        print(f"  -> Check file: {os.path.abspath(save_path_spectra)}")
    except Exception as e:
        print(f"  -> Export failed: {e}")

    # --- Test Export Integrated Data (for T=10K) ---
    save_path_integrated = 'temp_export_test/integrated_data_export_T10K.txt' # Use .txt for tab-separated
    integration_range_test = (2.06, 2.07) # Corresponds roughly to 600nm range
    target_temp_test = 10.0
    print(f"\nAttempting to export integrated data for T={target_temp_test}K to: {save_path_integrated}")
    try:
        export_integrated_data(handler, target_temp_test, integration_range_test, save_path_integrated)
        print(f"  -> Check file: {os.path.abspath(save_path_integrated)}")
    except Exception as e:
        print(f"  -> Export failed: {e}")

    # Clean up
    import shutil
    # shutil.rmtree('temp_export_test') # Keep files for inspection
    print("\n--- Export Test Complete (check temp_export_test folder) ---")
