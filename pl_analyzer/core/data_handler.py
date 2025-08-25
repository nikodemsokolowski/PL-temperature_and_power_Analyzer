import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from tkinter import filedialog

from .file_parser import parse_filename

logger = logging.getLogger(__name__)

# Constants
HC_EV_NM = 1239.84198 # Planck's constant * speed of light (eV * nm)

class DataHandler:
    """
    Manages loading, storing, and providing access to PL spectral data.
    """
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.raw_spectra: Dict[str, pd.DataFrame] = {} # Store original Wavelength, Counts
        self.processed_spectra: Dict[str, pd.DataFrame] = {} # Store Energy, Processed Counts
        self.metadata: Dict[str, Dict[str, Any]] = {} # Store parameters per file_id
        self._file_id_counter = 0

    def _generate_file_id(self) -> str:
        """Generates a unique internal ID for each loaded file."""
        self._file_id_counter += 1
        return f"file_{self._file_id_counter}"

    def load_files(self, filepaths: Optional[List[str]] = None) -> int:
        """
        Loads PL data from specified .csv files or opens a dialog if None.

        Args:
            filepaths: A list of paths to the .csv files. If None, a file
                    dialog will be opened to select files.

        Returns:
            The number of files successfully loaded.
        """
        if filepaths is None:
            filepaths = filedialog.askopenfilenames(
                title="Select PL Data Files",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("Data files", "*.dat"), ("All files", "*.*")]
            )
            if not filepaths:
                logger.info("No files selected.")
                return 0

        loaded_count = 0
        new_metadata = {}
        new_raw_spectra = {}

        for fpath in filepaths:
            if not os.path.isfile(fpath):
                logger.warning(f"File not found: {fpath}. Skipping.")
                continue

            basename = os.path.basename(fpath)
            file_id = self._generate_file_id()
            logger.info(f"Loading file: {basename} (ID: {file_id})")

            try:
                # Parse filename for parameters
                params = parse_filename(basename)
                params['filepath'] = fpath
                params['filename'] = basename
                params['file_id'] = file_id

                # Load data using pandas
                # Assuming tab-separated, no header, columns: Wavelength (nm), Counts
                spectrum_df = pd.read_csv(
                    fpath,
                    sep='\t',
                    header=None,
                    names=['wavelength_nm', 'counts'],
                    comment='#' # Ignore comment lines if any
                )

                # Basic validation
                if spectrum_df.isnull().values.any() or spectrum_df.empty:
                    logger.warning(f"Empty or invalid data found in {basename}. Skipping.")
                    continue
                if not pd.api.types.is_numeric_dtype(spectrum_df['wavelength_nm']) or \
                    not pd.api.types.is_numeric_dtype(spectrum_df['counts']):
                    logger.warning(f"Non-numeric data found in {basename}. Skipping.")
                    continue

                # Calculate Energy (eV)
                # Avoid division by zero or negative wavelengths
                valid_wl = spectrum_df['wavelength_nm'] > 0
                spectrum_df['energy_ev'] = np.nan
                spectrum_df.loc[valid_wl, 'energy_ev'] = HC_EV_NM / spectrum_df.loc[valid_wl, 'wavelength_nm']

                # Store raw and initial processed data
                new_metadata[file_id] = params
                # Keep raw Wavelength vs Counts
                new_raw_spectra[file_id] = spectrum_df[['wavelength_nm', 'counts']].copy()
                # Initial processed is Energy vs Counts (will be modified by processing steps)
                self.processed_spectra[file_id] = spectrum_df[['energy_ev', 'counts']].copy().dropna()


                loaded_count += 1
                logger.info(f"Successfully loaded and parsed: {basename}. Params: {params}")

            except pd.errors.EmptyDataError:
                logger.error(f"File is empty: {basename}. Skipping.")
            except Exception as e:
                logger.error(f"Failed to load or parse file {basename}: {e}", exc_info=True)

        # Combine new data with existing data if any
        if loaded_count > 0:
            self.metadata.update(new_metadata)
            self.raw_spectra.update(new_raw_spectra)
            self._update_main_dataframe() # Rebuild the summary DataFrame

        logger.info(f"Finished loading. Successfully loaded {loaded_count} files.")
        return loaded_count

    def _update_main_dataframe(self):
        """Rebuilds the main summary DataFrame from the metadata."""
        if not self.metadata:
            self.data = pd.DataFrame() # Ensure empty DataFrame if no data
            return

        df = pd.DataFrame.from_dict(self.metadata, orient='index')
        # Reorder columns for better readability
        cols = ['file_id', 'filename', 'temperature_k', 'power_uw', 'time_s', 'gf_present', 'filepath']
        # Add any extra columns that might exist, just in case
        cols.extend([c for c in df.columns if c not in cols])
        self.data = df[cols].reset_index(drop=True)
        logger.debug("Main DataFrame updated.")

    def get_metadata(self) -> Optional[pd.DataFrame]:
        """Returns the DataFrame containing metadata for all loaded files."""
        return self.data

    def get_processed_spectrum(self, file_id: str) -> Optional[pd.DataFrame]:
        """Returns the processed spectrum (Energy, Counts) for a given file_id."""
        return self.processed_spectra.get(file_id)

    def get_raw_spectrum(self, file_id: str) -> Optional[pd.DataFrame]:
        """Returns the raw spectrum (Wavelength, Counts) for a given file_id."""
        return self.raw_spectra.get(file_id)

    def get_all_file_ids(self) -> List[str]:
        """Returns a list of all loaded file IDs."""
        return list(self.metadata.keys())

    def clear_data(self):
        """Clears all loaded data."""
        self.data = None
        self.raw_spectra = {}
        self.processed_spectra = {}
        self.metadata = {}
        self._file_id_counter = 0
        logger.info("All data cleared.")

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dummy data files for testing
    os.makedirs('temp_test_data', exist_ok=True)
    file1_path = 'temp_test_data/SampleA_10K_100uW_0.5s.csv'
    file2_path = 'temp_test_data/SampleB_5-9K_50uW_1s_GF4p8.csv'
    file3_path = 'temp_test_data/Invalid_format.csv'
    file4_path = 'temp_test_data/Empty.csv'

    with open(file1_path, 'w') as f:
        f.write("600\t1000\n")
        f.write("601\t1200\n")
        f.write("602\t1500\n")

    with open(file2_path, 'w') as f:
        f.write("700\t500\n")
        f.write("701\t600\n")
        f.write("702\t550\n")

    with open(file3_path, 'w') as f:
        f.write("Wavelength\tCounts\n") # Header makes it invalid for current read_csv
        f.write("800\t100\n")

    with open(file4_path, 'w') as f:
        pass # Empty file

    handler = DataHandler()
    # Test loading specific files
    num_loaded = handler.load_files([file1_path, file2_path, file3_path, file4_path, 'nonexistent.csv'])
    print(f"\nLoaded {num_loaded} files initially.")

    print("\nMetadata DataFrame:")
    print(handler.get_metadata())

    print("\nProcessed Spectrum for file_1:")
    print(handler.get_processed_spectrum('file_1'))

    print("\nRaw Spectrum for file_2:")
    print(handler.get_raw_spectrum('file_2'))

    # Clean up dummy files
    import shutil
    shutil.rmtree('temp_test_data')

    # Test clearing data
    handler.clear_data()
    print("\nMetadata after clearing:")
    print(handler.get_metadata())