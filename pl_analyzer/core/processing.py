import logging
import pandas as pd
from typing import Dict, Optional, Tuple
import numpy as np

from .data_handler import Dataset

logger = logging.getLogger(__name__)

# Spectrometer response time correction factors
RESPONSE_CORRECTION_FACTORS = {
    0.1: 2.67411052994443,
    0.2: 1.74082446540644,
    0.3: 1.44263778631468,
    0.5: 1.20406605805442,
}

def normalize_by_acquisition_time(dataset: Dataset):
    """
    Divides the counts of all spectra in the dataset by their acquisition time.
    """
    logger.info(f"Applying time normalization to dataset '{dataset.name}'...")
    for file_id, metadata in dataset.metadata.items():
        acq_time = metadata.get('time_s')
        spectrum_df = dataset.get_processed_spectrum(file_id)
        if spectrum_df is not None and acq_time and acq_time > 0:
            spectrum_df['counts'] /= acq_time

def rescale_by_grey_filter(dataset: Dataset, gf_transmission_factor: float):
    """
    Divides the counts of spectra with 'gf_present' by the given factor.
    """
    logger.info(f"Applying GF rescaling to dataset '{dataset.name}'...")
    if not isinstance(gf_transmission_factor, (int, float)) or gf_transmission_factor <= 0:
        raise ValueError("GF transmission factor must be a positive number.")

    for file_id, metadata in dataset.metadata.items():
        if metadata.get('gf_present', False):
            spectrum_df = dataset.get_processed_spectrum(file_id)
            if spectrum_df is not None:
                spectrum_df['counts'] /= gf_transmission_factor

def correct_spectrometer_response(dataset: Dataset):
    """
    Corrects for spectrometer response based on acquisition time.
    """
    logger.info(f"Applying response correction to dataset '{dataset.name}'...")
    for file_id, metadata in dataset.metadata.items():
        acq_time = metadata.get('time_s')
        correction_factor = RESPONSE_CORRECTION_FACTORS.get(acq_time)
        if correction_factor:
            spectrum_df = dataset.get_processed_spectrum(file_id)
            if spectrum_df is not None:
                spectrum_df['counts'] /= correction_factor

def normalize_spectrum(spectrum_df: pd.DataFrame, energy_range: Optional[Tuple[float, float]] = None) -> Optional[pd.DataFrame]:
    """
    Normalizes a spectrum's counts to a maximum of 1.0.
    """
    if spectrum_df is None or spectrum_df.empty:
        return None
    
    normalized_df = spectrum_df.copy()
    
    try:
        if energy_range:
            min_e, max_e = energy_range
            range_df = normalized_df[(normalized_df['energy_ev'] >= min_e) & (normalized_df['energy_ev'] <= max_e)]
            max_count = range_df['counts'].max() if not range_df.empty else 0
        else:
            max_count = normalized_df['counts'].max()

        if max_count > 0:
            normalized_df['counts'] /= max_count
        
        return normalized_df
    except Exception as e:
        logger.error(f"Error during spectrum normalization: {e}", exc_info=True)
        return None


def subtract_background(counts: np.ndarray) -> np.ndarray:
    """
    Subtract the minimum value from a spectrum to remove constant background.

    Args:
        counts: 1D array of intensity values.

    Returns:
        A new numpy array with the global minimum removed (values >= 0).
    """
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return arr.copy()
    try:
        min_val = np.nanmin(arr)
    except Exception as exc:
        logger.error(f"Background subtraction failed: {exc}", exc_info=True)
        return arr.copy()
    if not np.isfinite(min_val):
        return arr.copy()
    return arr - min_val


def subtract_baseline_from_dataset(dataset: Dataset):
    """
    Subtracts the baseline (minimum value) from all spectra in the dataset.
    """
    logger.info(f"Applying baseline subtraction to dataset '{dataset.name}'...")
    for file_id in dataset.metadata.keys():
        spectrum_df = dataset.get_processed_spectrum(file_id)
        if spectrum_df is not None and 'counts' in spectrum_df.columns:
            spectrum_df['counts'] = subtract_background(spectrum_df['counts'].values)