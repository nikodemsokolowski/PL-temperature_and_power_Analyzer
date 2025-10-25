import logging
import os
import pandas as pd
import numpy as np
from tkinter import filedialog
from typing import List, Tuple, Optional

from .data_handler import Dataset
from .analysis import integrate_spectrum

logger = logging.getLogger(__name__)

def export_integrated_data(dataset: Dataset, temperature: float, integration_range: Tuple[float, float], save_path: str):
    """
    Calculates and exports integrated intensity for a power series within a dataset.
    """
    logger.info(f"Exporting integrated data for T={temperature}K from dataset '{dataset.name}'...")
    metadata_df = dataset.get_metadata()
    if metadata_df is None or metadata_df.empty:
        raise ValueError("No data in the dataset to export.")

    series_df = metadata_df[metadata_df['temperature_k'] == temperature].sort_values(by='power_uw')
    if series_df.empty:
        raise ValueError(f"No files found for T={temperature}K in this dataset.")

    results = []
    for _, row in series_df.iterrows():
        spectrum_df = dataset.get_processed_spectrum(row['file_id'])
        if spectrum_df is not None and not spectrum_df.empty:
            integral = integrate_spectrum(
                spectrum_df['energy_ev'].values,
                spectrum_df['counts'].values,
                integration_range
            )
            if integral is not None:
                results.append({'Power (uW)': row['power_uw'], 'Integrated Intensity': integral})

    if not results:
        raise ValueError("Failed to calculate any integrated intensities.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, sep='\t', index=False, float_format='%.6g')
    logger.info(f"Successfully exported {len(results_df)} points.")

def export_processed_spectra(dataset: Dataset, file_ids: List[str], save_path: str):
    """
    Exports processed spectra for the given file IDs from a dataset.
    """
    logger.info(f"Exporting {len(file_ids)} spectra from dataset '{dataset.name}'...")
    if not file_ids:
        raise ValueError("No file IDs provided for export.")

    all_spectra_dict = {}
    energy_basis = None

    # Sort file_ids by temperature, then by power
    file_ids.sort(key=lambda file_id: (
        dataset.metadata.get(file_id, {}).get('temperature_k', 0),
        dataset.metadata.get(file_id, {}).get('power_uw', 0)
    ))

    for file_id in file_ids:
        spectrum_df = dataset.get_processed_spectrum(file_id)
        metadata = dataset.metadata.get(file_id, {})
        if spectrum_df is not None and not spectrum_df.empty:
            col_name = f"T{metadata.get('temperature_k', 'NA')}K_P{metadata.get('power_uw', 'NA')}uW"
            current_energy = spectrum_df['energy_ev'].values
            current_counts = spectrum_df['counts'].values

            if energy_basis is None:
                energy_basis = current_energy
                all_spectra_dict['Energy (eV)'] = energy_basis
                all_spectra_dict[col_name] = current_counts
            else:
                if not np.allclose(energy_basis, current_energy):
                    logger.warning(f"Energy axis for {file_id} differs. Interpolating.")
                    interp_counts = np.interp(energy_basis, current_energy, current_counts)
                    all_spectra_dict[col_name] = interp_counts
                else:
                    all_spectra_dict[col_name] = current_counts

    if len(all_spectra_dict) <= 1:
        raise ValueError("No valid spectra found for export.")

    export_df = pd.DataFrame(all_spectra_dict)
    export_df.to_csv(save_path, sep='\t', index=False, float_format='%.6g')
    logger.info(f"Successfully exported {len(export_df.columns) - 1} spectra.")


def export_all_processed_spectra(dataset: Dataset, save_path: str):
    """
    Exports all processed spectra from a dataset.
    """
    logger.info(f"Exporting all spectra from dataset '{dataset.name}'...")
    file_ids = list(dataset.metadata.keys())
    if not file_ids:
        raise ValueError("No files in the dataset to export.")
    
    export_processed_spectra(dataset, file_ids, save_path)


def export_integrated_data_all(dataset: Dataset, integration_range: Tuple[float, float], save_path: str, format: str):
    """
    Calculates and exports integrated intensity for all data in a dataset.
    """
    logger.info(f"Exporting all integrated data from dataset '{dataset.name}'...")
    metadata_df = dataset.get_metadata()
    if metadata_df is None or metadata_df.empty:
        raise ValueError("No data in the dataset to export.")

    results = []
    for _, row in metadata_df.iterrows():
        spectrum_df = dataset.get_processed_spectrum(row['file_id'])
        if spectrum_df is not None and not spectrum_df.empty:
            integral = integrate_spectrum(
                spectrum_df['energy_ev'].values,
                spectrum_df['counts'].values,
                integration_range
            )
            if integral is not None:
                results.append({
                    'Power (uW)': row['power_uw'],
                    'Temperature (K)': row['temperature_k'],
                    'Integrated Intensity': integral
                })

    if not results:
        raise ValueError("Failed to calculate any integrated intensities.")

    results_df = pd.DataFrame(results)

    if format == 'power':
        pivot_df = results_df.pivot(index='Power (uW)', columns='Temperature (K)', values='Integrated Intensity')
        pivot_df.to_csv(save_path, sep='\t', float_format='%.6g')
    else:
        pivot_df = results_df.pivot(index='Temperature (K)', columns='Power (uW)', values='Integrated Intensity')
        pivot_df.to_csv(save_path, sep='\t', float_format='%.6g')

    logger.info(f"Successfully exported {len(results_df)} points.")
