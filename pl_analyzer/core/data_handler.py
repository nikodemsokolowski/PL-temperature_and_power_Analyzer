import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from tkinter import filedialog

from .file_parser import parse_filename

logger = logging.getLogger(__name__)

# Constants
HC_EV_NM = 1239.84198  # Planck's constant * speed of light (eV * nm)


class Dataset:
    """
    Represents a single dataset, containing multiple PL spectra and their metadata.
    """
    def __init__(self, name: str):
        self.name = name
        self.data: Optional[pd.DataFrame] = None
        self.raw_spectra: Dict[str, pd.DataFrame] = {}
        self.processed_spectra: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._file_id_counter = 0
        self.time_normalized = False
        self.gf_rescaled = False
        self.response_corrected = False
        self.spike_cleaned = False
        self.background_subtracted = False
        self.is_virtual_sum = False
        self.source_dataset_name: Optional[str] = None
        self.source_pairs: Dict[str, tuple] = {}

    def _generate_file_id(self) -> str:
        """Generates a unique internal ID for each loaded file within this dataset."""
        self._file_id_counter += 1
        return f"{self.name}_file_{self._file_id_counter}"

    def load_files(self, filepaths: List[str]) -> int:
        """
        Loads PL data from specified .csv files into the dataset.
        """
        loaded_count = 0
        new_metadata = {}
        new_raw_spectra = {}

        try:
            for fpath in filepaths:
                if not os.path.isfile(fpath):
                    logger.warning(f"File not found: {fpath}. Skipping.")
                    continue

                basename = os.path.basename(fpath)
                file_id = self._generate_file_id()
                logger.info(f"Loading file into dataset '{self.name}': {basename} (ID: {file_id})")

                try:
                    params = parse_filename(basename)
                    params['filepath'] = fpath
                    params['filename'] = basename
                    params['file_id'] = file_id

                    spectrum_df = pd.read_csv(
                        fpath,
                        sep=None,
                        engine='python',
                        header=None,
                        comment='#'
                    )

                    if spectrum_df.shape[1] < 2:
                        logger.warning(f"File {basename} does not contain at least two columns. Skipping.")
                        continue

                    if spectrum_df.shape[1] > 2:
                        logger.debug(
                            f"{basename} contains {spectrum_df.shape[1]} columns. Only the first two will be used."
                        )
                    spectrum_df = spectrum_df.iloc[:, :2].copy()
                    spectrum_df.columns = ['wavelength_nm', 'counts']

                    if spectrum_df.isnull().values.any() or spectrum_df.empty:
                        logger.warning(f"Empty or invalid data in {basename}. Skipping.")
                        continue
                    if not pd.api.types.is_numeric_dtype(spectrum_df['wavelength_nm']) or \
                       not pd.api.types.is_numeric_dtype(spectrum_df['counts']):
                        logger.warning(f"Non-numeric data in {basename}. Skipping.")
                        continue

                    valid_wl = spectrum_df['wavelength_nm'] > 0
                    spectrum_df['energy_ev'] = np.nan
                    spectrum_df.loc[valid_wl, 'energy_ev'] = HC_EV_NM / spectrum_df.loc[valid_wl, 'wavelength_nm']

                    new_metadata[file_id] = params
                    new_raw_spectra[file_id] = spectrum_df[['wavelength_nm', 'counts']].copy()
                    self.processed_spectra[file_id] = spectrum_df[['energy_ev', 'counts']].copy().dropna()

                    loaded_count += 1
                    logger.info(f"Successfully loaded and parsed: {basename}. Params: {params}")

                except Exception as e:
                    logger.error(f"Failed to load or parse file {basename}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"An unexpected error occurred during file loading: {e}", exc_info=True)
            raise

        if loaded_count > 0:
            self.metadata.update(new_metadata)
            self.raw_spectra.update(new_raw_spectra)
            self._update_main_dataframe()

        logger.info(f"Finished loading for dataset '{self.name}'. Successfully loaded {loaded_count} files.")
        return loaded_count

    def _update_main_dataframe(self):
        """Rebuilds the main summary DataFrame from the metadata."""
        if not self.metadata:
            self.data = pd.DataFrame()
            return

        df = pd.DataFrame.from_dict(self.metadata, orient='index')
        cols = ['file_id', 'filename', 'temperature_k', 'power_uw', 'time_s', 'gf_present', 'bfield_t', 'polarization', 'filepath']
        cols.extend([c for c in df.columns if c not in cols])
        self.data = df[cols].reset_index(drop=True)
        logger.debug(f"Main DataFrame for dataset '{self.name}' updated.")

    def get_metadata(self) -> Optional[pd.DataFrame]:
        return self.data

    @staticmethod
    def _safe_meta_value(value):
        try:
            if pd.isna(value):
                return None
        except Exception:
            return value
        return value

    @staticmethod
    def _sort_meta_key(value):
        if value is None:
            return (1, 0)
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (0, str(value))

    def generate_sum_spectra_pairs(
        self,
        metadata_df: Optional[pd.DataFrame] = None
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build summed spectra for sigma+/sigma- pairs sharing the same conditions.

        Args:
            metadata_df: Optional subset of the dataset metadata to use.

        Returns:
            (entries, unmatched) where entries is a list of dictionaries containing:
                - 'energy': np.ndarray of energy values (eV)
                - 'counts': np.ndarray of summed counts
                - 'metadata': dict with temperature/power/bfield info
                - 'source_file_ids': tuple(sigma_plus_id, sigma_minus_id)
                - 'sum_key': string identifier for the pair
            and unmatched is a list describing incomplete pairs.
        """
        meta = metadata_df if metadata_df is not None else self.get_metadata()
        if meta is None or meta.empty:
            return [], []

        grouped: Dict[tuple, Dict[str, Any]] = {}
        for _, row in meta.iterrows():
            fid = row.get('file_id')
            pol = row.get('polarization')
            if not fid or pol not in ('sigma+', 'sigma-'):
                continue
            temp = self._safe_meta_value(row.get('temperature_k'))
            power = self._safe_meta_value(row.get('power_uw'))
            bfield = self._safe_meta_value(row.get('bfield_t'))
            key = (temp, power, bfield)
            entry = grouped.setdefault(key, {'rows': {}})
            entry['rows'].setdefault(pol, row)
            entry.setdefault(pol, fid)

        entries: List[Dict[str, Any]] = []
        unmatched: List[Dict[str, Any]] = []

        for key, entry in grouped.items():
            temp, power, bfield = key
            plus_id = entry.get('sigma+')
            minus_id = entry.get('sigma-')
            if not plus_id or not minus_id:
                unmatched.append({
                    'conditions': key,
                    'present': [pol for pol in ('sigma+', 'sigma-') if entry.get(pol)]
                })
                continue

            plus_spec = self.get_processed_spectrum(plus_id)
            minus_spec = self.get_processed_spectrum(minus_id)
            if plus_spec is None or plus_spec.empty or minus_spec is None or minus_spec.empty:
                unmatched.append({
                    'conditions': key,
                    'present': [
                        'sigma+' if plus_spec is not None and not plus_spec.empty else None,
                        'sigma-' if minus_spec is not None and not minus_spec.empty else None
                    ],
                    'reason': 'missing spectrum'
                })
                continue

            e_plus = np.asarray(plus_spec['energy_ev'].values)
            c_plus = np.asarray(plus_spec['counts'].values)
            e_minus = np.asarray(minus_spec['energy_ev'].values)
            c_minus = np.asarray(minus_spec['counts'].values)

            plus_order = np.argsort(e_plus)
            minus_order = np.argsort(e_minus)
            e_plus = e_plus[plus_order]
            c_plus = c_plus[plus_order]
            e_minus = e_minus[minus_order]
            c_minus = c_minus[minus_order]

            if e_plus.size == 0 or e_minus.size == 0:
                unmatched.append({
                    'conditions': key,
                    'present': [
                        'sigma+' if e_plus.size else None,
                        'sigma-' if e_minus.size else None
                    ],
                    'reason': 'empty spectrum'
                })
                continue

            if np.array_equal(e_plus, e_minus):
                common_energy = e_plus
                plus_interp = c_plus
                minus_interp = c_minus
            else:
                common_energy = np.unique(np.concatenate([e_plus, e_minus]))
                if common_energy.size == 0:
                    unmatched.append({
                        'conditions': key,
                        'present': ['sigma+', 'sigma-'],
                        'reason': 'no overlapping energy range'
                    })
                    continue
                plus_interp = np.interp(common_energy, e_plus, c_plus)
                minus_interp = np.interp(common_energy, e_minus, c_minus)

            summed_counts = plus_interp + minus_interp

            representative = entry['rows'].get('sigma+')
            if representative is None:
                representative = entry['rows'].get('sigma-')
            temp_val = self._safe_meta_value(representative.get('temperature_k')) if representative is not None else temp
            power_val = self._safe_meta_value(representative.get('power_uw')) if representative is not None else power
            bfield_val = self._safe_meta_value(representative.get('bfield_t')) if representative is not None else bfield

            entries.append({
                'energy': common_energy.astype(float),
                'counts': summed_counts.astype(float),
                'metadata': {
                    'temperature_k': temp_val,
                    'power_uw': power_val,
                    'bfield_t': bfield_val,
                    'polarization': 'sum'
                },
                'source_file_ids': (plus_id, minus_id),
                'sum_key': f"{plus_id}+{minus_id}"
            })

        entries.sort(key=lambda item: (
            self._sort_meta_key(item['metadata'].get('temperature_k')),
            self._sort_meta_key(item['metadata'].get('power_uw')),
            self._sort_meta_key(item['metadata'].get('bfield_t'))
        ))

        return entries, unmatched

    def get_processed_spectrum(self, file_id: str) -> Optional[pd.DataFrame]:
        return self.processed_spectra.get(file_id)

    def get_raw_spectrum(self, file_id: str) -> Optional[pd.DataFrame]:
        return self.raw_spectra.get(file_id)

    def get_all_file_ids(self) -> List[str]:
        return list(self.metadata.keys())

    def clear_files(self, file_ids: List[str]):
        """Removes specified files from the dataset."""
        for file_id in file_ids:
            if file_id in self.metadata:
                del self.metadata[file_id]
            if file_id in self.raw_spectra:
                del self.raw_spectra[file_id]
            if file_id in self.processed_spectra:
                del self.processed_spectra[file_id]
        self._update_main_dataframe()

    def add_spectra_from_dataframe(self, spectra_df: pd.DataFrame):
        """
        Adds spectra to the dataset from a DataFrame.
        """
        for _, row in spectra_df.iterrows():
            file_id = self._generate_file_id()
            self.metadata[file_id] = {
                'file_id': file_id,
                'filename': row['file_id'],
                'temperature_k': row['temperature_k'],
                'power_uw': row['power_uw'],
                'time_s': None,
                'gf_present': False,
                'bfield_t': row.get('bfield_t', None),
                'polarization': row.get('polarization', None),
                'filepath': 'imported'
            }
            self.processed_spectra[file_id] = row['spectrum']
        self._update_main_dataframe()


class DataHandler:
    """
    Manages multiple datasets.
    """
    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}
        self.active_dataset_name: Optional[str] = None
        self._dataset_id_counter = 0

    def _generate_dataset_name(self) -> str:
        """Generates a unique name for a new dataset."""
        self._dataset_id_counter += 1
        return f"Dataset_{self._dataset_id_counter}"

    def _ensure_unique_dataset_name(self, base_name: str) -> str:
        """Ensure the proposed dataset name does not collide with existing datasets."""
        name = base_name
        suffix = 1
        while name in self.datasets:
            name = f"{base_name}_{suffix}"
            suffix += 1
        return name

    def create_new_dataset(self, filepaths: List[str]) -> Optional[str]:
        """
        Creates a new dataset and loads files into it.
        """
        if not filepaths:
            return None
        
        dataset_name = self._generate_dataset_name()
        new_dataset = Dataset(name=dataset_name)
        loaded_count = new_dataset.load_files(filepaths)

        if loaded_count > 0:
            self.datasets[dataset_name] = new_dataset
            self.active_dataset_name = dataset_name
            logger.info(f"Created new dataset '{dataset_name}' with {loaded_count} files.")
            return dataset_name
        return None

    def create_empty_dataset(self) -> str:
        """Creates a new, empty dataset and sets it as active."""
        dataset_name = self._generate_dataset_name()
        new_dataset = Dataset(name=dataset_name)
        self.datasets[dataset_name] = new_dataset
        self.active_dataset_name = dataset_name
        logger.info(f"Created new empty dataset '{dataset_name}'.")
        return dataset_name

    def create_sum_dataset(
        self,
        source_dataset_name: Optional[str] = None
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Create or refresh a virtual dataset containing sigma+ + sigma- summed spectra.

        Args:
            source_dataset_name: Optional explicit source dataset. Defaults to active dataset.

        Returns:
            (dataset_name, unmatched) where dataset_name is the name of the virtual dataset created
            (or refreshed) and unmatched is a list of incomplete pair descriptors. If no valid pairs
            are found, dataset_name is None.
        """
        source_name = source_dataset_name or self.active_dataset_name
        if not source_name:
            raise ValueError("No source dataset specified for sum creation.")
        source_dataset = self.datasets.get(source_name)
        if source_dataset is None:
            raise KeyError(f"Dataset '{source_name}' not found.")

        metadata_df = source_dataset.get_metadata()
        entries, unmatched = source_dataset.generate_sum_spectra_pairs(metadata_df)
        if not entries:
            return None, unmatched

        base_name = f"{source_dataset.name}_sum"
        target_dataset = self.datasets.get(base_name)
        if target_dataset and target_dataset.is_virtual_sum and target_dataset.source_dataset_name == source_dataset.name:
            logger.info(f"Refreshing existing sum dataset '{base_name}'.")
            target_dataset.raw_spectra.clear()
            target_dataset.processed_spectra.clear()
            target_dataset.metadata.clear()
            target_dataset.source_pairs.clear()
            target_dataset._file_id_counter = 0
        else:
            dataset_name = self._ensure_unique_dataset_name(base_name)
            target_dataset = Dataset(name=dataset_name)
            self.datasets[dataset_name] = target_dataset
        dataset_name = target_dataset.name

        target_dataset.is_virtual_sum = True
        target_dataset.source_dataset_name = source_dataset.name
        target_dataset.source_pairs.clear()

        for idx, entry in enumerate(entries, start=1):
            file_id = target_dataset._generate_file_id()
            energy = np.asarray(entry['energy'], dtype=float)
            counts = np.asarray(entry['counts'], dtype=float)
            target_dataset.processed_spectra[file_id] = pd.DataFrame({
                'energy_ev': energy,
                'counts': counts
            })
            target_dataset.metadata[file_id] = {
                'file_id': file_id,
                'filename': f"{target_dataset.name}_{idx:03d}.sum",
                'temperature_k': entry['metadata'].get('temperature_k'),
                'power_uw': entry['metadata'].get('power_uw'),
                'time_s': None,
                'gf_present': False,
                'bfield_t': entry['metadata'].get('bfield_t'),
                'polarization': 'sum',
                'filepath': f"virtual:{source_dataset.name}",
                'source_dataset': source_dataset.name,
                'source_sigma_plus': entry['source_file_ids'][0],
                'source_sigma_minus': entry['source_file_ids'][1],
                'virtual_sum': True
            }
            target_dataset.source_pairs[file_id] = entry['source_file_ids']

        target_dataset._update_main_dataframe()
        self.active_dataset_name = target_dataset.name
        logger.info(
            f"Created virtual sum dataset '{target_dataset.name}' with {len(entries)} spectra from '{source_dataset.name}'."
        )
        return target_dataset.name, unmatched

    def get_active_dataset(self) -> Optional[Dataset]:
        """Returns the currently active dataset."""
        if self.active_dataset_name:
            return self.datasets.get(self.active_dataset_name)
        return None

    def set_active_dataset(self, name: str):
        """Sets the active dataset by name."""
        if name in self.datasets:
            self.active_dataset_name = name
            logger.info(f"Active dataset set to: {name}")
        else:
            logger.warning(f"Dataset '{name}' not found.")

    def get_dataset_names(self) -> List[str]:
        """Returns a list of all dataset names."""
        return list(self.datasets.keys())

    def clear_all_data(self):
        """Clears all datasets."""
        self.datasets = {}
        self.active_dataset_name = None
        self._dataset_id_counter = 0
        logger.info("All datasets cleared.")

    def clear_dataset(self, name: str):
        """Clears a single dataset by name."""
        if name in self.datasets:
            del self.datasets[name]
            if self.active_dataset_name == name:
                self.active_dataset_name = None
            logger.info(f"Dataset '{name}' cleared.")

    def get_all_filepaths(self) -> List[str]:
        """Returns a list of all filepaths from all datasets."""
        all_filepaths = []
        for dataset in self.datasets.values():
            if dataset.data is not None:
                all_filepaths.extend(dataset.data['filepath'].tolist())
        return all_filepaths

    def get_all_datasets_filepaths(self) -> Dict[str, List[str]]:
        """Returns a dictionary mapping dataset names to their filepaths."""
        datasets_filepaths = {}
        for name, dataset in self.datasets.items():
            if dataset.data is not None:
                datasets_filepaths[name] = dataset.data['filepath'].tolist()
        return datasets_filepaths
