import re
import os
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Regex patterns for parsing experimental parameters from filenames.
# Case-insensitive matching is used for units.
DELIMITER = r'[_,-]'
# Temperature: Handles formats like 5K, 5.5K, 5p5K, 10k.
TEMP_PATTERN = re.compile(f'{DELIMITER}(\\d+(?:[p.]\\d+)?)[Kk]', re.IGNORECASE)
# Power: Handles formats like 50uW, 0.5nW, 10UW, 5NW, with 'p' as a decimal separator.
# It explicitly captures 'uW' or 'nW' for unit conversion.
POWER_PATTERN = re.compile(f'{DELIMITER}(\\d+(?:[p.]\\d+)?)(uW|nW)', re.IGNORECASE)
# Acquisition Time: Handles formats like 1s, 0.5s, 0p5s, 2S.
TIME_PATTERN = re.compile(f'{DELIMITER}(\\d+(?:[p.]\\d+)?)s', re.IGNORECASE)
# Grey Filter: Detects the presence of 'GF' with an optional numeric value.
GF_PATTERN = re.compile(f'{DELIMITER}GF(\\d+(?:[p.]\\d+)?)?', re.IGNORECASE)
# Magnetic Field: Handles formats like 5T, 0T, B=1T, B1p5T, 1.5T
BFIELD_PATTERN = re.compile(f'{DELIMITER}(?:B=?)?(\\d+(?:[p.]\\d+)?)[Tt](?:{DELIMITER}|$)', re.IGNORECASE)

# Polarization configuration - will be loaded from config
_POLARIZATION_CONFIG = {
    'sigma_plus': ['pol1', 'ROI1', 'sig+', 'sigma+', 'CP_plus', 'sigmaplus'],
    'sigma_minus': ['pol2', 'ROI2', 'sig-', 'sigma-', 'CP_minus', 'sigmaminus']
}

_MAG_SWEEP_CONFIG = {
    'min_bfield_t': 0.0,
    'max_bfield_t': None,
    'step_t': 0.5,
    'temperature_k': None,
    'power_uw': None,
    'time_s': None,
    'time_ranges': [],
    'sweep_direction': 'low_to_high',
    'roi_map': {
        '1': 'sigma+',
        '2': 'sigma-'
    }
}

SWEEP_PATTERN = re.compile(r'pl[\s_]*(\d+)[\s_-]*roi[\s_-]*(\d+)', re.IGNORECASE)

def set_polarization_config(sigma_plus_strings: list, sigma_minus_strings: list):
    """Update the polarization configuration."""
    global _POLARIZATION_CONFIG
    _POLARIZATION_CONFIG = {
        'sigma_plus': sigma_plus_strings if sigma_plus_strings else [],
        'sigma_minus': sigma_minus_strings if sigma_minus_strings else []
    }

def get_polarization_config() -> Dict[str, list]:
    """Get current polarization configuration."""
    return _POLARIZATION_CONFIG.copy()


def _sanitize_time_ranges(ranges: Any) -> List[Dict[str, Optional[float]]]:
    """Ensure time range entries only contain numeric values."""
    sanitized: List[Dict[str, Optional[float]]] = []
    if not isinstance(ranges, list):
        return sanitized
    for entry in ranges:
        if not isinstance(entry, dict):
            continue
        try:
            time_val = float(str(entry.get('time_s')).replace('p', '.'))
        except (TypeError, ValueError):
            continue
        if time_val <= 0:
            continue
        try:
            b_min_val = float(str(entry.get('b_min')).replace('p', '.'))
        except (TypeError, ValueError):
            continue
        b_max_raw = entry.get('b_max')
        b_max_val: Optional[float] = None
        if b_max_raw not in (None, ""):
            try:
                b_max_val = float(str(b_max_raw).replace('p', '.'))
            except (TypeError, ValueError):
                b_max_val = None
        sanitized.append({'b_min': b_min_val, 'b_max': b_max_val, 'time_s': time_val})
    sanitized.sort(key=lambda item: item['b_min'])
    return sanitized


def set_magnetic_sweep_config(config: Dict[str, Any]):
    """Update the magnetic sweep defaults used for filename parsing."""
    global _MAG_SWEEP_CONFIG
    merged = _MAG_SWEEP_CONFIG.copy()
    merged['roi_map'] = _MAG_SWEEP_CONFIG['roi_map'].copy()
    merged['time_ranges'] = list(_MAG_SWEEP_CONFIG.get('time_ranges', []))
    for key, value in config.items():
        if key == 'roi_map' and isinstance(value, dict):
            merged['roi_map'] = {k: v for k, v in value.items() if v}
        elif key == 'time_ranges':
            merged['time_ranges'] = _sanitize_time_ranges(value)
        elif key == 'sweep_direction':
            merged['sweep_direction'] = value if value in ('low_to_high', 'high_to_low') else 'low_to_high'
        else:
            merged[key] = value
    if not isinstance(merged.get('time_ranges'), list):
        merged['time_ranges'] = []
    if merged.get('sweep_direction') not in ('low_to_high', 'high_to_low'):
        merged['sweep_direction'] = 'low_to_high'
    _MAG_SWEEP_CONFIG = merged


def get_magnetic_sweep_config() -> Dict[str, Any]:
    """Return a copy of the current magnetic sweep defaults."""
    result = _MAG_SWEEP_CONFIG.copy()
    result['roi_map'] = _MAG_SWEEP_CONFIG['roi_map'].copy()
    result['time_ranges'] = [entry.copy() for entry in _MAG_SWEEP_CONFIG.get('time_ranges', [])]
    return result


def _get_time_for_bfield(bfield: Optional[float], config: Dict[str, Any]) -> Optional[float]:
    """Return acquisition time for the provided B-field based on configured ranges."""
    fallback = config.get('time_s')
    if bfield is None:
        if fallback is None:
            return None
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return None

    try:
        bfield_val = float(bfield)
    except (TypeError, ValueError):
        return None

    for entry in config.get('time_ranges', []):
        try:
            b_min = float(entry['b_min'])
        except (KeyError, TypeError, ValueError):
            continue
        b_max_val = entry.get('b_max')
        if b_max_val is not None:
            try:
                b_max = float(b_max_val)
            except (TypeError, ValueError):
                b_max = None
        else:
            b_max = None
        lower_ok = bfield_val >= b_min - 1e-9
        upper_ok = True if b_max is None else bfield_val <= b_max + 1e-9
        if lower_ok and upper_ok:
            try:
                return float(entry['time_s'])
            except (KeyError, TypeError, ValueError):
                return None

    if fallback is None:
        return None
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return None

def _detect_polarization(filename: str) -> Optional[str]:
    """
    Detects polarization from filename based on configured strings.
    Returns 'sigma+', 'sigma-', or None.
    """
    filename_lower = filename.lower()
    
    # Check for sigma plus indicators
    for indicator in _POLARIZATION_CONFIG['sigma_plus']:
        if indicator.lower() in filename_lower:
            return 'sigma+'
    
    # Check for sigma minus indicators
    for indicator in _POLARIZATION_CONFIG['sigma_minus']:
        if indicator.lower() in filename_lower:
            return 'sigma-'
    
    return None

def parse_filename(filename: str) -> Dict[str, Optional[Any]]:
    """
    Parses a filename to extract experimental parameters using robust regex.

    Args:
        filename: The base name of the file (without directory path).

    Returns:
        A dictionary containing the extracted parameters:
        'temperature_k': Temperature in Kelvin (float) or None.
        'power_uw': Laser power in microWatts (float) or None.
        'time_s': Acquisition time in seconds (float) or None.
        'gf_present': Boolean indicating if a Grey Filter pattern was found.
        'bfield_t': Magnetic field in Tesla (float) or None.
        'polarization': Circular polarization ('sigma+', 'sigma-') or None.
    """
    basename = os.path.basename(filename)
    params: Dict[str, Any] = {
        'temperature_k': None,
        'power_uw': None,
        'time_s': None,
        'gf_present': False,
        'bfield_t': None,
        'polarization': None,
    }

    # Helper to safely convert string to float, handling 'p' as decimal.
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s.replace('p', '.'))
        except (ValueError, TypeError):
            return None

    # Extract Temperature
    temp_match = TEMP_PATTERN.search(basename)
    if temp_match:
        params['temperature_k'] = _to_float(temp_match.group(1))

    # Extract Power
    power_match = POWER_PATTERN.search(basename)
    if power_match:
        value_str, unit = power_match.groups()
        power_val = _to_float(value_str)
        if power_val is not None:
            if unit.lower() == 'nw':
                params['power_uw'] = power_val / 1000.0
            else: # uW
                params['power_uw'] = power_val

    # Extract Time
    time_match = TIME_PATTERN.search(basename)
    if time_match:
        params['time_s'] = _to_float(time_match.group(1))

    # Detect Grey Filter
    if GF_PATTERN.search(basename):
        params['gf_present'] = True

    # Extract Magnetic Field
    bfield_match = BFIELD_PATTERN.search(basename)
    if bfield_match:
        params['bfield_t'] = _to_float(bfield_match.group(1))

    # Detect Polarization
    params['polarization'] = _detect_polarization(basename)

    sweep_match = SWEEP_PATTERN.search(basename)
    if sweep_match:
        index_str, roi_str = sweep_match.groups()
        try:
            index = int(index_str)
        except ValueError:
            index = None
        sweep_cfg = _MAG_SWEEP_CONFIG
        step = sweep_cfg.get('step_t') or 0.0
        min_b = sweep_cfg.get('min_bfield_t') or 0.0
        max_b = sweep_cfg.get('max_bfield_t')
        direction = sweep_cfg.get('sweep_direction', 'low_to_high')

        bfield_val = None
        if index is not None and step > 0:
            if direction == 'high_to_low' and max_b is not None:
                candidate = max_b - index * step
                if candidate >= min_b - 1e-9:
                    bfield_val = candidate
            else:
                candidate = min_b + index * step
                if max_b is None or candidate <= (max_b + 1e-9):
                    bfield_val = candidate
        if bfield_val is not None:
            params['bfield_t'] = bfield_val

        for key in ('temperature_k', 'power_uw'):
            cfg_val = sweep_cfg.get(key)
            if cfg_val is not None:
                params[key] = cfg_val
        if not params.get('polarization'):
            roi_map = sweep_cfg.get('roi_map', {})
            pol = roi_map.get(roi_str.strip())
            if pol:
                params['polarization'] = pol

    if params.get('time_s') is None:
        from_ranges = _get_time_for_bfield(params.get('bfield_t'), _MAG_SWEEP_CONFIG)
        if from_ranges is not None:
            params['time_s'] = from_ranges

    # Log a warning if essential parameters are missing
    if params['temperature_k'] is None:
        logger.debug(f"Could not parse temperature from {basename}")
    if params['power_uw'] is None:
        logger.debug(f"Could not parse power from {basename}")

    return params


def parse_exported_spectra(filepath: str) -> Optional[pd.DataFrame]:
    """
    Parses a file that was exported from this application.
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
        if 'Energy (eV)' not in df.columns:
            logger.error("Invalid format: 'Energy (eV)' column not found.")
            return None

        energy_col = df['Energy (eV)']
        spectra_data = []

        for col_name in df.columns:
            if col_name == 'Energy (eV)':
                continue

            match = re.match(r"T(\d+(?:\.\d+)?)K_P(\d+(?:\.\d+)?)uW", col_name)
            if match:
                temp_str, power_str = match.groups()
                temp = float(temp_str)
                power = float(power_str)
                
                file_id = f"imported_{os.path.basename(filepath)}_{col_name}"
                
                spectrum = pd.DataFrame({
                    'energy_ev': energy_col,
                    'counts': df[col_name]
                })

                spectra_data.append({
                    'file_id': file_id,
                    'temperature_k': temp,
                    'power_uw': power,
                    'spectrum': spectrum
                })

        return pd.DataFrame(spectra_data)

    except Exception as e:
        logger.error(f"Error parsing exported spectra file {filepath}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    # Example Usage for testing
    test_filenames = [
        "WS2_WSe2_Htype_Zeiss100x_grating300_slit20_confocal_5-9K_50uW_1s_GF4p8.csv",
        "SampleA_10K_100uW_0.5s.txt",
        "SampleB_300k_10uW_0.1s_noGF.dat",
        "Invalid_filename_structure.csv",
        "RangeTest_2.5K_1uW_0.2s.csv",
        "NoTemp_5uW_0.3s_GF1.csv",
        "NoPower_15K_10s.csv",
        "NoTime_20K_200uW.csv",
        "DecimalTest_12K_0p5uW_1p5s.csv", # Test 'p' decimal
        "TempDecimalTest_4p5K_10uW_1s.csv", # Test temperature 'p' decimal
        "DecimalTest_13K_100p0uW_2s.csv", # Test 'p' decimal with integer part
        "CaseTest_14k_5UW_0p1S_gf.csv",   # Test case insensitivity
        "NwTest_10K_500nW_1s.csv",         # Test nW power unit
        "NwCaseTest_11K_500NW_1s.csv",     # Test nW case insensitivity
        "GFOnlyTest_15K_10uW_1s_GF.csv",  # Test GF without number
    ]

    logging.basicConfig(level=logging.INFO) # Basic config for testing

    for fname in test_filenames:
        extracted_params = parse_filename(fname)
        print(f"Filename: {fname}")
        print(f"Extracted Params: {extracted_params}\n")
