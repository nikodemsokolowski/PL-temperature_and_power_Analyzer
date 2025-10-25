import json
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"


def _load_config_data() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}", exc_info=True)
        return {}


def _save_config_data(config: dict) -> None:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving config file: {e}", exc_info=True)


def save_last_datasets(datasets: dict[str, list[str]]):
    """Persist the dictionary of datasets and their file paths."""
    config = _load_config_data()
    config["last_datasets"] = datasets
    _save_config_data(config)
    logger.info(f"Saved {len(datasets)} datasets to {CONFIG_FILE}")


def load_last_datasets() -> dict[str, list[str]]:
    """Load the dictionary of datasets and their file paths."""
    config = _load_config_data()
    datasets = config.get("last_datasets", {})
    if datasets:
        logger.info(f"Loaded {len(datasets)} datasets from {CONFIG_FILE}")
    return datasets


# Default polarization configuration
DEFAULT_POLARIZATION_CONFIG = {
    'sigma_plus': ['pol1', 'ROI1', 'sig+', 'sigma+', 'CP_plus', 'sigmaplus'],
    'sigma_minus': ['pol2', 'ROI2', 'sig-', 'sigma-', 'CP_minus', 'sigmaminus']
}


def save_polarization_config(sigma_plus: list, sigma_minus: list):
    """Save polarization string mappings."""
    config = _load_config_data()
    config['polarization'] = {
        'sigma_plus': sigma_plus,
        'sigma_minus': sigma_minus
    }
    _save_config_data(config)
    logger.info("Saved polarization configuration")


def load_polarization_config() -> dict:
    """Load polarization string mappings, falling back to defaults."""
    config = _load_config_data()
    pol_config = config.get("polarization", {})
    if not pol_config or 'sigma_plus' not in pol_config or 'sigma_minus' not in pol_config:
        return DEFAULT_POLARIZATION_CONFIG.copy()
    return {
        'sigma_plus': pol_config['sigma_plus'],
        'sigma_minus': pol_config['sigma_minus']
    }


DEFAULT_MAGNETIC_SWEEP_CONFIG = {
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


def _sanitize_time_ranges(ranges: Any) -> List[Dict[str, Optional[float]]]:
    """Ensure time range entries are numeric and well-formed."""
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


def save_magnetic_sweep_config(config_updates: dict):
    """Save magnetic sweep defaults to the config file."""
    config = _load_config_data()
    sanitized_updates: Dict[str, Any] = {}
    for key, value in config_updates.items():
        if key == 'roi_map' and isinstance(value, dict):
            sanitized_updates['roi_map'] = {k: v for k, v in value.items() if v}
        elif key == 'time_ranges':
            sanitized_updates['time_ranges'] = _sanitize_time_ranges(value)
        elif key == 'sweep_direction':
            if value in ('low_to_high', 'high_to_low'):
                sanitized_updates['sweep_direction'] = value
            else:
                sanitized_updates['sweep_direction'] = 'low_to_high'
        else:
            sanitized_updates[key] = value
    sanitized_updates.setdefault('time_ranges', [])
    sanitized_updates.setdefault('sweep_direction', 'low_to_high')
    if 'time_s' not in sanitized_updates:
        sanitized_updates['time_s'] = config_updates.get('time_s')
    config['magnetic_sweep'] = sanitized_updates
    _save_config_data(config)
    logger.info("Saved magnetic sweep configuration")


def load_magnetic_sweep_config() -> dict:
    """Load magnetic sweep defaults, falling back to defaults."""
    config = _load_config_data()
    stored = config.get('magnetic_sweep', {})
    merged = DEFAULT_MAGNETIC_SWEEP_CONFIG.copy()
    merged['roi_map'] = DEFAULT_MAGNETIC_SWEEP_CONFIG['roi_map'].copy()
    merged['time_ranges'] = list(DEFAULT_MAGNETIC_SWEEP_CONFIG['time_ranges'])
    if stored:
        for key, value in stored.items():
            if key == 'roi_map' and isinstance(value, dict):
                merged['roi_map'].update(value)
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
    return merged


DEFAULT_WINDOW_RESOLUTION = "1200x800"


def save_window_resolution(resolution: str):
    """Save window resolution to the config file."""
    config = _load_config_data()
    config['window_resolution'] = resolution
    _save_config_data(config)
    logger.info(f"Saved window resolution: {resolution}")


def load_window_resolution() -> str:
    """Load window resolution, falling back to default."""
    config = _load_config_data()
    return config.get('window_resolution', DEFAULT_WINDOW_RESOLUTION)


def save_has_magnetic_polarization(has_mag_pol: bool):
    """Save whether dataset has magnetic/polarization data."""
    config = _load_config_data()
    config['has_magnetic_polarization'] = has_mag_pol
    _save_config_data(config)
    logger.info(f"Saved has_magnetic_polarization: {has_mag_pol}")


def load_has_magnetic_polarization() -> bool:
    """Load whether dataset has magnetic/polarization data, defaulting to True."""
    config = _load_config_data()
    return config.get('has_magnetic_polarization', True)  # Default to True for backward compatibility


def save_ui_scale(scale: float):
    """Save UI scaling factor to the config file."""
    config = _load_config_data()
    config['ui_scale'] = scale
    _save_config_data(config)
    logger.info(f"Saved UI scale: {scale}")


def load_ui_scale() -> float:
    """Load UI scaling factor, defaulting to 1.0 (100%)."""
    config = _load_config_data()
    return config.get('ui_scale', 1.0)
