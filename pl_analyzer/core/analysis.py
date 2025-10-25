import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

def integrate_spectrum(energy_ev: np.ndarray, counts: np.ndarray, range_ev: Tuple[float, float]) -> Optional[float]:
    """
    Integrates the spectrum counts within a specified energy range.

    Args:
        energy_ev: Numpy array of energy values (eV).
        counts: Numpy array of corresponding counts.
        range_ev: A tuple containing the (min_energy, max_energy) for integration.

    Returns:
        The integrated intensity (area under the curve) within the range,
        or None if the range is invalid or no data points fall within it.
    """
    min_e, max_e = range_ev
    if min_e >= max_e:
        logger.warning(f"Invalid integration range: min_energy ({min_e}) >= max_energy ({max_e}).")
        return None

    # Ensure data is sorted by energy for correct slicing and integration
    sort_indices = np.argsort(energy_ev)
    energy_ev_sorted = energy_ev[sort_indices]
    counts_sorted = counts[sort_indices]

    # Find indices within the specified energy range
    integration_indices = np.where((energy_ev_sorted >= min_e) & (energy_ev_sorted <= max_e))[0]

    if len(integration_indices) < 2:
        logger.warning(f"Less than 2 data points found within the integration range [{min_e:.3f}, {max_e:.3f}] eV. Cannot integrate.")
        return None

    # Select the data within the range
    energy_range = energy_ev_sorted[integration_indices]
    counts_range = counts_sorted[integration_indices]

    try:
        # --- Baseline Correction ---
        # Find the minimum count value in the range and subtract it.
        # This ensures the integration is performed on the peak signal above background.
        baseline = np.min(counts_range)
        counts_corrected = counts_range - baseline
        
        # Perform trapezoidal integration on the baseline-corrected counts
        integrated_intensity = np.trapz(counts_corrected, energy_range)
        
        print(f"Integrated intensity over [{min_e:.3f}, {max_e:.3f}] eV: {integrated_intensity}")
        logger.debug(f"Integrated intensity over [{min_e:.3f}, {max_e:.3f}] eV: {integrated_intensity}")
        return integrated_intensity
    except Exception as e:
        logger.error(f"Error during integration over [{min_e:.3f}, {max_e:.3f}] eV: {e}", exc_info=True)
        return None


def calculate_centroid(
    energy_ev: np.ndarray,
    counts: np.ndarray,
    energy_range: Optional[Tuple[float, float]] = None
) -> Optional[float]:
    """
    Calculate the intensity-weighted centroid (center of mass) of a spectrum.

    Args:
        energy_ev: Array of energy values (eV).
        counts: Array of corresponding counts.
        energy_range: Optional tuple (min_energy, max_energy) in eV restricting the
            centroid calculation. If None, the full spectrum is used.

    Returns:
        Centroid energy in eV, or None if the calculation is not possible.
    """
    if energy_ev.size == 0 or counts.size == 0:
        logger.warning("Centroid calculation skipped: empty spectrum.")
        return None

    energy = np.asarray(energy_ev, dtype=float)
    intensity = np.asarray(counts, dtype=float)

    if energy_range is not None:
        min_e, max_e = energy_range
        if min_e >= max_e:
            logger.warning("Centroid calculation skipped: invalid energy window.")
            return None
        mask = (energy >= min_e) & (energy <= max_e)
        if not np.any(mask):
            logger.warning("Centroid calculation skipped: no points inside requested range.")
            return None
        energy = energy[mask]
        intensity = intensity[mask]

    if energy.size == 0 or intensity.size == 0:
        return None

    total_intensity = np.sum(intensity)
    if total_intensity <= 0:
        logger.warning("Centroid calculation skipped: non-positive total intensity.")
        return None

    centroid = float(np.sum(energy * intensity) / total_intensity)
    logger.debug(f"Calculated centroid at {centroid:.6f} eV.")
    return centroid


def bfield_intensity_analysis(dataset, integration_range: Tuple[float, float], polarization_mode: str) -> pd.DataFrame:
    """
    Integrate spectra over the supplied range for each B-field value.

    Args:
        dataset: Active Dataset instance providing spectra and metadata.
        integration_range: (min_e, max_e) tuple describing the integration window.
        polarization_mode: One of 'all', 'sigma+', 'sigma-', 'both', 'sum'.

    Returns:
        DataFrame with columns: bfield_t, intensity_sigma_plus, intensity_sigma_minus, intensity_sum.
    """
    metadata_df = dataset.get_metadata()
    if metadata_df is None or metadata_df.empty:
        logger.info("B-field intensity analysis aborted: no metadata available.")
        return pd.DataFrame(columns=['bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum'])

    if 'bfield_t' not in metadata_df.columns:
        logger.info("B-field intensity analysis aborted: dataset has no B-field information.")
        return pd.DataFrame(columns=['bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum'])

    valid_md = metadata_df.dropna(subset=['bfield_t'])
    if valid_md.empty:
        logger.info("B-field intensity analysis aborted: no spectra with B-field annotations.")
        return pd.DataFrame(columns=['bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum'])

    records = []
    for _, row in valid_md.iterrows():
        file_id = row.get('file_id')
        if not file_id:
            continue
        spectrum = dataset.get_processed_spectrum(file_id)
        if spectrum is None or spectrum.empty:
            continue
        intensity = integrate_spectrum(
            spectrum['energy_ev'].values,
            spectrum['counts'].values,
            integration_range
        )
        if intensity is None:
            continue
        records.append({
            'bfield_t': row.get('bfield_t'),
            'polarization': row.get('polarization'),
            'intensity': intensity
        })

    if not records:
        logger.info("B-field intensity analysis produced no valid integrals.")
        return pd.DataFrame(columns=['bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum'])

    intensity_df = pd.DataFrame(records)
    results = []
    for bfield, group in intensity_df.groupby('bfield_t'):
        plus_vals = group.loc[group['polarization'] == 'sigma+', 'intensity']
        minus_vals = group.loc[group['polarization'] == 'sigma-', 'intensity']

        plus_mean = float(plus_vals.mean()) if not plus_vals.empty else np.nan
        minus_mean = float(minus_vals.mean()) if not minus_vals.empty else np.nan
        sum_val = plus_mean + minus_mean if not np.isnan(plus_mean) and not np.isnan(minus_mean) else np.nan

        results.append({
            'bfield_t': float(bfield),
            'intensity_sigma_plus': plus_mean,
            'intensity_sigma_minus': minus_mean,
            'intensity_sum': sum_val
        })

    if not results:
        return pd.DataFrame(columns=['bfield_t', 'intensity_sigma_plus', 'intensity_sigma_minus', 'intensity_sum'])

    result_df = pd.DataFrame(results).sort_values('bfield_t').reset_index(drop=True)

    mode = polarization_mode or 'all'
    if mode == 'sigma+':
        result_df = result_df.dropna(subset=['intensity_sigma_plus'])
    elif mode == 'sigma-':
        result_df = result_df.dropna(subset=['intensity_sigma_minus'])
    elif mode == 'sum':
        result_df = result_df.dropna(subset=['intensity_sum'])
    elif mode == 'both':
        # Retain rows with at least one polarized component; prefer entries containing both
        both_mask = (~result_df['intensity_sigma_plus'].isna()) | (~result_df['intensity_sigma_minus'].isna())
        result_df = result_df[both_mask]

    logger.debug(f"B-field intensity dataframe created with {len(result_df)} rows for mode '{mode}'.")
    return result_df.reset_index(drop=True)


def calculate_dcp(intensity_plus: np.ndarray, intensity_minus: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute the degree of circular polarization (DCP).

    DCP is defined as (I_sigma_plus - I_sigma_minus) / (I_sigma_plus + I_sigma_minus).
    A small epsilon is added to the denominator to avoid division by zero.

    Args:
        intensity_plus: Array-like of σ+ integrated intensities.
        intensity_minus: Array-like of σ- integrated intensities.
        epsilon: Small value to stabilise the division.

    Returns:
        NumPy array of DCP values (float) with NaN where calculation is not possible.
    """
    plus = np.asarray(intensity_plus, dtype=float)
    minus = np.asarray(intensity_minus, dtype=float)

    if plus.shape != minus.shape:
        raise ValueError("Intensity arrays for σ+ and σ- must share the same shape.")

    numerator = plus - minus
    denominator = plus + minus

    dcp = np.full_like(denominator, np.nan, dtype=float)
    finite_mask = np.isfinite(numerator) & np.isfinite(denominator)
    safe_mask = finite_mask & (np.abs(denominator) > epsilon)
    dcp[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    return dcp


def _apply_moving_average(counts: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a simple moving average smoothing to the counts array."""
    if window_size <= 1:
        return counts
    window = max(int(window_size), 1)
    if window % 2 == 0:
        window += 1
    if window >= counts.size:
        window = counts.size - 1 if counts.size % 2 == 0 else counts.size
        if window <= 1:
            return counts
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(counts, kernel, mode='same')


def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def _fit_gaussian_peak(energy_ev: np.ndarray, counts: np.ndarray) -> Optional[float]:
    if energy_ev.size < 3 or counts.size < 3:
        return None

    try:
        amplitude_guess = float(np.nanmax(counts))
        if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
            return None
        max_idx = int(np.nanargmax(counts))
        center_guess = float(energy_ev[max_idx])
        sigma_guess = float(np.std(energy_ev)) / 6.0 if energy_ev.size > 1 else 1e-3
        sigma_guess = max(sigma_guess, 1e-4)
        lower_bounds = [0.0, float(np.min(energy_ev)), 1e-6]
        upper_bounds = [np.inf, float(np.max(energy_ev)), np.inf]
        popt, _ = curve_fit(
            _gaussian,
            energy_ev,
            counts,
            p0=[amplitude_guess, center_guess, sigma_guess],
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000
        )
        return float(popt[1])
    except Exception as exc:
        logger.debug(f"Gaussian peak fit failed: {exc}")
        return None


def detect_peak_energy(
    energy_ev: np.ndarray,
    counts: np.ndarray,
    method: str = "centroid"
) -> Optional[float]:
    """
    Determine the characteristic peak energy for a spectrum using the requested method.

    Args:
        energy_ev: Energy axis in eV.
        counts: Counts/intensity axis.
        method: One of {'max', 'centroid', 'gaussian'}.

    Returns:
        Peak energy in eV or None if detection fails.
    """
    method = (method or "centroid").lower()
    energy = np.asarray(energy_ev, dtype=float)
    intensity = np.asarray(counts, dtype=float)

    valid_mask = np.isfinite(energy) & np.isfinite(intensity)
    if not np.any(valid_mask):
        return None
    energy = energy[valid_mask]
    intensity = intensity[valid_mask]
    if energy.size == 0 or intensity.size == 0:
        return None

    if method == "max":
        idx = int(np.argmax(intensity))
        return float(energy[idx])
    if method == "centroid":
        return calculate_centroid(energy, intensity)
    if method == "gaussian":
        return _fit_gaussian_peak(energy, intensity)

    raise ValueError(f"Unknown peak detection method '{method}'.")


def extract_bfield_peak_energies(
    dataset,
    energy_range: Optional[Tuple[float, float]],
    method: str = "centroid",
    smoothing_points: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract characteristic peak energies for sigma+/sigma- spectra across B-field.

    Args:
        dataset: Active dataset instance.
        energy_range: Tuple specifying (min_e, max_e) or None for full spectrum.
        method: Peak detection strategy ('max', 'centroid', 'gaussian').
        smoothing_points: Optional integer window size for moving-average smoothing.

    Returns:
        DataFrame with columns: bfield_t, polarization, peak_energy_ev, file_id, filename.
    """
    metadata_df = dataset.get_metadata()
    if metadata_df is None or metadata_df.empty:
        logger.info("Peak extraction aborted: dataset metadata unavailable.")
        return pd.DataFrame(columns=['bfield_t', 'polarization', 'peak_energy_ev', 'file_id', 'filename'])

    if 'bfield_t' not in metadata_df.columns:
        logger.info("Peak extraction aborted: no B-field annotations present.")
        return pd.DataFrame(columns=['bfield_t', 'polarization', 'peak_energy_ev', 'file_id', 'filename'])

    valid_md = metadata_df.dropna(subset=['bfield_t'])
    if valid_md.empty:
        logger.info("Peak extraction aborted: metadata lacks B-field values.")
        return pd.DataFrame(columns=['bfield_t', 'polarization', 'peak_energy_ev', 'file_id', 'filename'])

    smoothing = int(smoothing_points) if smoothing_points else None
    records: List[Dict[str, object]] = []

    for _, row in valid_md.iterrows():
        polarization = row.get('polarization')
        if polarization not in ('sigma+', 'sigma-'):
            continue
        file_id = row.get('file_id')
        spectrum = dataset.get_processed_spectrum(file_id)
        if spectrum is None or spectrum.empty:
            continue

        energy = spectrum['energy_ev'].to_numpy(dtype=float)
        intensity = spectrum['counts'].to_numpy(dtype=float)

        if energy_range is not None:
            min_e, max_e = energy_range
            if min_e >= max_e:
                continue
            mask = (energy >= min_e) & (energy <= max_e)
            if not np.any(mask):
                continue
            energy = energy[mask]
            intensity = intensity[mask]

        if energy.size == 0 or intensity.size == 0:
            continue

        if smoothing and smoothing > 1:
            intensity = _apply_moving_average(intensity, smoothing)

        peak_energy = detect_peak_energy(energy, intensity, method=method)
        if peak_energy is None:
            continue

        records.append({
            'bfield_t': float(row.get('bfield_t')),
            'polarization': polarization,
            'peak_energy_ev': float(peak_energy),
            'file_id': row.get('file_id'),
            'filename': row.get('filename'),
            'method': method,
            'smoothing_points': smoothing if smoothing else 0
        })

    if not records:
        logger.info("Peak extraction produced no results.")
        return pd.DataFrame(columns=['bfield_t', 'polarization', 'peak_energy_ev', 'file_id', 'filename', 'method', 'smoothing_points'])

    return pd.DataFrame(records)


def prepare_zeeman_dataframe(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct DataFrame containing sigma+/sigma- energies and Zeeman splitting per B-field.

    Args:
        peaks_df: Output of extract_bfield_peak_energies.

    Returns:
        DataFrame with columns: bfield_t, sigma_plus_ev, sigma_minus_ev, delta_mev.
    """
    if peaks_df is None or peaks_df.empty:
        return pd.DataFrame(columns=['bfield_t', 'sigma_plus_ev', 'sigma_minus_ev', 'delta_mev'])

    pivot = peaks_df.pivot_table(
        index='bfield_t',
        columns='polarization',
        values='peak_energy_ev',
        aggfunc=np.mean
    )
    pivot = pivot.rename(columns={'sigma+': 'sigma_plus_ev', 'sigma-': 'sigma_minus_ev'})
    pivot = pivot.reset_index()

    if 'sigma_plus_ev' not in pivot.columns or 'sigma_minus_ev' not in pivot.columns:
        logger.info("Incomplete sigma+/sigma- pairing. Cannot compute delta energies.")
        return pd.DataFrame(columns=['bfield_t', 'sigma_plus_ev', 'sigma_minus_ev', 'delta_mev'])

    pivot['delta_mev'] = (pivot['sigma_plus_ev'] - pivot['sigma_minus_ev']) * 1000.0
    return pivot.sort_values('bfield_t').reset_index(drop=True)


def calculate_g_factor(
    bfield_t: np.ndarray,
    centroid_plus: np.ndarray,
    centroid_minus: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Fit Zeeman splitting ΔE(B) = g * μ_B * B and extract the g-factor.

    Args:
        bfield_t: Array of magnetic field values (Tesla).
        centroid_plus: Array of centroid energies (eV) for σ+ polarization.
        centroid_minus: Array of centroid energies (eV) for σ- polarization.

    Returns:
        Dictionary with g-factor, uncertainties, fit data, or None if fitting fails.
    """
    if len(bfield_t) == 0:
        logger.warning("g-factor calculation aborted: no B-field data supplied.")
        return None

    bfield = np.asarray(bfield_t, dtype=float)
    e_plus = np.asarray(centroid_plus, dtype=float)
    e_minus = np.asarray(centroid_minus, dtype=float)
    valid_mask = (~np.isnan(bfield)) & (~np.isnan(e_plus)) & (~np.isnan(e_minus))
    valid_mask &= (~np.isinf(bfield)) & (~np.isinf(e_plus)) & (~np.isinf(e_minus))

    bfield_valid = bfield[valid_mask]
    e_plus_valid = e_plus[valid_mask]
    e_minus_valid = e_minus[valid_mask]

    if bfield_valid.size < 2:
        logger.warning("g-factor calculation aborted: need at least two matching σ+/σ- centroids.")
        return None

    delta_mev = (e_plus_valid - e_minus_valid) * 1000.0  # Convert eV to meV
    mu_b = 0.05788  # meV/T

    try:
        coeffs, cov = np.polyfit(bfield_valid, delta_mev, 1, cov=True)
    except Exception as exc:
        logger.error(f"Linear fit for g-factor failed: {exc}", exc_info=True)
        return None

    slope = coeffs[0]
    intercept = coeffs[1]
    slope_unc = float(np.sqrt(cov[0, 0])) if cov is not None and cov.size >= 1 else np.nan
    intercept_unc = float(np.sqrt(cov[1, 1])) if cov is not None and cov.shape[0] > 1 else np.nan

    g_factor = slope / mu_b
    g_uncertainty = slope_unc / mu_b if not np.isnan(slope_unc) else np.nan

    fit_delta = slope * bfield_valid + intercept
    ss_res = float(np.sum((delta_mev - fit_delta) ** 2))
    ss_tot = float(np.sum((delta_mev - np.mean(delta_mev)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    logger.info(f"g-factor fit complete: g = {g_factor:.4f} ± {g_uncertainty:.4f}, R² = {r_squared:.4f}")

    residuals = delta_mev - fit_delta

    return {
        'g_factor': g_factor,
        'g_uncertainty': g_uncertainty,
        'intercept_mev': intercept,
        'intercept_uncertainty': intercept_unc,
        'r_squared': r_squared,
        'bfield_t': bfield_valid,
        'delta_mev': delta_mev,
        'fit_delta_mev': fit_delta,
        'residuals_mev': residuals
    }
# --- Power Law Fitting (Placeholder for Step 11) ---

def power_law(p, a, k):
    """Power law function: I = a * P^k"""
    # Ensure inputs are positive for potential log operations or power calculations
    epsilon = 1e-10
    p_safe = np.maximum(p, epsilon)
    return a * p_safe**k

def fit_power_law(power_uw: np.ndarray, integrated_intensity: np.ndarray, fixed_params: Optional[Dict[str, float]] = None) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    Fits integrated intensity vs. power data to I = a * P^k using curve_fit.

    Args:
        power_uw: Array of laser power values (uW). Must be > 0.
        integrated_intensity: Array of corresponding integrated intensities. Must be > 0.
        fixed_params: Dictionary specifying parameters to fix (e.g., {'k': 1.0}).
                    Valid keys are 'a' or 'k'.

    Returns:
        A tuple containing:
        - Dictionary of fitted parameters {'a': value, 'k': value} or None if fit fails.
        - Dictionary of parameter uncertainties (std dev) {'a': std_a, 'k': std_k} or None.
    """
    if fixed_params is None:
        fixed_params = {}

    # --- Input Validation ---
    if len(power_uw) != len(integrated_intensity):
        logger.error("Power and Intensity arrays must have the same length.")
        return None, None
    if len(power_uw) < 2:
        logger.error("At least two data points are required for fitting.")
        return None, None
    if np.any(power_uw <= 0) or np.any(integrated_intensity <= 0):
        logger.warning("Power and Intensity values must be positive for power law fitting. Skipping fit.")
        # Consider filtering out non-positive points if appropriate, but for now, fail.
        return None, None

    # --- Parameter Fixing Logic ---
    fixed_a = fixed_params.get('a')
    fixed_k = fixed_params.get('k')

    params_to_fit = []
    initial_guess = []
    param_names = ['a', 'k']
    bounds = ([0, -np.inf], [np.inf, np.inf]) # Bounds: a > 0, k can be anything

    if fixed_a is not None and fixed_k is not None:
        logger.error("Cannot fix both 'a' and 'k' simultaneously.")
        return None, None
    elif fixed_a is not None:
        if fixed_a <= 0:
            logger.error("Fixed parameter 'a' must be positive.")
            return None, None
        # Fit only k
        def fit_func(p, k_fit):
            return power_law(p, fixed_a, k_fit)
        params_to_fit = ['k']
        initial_guess = [1.0] # Initial guess for k
        bounds = ([-np.inf], [np.inf])
    elif fixed_k is not None:
        # Fit only a
        def fit_func(p, a_fit):
            return power_law(p, a_fit, fixed_k)
        params_to_fit = ['a']
        initial_guess = [np.mean(integrated_intensity)] # Initial guess for a
        bounds = ([0], [np.inf])
    else:
        # Fit both a and k
        fit_func = power_law
        params_to_fit = ['a', 'k']
        # Simple initial guesses - might need refinement
        initial_guess = [np.mean(integrated_intensity) / np.mean(power_uw), 1.0] # Adjusted initial guess for 'a'
        bounds = ([0, -np.inf], [np.inf, np.inf])


    # --- Perform Fit ---
    try:
        popt, pcov = curve_fit(
            fit_func,
            power_uw,
            integrated_intensity,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000 # Increase max iterations if needed
        )

        # Extract results and uncertainties
        fitted_params = {}
        param_errors = {}
        perr = np.sqrt(np.diag(pcov)) # Standard deviations from covariance matrix

        fit_idx = 0
        for i, name in enumerate(param_names):
            if name in params_to_fit:
                fitted_params[name] = popt[fit_idx]
                param_errors[name] = perr[fit_idx]
                fit_idx += 1
            else: # Parameter was fixed
                fitted_params[name] = fixed_params[name]
                param_errors[name] = 0.0 # No uncertainty for fixed params

        logger.info(f"Power law fit successful. Params: {fitted_params}, Errors: {param_errors}")
        return fitted_params, param_errors

    except RuntimeError as e:
        logger.error(f"Optimal parameters not found during curve_fit: {e}")
        return None, None
    except ValueError as e:
        logger.error(f"Error during curve_fit (likely bounds or input data issue): {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during power law fitting: {e}", exc_info=True)
        return None, None

def arrhenius_1_exp(T, I0, Ea):
    """Arrhenius model with 1 exponential term."""
    k_B = 8.617333262145e-5 # Boltzmann constant in eV/K
    return I0 * np.exp(-Ea / (k_B * T))

def arrhenius_2_exp(T, I0, C1, Ea1, C2, Ea2):
    """Arrhenius model with 2 exponential terms."""
    k_B = 8.617333262145e-5 # Boltzmann constant in eV/K
    return I0 / (1 + C1 * np.exp(-Ea1 / (k_B * T)) + C2 * np.exp(-Ea2 / (k_B * T)))

def arrhenius_fit_1_exp(temperature, intensity):
    """
    Fits temperature-dependent intensity data to a single exponential Arrhenius model.
    """
    try:
        popt, pcov = curve_fit(arrhenius_1_exp, temperature, intensity, p0=[max(intensity), 10], bounds=([0, 0], [np.inf, np.inf]))
        return popt, pcov
    except RuntimeError:
        return None, None

def arrhenius_fit_2_exp(temperature, intensity, initial_guesses=None, fixed_params=None):
    """
    Fits temperature-dependent intensity data to a double exponential Arrhenius model.
    """
    if initial_guesses is None:
        initial_guesses = {}
    if fixed_params is None:
        fixed_params = {}

    param_names = ["I0", "C1", "Ea1", "C2", "Ea2"]
    
    # Start with default guesses and bounds
    p0 = [max(intensity), 1, 0.01, 1, 0.1]
    bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

    # Overwrite with user-provided initial guesses
    for i, name in enumerate(param_names):
        if name in initial_guesses:
            p0[i] = initial_guesses[name]

    # Adjust bounds for fixed parameters
    for i, name in enumerate(param_names):
        if name in fixed_params:
            bounds[0][i] = fixed_params[name]
            bounds[1][i] = fixed_params[name]
            p0[i] = fixed_params[name] # Also set the initial guess to the fixed value

    try:
        popt, pcov = curve_fit(arrhenius_2_exp, temperature, intensity, p0=p0, bounds=bounds)
        return popt, pcov
    except RuntimeError:
        return None, None


def arrhenius_fit(temperature, intensity):
    """
    Fits the temperature-dependent intensity data to the Arrhenius equation
    with one activation energy.

    I = I0 / (1 + C * exp(-Ea / (k_B * T)))
    """
    # Filter out non-positive data points
    valid_mask = intensity > 0
    if np.sum(valid_mask) < 3: # Need at least 3 points to fit 3 parameters
        logger.warning("Not enough positive data points to perform Arrhenius fit.")
        return None, None
    
    temp_fit = temperature[valid_mask]
    intensity_fit = intensity[valid_mask]

    def arrhenius_model(T, I0, C, Ea):
        k_B = 8.617333262145e-5 # Boltzmann constant in eV/K
        return I0 / (1 + C * np.exp(-Ea / (k_B * T)))

    try:
        # Provide reasonable initial guesses and bounds
        p0 = [np.max(intensity_fit), 1.0, 0.1]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(arrhenius_model, temp_fit, intensity_fit, p0=p0, bounds=bounds)
        return popt, pcov
    except RuntimeError:
        return None, None

def arrhenius_fit_double(temperature, intensity):
    """
    Fits the temperature-dependent intensity data to the Arrhenius equation
    with two activation energies.

    I = I0 / (1 + C1 * exp(-Ea1 / (k_B * T)) + C2 * exp(-Ea2 / (k_B * T)))
    """
    # Filter out non-positive data points
    valid_mask = intensity > 0
    if np.sum(valid_mask) < 5: # Need at least 5 points to fit 5 parameters
        logger.warning("Not enough positive data points to perform double Arrhenius fit.")
        return None, None

    temp_fit = temperature[valid_mask]
    intensity_fit = intensity[valid_mask]

    def arrhenius_model_double(T, I0, C1, Ea1, C2, Ea2):
        k_B = 8.617333262145e-5 # Boltzmann constant in eV/K
        return I0 / (1 + C1 * np.exp(-Ea1 / (k_B * T)) + C2 * np.exp(-Ea2 / (k_B * T)))

    try:
        # Provide reasonable initial guesses and bounds
        p0 = [np.max(intensity_fit), 1.0, 0.01, 1.0, 0.1]
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(arrhenius_model_double, temp_fit, intensity_fit, p0=p0, bounds=bounds)
        return popt, pcov
    except RuntimeError:
        return None, None


# Example usage (for testing integration)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Create dummy spectrum data
    test_energy = np.linspace(1.5, 2.5, 101) # 101 points for 100 intervals
    test_counts = np.exp(-(test_energy - 2.0)**2 / 0.05) * 1000 # Gaussian peak at 2.0 eV

    print("--- Testing Integration ---")
    range1 = (1.8, 2.2)
    integral1 = integrate_spectrum(test_energy, test_counts, range1)
    print(f"Integral over [{range1[0]}, {range1[1]}] eV: {integral1}")

    range2 = (2.1, 2.4)
    integral2 = integrate_spectrum(test_energy, test_counts, range2)
    print(f"Integral over [{range2[0]}, {range2[1]}] eV: {integral2}")

    range3 = (2.6, 2.7) # Outside peak
    integral3 = integrate_spectrum(test_energy, test_counts, range3)
    print(f"Integral over [{range3[0]}, {range3[1]}] eV: {integral3}")

    range4 = (2.1, 1.9) # Invalid range
    integral4 = integrate_spectrum(test_energy, test_counts, range4)
    print(f"Integral over [{range4[0]}, {range4[1]}] eV: {integral4}")

    # Test with unsorted data
    unsorted_indices = np.random.permutation(len(test_energy))
    unsorted_energy = test_energy[unsorted_indices]
    unsorted_counts = test_counts[unsorted_indices]
    integral5 = integrate_spectrum(unsorted_energy, unsorted_counts, range1)
    print(f"Integral over [{range1[0]}, {range1[1]}] eV (unsorted input): {integral5}")

    print("\n--- Integration Test Complete ---")


    # --- Test Power Law Fitting ---
    print("\n--- Testing Power Law Fitting ---")
    # Generate synthetic data: I = 5 * P^1.2 + noise
    test_power = np.array([1, 5, 10, 20, 50, 100], dtype=float)
    true_a = 5.0
    true_k = 1.2
    noise = np.random.normal(0, 0.1 * power_law(test_power, true_a, true_k), size=len(test_power))
    test_intensity = power_law(test_power, true_a, true_k) + noise
    test_intensity = np.maximum(test_intensity, 1e-5) # Ensure positive intensity

    print(f"Test Data (Power): {test_power}")
    print(f"Test Data (Intensity): {test_intensity}")

    # Test fitting both parameters
    print("\nFitting both a and k:")
    params_both, errors_both = fit_power_law(test_power, test_intensity)
    if params_both:
        print(f"  Fitted Params: {params_both}")
        print(f"  Param Errors: {errors_both}")

    # Test fitting with fixed k
    print("\nFitting a with fixed k=1.1:")
    params_fix_k, errors_fix_k = fit_power_law(test_power, test_intensity, fixed_params={'k': 1.1})
    if params_fix_k:
        print(f"  Fitted Params: {params_fix_k}")
        print(f"  Param Errors: {errors_fix_k}")

    # Test fitting with fixed a
    print("\nFitting k with fixed a=4.5:")
    params_fix_a, errors_fix_a = fit_power_law(test_power, test_intensity, fixed_params={'a': 4.5})
    if params_fix_a:
        print(f"  Fitted Params: {params_fix_a}")
        print(f"  Param Errors: {errors_fix_a}")

    # Test with invalid input (non-positive)
    print("\nFitting with non-positive intensity:")
    test_intensity_invalid = test_intensity.copy()
    test_intensity_invalid[0] = -1.0
    params_invalid, errors_invalid = fit_power_law(test_power, test_intensity_invalid)
    print(f"  Result: Params={params_invalid}, Errors={errors_invalid}")

    print("\n--- Power Law Fitting Test Complete ---")
