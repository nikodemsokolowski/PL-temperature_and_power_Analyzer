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
        # Perform trapezoidal integration on the absolute counts
        integrated_intensity = np.trapz(np.abs(counts_range), energy_range)
        print(f"Integrated intensity over [{min_e:.3f}, {max_e:.3f}] eV: {integrated_intensity}")
        logger.debug(f"Integrated intensity over [{min_e:.3f}, {max_e:.3f}] eV: {integrated_intensity}")
        return integrated_intensity
    except Exception as e:
        logger.error(f"Error during integration over [{min_e:.3f}, {max_e:.3f}] eV: {e}", exc_info=True)
        return None

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
