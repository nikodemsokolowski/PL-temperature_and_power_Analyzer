import re
import os
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Regex patterns based on the requirements (now case-insensitive)
# Temperature: _(\d+(\.\d+)?)[Kk] or _(\d+)-(\d+)[Kk] -> extract first number
TEMP_PATTERN = re.compile(r'_(\d+(?:\.\d+)?)-?\d*[Kk]', re.IGNORECASE) # 5K, 5.5K, 5-9K, 10k -> 5, 5.5, 5, 10
# Power: _(\d+(?:[p.]\d+)?)uW -> Handles 50uW, 0.5uW, 0p5uW, 10UW etc.
POWER_PATTERN = re.compile(r'_(\d+(?:[p.]\d+)?)uW', re.IGNORECASE)
# Acquisition Time: _(\d+(?:[p.]\d+)?)s -> Handles 1s, 0.5s, 0p5s, 2S etc.
TIME_PATTERN = re.compile(r'_(\d+(?:[p.]\d+)?)s', re.IGNORECASE)
# Grey Filter: _GF followed by optional numbers/p -> detect presence
GF_PATTERN = re.compile(r'_GF(\d+(?:[p.]\d+)?)?', re.IGNORECASE) # Allows GF, GF4, GF4p8, gf etc.

def parse_filename(filename: str) -> Dict[str, Optional[Any]]:
    """
    Parses a filename to extract experimental parameters.

    Args:
        filename: The base name of the file (without directory path).

    Returns:
        A dictionary containing the extracted parameters:
        'temperature_k': Temperature in Kelvin (float) or None.
        'power_uw': Laser power in microWatts (float) or None.
        'time_s': Acquisition time in seconds (float) or None.
        'gf_present': Boolean indicating if a Grey Filter pattern was found.
    """
    basename = os.path.basename(filename)
    params = {
        'temperature_k': None,
        'power_uw': None,
        'time_s': None,
        'gf_present': False,
    }

    # Extract Temperature
    temp_match = TEMP_PATTERN.search(basename)
    if temp_match:
        try:
            # Group 1 captures the first number before potential hyphen or K/k
            params['temperature_k'] = float(temp_match.group(1))
        except (ValueError, IndexError):
            logger.warning(f"Could not parse temperature from {basename}")

    # Extract Power
    power_match = POWER_PATTERN.search(basename)
    if power_match:
        try:
            power_str = power_match.group(1).replace('p', '.') # Replace 'p' with '.'
            params['power_uw'] = float(power_str)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse power from {basename}")

    # Extract Time
    time_match = TIME_PATTERN.search(basename)
    if time_match:
        try:
            time_str = time_match.group(1).replace('p', '.') # Replace 'p' with '.'
            params['time_s'] = float(time_str)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse time from {basename}")

    # Detect Grey Filter
    gf_match = GF_PATTERN.search(basename)
    if gf_match:
        params['gf_present'] = True

    return params

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
        "DecimalTest_13K_100p0uW_2s.csv", # Test 'p' decimal with integer part
        "CaseTest_14k_5UW_0p1S_gf.csv",   # Test case insensitivity
        "GFOnlyTest_15K_10uW_1s_GF.csv",  # Test GF without number
    ]

    logging.basicConfig(level=logging.INFO) # Basic config for testing

    for fname in test_filenames:
        extracted_params = parse_filename(fname)
        print(f"Filename: {fname}")
        print(f"Extracted Params: {extracted_params}\n")
