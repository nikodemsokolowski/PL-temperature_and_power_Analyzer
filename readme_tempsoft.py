import os

# --- README Content ---
# Using a multiline string to hold the entire Markdown content.
# This makes it easy to read and edit.

readme_content = """
# PL Analyzer

PL Analyzer is a desktop application designed for the analysis of photoluminescence (PL) spectroscopy data. It provides a graphical user interface to streamline the process of loading, processing, visualizing, and analyzing temperature- and power-dependent PL spectra, which is common in physics and materials science research.

![PL Analyzer Screenshot](image_7d789a.png)

The core analysis feature allows for robust power-law fitting on a log-log scale to determine exciton recombination mechanisms.

![Power Law Fitting Screenshot](fig2.png)

## 🔬 Core Features

* **Automatic Metadata Parsing**: Extracts experimental parameters (temperature, laser power, acquisition time) directly from filenames, eliminating manual data entry.
* **Batch Data Processing**: Apply processing steps to all loaded files simultaneously.
    * **Time Normalization**: Normalize signal counts by acquisition time (counts/s).
    * **Grey Filter Correction**: Rescale data from files taken with a neutral density (grey) filter.
    * **Spectrometer Response Correction**: Apply predefined correction factors based on acquisition time.
* **Interactive Visualization**:
    * Plot individual spectra or compare multiple datasets.
    * Automatically plot entire **power-dependent** or **temperature-dependent** series with a single click.
    * Toggle between linear and **logarithmic scales** for the Y-axis to easily view features at different intensity levels.
    * Standard plot controls (zoom, pan, save) via the Matplotlib toolbar.
* **Quantitative Analysis**:
    * **Spectral Integration**: Calculate the integrated intensity of selected spectra over a user-defined energy range.
    * **Power Dependence Analysis**: The primary analysis workflow. Automatically integrate a full power series over a specified range, plot the integrated intensity vs. laser power on a **log-log plot**, and fit the data to a power law model ($I = a \\cdot P^k$) to determine the exponent `k`.
* **Data Export**:
    * Export processed spectra (Energy vs. Counts) to a tab-separated file.
    * Export integrated intensity vs. power data for a specific series, ready for external plotting or analysis.

## 🚀 Getting Started

You can run the application from the source code.

### Prerequisites

* Python 3.8 or newer
* The required Python packages listed in `requirements.txt`

### Installation & Running from Source

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # On Windows
    python -m venv venv
    .\\venv\\Scripts\\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

### Dependencies

The application relies on the following libraries. A `requirements.txt` file should be included in your repository.

```
customtkinter
pandas
numpy
matplotlib
scipy
```

## 📖 How to Use

### 1. File Naming Convention

For the automatic metadata parsing to work, your data files **must** follow a specific naming convention. The parser is flexible but expects parameters to be delimited by underscores (`_`). **Decimals must be indicated with a 'p'**.

**Template:** `Prefix_<Temperature>K_<Power>uW_<Time>s_GF<optional>.csv`

**Component Breakdown:**
* **Prefix**: Any text that does not contain the parameter keys (e.g., `WSe2_SampleA`, `Data`).
* `<Temperature>K`: The temperature followed by `K`. For decimals, use 'p' (e.g., `5p5K` for 5.5 K).
* `<Power>uW`: The laser power followed by `uW`. For decimals, use 'p' (e.g., `0p1uW` for 0.1 uW).
* `<Time>s`: The acquisition time followed by `s`. For decimals, use 'p' (e.g., `0p5s` for 0.5 s).
* `GF`: An optional tag indicating a grey filter was used. The number following `GF` is ignored; its presence is what matters.

**Examples of valid filenames:**
* `WSe2_H5_K6_run1_5K_10uW_0p1s.csv`
* `Data_100p5K_50uW_1s_GF1.csv`
* `measurement_20K_0p2uW_2p5s.dat`

### 2. Workflow

1.  **Load Data**: Click the **"Load Files"** button and select all the data files for your experiment. They will appear in the table on the left.
2.  **Process Data**: In the "Processing Steps" panel, apply corrections as needed. It's recommended to apply them in order:
    1.  **Normalize by Time**: To get counts per second.
    2.  **Rescale by GF Factor**: If you used a grey filter. A dialog will ask for the transmission factor (e.g., 0.1 for a 10% filter).
    3.  **Correct Spectrometer Response**: If you have known correction factors for your setup.
4.  **Plot Spectra**:
    * Select one or more files in the table and click **"Plot Selected"**.
    * To see a full series, select any file from that series and click **"Plot Power Series"** or **"Plot Temp Series"**.
    * Use the **"Log Y-Axis"** checkbox to toggle the scale.
5.  **Analyze Power Dependence**:
    1.  Enter the **Min E (eV)** and **Max E (eV)** for your peak of interest.
    2.  Select **one** file from the temperature series you want to analyze.
    3.  Click **"Power Dependence Analysis"**.
    4.  A new window will open showing the **integrated intensity vs. power** on a log-log plot. Use the "Fit Power Law" button in this new window to perform the fit and extract the exponent `k`.

## 🖥️ Downloads

Pre-compiled executables for Windows can be made available on the [**Releases page**](https://github.com/your-username/your-repo-name/releases). It is recommended to distribute large `.exe` files this way rather than committing them to the repository.
"""

def create_readme_file():
    """Creates a README.md file in the current directory with the specified content."""
    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("Successfully created README.md")
        print(f"File saved in: {os.path.abspath('README.md')}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    # Make sure you have your images 'image_7d789a.png' and 'fig2.png'
    # in the same directory as this script before running it.
    create_readme_file()

