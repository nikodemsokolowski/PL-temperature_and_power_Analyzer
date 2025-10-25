# Product Context: PL Analyzer

## Problem Space
The PL Analyzer application serves scientists and researchers who work with photoluminescence (PL) spectroscopy data. A common task in this field is to analyze how PL spectra change with experimental conditions like temperature and laser power. Manually extracting this information from filenames and processing the data is tedious and prone to error.

## Core Purpose
The application automates the analysis of temperature- and power-dependent PL data. It streamlines the workflow by:
1.  **Automating Metadata Extraction**: Parsing filenames to get experimental parameters.
2.  **Simplifying Data Processing**: Applying common corrections and normalizations in batches.
3.  **Providing Clear Visualizations**: Plotting spectra for easy comparison and analysis.
4.  **Enabling Quantitative Analysis**: Fitting data to physical models (e.g., power law) to extract key parameters.

## User Goals
-   Quickly load and visualize experimental data.
-   Avoid manual data entry and reduce errors.
-   Easily compare spectra from a series (power or temperature).
-   Perform power-law fits to determine exciton recombination mechanisms.
-   Export processed data for use in other software or publications.
