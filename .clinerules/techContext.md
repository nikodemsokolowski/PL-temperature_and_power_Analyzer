# Tech Context: PL Analyzer

## Technologies Used
-   **Python 3**: The core programming language.
-   **customtkinter**: The UI framework for the desktop application.
-   **pandas**: For data manipulation and storage (DataFrames).
-   **NumPy**: For numerical operations, especially on arrays.
-   **Matplotlib**: For plotting and data visualization.
-   **SciPy**: For scientific computing, including curve fitting.

## Development Setup
-   A Python virtual environment is recommended to manage dependencies.
-   Dependencies are listed in `requirements.txt` and can be installed using `pip`.
-   The application is run from the root directory using `python run_app.py`.

## Technical Constraints
-   The application is a desktop application and is not web-based.
-   File parsing is dependent on a specific, though now more flexible, filename convention.

## Dependencies
-   `customtkinter`
-   `pandas`
-   `numpy`
-   `matplotlib`
-   `scipy`

## Tool Usage Patterns
-   The `DataHandler` class is used to manage all data, ensuring a single source of truth.
-   The `MainWindow` class acts as the central controller, orchestrating UI events and data processing.
-   `FileTable` and `PlotCanvas` are specialized widgets for their respective functions, keeping the UI modular.
