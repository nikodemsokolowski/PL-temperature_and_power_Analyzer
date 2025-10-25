# System Patterns: PL Analyzer

## System Architecture
The PL Analyzer is a monolithic desktop application built on Python. Its architecture follows a standard Model-View-Controller (MVC) pattern, though not strictly enforced.

-   **Model**: The `core` module (`data_handler.py`, `file_parser.py`, `processing.py`, `analysis.py`) is responsible for data management, business logic, and computations. It is decoupled from the user interface.
-   **View**: The `gui` module, built with `customtkinter`, defines the user interface. It consists of the main window (`main_window.py`) and various widgets (`widgets/`). The `PlotCanvas` widget encapsulates Matplotlib for plotting.
-   **Controller**: The `MainWindow` class in `main_window.py` acts as the primary controller, handling user input, invoking model functions, and updating the view.

## Key Technical Decisions
-   **GUI Framework**: `customtkinter` was chosen for its modern look and feel and its similarity to the standard `tkinter` library.
-   **Data Handling**: `pandas` DataFrames are used for managing tabular data (spectra and metadata), providing efficient data manipulation and analysis capabilities.
-   **Plotting**: `Matplotlib` is embedded within the `customtkinter` application for its power and flexibility in creating scientific plots.

## Component Relationships
-   `run_app.py`: The main entry point of the application. It initializes the `customtkinter` app and the `MainWindow`.
-   `MainWindow`: The central hub of the application. It creates and manages all other GUI components.
-   `DataHandler`: A central class for loading, storing, and providing access to the spectral data and metadata.
-   `FileTable`: A custom widget for displaying the list of loaded files and their metadata.
-   `PlotCanvas`: A custom widget for displaying the Matplotlib plots.

## Refactoring Guidelines
-   To maintain code quality and readability, files should be kept concise, ideally under 300 lines. This will be a consideration for future refactoring efforts as new features are added.
