import customtkinter as ctk
import logging
import matplotlib
matplotlib.use('TkAgg') # Necessary backend for tkinter embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

logger = logging.getLogger(__name__)

class PlotCanvas(ctk.CTkFrame):
    """
    A CustomTkinter frame containing an embedded Matplotlib plot and toolbar.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create Matplotlib Figure and Axes
        # Adjust dpi for clarity if needed
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Intensity (arb. units)")
        self.figure.tight_layout() # Adjust layout to prevent labels overlapping

        # Create Canvas to display the figure
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Create Matplotlib Navigation Toolbar
        # Frame to hold the toolbar (optional, for better layout control)
        self.toolbar_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Initial plot setup (e.g., empty plot)
        self.axes.plot([], []) # Plot empty data initially
        self.canvas.draw()
        logger.debug("PlotCanvas initialized.")

    def set_yscale(self, scale='linear'):
        """Sets the Y-axis scale ('linear' or 'log')."""
        try:
            self.axes.set_yscale(scale)
            self.canvas.draw()
            logger.debug(f"Plot Y-axis scale set to {scale}")
        except ValueError as e:
            logger.error(f"Invalid scale value '{scale}': {e}")
        except Exception as e:
            logger.error(f"Error setting Y-axis scale: {e}", exc_info=True)


    def plot_data(self, energy_ev_list, counts_list, labels_list=None, title=None, y_scale='linear'):
        """
        Clears the current plot and plots new data.

        Args:
            energy_ev_list: A list of numpy arrays or lists containing energy values (eV).
            counts_list: A list of numpy arrays or lists containing intensity counts.
                        Must have the same length as energy_ev_list.
            labels_list: An optional list of labels for each dataset.
            title: An optional title for the plot.
        """
        self.axes.clear() # Clear previous plot content

        if not energy_ev_list or not counts_list or len(energy_ev_list) != len(counts_list):
            logger.warning("Invalid data provided for plotting. Plotting empty.")
            self.axes.plot([], []) # Plot empty if data is invalid
        else:
            num_plots = len(energy_ev_list)
            use_labels = labels_list is not None and len(labels_list) == num_plots

            for i in range(num_plots):
                label = labels_list[i] if use_labels else None
                try:
                    self.axes.plot(energy_ev_list[i], counts_list[i], label=label)
                except Exception as e:
                    logger.error(f"Error plotting dataset {i}: {e}", exc_info=True)

            if use_labels and any(labels_list): # Add legend only if labels were provided
                self.axes.legend()

        # Set labels and title
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Intensity (arb. units)")
        if title:
            self.axes.set_title(title)
        else:
            self.axes.set_title("PL Spectra") # Default title

        self.axes.set_yscale(y_scale) # Apply the specified scale
        self.axes.grid(True, linestyle='--', alpha=0.6) # Add a grid
        self.figure.tight_layout()
        self.canvas.draw() # Redraw the canvas
        logger.info(f"Plotted {len(energy_ev_list)} dataset(s) with y_scale='{y_scale}'.")

    def clear_plot(self, y_scale='linear'):
        """Clears the plot area."""
        self.axes.clear()
        self.axes.plot([], []) # Plot empty data
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Intensity (arb. units)")
        self.axes.set_title("PL Spectra")
        self.axes.set_yscale(y_scale) # Apply the specified scale on clear
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.figure.tight_layout()
        self.canvas.draw()
        logger.info(f"Plot cleared with y_scale='{y_scale}'.")

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running plot_canvas.py directly for testing...")

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("PlotCanvas Test")
    root.geometry("700x600")

    plot_frame = PlotCanvas(master=root)
    plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Example plotting after a delay
    def test_plot():
        import numpy as np
        print("Testing plot_data...")
        energies1 = np.linspace(1.5, 2.5, 100)
        counts1 = np.exp(-(energies1 - 1.8)**2 / 0.1) * 1000 + np.random.rand(100) * 50
        energies2 = np.linspace(1.5, 2.5, 100)
        counts2 = np.exp(-(energies2 - 2.1)**2 / 0.05) * 800 + np.random.rand(100) * 40

        plot_frame.plot_data(
            [energies1, energies2],
            [counts1, counts2],
            labels_list=["Sample A", "Sample B"],
            title="Test Spectra"
        )

    # Example clearing after another delay
    def test_clear():
        print("Testing clear_plot...")
        plot_frame.clear_plot()

    root.after(2000, test_plot) # Plot after 2 seconds
    root.after(5000, test_clear) # Clear after 5 seconds

    root.mainloop()
