import customtkinter as ctk
from tkinter import ttk
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FileTable(ctk.CTkFrame):
    """
    A widget to display loaded file metadata in a table (Treeview).
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Define columns
        self.columns = {
            "file_id": {"text": "ID", "width": 50, "anchor": "w"},
            "filename": {"text": "Filename", "width": 200, "anchor": "w"},
            "temperature_k": {"text": "Temp (K)", "width": 70, "anchor": "center"},
            "power_uw": {"text": "Power (uW)", "width": 80, "anchor": "center"},
            "time_s": {"text": "Time (s)", "width": 60, "anchor": "center"},
            "gf_present": {"text": "GF", "width": 40, "anchor": "center"},
            "bfield_t": {"text": "B-Field (T)", "width": 80, "anchor": "center"},
            "polarization": {"text": "Polarization", "width": 90, "anchor": "center"}
        }

        # Style for Treeview
        self.style = ttk.Style(self)
        self.style.map("Treeview")
        self.style.configure("Treeview", rowheight=25)
        self.style.configure("Treeview.Heading", font=('Calibri', 10,'bold'))

        # Create Treeview
        self.tree = ttk.Treeview(
            self,
            columns=list(self.columns.keys()),
            show='headings',
            selectmode='extended' # Allows multiple selection
        )

        self.tree.tag_configure('duplicate', background='red')

        # Configure headings
        for col_id, col_info in self.columns.items():
            self.tree.heading(col_id, text=col_info["text"], anchor=col_info.get("anchor", "center"))
            self.tree.column(col_id, width=col_info["width"], anchor=col_info.get("anchor", "center"), stretch=False) # No stretch initially

        # Add Scrollbars
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        # Grid layout for Treeview and Scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.vsb.grid(row=0, column=1, sticky='ns')
        self.hsb.grid(row=1, column=0, sticky='ew')

        # Store item IDs for easy access/update if needed later
        self._item_ids = []

    def update_data(self, dataframe: Optional[pd.DataFrame]):
        """
        Clears the table and populates it with data from the DataFrame.

        Args:
            dataframe: A pandas DataFrame containing the file metadata.
                        Expected columns match the keys in self.columns.
                        If None or empty, the table is cleared.
        """
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._item_ids = []

        if dataframe is None or dataframe.empty:
            logger.debug("FileTable received empty data, table cleared.")
            return

        logger.debug(f"Updating FileTable with {len(dataframe)} rows.")

        # Ensure all expected columns exist, fill missing with default values
        for col_id in self.columns.keys():
            if col_id not in dataframe.columns:
                dataframe[col_id] = None # Or appropriate default

        # Populate table
        duplicates = dataframe[dataframe.duplicated(['temperature_k', 'power_uw', 'bfield_t', 'polarization'], keep=False)]
        for index, row in dataframe.iterrows():
            values = []
            for col_id in self.columns.keys():
                value = row[col_id]
                # Format values for display
                if pd.isna(value):
                    display_value = ""
                elif isinstance(value, bool):
                    display_value = "Yes" if value else "No"
                elif isinstance(value, float):
                    # Special formatting for power column
                    if col_id == "power_uw":
                        display_value = f"{value:.3f}" # Show 3 decimal places for power
                    else:
                        display_value = f"{value:.1f}" # Default to 1 decimal place
                else:
                    display_value = str(value)
                values.append(display_value)

            tags = ()
            if not duplicates.empty and duplicates.apply(lambda x: x.equals(row), axis=1).any():
                tags = ('duplicate',)

            item_id = self.tree.insert('', 'end', values=values, tags=tags)
            self._item_ids.append(item_id)

    def get_selected_file_ids(self) -> list[str]:
        """
        Returns the 'file_id' values for the currently selected rows.
        """
        selected_items = self.tree.selection()
        selected_ids = []
        if not selected_items:
            return []

        for item_iid in selected_items:
            item_values = self.tree.item(item_iid, 'values')
            if item_values:
                # Assuming 'file_id' is the first column (index 0)
                selected_ids.append(item_values[0])
        return selected_ids


# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running file_table.py directly for testing...")

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("FileTable Test")
    root.geometry("700x400")

    table_frame = FileTable(master=root)
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Create dummy data
    dummy_data = {
        'file_id': ['file_1', 'file_2', 'file_3'],
        'filename': ['SampleA_10K_100uW_0.5s.csv', 'SampleB_5-9K_50uW_1s_GF4p8.csv', 'SampleC_15K_20uW_0.2s.dat'],
        'temperature_k': [10.0, 5.0, 15.0],
        'power_uw': [100.0, 50.0, 20.0],
        'time_s': [0.5, 1.0, 0.2],
        'gf_present': [False, True, False],
        'filepath': ['path/1', 'path/2', 'path/3'] # Extra column, should be ignored by table
    }
    dummy_df = pd.DataFrame(dummy_data)

    # Populate table
    table_frame.update_data(dummy_df)

    # Example of getting selected items after a delay
    def print_selection():
        selected = table_frame.get_selected_file_ids()
        print("Selected file IDs:", selected)
    root.after(5000, print_selection) # Print selection after 5 seconds

    root.mainloop()
