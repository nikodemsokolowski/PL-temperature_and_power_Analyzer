import customtkinter as ctk
from pl_analyzer.gui.dialogs import MagneticSweepSettingsDialog

ctk.set_appearance_mode("Dark")
root = ctk.CTk()
root.withdraw()
dialog = MagneticSweepSettingsDialog(root, {})
print('rows:', len(dialog.time_range_rows))
dialog._add_time_range_row()
print('rows after add:', len(dialog.time_range_rows))
dialog.destroy()
root.destroy()
