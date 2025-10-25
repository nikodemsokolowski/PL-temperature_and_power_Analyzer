"""Placeholder Qt GUI components for future development."""
try:
    from PySide6 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - optional dependency
    QtCore = None
    QtWidgets = None


class MainWindow(QtWidgets.QMainWindow if QtWidgets else object):
    def __init__(self, controller, parent=None):
        if QtWidgets is None or QtCore is None:
            raise RuntimeError("PySide6 is required for the GUI but is not installed.")
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("SHG Analyzer (Prototype)")
        label = QtWidgets.QLabel("GUI implementation pending")
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(label)


__all__ = ["MainWindow"]
