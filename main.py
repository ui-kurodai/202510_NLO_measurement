from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QStackedWidget, QLabel
)
from PyQt6.QtCore import QLocale, Qt, QSize

# Workflow widgets
from widgets.measurement_widget import SHGMeasurementWidget
from widgets.analysis_widget import FittingAnalysisWidget

# Device widgets
from widgets.crylasQlaser_widget import CrylasQlaserWidget
from widgets.elliptec_rotator_widget import ElliptecRotatorWidget
from widgets.gsc02_stage_widget import OSMS2035Widget, OSMS60YAWWidget
from widgets.boxcar_widget import BoxcarWidget


class DevicesPanel(QWidget):
    """
    Devices tab:
    - Left: device names as a vertical list (acts like a menu).
    - Right: the selected device widget displayed via QStackedWidget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Left: device list (menu-like) ---
        self.device_list = QListWidget()
        self.device_list.setUniformItemSizes(True)
        self.device_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.device_list.setMinimumWidth(180)  # keeps ~1/5 width in typical 1100px window
        self.device_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.device_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.device_list.setSpacing(2)
        self.device_list.setStyleSheet("QListWidget { border: none; }")

        # --- Right: device widgets as pages ---
        self.stack = QStackedWidget()

        # Instantiate device widgets (same as before)
        self.laser_widget = CrylasQlaserWidget()
        self.stage_tr_widget = OSMS2035Widget(axis=1)   # translation stage
        self.stage_rot_widget = OSMS60YAWWidget(axis=2) # rotation stage
        self.boxcar_widget = BoxcarWidget()
        self.elliptec_widget = ElliptecRotatorWidget()

        # Ordered list of (name, widget)
        self._devices = [
            ("Laser", self.laser_widget),
            ("Stage (Translation)", self.stage_tr_widget),
            ("Stage (Rotation)", self.stage_rot_widget),
            ("Boxcar", self.boxcar_widget),
            ("Analyzer", self.elliptec_widget),
        ]

        # Populate list and stack in the same order
        for name, widget in self._devices:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(item.sizeHint().width(), 30))  # taller row for readability
            self.device_list.addItem(item)
            self.stack.addWidget(widget)

        # Default selection = first device
        if self.device_list.count() > 0:
            self.device_list.setCurrentRow(0)
            self.stack.setCurrentIndex(0)

        # Connect selection change
        self.device_list.currentRowChanged.connect(self.stack.setCurrentIndex)

        # Layout via splitter (left: list, right: stacked content)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.device_list)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 1)  # left
        splitter.setStretchFactor(1, 4)  # right

        # Root layout
        root = QVBoxLayout(self)
        root.addWidget(splitter)

    # Optional convenience accessors for shutdown usage
    @property
    def all_device_widgets(self):
        """Return all instantiated device widgets for unified shutdown."""
        return [
            self.laser_widget,
            self.stage_tr_widget,
            self.stage_rot_widget,
            self.boxcar_widget,
            self.elliptec_widget,
        ]


class MainWindow(QWidget):
    """
    Main window with three tabs:
    - Home
    - Fitting analysis
    - Devices (left menu + right content)
    """
    def __init__(self):
        super().__init__()

        # Always use "." as decimal separator
        QLocale.setDefault(QLocale.c())

        self.setWindowTitle("SHG Laser Control")
        self.resize(1100, 720)

        # Central tabs (Workflow + Devices)
        self.tabs = QTabWidget()

        # Workflow tabs
        self.home_widget = SHGMeasurementWidget(main_window=self)
        self.analysis_widget = FittingAnalysisWidget()

        # Devices tab (menu + content)
        self.devices_tab = DevicesPanel()

        # Add tabs in the requested order
        self.tabs.addTab(self.home_widget, "Home")
        self.tabs.addTab(self.analysis_widget, "Fitting analysis")
        self.tabs.addTab(self.devices_tab, "Devices")

        # Root layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def closeEvent(self, event):
        """
        Keep the same shutdown behavior:
        call shutdown() for each device widget if available.
        """
        try:
            # Access through devices_tab to ensure single source of truth
            for w in self.devices_tab.all_device_widgets:
                # Call shutdown if widget implements it
                if hasattr(w, "shutdown") and callable(getattr(w, "shutdown")):
                    w.shutdown()
        except Exception as e:
            print(f"Shutdown error: {e}")
        event.accept()


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
