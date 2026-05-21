from __future__ import annotations

import logging
import time
import importlib.util
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

_OPHIR_CONTROL_PATH = Path(__file__).resolve().parents[1] / "devices" / "ophir-3A_powermeter_control.py"
_OPHIR_CONTROL_SPEC = importlib.util.spec_from_file_location(
    "ophir_3A_powermeter_control",
    _OPHIR_CONTROL_PATH,
)
if _OPHIR_CONTROL_SPEC is None or _OPHIR_CONTROL_SPEC.loader is None:
    raise ImportError(f"Cannot load Ophir control module from {_OPHIR_CONTROL_PATH}")
_ophir_control_module = importlib.util.module_from_spec(_OPHIR_CONTROL_SPEC)
sys.modules[_OPHIR_CONTROL_SPEC.name] = _ophir_control_module
_OPHIR_CONTROL_SPEC.loader.exec_module(_ophir_control_module)
Ophir3APowerMeterController = _ophir_control_module.Ophir3APowerMeterController

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class OphirPowerMeterWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Ophir 3A Power Meter", parent)
        self.controller = None
        self.polling_thread = None

        self.scan_btn = QPushButton("Scan Ophir USB")
        self.scan_btn.clicked.connect(self.scan_devices)
        self.device_combo = QComboBox()

        self.connect_btn = QPushButton("Open Device")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.power_label = QLabel("Power: ---- W")
        font = self.power_label.font()
        font.setPointSizeF(font.pointSizeF() * 2)
        self.power_label.setFont(font)

        self.wavelength_label = QLabel("Wavelengths: ---")
        self.sensor_label = QLabel("Sensor: ---")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.scan_btn)
        layout.addWidget(self.device_combo)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.power_label)
        layout.addWidget(self.sensor_label)
        layout.addWidget(self.wavelength_label)
        self.setLayout(layout)

    def scan_devices(self):
        self.device_combo.clear()
        try:
            controller = self.controller or Ophir3APowerMeterController()
            for serial in controller.scan_usb():
                self.device_combo.addItem(serial, serial)
            if self.device_combo.count() == 0:
                QMessageBox.information(self, "Device Not Found", "No Ophir USB device was detected.")
        except Exception as exc:
            QMessageBox.critical(self, "Ophir Scan Error", str(exc))

    def toggle_connection(self):
        if self.controller is None:
            serial = self.device_combo.currentData()
            try:
                self.controller = Ophir3APowerMeterController(channel=0)
                self.controller.connect(serial)
                self.controller.set_power_mode()
                self.connect_btn.setText("Close Device")
                self._refresh_device_labels()
                self.polling_thread = OphirPollingThread(self.controller)
                self.polling_thread.power_updated.connect(self.update_power_label)
                self.polling_thread.start()
            except Exception as exc:
                self.controller = None
                QMessageBox.critical(self, "Connection Error", str(exc))
        else:
            self.shutdown()
            self.connect_btn.setText("Open Device")
            self.power_label.setText("Power: ---- W")
            self.sensor_label.setText("Sensor: ---")
            self.wavelength_label.setText("Wavelengths: ---")

    def _refresh_device_labels(self):
        if self.controller is None:
            return
        try:
            self.sensor_label.setText(f"Sensor: {self.controller.get_sensor_info()}")
        except Exception as exc:
            self.sensor_label.setText(f"Sensor: unavailable ({exc})")
        try:
            current, wavelengths = self.controller.get_wavelengths()
            current_text = "---" if current is None else str(current)
            self.wavelength_label.setText(f"Wavelengths: current {current_text}; {', '.join(wavelengths)}")
        except Exception as exc:
            self.wavelength_label.setText(f"Wavelengths: unavailable ({exc})")

    def update_power_label(self, power_w: float):
        self.power_label.setText(f"Power: {power_w:.6g} W")

    def shutdown(self):
        if self.polling_thread is not None:
            self.polling_thread.stop()
            self.polling_thread = None
        if self.controller is not None:
            try:
                self.controller.shutdown()
            except Exception as exc:
                logging.warning("Failed to shutdown Ophir controller: %s", exc)
            self.controller = None


class OphirPollingThread(QThread):
    power_updated = pyqtSignal(float)

    def __init__(self, controller, interval_s: float = 1.0, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.interval_s = interval_s
        self._running = True

    def run(self):
        while self._running:
            try:
                self.controller.start_stream()
                self.power_updated.emit(self.controller.read_power())
            except Exception as exc:
                logging.debug("Ophir polling failed: %s", exc)
            finally:
                try:
                    self.controller.stop_stream()
                except Exception:
                    pass
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False
        self.wait()
