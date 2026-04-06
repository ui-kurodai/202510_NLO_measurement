from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal
import logging
import time

from devices.power_meter_control import ThorlabsPowerMeterController


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class PowerMeterWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Thorlabs Power Meter", parent)
        self.controller = None
        self.polling_thread = None

        self.scan_resource_btn = QPushButton("Scan VISA Resources")
        self.scan_resource_btn.clicked.connect(self.scan_resources)
        self.resource_combo = QComboBox()

        self.connect_btn = QPushButton("Open Port")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.idn_label = QLabel("IDN: ---")
        self.power_label = QLabel("Power: ---")

        layout = QVBoxLayout()
        layout.addWidget(self.scan_resource_btn)
        layout.addWidget(self.resource_combo)
        layout.addWidget(self.connect_btn)

        form = QFormLayout()
        form.addRow(self.idn_label)
        form.addRow(self.power_label)
        layout.addLayout(form)

        self.setLayout(layout)

    def scan_resources(self):
        self.resource_combo.clear()
        try:
            controller = ThorlabsPowerMeterController()
            resources = controller.list_resources()
        except Exception as exc:
            QMessageBox.critical(self, "VISA Error", str(exc))
            return

        for resource in resources:
            self.resource_combo.addItem(resource, resource)

    def toggle_connection(self):
        if self.controller is None:
            resource = self.resource_combo.currentData()
            if not resource:
                QMessageBox.warning(self, "Device Not Found", "No VISA resource is selected.")
                return
            try:
                self.controller = ThorlabsPowerMeterController()
                self.controller.connect(resource)
                idn = self.controller.idn
            except Exception as exc:
                self.controller = None
                logging.error(f"Failed to connect power meter: {exc}")
                QMessageBox.critical(self, "Connection Error", str(exc))
                return

            self.connect_btn.setText("Close Port")
            self.idn_label.setText(f"IDN: {idn}")
            self.polling_thread = PowerMeterPollingThread(self.controller, interval=1.0)
            self.polling_thread.status_updated.connect(self.update_power_display)
            self.polling_thread.start()
        else:
            self.shutdown()

    def update_power_display(self, power_watts: float):
        if power_watts >= 1.0:
            self.power_label.setText(f"Power: {power_watts:.4f} W")
        elif power_watts >= 1e-3:
            self.power_label.setText(f"Power: {power_watts * 1e3:.4f} mW")
        else:
            self.power_label.setText(f"Power: {power_watts * 1e6:.4f} uW")

    def shutdown(self):
        if self.polling_thread is not None:
            try:
                self.polling_thread.stop()
            except Exception as exc:
                logging.error(f"Failed to stop power meter polling: {exc}")
        self.polling_thread = None

        if self.controller is not None:
            try:
                self.controller.disconnect()
            except Exception as exc:
                logging.error(f"Failed to disconnect power meter: {exc}")
            self.controller = None

        self.connect_btn.setText("Open Port")
        self.idn_label.setText("IDN: ---")
        self.power_label.setText("Power: ---")


class PowerMeterPollingThread(QThread):
    status_updated = pyqtSignal(float)

    def __init__(self, controller, interval=1.0, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            try:
                power = self.controller.read_power_watts()
                self.status_updated.emit(power)
            except Exception as exc:
                logging.debug(f"Power meter polling skipped: {exc}")
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait()
