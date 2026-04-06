from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal
import logging
import serial.tools.list_ports
import time

from devices.polarization_control import ElliptecPolarizerController


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class ElliptecRotatorWidget(QGroupBox):
    """
    Polarizer control widget for a Thorlabs Elliptec rotator.
    The UI exposes logical polarization angles while the controller
    keeps track of the raw Elliptec angle plus a saved calibration offset.
    """

    def __init__(self, title: str, calibration_key: str, default_address: str = "0", parent=None):
        super().__init__(title, parent)
        self.controller = ElliptecPolarizerController(
            name=title,
            calibration_key=calibration_key,
            default_address=default_address,
        )
        self.polling_thread = None

        self.scan_port_btn = QPushButton("Scan COM Port")
        self.scan_port_btn.clicked.connect(self.scan_com_port)
        self.ports_combo = QComboBox()

        self.address_combo = QComboBox()
        for address in "0123456789ABCDEF":
            self.address_combo.addItem(address, address)
        default_index = max(self.address_combo.findData(default_address.upper()), 0)
        self.address_combo.setCurrentIndex(default_index)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connect)

        self.home_btn = QPushButton("Homing")
        self.home_btn.setEnabled(False)
        self.home_btn.clicked.connect(self.home)

        self.target_angle_spin = QDoubleSpinBox()
        self.target_angle_spin.setSuffix(" deg")
        self.target_angle_spin.setSingleStep(0.1)
        self.target_angle_spin.setDecimals(2)
        self.target_angle_spin.setRange(0.0, 180.0)
        self.target_angle_spin.setEnabled(False)
        self.target_angle_spin.editingFinished.connect(self.go_to)

        self.logical_angle_label = QLabel("---")
        self.raw_angle_label = QLabel("---")
        self.zero_offset_label = QLabel("---")
        self.connection_label = QLabel("Status: disconnected")
        self.refresh_calibration_display()

        layout = QVBoxLayout()
        layout.addWidget(self.scan_port_btn)
        layout.addWidget(self.ports_combo)

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Address:"))
        port_layout.addWidget(self.address_combo)
        layout.addLayout(port_layout)

        layout.addWidget(self.connect_btn)
        layout.addWidget(self.home_btn)
        layout.addWidget(self.connection_label)

        angle_form = QFormLayout()
        angle_form.addRow("Target pol.:", self.target_angle_spin)
        angle_form.addRow("Current pol.:", self.logical_angle_label)
        angle_form.addRow("Raw angle:", self.raw_angle_label)
        angle_form.addRow("Zero offset:", self.zero_offset_label)
        layout.addLayout(angle_form)

        self.setLayout(layout)

    def scan_com_port(self):
        self.ports_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.ports_combo.addItem(f"{port.description}", port.device)

    def toggle_connect(self):
        if not self.controller.is_connected:
            port = self.ports_combo.currentData()
            if not port:
                QMessageBox.warning(self, "Device Not Found", "No COM port is selected.")
                return
            address = self.address_combo.currentData()
            try:
                self.controller.connect(port=port, address=address, debug=False)
                raw_angle = self.controller.get_raw_angle()
                logical_angle = self.controller.get_logical_angle()
            except Exception as exc:
                logging.error(f"Failed to connect Elliptec polarizer: {exc}")
                QMessageBox.critical(self, "Connection Error", str(exc))
                return

            self.connect_btn.setText("Disconnect")
            self.enable_control_uis(True)
            self.connection_label.setText(f"Status: connected on {port} / addr {address}")
            self.target_angle_spin.setValue(logical_angle)
            self.update_angle_display(raw_angle, logical_angle)

            self.polling_thread = RotatorPollingThread(self.controller, interval=0.5)
            self.polling_thread.updated.connect(self.update_angle_display)
            self.polling_thread.start()
        else:
            self.shutdown()
            self.connection_label.setText("Status: disconnected")

    def enable_control_uis(self, enable: bool):
        self.home_btn.setEnabled(enable)
        self.target_angle_spin.setEnabled(enable)
        self.ports_combo.setEnabled(not enable)
        self.address_combo.setEnabled(not enable)

    def home(self):
        if not self.controller.is_connected:
            return
        try:
            raw_angle = self.controller.home()
            logical_angle = self.controller.get_logical_angle()
            self.target_angle_spin.setValue(logical_angle)
            self.update_angle_display(raw_angle, logical_angle)
        except Exception as exc:
            logging.error(f"Failed to home {self.controller.name}: {exc}")
            QMessageBox.critical(self, "Homing Error", str(exc))

    def go_to(self):
        if not self.controller.is_connected:
            return
        target_angle = self.target_angle_spin.value()
        try:
            raw_angle = self.controller.move_to_polarization(target_angle)
            logical_angle = self.controller.get_logical_angle()
            self.update_angle_display(raw_angle, logical_angle)
        except Exception as exc:
            logging.error(f"Failed to move {self.controller.name}: {exc}")
            QMessageBox.critical(self, "Move Error", str(exc))

    def refresh_calibration_display(self):
        self.zero_offset_label.setText(f"{self.controller.zero_offset_deg:.2f} deg")

    def update_angle_display(self, raw_angle: float, logical_angle: float):
        self.logical_angle_label.setText(f"{logical_angle:.2f} deg")
        self.raw_angle_label.setText(f"{raw_angle:.2f} deg")
        self.refresh_calibration_display()

    def shutdown(self):
        if self.polling_thread is not None:
            try:
                self.polling_thread.stop()
            except Exception as exc:
                logging.error(f"Failed to stop Elliptec polling: {exc}")
        self.polling_thread = None

        if self.controller.is_connected:
            try:
                self.controller.disconnect()
            except Exception as exc:
                logging.error(f"Failed to disconnect Elliptec polarizer: {exc}")

        self.connect_btn.setText("Connect")
        self.enable_control_uis(False)
        self.connection_label.setText("Status: disconnected")
        self.logical_angle_label.setText("---")
        self.raw_angle_label.setText("---")
        self.refresh_calibration_display()


class RotatorPollingThread(QThread):
    updated = pyqtSignal(float, float)

    def __init__(self, controller, interval=0.5, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            try:
                raw_angle = self.controller.get_raw_angle()
                logical_angle = self.controller.get_logical_angle()
                self.updated.emit(raw_angle, logical_angle)
            except Exception as exc:
                logging.debug(f"Rotator polling skipped: {exc}")
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait()
