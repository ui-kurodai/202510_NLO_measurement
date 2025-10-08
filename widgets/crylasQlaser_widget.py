from PyQt6.QtWidgets import (
    QGroupBox, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from devices.crylasQlaser_control import CrylasQLaserController
import serial.tools.list_ports
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')

class CrylasQlaserWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("CryLaS Q-Series Laser", parent)

        self.controller = None
        self.polling_thread = None

        # UI Elements
        # COM port selection
        self.scan_port_btn = QPushButton("Scan COM Port")
        self.scan_port_btn.clicked.connect(self.scan_com_port)
        self.ports_combo = QComboBox()

        # Establish connection
        self.connect_btn = QPushButton("Open Port")
        self.connect_btn.clicked.connect(self.toggle_connection)

        # Toggle laser emission
        self.laser_btn = QPushButton("Turn Laser ON")
        self.laser_btn.clicked.connect(self.toggle_laser)
        self.laser_btn.setEnabled(False)

        # Display laser status
        self.status_label = QLabel("Status: ---")
        self.rep_rate_label = QLabel("Repetition Rate: --- Hz")
        self.pd_voltage_label = QLabel("PD Voltage: --- mV")
        self.error_label = QLabel("Error: ---")

        # Repetition Rate Control
        self.rep_rate_spin = QDoubleSpinBox()
        self.rep_rate_spin.setSuffix(" Hz")
        self.rep_rate_spin.setDecimals(0)
        self.rep_rate_spin.setRange(1, 10000)
        self.rep_rate_spin.setSingleStep(10)
        self.rep_rate_spin.setEnabled(False)
        self.rep_rate_spin.valueChanged.connect(self.update_rep_rate)

        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.scan_port_btn)
        layout.addWidget(self.ports_combo)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.laser_btn)
        layout.addWidget(self.status_label)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.rep_rate_label)
        hlayout.addWidget(self.rep_rate_spin)
        layout.addLayout(hlayout)

        layout.addWidget(self.pd_voltage_label)
        layout.addWidget(self.error_label)

        self.setLayout(layout)

    def scan_com_port(self):
        self.ports_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.ports_combo.addItem(f"{port.description}", port.device)

    def toggle_connection(self):
        # Connection ON
        if self.controller is None:
            port = self.ports_combo.currentData()
            if port == "":
                QMessageBox.warning(self, "Device Not Found", "No COM port is selected.")
                return
            try:
                self.controller = CrylasQLaserController()
                self.controller.open(port=port)
                if not self.controller.is_connected:
                    QMessageBox.warning(self, "Failed to connect to CrylasQLaser")
                    return
            except Exception as e:
                logging.error(f"Failed to connect to CrylasQLaser: {e}")
                return
            else:
                logging.info("CrylasQLaser device connected")
                self.connect_btn.setText("Connected")
                self.set_controls_enabled(True)

                self.polling_thread = LaserPollingThread(self.controller, interval=1.0)
                self.polling_thread.status_updated.connect(self.update_status)
                self.polling_thread.start() 
        
        # Connection OFF
        else:
            # stop polling
            if self.polling_thread is not None:
                try:
                    self.polling_thread.stop()
                except Exception as e:
                    logging.error(f"Failed to stop polling: {e}")
            # stop serial communication
            if not self.controller.is_connected:
                logging.debug("Attempted to close connection when not connected")
                return
            try:
                self.controller.close()
            except Exception as e:
                logging.error(f"Failed to close communication with CrylasQLaser: {e}")
            
            # delete instance anyway
            self.controller = None    
            self.polling_thread = None
            self.connect_btn.setText("Disconnected")
            self.set_controls_enabled(False)
            self.clear_status()

    def toggle_laser(self):
        if self.controller is None:
            return
        if self.controller.is_emission_on:
            self.controller.stop()
            self.laser_btn.setText("Turn laser ON")
        else:
            self.controller.start()
            self.laser_btn.setText("Turn laser OFF")

    def set_controls_enabled(self, enabled):
        self.laser_btn.setEnabled(enabled)
        self.rep_rate_spin.setEnabled(enabled)

    def update_rep_rate(self, value):
        try:
            self.controller.rep_rate = int(value)
        except Exception as e:
            logging.error(f"Failed to set rep rate: {e}")

    def update_status(self, status_dict):
        if status_dict['emission']: emission = "ON" 
        else: emission = "OFF"
        self.status_label.setText(f"Emission: {emission}")
        self.rep_rate_label.setText(f"Repetition Rate: {status_dict['rep_rate']} Hz")
        self.pd_voltage_label.setText(f"PD Voltage: {status_dict['pd_voltage']} mV")
        self.error_label.setText(f"Error: {status_dict['error']}")

    def clear_status(self):
        self.status_label.setText("Emission: ---")
        self.rep_rate_label.setText("Repetition Rate: --- Hz")
        self.pd_voltage_label.setText("PD Voltage: --- mV")
        self.error_label.setText("Error: ---")

    def shutdown(self):
        if self.controller is not None:
            if self.controller.is_connected:
                try:
                    self.controller.close()
                    self.controller = None
                    logging.info("Closed ports before shutdown")
                except Exception as e:
                    logging.warning(f"Failed to disconnect: {e}")


class LaserPollingThread(QThread):
    status_updated = pyqtSignal(dict)

    def __init__(self, controller, interval=1.0, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            try:
                status = {
                    "emission": self.controller.is_emission_on,
                    "rep_rate": self.controller.rep_rate,
                    "pd_voltage": self.controller.photodiode_voltage,
                    "error": self.controller.last_error_message
                }
                self.status_updated.emit(status)
            except Exception as e:
                logging.error(f"Polling failed: {e}")
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        self.wait()
