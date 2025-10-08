from PyQt6.QtWidgets import (
    QGroupBox, QPushButton, QLabel, QVBoxLayout,
    QComboBox, QDoubleSpinBox, QFormLayout, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal
import elliptec
import serial.tools.list_ports
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


class ElliptecRotatorWidget(QGroupBox):
    """
    Control Widget for Thorlabs Elliptec Rotator ELL14;
    use elliptec library https://github.com/roesel/elliptec
    """

    def __init__(self, parent=None):
        super().__init__("Analyzer angle Control", parent)
        self.controller = None
        self.rotator = None
        self.polling_thread = None

        # UI Elements
        self.scan_port_btn = QPushButton("Scan COM Port")
        self.scan_port_btn.clicked.connect(self.scan_com_port)
        self.ports_combo = QComboBox()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connect)

        self.home_btn = QPushButton("Homing")
        self.home_btn.setEnabled(False)
        self.home_btn.clicked.connect(self.home)

        self.target_angle_spin = QDoubleSpinBox()
        self.target_angle_spin.setSuffix("°")
        self.target_angle_spin.setSingleStep(0.1)
        self.target_angle_spin.setDecimals(2)
        self.target_angle_spin.setRange(0.0, 360.0) # not sure if this is correct
        self.target_angle_spin.setEnabled(False)
        self.target_angle_spin.editingFinished.connect(self.go_to) # too much communication with valueChanged. editingFinished signal doesn't emit a value!
        self.angle_label = QLabel("---")


        #layout
        layout = QVBoxLayout()
        layout.addWidget(self.scan_port_btn)
        layout.addWidget(self.ports_combo)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.home_btn)

        angle_form = QFormLayout()
        angle_form.addRow(self.target_angle_spin)
        angle_form.addRow("Angle:", self.angle_label)
        layout.addLayout(angle_form)

        self.setLayout(layout)
    

    def scan_com_port(self):
        self.ports_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.ports_combo.addItem(f"{port.description}", port.device)

    
    def toggle_connect(self):
        if self.controller is None:
            # connect
            port = self.ports_combo.currentData()
            if port == "":
                QMessageBox.warning(self, "Device Not Found", "No COM port is selected.")
                return
            try:
                self.controller = elliptec.Controller(port, debug=False)
            except Exception as e:
                logging.error(f"Failed to connect to Elliptec controller device: {e}")
                return
            try:
                self.rotator = elliptec.Rotator(self.controller, debug=False)
            except Exception as e:
                self.controller = None
                logging.error(f"Failed to connect Elliptec rotator: {e}")
                return
            logging.info("Elliptec device connected")
            self.connect_btn.setText("Disconnect")
            self.enable_control_uis(enable=True)
            self.target_angle_spin.setValue(self.rotator.get_angle())
            self.polling_thread = RotatorPollingThread(self.rotator, interval=0.5)
            self.polling_thread.updated.connect(self.update_angle_display) # emit polling_thread.status_updated -> execute self.update_status_display
            self.polling_thread.start()
        else:
            # disconnect
            if not self.polling_thread is None:
                try:
                    self.polling_thread.stop()
                except Exception as e:
                    logging.error(f"Failed to stop polling: {e}")
            self.polling_thread = None
            self.rotator = None
            self.controller = None
            logging.info("Elliptec device disconnected")
            self.connect_btn.setText("Connect")
            self.enable_control_uis(enable=False)
    

    def enable_control_uis(self, enable:bool):
        self.home_btn.setEnabled(enable)
        self.target_angle_spin.setEnabled(enable)
        self.ports_combo.setEnabled(not enable)


    def home(self):
        if self.rotator is None:
            return
        try:
            self.rotator.home()
            self.target_angle_spin.setValue(self.rotator.get_angle())
        except Exception as e:
            logging.error(f"Failed to move to home: {e}")
    

    def go_to(self):
        if self.rotator is None:
            return
        try:
            target_angle = self.target_angle_spin.value()
            self.rotator.set_angle(target_angle)
        except Exception as e:
            logging.error(f"Failed to move to {target_angle} deg: {e}")
    

    def update_angle_display(self, current_angle:float):
        self.angle_label.setText(f"{current_angle:.2f}°")



class RotatorPollingThread(QThread):
    updated = pyqtSignal(float)

    def __init__(self, rotator, interval=0.5, parent=None):
        super().__init__(parent)
        self.rotator = rotator
        self.interval = interval
        self._running = True

    
    def run(self):
        while self._running:
            try:
                angle = self.rotator.get_angle()
                if not angle is None:
                    self.updated.emit(angle)
            except Exception as e:
                logging.error(f"Rotator polling failed: {e}")
            time.sleep(self.interval)


    def stop(self):
        self._running = False
        self.wait()