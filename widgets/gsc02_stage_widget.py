from PyQt6.QtWidgets import (
    QGroupBox, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QMessageBox, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from devices.osms2035_control import OSMS2035Controller
from devices.osms60yaw_control import OSMS60YAWController
import serial.tools.list_ports
import logging
from threading import Lock
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


class StageCommonWidget(QGroupBox):
    def __init__(self, title, controller_class, unit, axis=1, parent=None):
        super().__init__(title, parent)
        self.controller_class = controller_class
        self.controller = None
        self.axis = axis
        self.unit = unit
        self.polling_thread = None
        self._serial_Lock = Lock()

        # UI Elements
        # COM port selection
        self.scan_port_btn = QPushButton("Scan COM Port")
        self.scan_port_btn.clicked.connect(self.scan_com_port)
        self.ports_combo = QComboBox()

        # Establish connection
        self.connect_btn = QPushButton("Open Port")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.connect_btn.setEnabled(False)

        # stage status
        self.position_label = QLabel(f"Position: --- {self.unit}")
        self.ready_label = QLabel("Busy/Ready: ---")
        self.motor_label = QLabel("Manual operation: ---")
        self.status_label = QLabel("Status message: ---")

        # stage control
        self.return_origin_btn = QPushButton("Origin")
        self.return_origin_btn.clicked.connect(self.reset)
        self.return_origin_btn.setEnabled(False)

        self.jog_label = QLabel("Jogging:")
        self.jog_plus_btn = QPushButton("+")
        self.jog_minus_btn = QPushButton("-")
        self.jog_plus_btn.clicked.connect(lambda: self.jog("+"))
        self.jog_minus_btn.clicked.connect(lambda: self.jog("-"))
        self.jog_plus_btn.setEnabled(False)
        self.jog_minus_btn.setEnabled(False)

        self.stop_btn = QPushButton("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)

        self.emergency_btn = QPushButton("Immediate Stop")
        self.emergency_btn.clicked.connect(self.immediate_stop)
        self.emergency_btn.setEnabled(False)

        self.motor_energize_btn = QPushButton("Deactivate motor")
        self.motor_energize_btn.clicked.connect(self.toggle_motor)
        self.motor_energize_btn.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.scan_port_btn)
        layout.addWidget(self.ports_combo)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.position_label)
        layout.addWidget(self.ready_label)

        jog_layout = QHBoxLayout()
        jog_layout.addWidget(self.jog_label)
        jog_layout.addWidget(self.jog_plus_btn)
        jog_layout.addWidget(self.jog_minus_btn)
        jog_layout.addWidget(self.stop_btn)
        layout.addLayout(jog_layout)
        
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(self.motor_label)
        manual_layout.addWidget(self.motor_energize_btn)
        layout.addLayout(manual_layout)

        layout.addWidget(self.return_origin_btn)
        layout.addWidget(self.emergency_btn)
        layout.addWidget(self.status_label)

        self.setLayout(layout)


    def scan_com_port(self):
        self.ports_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.ports_combo.addItem(f"{port.description}", port.device)
        self.connect_btn.setEnabled(True)


    def set_controls_enabled(self, enabled):
        self.return_origin_btn.setEnabled(enabled)
        self.motor_energize_btn.setEnabled(enabled)
        self.emergency_btn.setEnabled(enabled)
        self.jog_plus_btn.setEnabled(enabled)
        self.jog_minus_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)


    def toggle_connection(self):
        # Connection ON
        if self.controller is None:
            port = self.ports_combo.currentData()
            if port == "":
                QMessageBox.warning(self, "Device Not Found", "No COM port is selected.")
                return
            try:
                self.controller = self.controller_class(port=port, axis=self.axis)
                logging.info(f"{self.controller.position1}, {self.controller.position2}")
                              
                # print("controller:", self.controller_class)
                # print("connection:", self.controller.is_connected)
                # if not self.controller.is_connected:
                #     QMessageBox.warning(self, "Connection Error", f"Failed to connect to Stage{self.axis}")
                #     return
            except Exception as e:
                logging.error(f"Failed to connect to Optosigma stage: {e}")
                return
            else:
                if self.controller is not None:
                    logging.info(f"Optosiga stage{self.axis} connected")
                else:
                    logging.error(f"Failed to connect the stage{self.axis}")
                self.connect_btn.setText("Connected")
                self.set_controls_enabled(True)

                self.polling_thread = StagePollingThread(self.controller, self._serial_Lock, interval=1.0)
                self.polling_thread.status_updated.connect(self.update_status)
                # self.polling_thread.start() 
        
        # Connection OFF
        else:
            # stop polling
            if self.polling_thread is not None:
                try:
                    self.polling_thread.stop()
                except Exception as e:
                    logging.error(f"Failed to stop polling: {e}")
            # stop serial communication
            if self.controller.is_connected:
                with self._serial_Lock:
                    try:
                        self.controller.close()
                        self.controller = None
                    except Exception as e:
                        logging.error(f"Failed to close communication with stage: {e}")
                self.polling_thread = None
                self.connect_btn.setText("Disconnected")
                self.set_controls_enabled(False)
                self.clear_status()


    def toggle_motor(self):
        if self.controller is not None:
            if self.controller.is_connected:
                # Deactivate
                if self.controller.is_energized:
                    try:
                        self.motor_energize_btn.setEnabled(False)
                        with self._serial_Lock:
                            self.controller.energize_motor(False, axis=self.axis)
                        self.set_controls_enabled(False)
                        self.motor_energize_btn.setEnabled(True)
                        self.motor_energize_btn.setText("Activate motor")
                    except Exception as e:
                        QMessageBox.critical(self, "Failed to free the motor", str(e))
                # Activate
                else:
                    try:
                        self.motor_energize_btn.setEnabled(False)
                        with self._serial_Lock:
                            self.controller.energize_motor(True, axis=self.axis)
                        self.set_controls_enabled(True)
                        self.motor_energize_btn.setText("Deactivate motor")
                    except Exception as e:
                        QMessageBox.critical(self, "Failed to hold the motor", str(e))


    def reset(self):
        if self.controller is not None:
            if self.controller.is_connected:
                with self._serial_Lock:
                    try:
                        self.controller.reset()
                    except Exception as e:
                        QMessageBox.critical(self, "Failed to return to origin", str(e))


    def jog(self, direction: str):
        if self.controller is not None:
            if self.controller.is_connected:
                self.jog_plus_btn.setEnabled(False)
                self.jog_minus_btn.setEnabled(False)
                with self._serial_Lock:
                    try:
                        self.controller.set_speed(2, (2000, 2000), (5000, 5000), (200, 200))
                        self.controller.jog(direction=direction, axis=self.axis)
                        self.controller.driving()
                    except Exception as e:
                        QMessageBox.critical(self, "Jog Error", str(e))
                    finally:
                        self.set_controls_enabled(False)
                        self.stop_btn.setEnabled(True)
                        self.emergency_btn.setEnabled(True)


    def stop(self):
        if self.controller is not None:
            if self.controller.is_connected:
                with self._serial_Lock:
                    try:
                        self.controller.stop()
                    except Exception as e:
                        QMessageBox.critical(self, "Failed to stop", str(e))
                    finally:
                        self.set_controls_enabled(True)
    

    def immediate_stop(self):
        if self.controller is not None:
            if self.controller.is_connected:
                with self._serial_Lock:
                    try:
                        self.controller.immediate_stop()
                    except Exception as e:
                        QMessageBox.critical(self, "Fatal: failed to stop", str(e))
                    finally:
                        self.set_controls_enabled(True)


    def update_status(self, status_dict):
        self.position_label.setText(f"Position: {status_dict['position']} {self.unit}")
        self.ready_label.setText(f"Ready: {status_dict['ready']}")
        self.motor_label.setText(f"Motor energized: {status_dict['motor']}")
        self.status_label.setText(f"Last command state: {status_dict['status']}")


    def clear_status(self):
        self.position_label = QLabel(f"Position: --- {self.unit}")
        self.ready_label = QLabel("Busy/Ready: ---")
        self.motor_label = QLabel("Manual operation: ---")
        self.status_label = QLabel("Status message: ---")


    def shutdown(self):
        if self.controller is not None:
            if self.controller.is_connected:
                try:
                    self.controller.close()
                    self.controller = None
                    logging.info("Closed ports before shutdown")
                except Exception as e:
                    logging.warning(f"Failed to disconnect: {e}")
                

class StagePollingThread(QThread):
    status_updated = pyqtSignal(dict)

    def __init__(self, controller, lock, interval=1.0, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.lock = lock
        self.interval = interval
        self._running = True


    def run(self):
        while self._running:
            with self.lock:
                try:
                    if self.controller.axis == 1:
                        status = {
                        "position": self.controller.millimeter,
                        "ready": self.controller.is_ready,
                        "motor": self.controller.is_energized,
                        "status": self.controller.is_last_command_success
                        }
                    elif self.controller.axis == 2:
                        status = {
                        "position": self.controller.degree,
                        "ready": self.controller.is_ready,
                        "motor": self.controller.is_energized,
                        "status": self.controller.is_last_command_success
                        }
                    else: status = {}
                    self.status_updated.emit(status)
                except Exception as e:
                    logging.error(f"Polling failed: {e}")
            time.sleep(self.interval)


    def stop(self):
        self._running = False
        self.wait()


class OSMS2035Widget(StageCommonWidget):
    def __init__(self, axis):
        super().__init__(
            title="Stage 1: OSMS2035 (Translation)",
            controller_class=OSMS2035Controller,
            unit="mm",
            axis=axis,
            parent=None
        )
    
    def shutdown(self):
        super().shutdown()


class OSMS60YAWWidget(StageCommonWidget):
    def __init__(self, axis):
        super().__init__(
            title="Stage 2: OSMS60YAW (Rotation)",
            controller_class=OSMS60YAWController,
            unit="deg",
            axis=axis,
            parent=None
        )

    def shutdown(self):
        super().shutdown()

# class Gsc02StageWidget(QGroupBox):
#     def __init__(self, parent=None):
#         super().__init__("Sample Stage Control", parent)

#         self.stage1 = Stage1Widget()
#         self.stage2 = Stage2Widget()

#         layout = QVBoxLayout()
#         layout.addWidget(Stage1Widget())    # translation stage
#         layout.addWidget(Stage2Widget())    # rotation stage

#         self.setLayout(layout)

#     def shutdown(self):
#         self.stage1.shutdown()
#         self.stage2.shutdown()