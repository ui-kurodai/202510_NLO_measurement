from PyQt6.QtWidgets import (
    QGroupBox, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QMessageBox, QComboBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from devices.osms2035_control import OSMS2035Controller
from devices.osms60yaw_control import OSMS60YAWController
import serial.tools.list_ports
import logging
from threading import Lock
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')

_STAGE_PORT_LOCKS = {}


def _lock_for_port(port):
    if port not in _STAGE_PORT_LOCKS:
        _STAGE_PORT_LOCKS[port] = Lock()
    return _STAGE_PORT_LOCKS[port]


class StageCommonWidget(QGroupBox):
    def __init__(self, title, controller_class, unit, axis=1, parent=None):
        super().__init__(title, parent)
        self.controller_class = controller_class
        self.controller = None
        self.axis = axis
        self.unit = unit
        self.polling_thread = None
        self.command_thread = None
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
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setDecimals(4)
        self.target_spin.setSingleStep(0.1)
        self.target_spin.setSuffix(f" {self.unit}")
        self.target_spin.setEnabled(False)
        self._configure_target_spin()

        self.move_to_btn = QPushButton("Move")
        self.move_to_btn.clicked.connect(self.move_to_target)
        self.move_to_btn.setEnabled(False)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_status)
        self.refresh_btn.setEnabled(False)

        self.live_read_checkbox = QCheckBox("Live read")
        self.live_read_checkbox.setToolTip(
            "Polls stage status periodically. Commands are serialized per COM port, but keep this off during measurements."
        )
        self.live_read_checkbox.toggled.connect(self.toggle_live_read)
        self.live_read_checkbox.setEnabled(False)

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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.scan_port_btn)
        layout.addWidget(self.ports_combo)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.position_label)
        layout.addWidget(self.ready_label)

        move_layout = QHBoxLayout()
        move_layout.addWidget(QLabel("Target:"))
        move_layout.addWidget(self.target_spin)
        move_layout.addWidget(self.move_to_btn)
        move_layout.addWidget(self.refresh_btn)
        move_layout.addWidget(self.live_read_checkbox)
        layout.addLayout(move_layout)

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


    def _configure_target_spin(self):
        if self.unit == "deg":
            self.target_spin.setRange(-180.0, 179.9975)
            self.target_spin.setSingleStep(1.0)
        else:
            self.target_spin.setRange(-1000000.0, 1000000.0)


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
        self.target_spin.setEnabled(enabled)
        self.move_to_btn.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.live_read_checkbox.setEnabled(enabled)


    def set_busy(self, busy, message=None):
        enabled = not busy and self.controller is not None and self.controller.is_connected
        self.set_controls_enabled(enabled)
        self.connect_btn.setEnabled(not busy)
        if busy:
            self.stop_btn.setEnabled(True)
            self.emergency_btn.setEnabled(True)
        if message:
            self.status_label.setText(f"Status message: {message}")


    def toggle_connection(self):
        # Connection ON
        if self.controller is None:
            port = self.ports_combo.currentData()
            if port == "":
                QMessageBox.warning(self, "Device Not Found", "No COM port is selected.")
                return
            self._serial_Lock = _lock_for_port(port)
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

                self.refresh_status()
        
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
                self.command_thread = None
                self.connect_btn.setText("Disconnected")
                self.set_controls_enabled(False)
                self.clear_status()


    def toggle_motor(self):
        if not self._has_controller():
            return

        energize = not self.controller.is_energized
        label = "Activating motor..." if energize else "Deactivating motor..."

        def command():
            self.controller.energize_motor(energize, axis=self.axis)

        self._start_command(command, label)


    def reset(self):
        if not self._has_controller():
            return
        self._start_command(self.controller.reset, "Moving to origin...")


    def move_to_target(self):
        if not self._has_controller():
            return
        target = self.target_spin.value()

        def command():
            if self.axis == 1:
                self.controller.millimeter = target
            elif hasattr(self.controller, "move_to_angle"):
                self.controller.move_to_angle(target, "auto")
            else:
                self.controller.degree = target

        self._start_command(command, f"Moving to {target:.4f} {self.unit}...")


    def jog(self, direction: str):
        if self.controller is not None:
            if self.controller.is_connected:
                self.jog_plus_btn.setEnabled(False)
                self.jog_minus_btn.setEnabled(False)

                def command():
                    self.controller.set_speed(2, (2000, 2000), (5000, 5000), (200, 200))
                    self.controller.jog(direction=direction, axis=self.axis)
                    self.controller.driving()

                self._start_command(command, f"Jogging {direction}...", keep_busy=True)


    def stop(self):
        if self.controller is not None:
            if self.controller.is_connected:
                self._start_command(self.controller.stop, "Stopping...")
    

    def immediate_stop(self):
        if self.controller is not None:
            if self.controller.is_connected:
                self._start_command(self.controller.immediate_stop, "Stopping immediately...")


    def refresh_status(self):
        if not self._has_controller():
            return
        self._start_command(lambda: None, "Reading status...")


    def toggle_live_read(self, enabled):
        if enabled:
            if not self._has_controller():
                self.live_read_checkbox.setChecked(False)
                return
            self.polling_thread = StagePollingThread(self.controller, self._serial_Lock, interval=2.0)
            self.polling_thread.status_updated.connect(self.update_status)
            self.polling_thread.polling_failed.connect(self.on_command_failed)
            self.polling_thread.start()
            self.status_label.setText("Status message: Live read enabled")
        else:
            if self.polling_thread is not None:
                self.polling_thread.stop()
                self.polling_thread = None
            if self.controller is not None:
                self.status_label.setText("Status message: Live read disabled")


    def update_status(self, status_dict):
        if not status_dict:
            return
        self.position_label.setText(f"Position: {status_dict['position']} {self.unit}")
        self.ready_label.setText(f"Ready: {status_dict['ready']}")
        self.motor_label.setText(f"Motor energized: {status_dict['motor']}")
        self.status_label.setText(f"Last command state: {status_dict['status']}")
        if "motor" in status_dict:
            self.motor_energize_btn.setText(
                "Deactivate motor" if status_dict["motor"] else "Activate motor"
            )


    def clear_status(self):
        self.position_label.setText(f"Position: --- {self.unit}")
        self.ready_label.setText("Busy/Ready: ---")
        self.motor_label.setText("Manual operation: ---")
        self.status_label.setText("Status message: ---")


    def _has_controller(self):
        return self.controller is not None and self.controller.is_connected


    def _read_status_unlocked(self):
        if self.controller.axis == 1:
            position = self.controller.millimeter
        elif self.controller.axis == 2:
            position = self.controller.degree
        else:
            position = None
        return {
            "position": position,
            "ready": self.controller.is_ready,
            "motor": self.controller.is_energized,
            "status": self.controller.is_last_command_success,
        }


    def _start_command(self, command, message, keep_busy=False):
        if self.command_thread is not None and self.command_thread.isRunning():
            QMessageBox.information(self, "Stage Busy", "Another stage command is still running.")
            return
        self.set_busy(True, message)
        self.command_thread = StageCommandThread(
            command=command,
            lock=self._serial_Lock,
            status_reader=self._read_status_unlocked,
        )
        self.command_thread.status_updated.connect(self.update_status)
        self.command_thread.command_failed.connect(self.on_command_failed)
        self.command_thread.command_finished.connect(
            lambda: self.on_command_finished(keep_busy=keep_busy)
        )
        self.command_thread.start()


    def on_command_failed(self, message):
        QMessageBox.critical(self, "Stage Error", message)
        self.status_label.setText(f"Status message: {message}")


    def on_command_finished(self, keep_busy=False):
        if keep_busy:
            self.set_controls_enabled(False)
            self.stop_btn.setEnabled(True)
            self.emergency_btn.setEnabled(True)
        else:
            self.set_busy(False)
            self.connect_btn.setEnabled(True)


    def shutdown(self):
        if self.polling_thread is not None:
            self.polling_thread.stop()
            self.polling_thread = None
        if self.command_thread is not None and self.command_thread.isRunning():
            self.command_thread.wait()
        if self.controller is not None:
            if self.controller.is_connected:
                try:
                    self.controller.close()
                    self.controller = None
                    logging.info("Closed ports before shutdown")
                except Exception as e:
                    logging.warning(f"Failed to disconnect: {e}")
                

class StageCommandThread(QThread):
    status_updated = pyqtSignal(dict)
    command_failed = pyqtSignal(str)
    command_finished = pyqtSignal()

    def __init__(self, command, lock, status_reader=None, parent=None):
        super().__init__(parent)
        self.command = command
        self.lock = lock
        self.status_reader = status_reader

    def run(self):
        try:
            with self.lock:
                self.command()
                if self.status_reader is not None:
                    self.status_updated.emit(self.status_reader())
        except Exception as e:
            logging.error(f"Stage command failed: {e}")
            self.command_failed.emit(str(e))
        finally:
            self.command_finished.emit()


class StagePollingThread(QThread):
    status_updated = pyqtSignal(dict)
    polling_failed = pyqtSignal(str)

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
                        position = self.controller.millimeter
                    elif self.controller.axis == 2:
                        position = self.controller.degree
                    else:
                        position = None
                    status = {
                        "position": position,
                        "ready": self.controller.is_ready,
                        "motor": self.controller.is_energized,
                        "status": self.controller.is_last_command_success,
                    }
                    self.status_updated.emit(status)
                except Exception as e:
                    logging.error(f"Polling failed: {e}")
                    self.polling_failed.emit(str(e))
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
