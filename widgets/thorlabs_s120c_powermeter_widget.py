from __future__ import annotations

from collections import deque
import importlib.util
import logging
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QLocale, QThread, Qt, QTime, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTimeEdit,
    QVBoxLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

_CONTROL_PATH = Path(__file__).resolve().parents[1] / "devices" / "thorlabs_s120c_powermeter_control.py"
_CONTROL_SPEC = importlib.util.spec_from_file_location("thorlabs_s120c_powermeter_control", _CONTROL_PATH)
if _CONTROL_SPEC is None or _CONTROL_SPEC.loader is None:
    raise ImportError(f"Cannot load Thorlabs control module from {_CONTROL_PATH}")
_control_module = importlib.util.module_from_spec(_CONTROL_SPEC)
sys.modules[_CONTROL_SPEC.name] = _control_module
_CONTROL_SPEC.loader.exec_module(_control_module)
ThorlabsS120CPowerMeterController = _control_module.ThorlabsS120CPowerMeterController

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")

POWER_UNITS = [
    ("Auto", None),
    ("nW", 1e9),
    ("uW", 1e6),
    ("mW", 1e3),
    ("W", 1.0),
    ("kW", 1e-3),
    ("MW", 1e-6),
    ("GW", 1e-9),
]


class ThorlabsS120CPowerMeterWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Thorlabs S120C Power Meter", parent)
        self.controller = None
        self.polling_thread = None
        self.zero_thread = None
        self._history = deque()
        self._last_power_w = None

        self.scan_btn = QPushButton("Scan Thorlabs USB")
        self.scan_btn.clicked.connect(self.scan_devices)
        self.device_combo = QComboBox()
        self.library_path_edit = QLineEdit()
        self.library_path_edit.setPlaceholderText("Optional folder containing Thorlabs.TLPM_64.Interop.dll")
        self.browse_library_btn = QPushButton("Browse DLL Folder")
        self.browse_library_btn.clicked.connect(self.browse_library_path)

        self.connect_btn = QPushButton("Open Device")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.power_label = QLabel("Power: ---- W")
        font = self.power_label.font()
        font.setPointSizeF(font.pointSizeF() * 2)
        self.power_label.setFont(font)

        self.sensor_label = QLabel("Sensor: ---")
        self.version_label = QLabel("Driver version: ---")
        self.zero_label = QLabel("Zero offset: 0 W")
        self.status_label = QLabel("Status: idle")

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setLocale(QLocale.c())
        self.wavelength_spin.setDecimals(3)
        self.wavelength_spin.setRange(1.0, 100000.0)
        self.wavelength_spin.setValue(532.0)
        self.wavelength_spin.setSuffix(" nm")

        self.average_time_spin = QDoubleSpinBox()
        self.average_time_spin.setLocale(QLocale.c())
        self.average_time_spin.setDecimals(4)
        self.average_time_spin.setRange(0.0001, 10.0)
        self.average_time_spin.setValue(0.001)
        self.average_time_spin.setSuffix(" s")

        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.zero_btn = QPushButton("Zero Offset")
        self.zero_btn.clicked.connect(self.zero_device)

        self.history_window_edit = QTimeEdit()
        self.history_window_edit.setDisplayFormat("HH:mm:ss")
        self.history_window_edit.setTime(QTime(0, 1, 0))
        self.history_window_edit.timeChanged.connect(lambda _: self.update_plot())
        self.poll_interval_spin = QDoubleSpinBox()
        self.poll_interval_spin.setLocale(QLocale.c())
        self.poll_interval_spin.setDecimals(2)
        self.poll_interval_spin.setRange(0.02, 10.0)
        self.poll_interval_spin.setValue(0.1)
        self.poll_interval_spin.setSuffix(" s")
        self.poll_interval_spin.valueChanged.connect(self.update_poll_interval)
        self.clear_stream_btn = QPushButton("Clear Stream")
        self.clear_stream_btn.clicked.connect(self.clear_stream_history)
        self.auto_y_checkbox = QCheckBox("Auto Y")
        self.auto_y_checkbox.setChecked(True)
        self.power_unit_combo = QComboBox()
        for unit, scale in POWER_UNITS:
            self.power_unit_combo.addItem(unit, scale)
        self.power_unit_combo.setCurrentText("Auto")
        self.power_unit_combo.currentIndexChanged.connect(lambda _: self.update_power_display())
        self.power_unit_combo.currentIndexChanged.connect(lambda _: self.update_plot())

        self.figure = Figure(figsize=(5, 2.8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(260)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Power (W)")
        self.line, = self.ax.plot([], [], "-", color="blue")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.scan_btn)
        layout.addWidget(self.device_combo)
        library_layout = QHBoxLayout()
        library_layout.addWidget(self.library_path_edit, 1)
        library_layout.addWidget(self.browse_library_btn)
        layout.addLayout(library_layout)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.power_label)
        layout.addWidget(self.sensor_label)
        layout.addWidget(self.version_label)
        layout.addWidget(self.zero_label)
        layout.addWidget(self.status_label)

        settings_layout = QFormLayout()
        settings_layout.addRow("Wavelength:", self.wavelength_spin)
        settings_layout.addRow("Average time:", self.average_time_spin)
        layout.addLayout(settings_layout)

        settings_buttons = QHBoxLayout()
        settings_buttons.addWidget(self.apply_settings_btn)
        settings_buttons.addWidget(self.zero_btn)
        layout.addLayout(settings_buttons)

        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("History:"))
        plot_controls.addWidget(self.history_window_edit)
        plot_controls.addWidget(QLabel("Poll:"))
        plot_controls.addWidget(self.poll_interval_spin)
        plot_controls.addWidget(QLabel("Unit:"))
        plot_controls.addWidget(self.power_unit_combo)
        plot_controls.addWidget(self.auto_y_checkbox)
        plot_controls.addWidget(self.clear_stream_btn)
        layout.addLayout(plot_controls)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.set_controls_enabled(False)

    def scan_devices(self):
        self.device_combo.clear()
        try:
            controller = self.controller or self._make_controller()
            for resource in controller.scan_usb():
                self.device_combo.addItem(resource, resource)
            if self.device_combo.count() == 0:
                QMessageBox.information(self, "Device Not Found", "No Thorlabs power meter was detected.")
        except Exception as exc:
            QMessageBox.critical(self, "Thorlabs Scan Error", str(exc))

    def browse_library_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Thorlabs TLPM DLL Folder")
        if folder:
            self.library_path_edit.setText(folder)

    def _library_path(self) -> str | None:
        text = self.library_path_edit.text().strip()
        return text or None

    def _make_controller(self):
        return ThorlabsS120CPowerMeterController(library_path=self._library_path())

    def toggle_connection(self):
        if self.controller is None:
            resource = self.device_combo.currentData()
            try:
                self.controller = self._make_controller()
                self.controller.connect(resource)
                self.controller.set_wavelength_nm(self.wavelength_spin.value())
                self.controller.set_average_time(self.average_time_spin.value())
                self.connect_btn.setText("Close Device")
                self.status_label.setText("Status: connected")
                self._refresh_device_labels()
                self.set_controls_enabled(True)
                self.start_polling()
            except Exception as exc:
                self.controller = None
                QMessageBox.critical(self, "Connection Error", str(exc))
        else:
            self.shutdown()
            self.connect_btn.setText("Open Device")
            self.power_label.setText("Power: ---- W")
            self.sensor_label.setText("Sensor: ---")
            self.version_label.setText("Driver version: ---")
            self.zero_label.setText("Zero offset: 0 W")
            self.status_label.setText("Status: disconnected")
            self.clear_stream_history()
            self.set_controls_enabled(False)

    def _refresh_device_labels(self):
        if self.controller is None:
            return
        self.sensor_label.setText(f"Sensor: {self.controller.get_sensor_info()}")
        self.version_label.setText(f"Driver version: {self.controller.get_version() or '---'}")
        self.zero_label.setText(f"Zero offset: {self.controller.zero_offset_w:.6g} W")

    def set_controls_enabled(self, enabled: bool):
        for widget in (
            self.wavelength_spin,
            self.average_time_spin,
            self.apply_settings_btn,
            self.zero_btn,
            self.clear_stream_btn,
            self.poll_interval_spin,
            self.power_unit_combo,
            self.auto_y_checkbox,
        ):
            widget.setEnabled(enabled)

    def apply_settings(self):
        if self.controller is None:
            return
        was_polling = self.stop_polling()
        try:
            self.controller.set_wavelength_nm(self.wavelength_spin.value())
            self.controller.set_average_time(self.average_time_spin.value())
            self.status_label.setText("Status: settings applied")
        except Exception as exc:
            QMessageBox.critical(self, "Apply Settings Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def zero_device(self):
        if self.controller is None:
            return
        if self.zero_thread is not None and self.zero_thread.isRunning():
            return
        was_polling = self.stop_polling()
        self.set_controls_enabled(False)
        self.zero_btn.setText("Zeroing...")
        self.status_label.setText("Status: measuring background offset")
        self.zero_thread = ThorlabsZeroThread(
            self.controller,
            wait_s=self.controller.zero_wait_s,
            restart_polling=was_polling,
        )
        self.zero_thread.finished_ok.connect(self.finish_zero)
        self.zero_thread.failed.connect(self.zero_failed)
        self.zero_thread.start()

    def finish_zero(self, restart_polling: bool):
        self.zero_btn.setText("Zero Offset")
        self.set_controls_enabled(True)
        self._refresh_device_labels()
        self.status_label.setText("Status: zero offset updated")
        if restart_polling:
            self.start_polling()

    def zero_failed(self, message: str, restart_polling: bool):
        self.zero_btn.setText("Zero Offset")
        self.set_controls_enabled(True)
        self.status_label.setText("Status: zero failed")
        if restart_polling:
            self.start_polling()
        QMessageBox.critical(self, "Zero Error", message)

    def start_polling(self):
        if self.controller is None or self.polling_thread is not None:
            return
        self.polling_thread = ThorlabsPollingThread(self.controller, interval_s=self.poll_interval_spin.value())
        self.polling_thread.readings_updated.connect(self.update_readings)
        self.polling_thread.start()

    def stop_polling(self) -> bool:
        if self.polling_thread is None:
            return False
        self.polling_thread.stop()
        self.polling_thread = None
        return True

    def update_poll_interval(self, value: float):
        if self.polling_thread is not None:
            self.polling_thread.interval_s = float(value)

    def clear_stream_history(self):
        self._history.clear()
        self._last_power_w = None
        self.update_power_display()
        self.update_plot()

    def update_readings(self, readings: list):
        now = time.monotonic()
        for reading in readings:
            self._history.append((now, reading.power_w))
            now += 1e-6
            self._last_power_w = reading.power_w
        self.trim_history()
        self.update_power_display()
        self.update_plot()

    def trim_history(self):
        cutoff = time.monotonic() - self.history_window_seconds()
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def history_window_seconds(self) -> float:
        qtime = self.history_window_edit.time()
        return max(1.0, float(qtime.hour() * 3600 + qtime.minute() * 60 + qtime.second()))

    def current_power_unit(self) -> str:
        selected = self.power_unit_combo.currentText() or "W"
        if selected != "Auto":
            return selected
        return self.auto_power_unit()

    def power_scale(self) -> float:
        selected = self.power_unit_combo.currentText() or "W"
        if selected == "Auto":
            return dict(POWER_UNITS[1:]).get(self.auto_power_unit(), 1.0)
        return float(self.power_unit_combo.currentData() or 1.0)

    def scale_power(self, power_w: float) -> float:
        return float(power_w) * self.power_scale()

    def auto_power_unit(self) -> str:
        values = [abs(power) for _, power in self._history if power is not None]
        if self._last_power_w is not None:
            values.append(abs(self._last_power_w))
        peak = max(values, default=0.0)
        if peak < 1e-6:
            return "nW"
        if peak < 1e-3:
            return "uW"
        if peak < 1.0:
            return "mW"
        if peak < 1e3:
            return "W"
        if peak < 1e6:
            return "kW"
        if peak < 1e9:
            return "MW"
        return "GW"

    def update_power_display(self):
        unit = self.current_power_unit()
        if self._last_power_w is None:
            self.power_label.setText(f"Power: ---- {unit}")
            self.ax.set_ylabel(f"Power ({unit})")
            return
        self.power_label.setText(f"Power: {self.scale_power(self._last_power_w):.6g} {unit}")
        self.ax.set_ylabel(f"Power ({unit})")

    def update_plot(self):
        if not self._history:
            self.line.set_data([], [])
        else:
            latest = self._history[-1][0]
            xs = [timestamp - latest for timestamp, _ in self._history]
            ys = [self.scale_power(power) for _, power in self._history]
            self.line.set_data(xs, ys)
            self.ax.set_xlim(-self.history_window_seconds(), 0.0)
            if self.auto_y_checkbox.isChecked():
                ymin = min(ys)
                ymax = max(ys)
                pad = max(abs(ymax - ymin) * 0.1, 1e-9)
                self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(f"Power ({self.current_power_unit()})")
        self.canvas.draw_idle()

    def shutdown(self):
        if self.zero_thread is not None and self.zero_thread.isRunning():
            self.zero_thread.wait(1000)
        self.stop_polling()
        if self.controller is not None:
            try:
                self.controller.shutdown()
            except Exception as exc:
                logging.warning("Failed to shutdown Thorlabs controller: %s", exc)
            self.controller = None


class ThorlabsPollingThread(QThread):
    readings_updated = pyqtSignal(list)

    def __init__(self, controller, interval_s: float = 0.1, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.interval_s = interval_s
        self._running = True

    def run(self):
        while self._running:
            try:
                self.readings_updated.emit(self.controller.read_available_data())
            except Exception as exc:
                logging.debug("Thorlabs polling failed: %s", exc)
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False
        self.wait()


class ThorlabsZeroThread(QThread):
    finished_ok = pyqtSignal(bool)
    failed = pyqtSignal(str, bool)

    def __init__(self, controller, wait_s: float, restart_polling: bool, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.wait_s = wait_s
        self.restart_polling = restart_polling

    def run(self):
        try:
            self.controller.zero(wait_s=self.wait_s)
        except Exception as exc:
            self.failed.emit(str(exc), self.restart_polling)
            return
        self.finished_ok.emit(self.restart_polling)
