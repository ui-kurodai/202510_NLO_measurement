from __future__ import annotations

from collections import deque
import logging
import time
import importlib.util
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, QTime, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTimeEdit,
    QVBoxLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

POWER_UNITS = [
    ("nW", 1e9),
    ("uW", 1e6),
    ("mW", 1e3),
    ("W", 1.0),
    ("kW", 1e-3),
    ("MW", 1e-6),
    ("GW", 1e-9),
]


class OphirPowerMeterWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Ophir 3A Power Meter", parent)
        self.controller = None
        self.polling_thread = None
        self._history = deque()
        self._last_power_w = None

        self.scan_btn = QPushButton("Scan Ophir USB")
        self.scan_btn.clicked.connect(self.scan_devices)
        self.device_combo = QComboBox()

        self.connect_btn = QPushButton("Open Device")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.power_label = QLabel("Power: ---- W")
        font = self.power_label.font()
        font.setPointSizeF(font.pointSizeF() * 2)
        self.power_label.setFont(font)

        self.sensor_label = QLabel("Sensor: ---")
        self.version_label = QLabel("COM version: ---")

        self.mode_combo = QComboBox()
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.setEditable(True)
        self.wavelength_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.wavelength_combo.lineEdit().setPlaceholderText("Select wavelength or type nm + Enter")
        self.wavelength_combo.lineEdit().returnPressed.connect(self.add_typed_wavelength)
        self.wavelength_combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.wavelength_combo.customContextMenuRequested.connect(self.show_wavelength_context_menu)
        self.wavelength_combo.view().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.wavelength_combo.view().customContextMenuRequested.connect(self.show_wavelength_view_context_menu)
        self.range_combo = QComboBox()
        self.pulse_length_combo = QComboBox()

        self.refresh_settings_btn = QPushButton("Refresh Settings")
        self.refresh_settings_btn.clicked.connect(self.refresh_settings)
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.zero_btn = QPushButton("Zero")
        self.zero_btn.clicked.connect(self.zero_device)
        self.reset_btn = QPushButton("Reset Device")
        self.reset_btn.clicked.connect(self.reset_device)

        self.stream_mode_combo = QComboBox()
        self.stream_mode_combo.addItems(["Default", "Immediate", "Turbo"])
        self.stream_mode_combo.currentTextChanged.connect(self.update_stream_mode_hint)
        self.turbo_freq_spin = QDoubleSpinBox()
        self.turbo_freq_spin.setDecimals(2)
        self.turbo_freq_spin.setRange(0.1, 1000.0)
        self.turbo_freq_spin.setValue(15.0)
        self.turbo_freq_spin.setSuffix(" Hz")
        self.apply_stream_btn = QPushButton("Apply Stream Mode")
        self.apply_stream_btn.clicked.connect(self.apply_stream_mode)
        self.stream_hint_label = QLabel("Turbo sets the requested high-speed sampling frequency when supported.")

        self.history_window_edit = QTimeEdit()
        self.history_window_edit.setDisplayFormat("HH:mm:ss")
        self.history_window_edit.setTime(QTime(0, 1, 0))
        self.history_window_edit.timeChanged.connect(lambda _: self.update_plot())
        self.poll_interval_spin = QDoubleSpinBox()
        self.poll_interval_spin.setDecimals(2)
        self.poll_interval_spin.setRange(0.02, 10.0)
        self.poll_interval_spin.setValue(0.2)
        self.poll_interval_spin.setSuffix(" s")
        self.poll_interval_spin.valueChanged.connect(self.update_poll_interval)
        self.auto_y_checkbox = QCheckBox("Auto Y")
        self.auto_y_checkbox.setChecked(True)
        self.auto_y_checkbox.toggled.connect(lambda _: self.update_plot())
        self.power_unit_combo = QComboBox()
        for unit, scale in POWER_UNITS:
            self.power_unit_combo.addItem(unit, scale)
        self.power_unit_combo.setCurrentText("W")
        self.power_unit_combo.currentIndexChanged.connect(lambda _: self.update_power_display())
        self.power_unit_combo.currentIndexChanged.connect(lambda _: self.update_plot())
        self.y_min_spin = QDoubleSpinBox()
        self.y_max_spin = QDoubleSpinBox()
        for spin in (self.y_min_spin, self.y_max_spin):
            spin.setDecimals(6)
            spin.setRange(-1e6, 1e6)
        self.y_min_spin.setValue(0.0)
        self.y_max_spin.setValue(1.0)

        self.legacy_command_edit = QLineEdit()
        self.legacy_command_edit.setPlaceholderText("Legacy command, e.g. EF or SP")
        self.legacy_ask_btn = QPushButton("Ask")
        self.legacy_ask_btn.clicked.connect(self.ask_legacy_command)
        self.legacy_response_label = QLabel("Response: ---")

        self.figure = Figure(figsize=(5, 2.8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(260)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Power (W)")
        self.line, = self.ax.plot([], [], "-", color="blue")
        self.y_min_spin.valueChanged.connect(lambda _: self.update_plot())
        self.y_max_spin.valueChanged.connect(lambda _: self.update_plot())

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.scan_btn)
        layout.addWidget(self.device_combo)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.power_label)
        layout.addWidget(self.sensor_label)
        layout.addWidget(self.version_label)

        settings_layout = QFormLayout()
        settings_layout.addRow("Mode:", self.mode_combo)
        settings_layout.addRow("Wavelength:", self.wavelength_combo)
        settings_layout.addRow("Range:", self.range_combo)
        settings_layout.addRow("Pulse length:", self.pulse_length_combo)
        layout.addLayout(settings_layout)

        settings_buttons = QHBoxLayout()
        settings_buttons.addWidget(self.refresh_settings_btn)
        settings_buttons.addWidget(self.apply_settings_btn)
        settings_buttons.addWidget(self.zero_btn)
        settings_buttons.addWidget(self.reset_btn)
        layout.addLayout(settings_buttons)

        stream_layout = QGridLayout()
        stream_layout.addWidget(QLabel("Stream mode:"), 0, 0)
        stream_layout.addWidget(self.stream_mode_combo, 0, 1)
        stream_layout.addWidget(QLabel("Turbo freq:"), 0, 2)
        stream_layout.addWidget(self.turbo_freq_spin, 0, 3)
        stream_layout.addWidget(self.apply_stream_btn, 0, 4)
        layout.addLayout(stream_layout)
        layout.addWidget(self.stream_hint_label)

        plot_controls = QGridLayout()
        plot_controls.addWidget(QLabel("History:"), 0, 0)
        plot_controls.addWidget(self.history_window_edit, 0, 1)
        plot_controls.addWidget(QLabel("Poll:"), 0, 2)
        plot_controls.addWidget(self.poll_interval_spin, 0, 3)
        plot_controls.addWidget(QLabel("Unit:"), 1, 0)
        plot_controls.addWidget(self.power_unit_combo, 1, 1)
        plot_controls.addWidget(self.auto_y_checkbox, 1, 2)
        plot_controls.addWidget(QLabel("Y min:"), 2, 0)
        plot_controls.addWidget(self.y_min_spin, 2, 1)
        plot_controls.addWidget(QLabel("Y max:"), 2, 2)
        plot_controls.addWidget(self.y_max_spin, 2, 3)
        layout.addLayout(plot_controls)
        layout.addWidget(self.canvas)

        legacy_layout = QHBoxLayout()
        legacy_layout.addWidget(self.legacy_command_edit, 1)
        legacy_layout.addWidget(self.legacy_ask_btn)
        layout.addLayout(legacy_layout)
        layout.addWidget(self.legacy_response_label)
        self.setLayout(layout)
        self.set_controls_enabled(False)

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
                self.refresh_settings()
                self.set_controls_enabled(True)
                self.polling_thread = OphirPollingThread(self.controller)
                self.polling_thread.readings_updated.connect(self.update_readings)
                self.polling_thread.start()
            except Exception as exc:
                self.controller = None
                QMessageBox.critical(self, "Connection Error", str(exc))
        else:
            self.shutdown()
            self.connect_btn.setText("Open Device")
            self.power_label.setText("Power: ---- W")
            self.sensor_label.setText("Sensor: ---")
            self.version_label.setText("COM version: ---")
            self.clear_settings()
            self.set_controls_enabled(False)

    def _refresh_device_labels(self):
        if self.controller is None:
            return
        try:
            self.sensor_label.setText(f"Sensor: {self.controller.get_sensor_info()}")
        except Exception as exc:
            self.sensor_label.setText(f"Sensor: unavailable ({exc})")
        version = self.controller.get_version()
        self.version_label.setText(f"COM version: {version or '---'}")

    def set_controls_enabled(self, enabled: bool):
        for widget in (
            self.mode_combo,
            self.wavelength_combo,
            self.range_combo,
            self.pulse_length_combo,
            self.refresh_settings_btn,
            self.apply_settings_btn,
            self.zero_btn,
            self.reset_btn,
            self.stream_mode_combo,
            self.turbo_freq_spin,
            self.apply_stream_btn,
            self.legacy_command_edit,
            self.legacy_ask_btn,
        ):
            widget.setEnabled(enabled)

    def clear_settings(self):
        for combo in (self.mode_combo, self.wavelength_combo, self.range_combo, self.pulse_length_combo):
            combo.clear()
        self._history.clear()
        self.update_plot()

    def _populate_combo(self, combo: QComboBox, current_index: int | None, options: list[str]):
        combo.blockSignals(True)
        line_edit = combo.lineEdit() if combo.isEditable() else None
        line_text = line_edit.text() if line_edit is not None else ""
        combo.clear()
        for index, option in enumerate(options):
            combo.addItem(option, index)
        if current_index is not None and 0 <= current_index < combo.count():
            combo.setCurrentIndex(current_index)
        elif line_edit is not None:
            line_edit.setText(line_text)
        combo.blockSignals(False)
        combo.setEnabled(combo.count() > 0 and self.controller is not None)

    def refresh_settings(self):
        if self.controller is None:
            return
        try:
            mode = self.controller.get_measurement_modes()
            wavelength = self.controller.get_wavelength_options()
            ranges = self.controller.get_range_options()
            pulse_lengths = self.controller.get_pulse_length_options()
            self._populate_combo(self.mode_combo, mode.current_index, mode.options)
            self._populate_combo(self.wavelength_combo, wavelength.current_index, wavelength.options)
            self._populate_combo(self.range_combo, ranges.current_index, ranges.options)
            self._populate_combo(self.pulse_length_combo, pulse_lengths.current_index, pulse_lengths.options)
            self._refresh_device_labels()
        except Exception as exc:
            QMessageBox.critical(self, "Settings Error", str(exc))

    def apply_settings(self):
        if self.controller is None:
            return
        was_polling = self.stop_polling()
        try:
            if self.mode_combo.count():
                self.controller.set_measurement_mode_index(int(self.mode_combo.currentData()))
            if self.wavelength_combo.count():
                wavelength_data = self.wavelength_combo.currentData()
                if wavelength_data is None:
                    self.controller.add_or_select_wavelength_nm(self._parse_wavelength_text(self.wavelength_combo.currentText()))
                else:
                    self.controller.set_wavelength_index(int(wavelength_data))
            if self.range_combo.count():
                self.controller.set_range_index(int(self.range_combo.currentData()))
            if self.pulse_length_combo.count():
                self.controller.set_pulse_length_index(int(self.pulse_length_combo.currentData()))
            self.refresh_settings()
        except Exception as exc:
            QMessageBox.critical(self, "Apply Settings Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def add_typed_wavelength(self):
        if self.controller is None:
            return
        text = self.wavelength_combo.currentText().strip()
        if not text:
            return
        was_polling = self.stop_polling()
        try:
            selected = self.controller.add_or_select_wavelength_nm(self._parse_wavelength_text(text))
            self.refresh_settings()
            index = self.wavelength_combo.findText(selected)
            if index >= 0:
                self.wavelength_combo.setCurrentIndex(index)
        except Exception as exc:
            QMessageBox.critical(self, "Add Wavelength Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def show_wavelength_context_menu(self, position):
        self._show_wavelength_context_menu(self.wavelength_combo.currentIndex(), self.wavelength_combo.mapToGlobal(position))

    def show_wavelength_view_context_menu(self, position):
        index = self.wavelength_combo.view().indexAt(position)
        row = index.row() if index.isValid() else self.wavelength_combo.currentIndex()
        self._show_wavelength_context_menu(row, self.wavelength_combo.view().mapToGlobal(position))

    def _show_wavelength_context_menu(self, index: int, global_position):
        menu = QMenu(self)
        add_action = QAction("Add typed wavelength", self)
        add_action.triggered.connect(self.add_typed_wavelength)
        menu.addAction(add_action)
        if index >= 0:
            delete_action = QAction(f"Delete {self.wavelength_combo.itemText(index)}", self)
            delete_action.triggered.connect(lambda: self.delete_wavelength(index))
            menu.addAction(delete_action)
        menu.exec(global_position)

    def delete_wavelength(self, index: int):
        if self.controller is None:
            return
        was_polling = self.stop_polling()
        try:
            self.controller.delete_wavelength_index(index)
            self.refresh_settings()
        except NotImplementedError as exc:
            QMessageBox.information(self, "Delete Wavelength", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Delete Wavelength Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def _parse_wavelength_text(self, text: str) -> float:
        cleaned = text.strip().lower().replace("nm", "").strip()
        return float(cleaned)

    def apply_stream_mode(self):
        if self.controller is None:
            return
        was_polling = self.stop_polling()
        try:
            mode = self.stream_mode_combo.currentText()
            if mode == "Default":
                self.controller.configure_default_stream_mode()
            elif mode == "Immediate":
                self.controller.configure_immediate_stream_mode()
            elif mode == "Turbo":
                self.controller.configure_turbo_stream_mode(self.turbo_freq_spin.value())
        except Exception as exc:
            QMessageBox.critical(self, "Stream Mode Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def update_stream_mode_hint(self, mode: str):
        if mode == "Turbo":
            self.stream_hint_label.setText(
                "Turbo requests high-speed streaming at the selected frequency; it only works when the meter/sensor supports it."
            )
        elif mode == "Immediate":
            self.stream_hint_label.setText("Immediate asks the device for the newest reading without default buffered averaging.")
        else:
            self.stream_hint_label.setText("Default uses Ophir's standard stream behavior for the current measurement mode.")

    def zero_device(self):
        if self.controller is None:
            return
        was_polling = self.stop_polling()
        try:
            self.controller.zero(wait_s=30.0)
        except Exception as exc:
            QMessageBox.critical(self, "Zero Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def reset_device(self):
        if self.controller is None:
            return
        was_polling = self.stop_polling()
        try:
            self.controller.reset_device()
            self.refresh_settings()
        except Exception as exc:
            QMessageBox.critical(self, "Reset Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def ask_legacy_command(self):
        if self.controller is None:
            return
        command = self.legacy_command_edit.text().strip()
        if not command:
            return
        was_polling = self.stop_polling()
        try:
            response = self.controller.ask_legacy_command(command)
            self.legacy_response_label.setText(f"Response: {response}")
        except Exception as exc:
            QMessageBox.critical(self, "Legacy Command Error", str(exc))
        finally:
            if was_polling:
                self.start_polling()

    def start_polling(self):
        if self.controller is None or self.polling_thread is not None:
            return
        self.polling_thread = OphirPollingThread(self.controller, interval_s=self.poll_interval_spin.value())
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

    def update_power_label(self, power_w: float):
        self._last_power_w = power_w
        self.update_power_display()

    def update_power_display(self):
        unit = self.current_power_unit()
        if self._last_power_w is None:
            self.power_label.setText(f"Power: ---- {unit}")
            self.ax.set_ylabel(f"Power ({unit})")
            return
        self.power_label.setText(f"Power: {self.scale_power(self._last_power_w):.6g} {unit}")

    def update_readings(self, readings: list):
        now = time.monotonic()
        for reading in readings:
            self._history.append((now, reading.power_w))
            now += 1e-6
            self.update_power_label(reading.power_w)
        self.trim_history()
        self.update_plot()

    def trim_history(self):
        cutoff = time.monotonic() - self.history_window_seconds()
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def history_window_seconds(self) -> float:
        qtime = self.history_window_edit.time()
        seconds = qtime.hour() * 3600 + qtime.minute() * 60 + qtime.second()
        return max(1.0, float(seconds))

    def current_power_unit(self) -> str:
        return self.power_unit_combo.currentText() or "W"

    def power_scale(self) -> float:
        return float(self.power_unit_combo.currentData() or 1.0)

    def scale_power(self, power_w: float) -> float:
        return power_w * self.power_scale()

    def update_plot(self):
        if not hasattr(self, "line"):
            return
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
                if ymin == ymax:
                    pad = max(abs(ymin) * 0.1, 1e-9)
                else:
                    pad = (ymax - ymin) * 0.1
                self.ax.set_ylim(ymin - pad, ymax + pad)
            else:
                self.ax.set_ylim(self.y_min_spin.value(), self.y_max_spin.value())
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(f"Power ({self.current_power_unit()})")
        self.canvas.draw_idle()

    def shutdown(self):
        self.stop_polling()
        if self.controller is not None:
            try:
                self.controller.shutdown()
            except Exception as exc:
                logging.warning("Failed to shutdown Ophir controller: %s", exc)
            self.controller = None


class OphirPollingThread(QThread):
    readings_updated = pyqtSignal(list)

    def __init__(self, controller, interval_s: float = 1.0, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.interval_s = interval_s
        self._running = True

    def run(self):
        try:
            self.controller.start_stream()
            while self._running:
                try:
                    readings = self.controller.read_available_data()
                    if readings:
                        self.readings_updated.emit(readings)
                except Exception as exc:
                    logging.debug("Ophir polling failed: %s", exc)
                time.sleep(self.interval_s)
        finally:
            try:
                self.controller.stop_stream()
            except Exception:
                pass

    def stop(self):
        self._running = False
        self.wait()
