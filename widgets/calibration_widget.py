from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging

from devices.polarizer_calibration import PolarizerCalibrationResult, run_polarizer_calibration


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class PolarizerCalibrationWidget(QWidget):
    def __init__(self, devices_tab=None, home_widget=None, parent=None):
        super().__init__(parent)
        self.devices_tab = devices_tab
        self.home_widget = home_widget
        self.thread = None
        self._active_key = None

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 360.0)
        self.start_spin.setDecimals(2)
        self.start_spin.setValue(0.0)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.0, 360.0)
        self.end_spin.setDecimals(2)
        self.end_spin.setValue(180.0)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 90.0)
        self.step_spin.setDecimals(2)
        self.step_spin.setValue(5.0)

        self.settle_spin = QDoubleSpinBox()
        self.settle_spin.setRange(0.0, 10.0)
        self.settle_spin.setDecimals(2)
        self.settle_spin.setValue(0.30)
        self.settle_spin.setSuffix(" s")

        self.sample_count_spin = QDoubleSpinBox()
        self.sample_count_spin.setRange(1, 20)
        self.sample_count_spin.setDecimals(0)
        self.sample_count_spin.setValue(3)

        self.input_status_label = QLabel("Input polarizer zero: ---")
        self.analyzer_status_label = QLabel("Analyzer polarizer zero: ---")
        self.progress_label = QLabel("Status: idle")

        self.input_btn = QPushButton("Calibrate Input Polarizer")
        self.input_btn.clicked.connect(self.start_input_calibration)
        self.analyzer_btn = QPushButton("Calibrate Analyzer Polarizer")
        self.analyzer_btn.clicked.connect(self.start_analyzer_calibration)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        settings_box = QGroupBox("Scan Settings")
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Start:"))
        settings_layout.addWidget(self.start_spin)
        settings_layout.addWidget(QLabel("End:"))
        settings_layout.addWidget(self.end_spin)
        settings_layout.addWidget(QLabel("Step:"))
        settings_layout.addWidget(self.step_spin)
        settings_layout.addWidget(QLabel("Settle:"))
        settings_layout.addWidget(self.settle_spin)
        settings_layout.addWidget(QLabel("Samples:"))
        settings_layout.addWidget(self.sample_count_spin)
        settings_box.setLayout(settings_layout)
        layout.addWidget(settings_box)

        input_box = QGroupBox("Input Polarizer")
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Signal source: Thorlabs power meter"))
        input_layout.addWidget(self.input_status_label)
        input_layout.addWidget(self.input_btn)
        input_box.setLayout(input_layout)
        layout.addWidget(input_box)

        analyzer_box = QGroupBox("Analyzer Polarizer")
        analyzer_layout = QVBoxLayout()
        analyzer_layout.addWidget(QLabel("Signal source: Boxcar PMT value from Home tab signal channel"))
        analyzer_layout.addWidget(self.analyzer_status_label)
        analyzer_layout.addWidget(self.analyzer_btn)
        analyzer_box.setLayout(analyzer_layout)
        layout.addWidget(analyzer_box)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.refresh_labels()

    def refresh_labels(self):
        input_zero = self.devices_tab.input_polarizer_widget.controller.zero_offset_deg
        analyzer_zero = self.devices_tab.analyzer_polarizer_widget.controller.zero_offset_deg
        self.input_status_label.setText(f"Input polarizer zero: {input_zero:.2f} deg")
        self.analyzer_status_label.setText(f"Analyzer polarizer zero: {analyzer_zero:.2f} deg")
        self.devices_tab.input_polarizer_widget.refresh_calibration_display()
        self.devices_tab.analyzer_polarizer_widget.refresh_calibration_display()

    def set_buttons_enabled(self, enabled: bool):
        self.input_btn.setEnabled(enabled)
        self.analyzer_btn.setEnabled(enabled)

    def _scan_kwargs(self) -> dict:
        return {
            "start_deg": float(self.start_spin.value()),
            "end_deg": float(self.end_spin.value()),
            "step_deg": float(self.step_spin.value()),
            "settle_time_s": float(self.settle_spin.value()),
            "sample_count": int(self.sample_count_spin.value()),
        }

    def _ensure_idle(self) -> bool:
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Calibration is already running.")
            return False
        return True

    def start_input_calibration(self):
        if not self._ensure_idle():
            return
        polarizer = self.devices_tab.input_polarizer_widget.controller
        power_meter = self.devices_tab.power_meter_widget.controller
        if not polarizer.is_connected:
            QMessageBox.warning(self, "Not Ready", "Input polarizer is not connected.")
            return
        if power_meter is None or not power_meter.is_connected:
            QMessageBox.warning(self, "Not Ready", "Power meter is not connected.")
            return

        self._active_key = "input"
        self.thread = PolarizerCalibrationThread(
            polarizer_controller=polarizer,
            read_signal=power_meter.read_power_watts,
            label="Input polarizer",
            **self._scan_kwargs(),
        )
        self._wire_thread()

    def start_analyzer_calibration(self):
        if not self._ensure_idle():
            return
        polarizer = self.devices_tab.analyzer_polarizer_widget.controller
        boxcar = self.devices_tab.boxcar_widget.controller
        if not polarizer.is_connected:
            QMessageBox.warning(self, "Not Ready", "Analyzer polarizer is not connected.")
            return
        if boxcar is None or not boxcar.is_connected:
            QMessageBox.warning(self, "Not Ready", "Boxcar is not connected.")
            return

        signal_channel = int(self.home_widget.channel_combo_2.currentText().replace("CH", ""))
        self._active_key = "analyzer"
        self.thread = PolarizerCalibrationThread(
            polarizer_controller=polarizer,
            read_signal=lambda: boxcar.read_analog(signal_channel),
            label=f"Analyzer polarizer (CH{signal_channel})",
            **self._scan_kwargs(),
        )
        self._wire_thread()

    def _wire_thread(self):
        self.set_buttons_enabled(False)
        self.progress_label.setText("Status: scanning...")
        self.thread.progress_updated.connect(self.on_progress)
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_error.connect(self.on_error)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def on_progress(self, angle_deg: float, signal_value: float):
        self.progress_label.setText(f"Status: scanning raw {angle_deg:.2f} deg, signal {signal_value:.6g}")

    def on_success(self, result: PolarizerCalibrationResult):
        if self._active_key == "input":
            controller = self.devices_tab.input_polarizer_widget.controller
        else:
            controller = self.devices_tab.analyzer_polarizer_widget.controller

        controller.set_zero_offset(
            result.zero_offset_deg,
            persist=True,
            extra_metadata={
                "baseline": round(result.baseline, 8),
                "amplitude": round(result.amplitude, 8),
                "fit_success": result.fit_success,
            },
        )
        self.refresh_labels()
        self.draw_result(result)
        self.progress_label.setText(f"Status: done. Zero angle = {result.zero_offset_deg:.2f} deg")

    def on_error(self, message: str):
        self.progress_label.setText("Status: calibration failed")
        QMessageBox.critical(self, "Calibration Error", message)

    def on_thread_finished(self):
        self.set_buttons_enabled(True)
        self.thread = None
        self._active_key = None

    def draw_result(self, result: PolarizerCalibrationResult):
        self.ax.clear()
        self.ax.plot(result.angles_deg, result.signals, "o", label="measurement", color="tab:blue")
        self.ax.plot(result.fit_angles_deg, result.fit_signals, "-", label="fit", color="tab:red")
        self.ax.set_xlabel("Raw Elliptec angle (deg)")
        self.ax.set_ylabel("Signal (a.u.)")
        self.ax.set_title(f"Zero offset = {result.zero_offset_deg:.2f} deg")
        self.ax.legend()
        self.canvas.draw()


class PolarizerCalibrationThread(QThread):
    progress_updated = pyqtSignal(float, float)
    finished_success = pyqtSignal(object)
    finished_error = pyqtSignal(str)

    def __init__(
        self,
        polarizer_controller,
        read_signal,
        label: str,
        start_deg: float,
        end_deg: float,
        step_deg: float,
        settle_time_s: float,
        sample_count: int,
        parent=None,
    ):
        super().__init__(parent)
        self.polarizer_controller = polarizer_controller
        self.read_signal = read_signal
        self.label = label
        self.start_deg = start_deg
        self.end_deg = end_deg
        self.step_deg = step_deg
        self.settle_time_s = settle_time_s
        self.sample_count = sample_count

    def run(self):
        try:
            result = run_polarizer_calibration(
                polarizer_controller=self.polarizer_controller,
                read_signal=self.read_signal,
                start_deg=self.start_deg,
                end_deg=self.end_deg,
                step_deg=self.step_deg,
                settle_time_s=self.settle_time_s,
                sample_count=self.sample_count,
                progress_callback=lambda angle, signal: self.progress_updated.emit(angle, signal),
            )
            self.finished_success.emit(result)
        except Exception as exc:
            logging.error(f"{self.label} calibration failed: {exc}")
            self.finished_error.emit(str(exc))
