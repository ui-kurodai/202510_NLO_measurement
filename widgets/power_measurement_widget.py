from __future__ import annotations

import csv
import json
import os
from datetime import datetime

from PyQt6.QtCore import QLocale, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from measurement_metadata import (
    build_sample_catalog_key,
    format_beam_profile_display,
    format_sample_display,
    load_beam_profile_catalog,
    load_sample_catalog,
    normalize_crystal_orientation,
)
from power_measurement import PowerMeasurementRunner

from crystaldatabase import CRYSTALS

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


class PowerMeasurementWidget(QGroupBox):
    def __init__(self, devices_tab=None, parent=None):
        super().__init__("Power Measurement", parent)
        self.devices_tab = devices_tab
        self.runner = None
        self.thread = None
        self._angle_spins = []
        self._sample_catalog_map = {}
        self._beam_profile_catalog_map = {}
        self._fundamental_completed = False
        self._shg_completed = False
        self._laser_controller = None
        self._last_fundamental_stats = None
        self._last_shg_scans = []
        self._last_metadata = {}
        self._last_saved_dir = None

        self.sample_preset_combo = QComboBox()
        self.sample_preset_combo.currentIndexChanged.connect(self._apply_selected_sample_preset)
        self.reload_samples_btn = QPushButton("Reload Samples")
        self.reload_samples_btn.clicked.connect(self.reload_sample_catalog)

        self.sample_edit = QLineEdit()
        self.sample_edit.setPlaceholderText("<sample id>_<cut axis>_<measurement id>")
        self.material_combo = QComboBox()
        self.material_combo.addItems(CRYSTALS.keys())

        self.beam_profile_combo = QComboBox()
        self.reload_beams_btn = QPushButton("Reload Beams")
        self.reload_beams_btn.clicked.connect(self.reload_beam_profile_catalog)

        self.axis_edit = QLineEdit()
        self.axis_edit.setPlaceholderText("100")

        self.fundamental_wavelength_spin = QDoubleSpinBox()
        self.shg_wavelength_spin = QDoubleSpinBox()
        for spin in (self.fundamental_wavelength_spin, self.shg_wavelength_spin):
            spin.setLocale(QLocale.c())
            spin.setDecimals(3)
            spin.setRange(1.0, 100000.0)
            spin.setSuffix(" nm")
        self.fundamental_wavelength_spin.setValue(1064.0)
        self.shg_wavelength_spin.setValue(532.0)
        self.fundamental_wavelength_spin.valueChanged.connect(
            lambda value: self.shg_wavelength_spin.setValue(value / 2.0)
        )

        self.angles_layout = QVBoxLayout()
        self.add_angle_btn = QPushButton("+")
        self.add_angle_btn.clicked.connect(self.add_angle_input)
        self.add_angle_input(default_value=0.0)

        self.scan_range_spin = QDoubleSpinBox()
        self.step_spin = QDoubleSpinBox()
        for spin in (self.scan_range_spin, self.step_spin):
            spin.setLocale(QLocale.c())
            spin.setDecimals(4)
            spin.setRange(0.0001, 360.0)
            spin.setSuffix(" deg")
        self.scan_range_spin.setValue(1.0)
        self.step_spin.setValue(0.05)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Experiment notes (optional)...")
        self.notes_edit.setFixedHeight(60)

        self.dry_run_checkbox = QCheckBox("Dry Run")
        self.powermeter_combo = QComboBox()
        self.powermeter_combo.addItem("Ophir 3A", "ophir")
        self.powermeter_combo.addItem("Thorlabs S120C", "thorlabs_s120c")
        self.powermeter_combo.currentIndexChanged.connect(lambda _: self.refresh_measurement_ranges(show_errors=False))
        self.power_unit_combo = QComboBox()
        for unit, scale in POWER_UNITS:
            self.power_unit_combo.addItem(unit, scale)
        self.power_unit_combo.setCurrentText("Auto")
        self.power_unit_combo.currentIndexChanged.connect(self.refresh_plot_units)
        self.fundamental_range_combo = QComboBox()
        self.shg_range_combo = QComboBox()
        for combo in (self.fundamental_range_combo, self.shg_range_combo):
            combo.addItem("Auto", None)
        self.refresh_ranges_btn = QPushButton("Refresh Ranges")
        self.refresh_ranges_btn.clicked.connect(self.refresh_measurement_ranges)

        self.measure_fundamental_btn = QPushButton("Measure Fundamental Power")
        self.measure_fundamental_btn.clicked.connect(self.start_fundamental_measurement)
        self.measure_shg_btn = QPushButton("Measure SHG Power")
        self.measure_shg_btn.clicked.connect(self.start_shg_measurement)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self.abort_measurement)
        self.fundamental_result_label = QLabel("Fundamental power: ---")

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(320)
        self.axes = []

        self.reload_reference_catalogs()

        root_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        root_layout.addWidget(scroll_area)
        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        sample_preset_layout = QHBoxLayout()
        sample_preset_layout.addWidget(QLabel("Sample preset:"))
        sample_preset_layout.addWidget(self.sample_preset_combo, 1)
        sample_preset_layout.addWidget(self.reload_samples_btn)
        layout.addLayout(sample_preset_layout)

        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Measurement ID:"))
        sample_layout.addWidget(self.sample_edit)
        sample_layout.addWidget(QLabel("material:"))
        sample_layout.addWidget(self.material_combo)
        layout.addLayout(sample_layout)

        beam_profile_layout = QHBoxLayout()
        beam_profile_layout.addWidget(QLabel("Beam profile:"))
        beam_profile_layout.addWidget(self.beam_profile_combo, 1)
        beam_profile_layout.addWidget(self.reload_beams_btn)
        layout.addLayout(beam_profile_layout)

        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(QLabel("Fundamental:"))
        wavelength_layout.addWidget(self.fundamental_wavelength_spin)
        wavelength_layout.addWidget(QLabel("SHG:"))
        wavelength_layout.addWidget(self.shg_wavelength_spin)
        layout.addLayout(wavelength_layout)

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Rotation axis:"))
        axis_layout.addWidget(self.axis_edit)
        layout.addLayout(axis_layout)

        angle_header = QHBoxLayout()
        angle_header.addWidget(QLabel("Estimated PM angle:"))
        angle_header.addStretch(1)
        angle_header.addWidget(self.add_angle_btn)
        layout.addLayout(angle_header)
        layout.addLayout(self.angles_layout)

        scan_layout = QHBoxLayout()
        scan_layout.addWidget(QLabel("Scan range (+/-):"))
        scan_layout.addWidget(self.scan_range_spin)
        scan_layout.addWidget(QLabel("Step:"))
        scan_layout.addWidget(self.step_spin)
        layout.addLayout(scan_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.dry_run_checkbox)
        mode_layout.addWidget(QLabel("Power meter:"))
        mode_layout.addWidget(self.powermeter_combo)
        mode_layout.addStretch(1)
        layout.addLayout(mode_layout)
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Power unit:"))
        unit_layout.addWidget(self.power_unit_combo)
        unit_layout.addStretch(1)
        layout.addLayout(unit_layout)
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Fundamental range:"))
        range_layout.addWidget(self.fundamental_range_combo)
        range_layout.addWidget(QLabel("SHG range:"))
        range_layout.addWidget(self.shg_range_combo)
        range_layout.addWidget(self.refresh_ranges_btn)
        layout.addLayout(range_layout)
        layout.addWidget(QLabel("Notes:"))
        layout.addWidget(self.notes_edit)
        run_buttons = QHBoxLayout()
        run_buttons.addWidget(self.measure_fundamental_btn)
        run_buttons.addWidget(self.measure_shg_btn)
        layout.addLayout(run_buttons)
        layout.addWidget(self.abort_btn)
        layout.addWidget(self.fundamental_result_label)
        layout.addWidget(self.canvas)
        self.setLayout(root_layout)

    def reload_reference_catalogs(self):
        self.reload_sample_catalog()
        self.reload_beam_profile_catalog()

    def reload_sample_catalog(self):
        current_key = self.sample_preset_combo.currentData()
        catalog = load_sample_catalog()
        self._sample_catalog_map = {
            build_sample_catalog_key(entry["sample"], entry["crystal_orientation"]): entry
            for entry in catalog["samples"]
        }
        self.sample_preset_combo.blockSignals(True)
        self.sample_preset_combo.clear()
        self.sample_preset_combo.addItem("Manual entry", None)
        for entry in catalog["samples"]:
            sample_key = build_sample_catalog_key(entry["sample"], entry["crystal_orientation"])
            self.sample_preset_combo.addItem(format_sample_display(entry), sample_key)
        restored_index = 0
        if current_key in self._sample_catalog_map:
            for index in range(self.sample_preset_combo.count()):
                if self.sample_preset_combo.itemData(index) == current_key:
                    restored_index = index
                    break
        self.sample_preset_combo.setCurrentIndex(restored_index)
        self.sample_preset_combo.blockSignals(False)
        self._apply_selected_sample_preset()

    def reload_beam_profile_catalog(self):
        current_id = self.beam_profile_combo.currentData()
        catalog = load_beam_profile_catalog()
        self._beam_profile_catalog_map = {entry["id"]: entry for entry in catalog["beam_profiles"]}
        self.beam_profile_combo.clear()
        self.beam_profile_combo.addItem("No preset", None)
        for entry in catalog["beam_profiles"]:
            self.beam_profile_combo.addItem(format_beam_profile_display(entry), entry["id"])
        if current_id in self._beam_profile_catalog_map:
            for index in range(self.beam_profile_combo.count()):
                if self.beam_profile_combo.itemData(index) == current_id:
                    self.beam_profile_combo.setCurrentIndex(index)
                    break

    def add_angle_input(self, default_value: float | None = None):
        row = QHBoxLayout()
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale.c())
        spin.setDecimals(4)
        spin.setRange(-180.0, 179.9975)
        spin.setSuffix(" deg")
        spin.setValue(0.0 if default_value is None else float(default_value))
        remove_btn = QPushButton("-")
        remove_btn.clicked.connect(lambda: self.remove_angle_input(row, spin))
        row.addWidget(spin)
        row.addWidget(remove_btn)
        self._angle_spins.append(spin)
        self.angles_layout.addLayout(row)

    def remove_angle_input(self, row, spin):
        if len(self._angle_spins) <= 1:
            QMessageBox.information(self, "Angle Required", "At least one estimated PM angle is required.")
            return
        self._angle_spins.remove(spin)
        for index in reversed(range(row.count())):
            widget = row.itemAt(index).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.angles_layout.removeItem(row)

    def _selected_sample_entry(self):
        sample_key = self.sample_preset_combo.currentData()
        if sample_key is None:
            return None
        return self._sample_catalog_map.get(sample_key)

    def _selected_beam_profile_entry(self):
        profile_id = self.beam_profile_combo.currentData()
        if profile_id is None:
            return None
        return self._beam_profile_catalog_map.get(profile_id)

    def _sample_measurement_prefix(self, entry):
        return f"{entry['sample']}_{entry['crystal_orientation']}_"

    def _apply_selected_sample_preset(self):
        entry = self._selected_sample_entry()
        self.material_combo.setEnabled(entry is None)
        if entry is None:
            return
        material_index = self.material_combo.findText(entry["material"])
        if material_index >= 0:
            self.material_combo.setCurrentIndex(material_index)
        prefix = self._sample_measurement_prefix(entry)
        current_text = self.sample_edit.text().strip()
        suffix = ""
        if current_text.startswith(prefix):
            suffix = current_text[len(prefix):].strip()
        elif current_text:
            parts = current_text.split("_")
            if len(parts) >= 3:
                suffix = "_".join(parts[2:]).strip()
        self.sample_edit.setText(prefix + suffix)

    def _parse_measurement_id_text(self, measurement_id: str):
        parts = [part.strip() for part in measurement_id.split("_")]
        if len(parts) < 3:
            return None
        sample_id = parts[0]
        crystal_orientation = normalize_crystal_orientation(parts[1])
        suffix = "_".join(parts[2:]).strip()
        if not sample_id or crystal_orientation not in {"100", "010", "001"} or not suffix:
            return None
        return sample_id, crystal_orientation, suffix

    def start_fundamental_measurement(self):
        self.start_measurement("fundamental")

    def start_shg_measurement(self):
        self.start_measurement("shg")

    def start_measurement(self, measurement_task: str):
        dry_run = self.dry_run_checkbox.isChecked()
        try:
            stage_rot = self.devices_tab.stage_rot_widget.controller if measurement_task == "shg" else None
            powermeter = None if dry_run else self._selected_powermeter_controller()
            laser = None if dry_run else self.devices_tab.laser_widget.controller
        except AttributeError:
            QMessageBox.warning(self, "Not Ready", "MainWindow does not have required controller widgets.")
            return
        if measurement_task == "shg" and stage_rot is None:
            QMessageBox.warning(self, "Not Ready", "Rotation stage is not connected.")
            return
        if not dry_run and (powermeter is None or not getattr(powermeter, "is_connected", True)):
            QMessageBox.warning(self, "Not Ready", f"{self._selected_powermeter_label()} is not connected.")
            return
        if not dry_run and laser is None:
            QMessageBox.warning(self, "Not Ready", "Laser controller is not connected.")
            return
        if not dry_run:
            self.refresh_measurement_ranges(show_errors=False)
        if not self._prepare_stage_widgets_for_measurement():
            return

        measurement_id = self.sample_edit.text().strip()
        selected_sample = self._selected_sample_entry()
        if selected_sample is not None:
            sample_id = selected_sample["sample"]
            crystal_orientation = selected_sample["crystal_orientation"]
            material = selected_sample["material"]
            prefix = self._sample_measurement_prefix(selected_sample)
            suffix = measurement_id[len(prefix):].strip() if measurement_id.startswith(prefix) else ""
            if not suffix:
                QMessageBox.warning(self, "Input Error", f"Please append a measurement id after '{prefix}'.")
                return
        else:
            parsed = self._parse_measurement_id_text(measurement_id)
            if parsed is None:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    "Measurement ID must be '<sample id>_<cut axis>_<measurement id>' with cut axis 100, 010, or 001.",
                )
                return
            sample_id, crystal_orientation, suffix = parsed
            material = self.material_combo.currentText()

        estimated_angles = [float(spin.value()) for spin in self._angle_spins]
        self.runner = PowerMeasurementRunner(stage_rot=stage_rot, powermeter=powermeter, laser=laser)
        if not dry_run:
            self._laser_controller = laser
        if measurement_task == "shg":
            self._setup_plot(len(estimated_angles))
        elif not self.axes:
            self._setup_plot(len(estimated_angles))
        self.thread = PowerMeasurementThread(
            runner=self.runner,
            measurement_task=measurement_task,
            sample=sample_id,
            material=material,
            crystal_orientation=crystal_orientation,
            measurement_id=suffix,
            estimated_angles=estimated_angles,
            scan_range=float(self.scan_range_spin.value()),
            step=float(self.step_spin.value()),
            axis=self.axis_edit.text().strip(),
            fundamental_wavelength_nm=float(self.fundamental_wavelength_spin.value()),
            shg_wavelength_nm=float(self.shg_wavelength_spin.value()),
            fundamental_range_index=self.fundamental_range_combo.currentData(),
            shg_range_index=self.shg_range_combo.currentData(),
            fundamental_range_label=self.fundamental_range_combo.currentText(),
            shg_range_label=self.shg_range_combo.currentText(),
            operator="user",
            notes=self.notes_edit.toPlainText(),
            sample_entry=selected_sample,
            beam_profile_entry=self._selected_beam_profile_entry(),
            dry_run=dry_run,
        )
        self.thread.progress_updated.connect(self.update_plot)
        self.thread.finished.connect(self.finish_measurement)
        self.thread.failed.connect(self.measurement_failed)
        self._stop_device_polling_for_measurement()
        self._set_run_buttons_enabled(False)
        self.abort_btn.setEnabled(True)
        self.thread.start()

    def refresh_measurement_ranges(self, show_errors: bool = True):
        if self.dry_run_checkbox.isChecked():
            return
        try:
            powermeter = self._selected_powermeter_controller()
        except AttributeError:
            if show_errors:
                QMessageBox.warning(self, "Not Ready", f"{self._selected_powermeter_label()} widget is not available.")
            return
        if powermeter is None or not getattr(powermeter, "is_connected", True):
            if show_errors:
                QMessageBox.warning(self, "Not Ready", f"{self._selected_powermeter_label()} is not connected.")
            return
        try:
            ranges = powermeter.get_range_options()
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "Range Error", str(exc))
            return
        self._set_range_combo_options(self.fundamental_range_combo, ranges.options)
        self._set_range_combo_options(self.shg_range_combo, ranges.options)

    def _set_range_combo_options(self, combo: QComboBox, labels: list[str]):
        current_data = combo.currentData()
        current_text = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto", None)
        for index, label in enumerate(labels):
            if "auto" in label.lower():
                continue
            combo.addItem(label, index)
        restored_index = 0
        for index in range(combo.count()):
            if combo.itemData(index) == current_data or combo.itemText(index) == current_text:
                restored_index = index
                break
        combo.setCurrentIndex(restored_index)
        combo.blockSignals(False)

    def _selected_powermeter_key(self) -> str:
        return str(self.powermeter_combo.currentData() or "ophir")

    def _selected_powermeter_label(self) -> str:
        return self.powermeter_combo.currentText() or "power meter"

    def _selected_powermeter_widget(self):
        if self.devices_tab is None:
            return None
        if self._selected_powermeter_key() == "thorlabs_s120c":
            return getattr(self.devices_tab, "thorlabs_powermeter_widget", None)
        return getattr(self.devices_tab, "powermeter_widget", None)

    def _selected_powermeter_controller(self):
        widget = self._selected_powermeter_widget()
        return None if widget is None else getattr(widget, "controller", None)

    def _prepare_stage_widgets_for_measurement(self):
        stage_widget = getattr(self.devices_tab, "stage_rot_widget", None)
        command_thread = getattr(stage_widget, "command_thread", None)
        if command_thread is not None and command_thread.isRunning():
            QMessageBox.warning(self, "Stage Busy", "Please wait for the manual stage command to finish.")
            return False
        return True

    def _setup_plot(self, count: int):
        self.figure.clear()
        self.axes = []
        if count <= 0:
            ax = self.figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Fundamental power measurement", ha="center", va="center")
            self.canvas.draw()
            return
        for index in range(count):
            ax = self.figure.add_subplot(1, count, index + 1)
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel(f"Power ({self.current_power_unit()})")
            ax.set_title(f"theta{index + 1}")
            self.axes.append(ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot(self, scan_index: int, pos: float, power: float):
        del pos, power
        if self.runner is None or scan_index >= len(self.axes):
            return
        scans = self.runner.scans if scan_index < len(self.runner.scans) else self._last_shg_scans
        if scan_index >= len(scans):
            return
        scan = scans[scan_index]
        ax = self.axes[scan_index]
        ax.clear()
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel(f"Power ({self.current_power_unit()})")
        ax.set_title(scan["label"])
        ax.plot(scan["positions"], [self.scale_power(power) for power in scan["powers"]], "-*", color="blue")
        self.figure.tight_layout()
        self.canvas.draw()

    def abort_measurement(self):
        if self.runner is not None and self.runner.is_running:
            self.runner.abort()
        powermeter = None if self.dry_run_checkbox.isChecked() else self._selected_powermeter_controller()
        if powermeter is not None and hasattr(powermeter, "interrupt_pending_read"):
            try:
                powermeter.interrupt_pending_read()
            except Exception:
                pass
        self.abort_btn.setEnabled(False)

    def finish_measurement(self, result_dict):
        self._set_run_buttons_enabled(True)
        self.abort_btn.setEnabled(False)
        self._last_metadata.update(result_dict.get("metadata") or {})
        if result_dict.get("aborted"):
            QMessageBox.information(self, "Aborted", "Power measurement aborted.")
        elif result_dict.get("fundamental_power"):
            self._fundamental_completed = True
            stats = result_dict["fundamental_power"]
            self._last_fundamental_stats = stats
            self.fundamental_result_label.setText(
                f"Fundamental power measurement complete.\n\n"
                f"Mean: {self.scale_power(stats.get('mean_w', 0.0)):.6g} {self.current_power_unit()}\n"
                f"Std: {self.scale_power(stats.get('std_w', 0.0)):.3g} {self.current_power_unit()}\n"
                f"Min: {self.scale_power(stats.get('min_w', 0.0)):.6g} {self.current_power_unit()}\n"
                f"Max: {self.scale_power(stats.get('max_w', 0.0)):.6g} {self.current_power_unit()}",
            )
        else:
            self._shg_completed = True
            self._last_shg_scans = result_dict.get("scans", [])
        self._stop_laser_if_measurement_pair_complete()
        self._restart_device_polling_after_measurement()
        self._prompt_save_or_retake_if_ready()

    def measurement_failed(self, message: str):
        self._set_run_buttons_enabled(True)
        self.abort_btn.setEnabled(False)
        self._restart_device_polling_after_measurement()
        if self.runner is not None and getattr(self.runner, "_abort", False):
            QMessageBox.information(self, "Aborted", "Power measurement aborted.")
            return
        QMessageBox.critical(self, "Measurement Error", message)

    def _prompt_save_or_retake_if_ready(self):
        if not (self._last_fundamental_stats and self._last_shg_scans):
            return
        message = QMessageBox(self)
        message.setWindowTitle("Power Measurement Complete")
        message.setText("Fundamental and SHG power data are both available. Do you want to save the results?")
        save_button = message.addButton("Save Results", QMessageBox.ButtonRole.AcceptRole)
        retake_fundamental_button = message.addButton("Retake Fundamental", QMessageBox.ButtonRole.ActionRole)
        retake_shg_button = message.addButton("Retake SHG", QMessageBox.ButtonRole.ActionRole)
        retake_both_button = message.addButton("Retake Both", QMessageBox.ButtonRole.ActionRole)
        message.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        message.exec()

        clicked = message.clickedButton()
        if clicked == save_button:
            self.save_combined_results()
        elif clicked == retake_fundamental_button:
            self._last_fundamental_stats = None
            self._fundamental_completed = False
            self.fundamental_result_label.setText("Fundamental power: ---")
            self.start_fundamental_measurement()
        elif clicked == retake_shg_button:
            self._last_shg_scans = []
            self._shg_completed = False
            self.start_shg_measurement()
        elif clicked == retake_both_button:
            self._last_fundamental_stats = None
            self._last_shg_scans = []
            self._fundamental_completed = False
            self._shg_completed = False
            self.fundamental_result_label.setText("Fundamental power: ---")
            self.figure.clear()
            self.axes = []
            self.canvas.draw()
            self.start_fundamental_measurement()

    def save_combined_results(self):
        if not self._last_fundamental_stats or not self._last_shg_scans:
            QMessageBox.warning(self, "Save Error", "Fundamental and SHG data are both required before saving.")
            return

        metadata = dict(self._last_metadata)
        metadata["measurement_kind"] = "fundamental_and_shg_power"
        metadata["saved_timestamp"] = datetime.now().isoformat()
        metadata["fundamental_power"] = self._last_fundamental_stats

        sample = str(metadata.get("sample") or "sample")
        measurement_id = str(metadata.get("measurement_id") or "power")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join("PM_power_results", f"{timestamp}_{sample}_power_{measurement_id}")
        os.makedirs(base_dir, exist_ok=True)

        meta_path = os.path.join(base_dir, "power_measurement.json")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

        csv_paths = []
        for scan in self._last_shg_scans:
            label = scan.get("label") or "theta"
            csv_path = os.path.join(base_dir, f"{label}.csv")
            csv_paths.append(csv_path)
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["angle_deg", "power_w", "std_w", "n"])
                stats_list = scan.get("stats", [])
                for index, (pos, power) in enumerate(zip(scan.get("positions", []), scan.get("powers", []))):
                    stats = stats_list[index] if index < len(stats_list) else {}
                    writer.writerow([
                        pos,
                        power,
                        stats.get("std_w", ""),
                        stats.get("n", ""),
                    ])

        fig_path = os.path.join(base_dir, "power_measurement.png")
        self.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        self._last_saved_dir = base_dir
        QMessageBox.information(
            self,
            "Saved",
            f"Saved power measurement results.\n\nFolder: {base_dir}\nJSON: {meta_path}\nCSV files: {len(csv_paths)}",
        )

    def _stop_laser_if_measurement_pair_complete(self):
        if not (self._fundamental_completed and self._shg_completed):
            return
        if self._laser_controller is None:
            return
        try:
            if self._laser_controller.is_emission_on:
                self._laser_controller.stop()
        except Exception:
            try:
                self._laser_controller.stop()
            except Exception:
                pass
        self._fundamental_completed = False
        self._shg_completed = False

    def _set_run_buttons_enabled(self, enabled: bool):
        self.measure_fundamental_btn.setEnabled(enabled)
        self.measure_shg_btn.setEnabled(enabled)

    def _stop_device_polling_for_measurement(self):
        if self.dry_run_checkbox.isChecked() or self.devices_tab is None:
            return
        powermeter_widget = self._selected_powermeter_widget()
        if powermeter_widget is not None and hasattr(powermeter_widget, "stop_polling"):
            powermeter_widget.stop_polling()
        laser_widget = getattr(self.devices_tab, "laser_widget", None)
        laser_polling = getattr(laser_widget, "polling_thread", None)
        if laser_polling is not None and hasattr(laser_polling, "stop"):
            laser_polling.stop()
            laser_widget.polling_thread = None

    def _restart_device_polling_after_measurement(self):
        if self.dry_run_checkbox.isChecked() or self.devices_tab is None:
            return
        powermeter_widget = self._selected_powermeter_widget()
        if powermeter_widget is not None and hasattr(powermeter_widget, "start_polling"):
            powermeter_widget.start_polling()
        laser_widget = getattr(self.devices_tab, "laser_widget", None)
        if (
            laser_widget is not None
            and getattr(laser_widget, "controller", None) is not None
            and getattr(laser_widget, "polling_thread", None) is None
        ):
            from widgets.crylasQlaser_widget import LaserPollingThread

            laser_widget.polling_thread = LaserPollingThread(laser_widget.controller, interval=1.0)
            laser_widget.polling_thread.status_updated.connect(laser_widget.update_status)
            laser_widget.polling_thread.start()

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
        values = []
        if self.runner is not None:
            for scan in self.runner.scans:
                values.extend(abs(power) for power in scan.get("powers", []))
            if self.runner.fundamental_power:
                values.append(abs(float(self.runner.fundamental_power.get("mean_w", 0.0))))
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

    def refresh_plot_units(self):
        if self._last_fundamental_stats:
            stats = self._last_fundamental_stats
            self.fundamental_result_label.setText(
                f"Fundamental power measurement complete.\n\n"
                f"Mean: {self.scale_power(stats.get('mean_w', 0.0)):.6g} {self.current_power_unit()}\n"
                f"Std: {self.scale_power(stats.get('std_w', 0.0)):.3g} {self.current_power_unit()}\n"
                f"Min: {self.scale_power(stats.get('min_w', 0.0)):.6g} {self.current_power_unit()}\n"
                f"Max: {self.scale_power(stats.get('max_w', 0.0)):.6g} {self.current_power_unit()}"
            )
        for index, _ in enumerate(self.axes):
            self.update_plot(index, 0.0, 0.0)


class PowerMeasurementThread(QThread):
    progress_updated = pyqtSignal(int, float, float)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, runner, measurement_task: str, **kwargs):
        super().__init__()
        self.runner = runner
        self.measurement_task = measurement_task
        self.kwargs = kwargs

    def run(self):
        def on_progress(scan_index, pos, power):
            self.progress_updated.emit(scan_index, pos, power)

        try:
            if self.measurement_task == "fundamental":
                self.runner.run_fundamental_power(**self.kwargs)
            elif self.measurement_task == "shg":
                self.runner.run_shg_power_scan(on_progress=on_progress, **self.kwargs)
            else:
                raise RuntimeError(f"Unknown power measurement task: {self.measurement_task}")
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(self.runner.result)
