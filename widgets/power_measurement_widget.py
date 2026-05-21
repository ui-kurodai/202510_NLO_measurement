from __future__ import annotations

from threading import Event

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


class PowerMeasurementWidget(QGroupBox):
    def __init__(self, devices_tab=None, parent=None):
        super().__init__("Power Measurement", parent)
        self.devices_tab = devices_tab
        self.runner = None
        self.thread = None
        self._angle_spins = []
        self._sample_catalog_map = {}
        self._beam_profile_catalog_map = {}

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

        self.run_btn = QPushButton("Start Step 1")
        self.run_btn.clicked.connect(self.start_measurement)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self.abort_measurement)

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

        layout.addWidget(self.dry_run_checkbox)
        layout.addWidget(QLabel("Notes:"))
        layout.addWidget(self.notes_edit)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.abort_btn)
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

    def start_measurement(self):
        dry_run = self.dry_run_checkbox.isChecked()
        try:
            stage_rot = self.devices_tab.stage_rot_widget.controller
            powermeter = None if dry_run else self.devices_tab.powermeter_widget.controller
        except AttributeError:
            QMessageBox.warning(self, "Not Ready", "MainWindow does not have required controller widgets.")
            return
        if stage_rot is None:
            QMessageBox.warning(self, "Not Ready", "Rotation stage is not connected.")
            return
        if not dry_run and powermeter is None:
            QMessageBox.warning(self, "Not Ready", "Ophir power meter is not connected.")
            return
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
        self.runner = PowerMeasurementRunner(stage_rot=stage_rot, powermeter=powermeter)
        self._setup_plot(len(estimated_angles))
        self.thread = PowerMeasurementThread(
            runner=self.runner,
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
            operator="user",
            notes=self.notes_edit.toPlainText(),
            sample_entry=selected_sample,
            beam_profile_entry=self._selected_beam_profile_entry(),
            dry_run=dry_run,
        )
        self.thread.fundamental_finished.connect(self.confirm_step2)
        self.thread.progress_updated.connect(self.update_plot)
        self.thread.finished.connect(self.finish_measurement)
        self.thread.failed.connect(self.measurement_failed)
        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.thread.start()

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
        for index in range(count):
            ax = self.figure.add_subplot(count, 1, index + 1)
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Power (W)")
            ax.set_title(f"theta{index + 1}")
            self.axes.append(ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def confirm_step2(self, stats: dict):
        mean_w = stats.get("mean_w", 0.0)
        std_w = stats.get("std_w", 0.0)
        message = f"Fundamental power: {mean_w:.6g} W (std {std_w:.3g} W).\nStart SHG power scan?"
        answer = QMessageBox.question(self, "Step 1 Complete", message)
        if answer == QMessageBox.StandardButton.Yes:
            self.thread.allow_step2()
        else:
            self.abort_measurement()

    def update_plot(self, scan_index: int, pos: float, power: float):
        del pos, power
        if self.runner is None or scan_index >= len(self.axes):
            return
        scan = self.runner.scans[scan_index]
        ax = self.axes[scan_index]
        ax.clear()
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Power (W)")
        ax.set_title(scan["label"])
        ax.plot(scan["positions"], scan["powers"], "-*", color="blue")
        self.figure.tight_layout()
        self.canvas.draw()

    def abort_measurement(self):
        if self.runner is not None and self.runner.is_running:
            self.runner.abort()
        if self.thread is not None:
            self.thread.allow_step2()
        self.abort_btn.setEnabled(False)

    def finish_measurement(self, result_dict):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        if result_dict.get("base_dir"):
            fig_path = f"{result_dict['base_dir']}/power_measurement.png"
            self.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        if result_dict.get("aborted"):
            QMessageBox.information(self, "Aborted", "Power measurement aborted.")
        else:
            QMessageBox.information(self, "Done", "Power measurement complete.")

    def measurement_failed(self, message: str):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        QMessageBox.critical(self, "Measurement Error", message)


class PowerMeasurementThread(QThread):
    fundamental_finished = pyqtSignal(dict)
    progress_updated = pyqtSignal(int, float, float)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, runner, **kwargs):
        super().__init__()
        self.runner = runner
        self.kwargs = kwargs
        self._continue_event = Event()

    def allow_step2(self):
        self._continue_event.set()

    def run(self):
        def on_step1_complete(stats):
            self.fundamental_finished.emit(stats)

        def on_progress(scan_index, pos, power):
            self.progress_updated.emit(scan_index, pos, power)

        try:
            self.runner.run(
                continue_event=self._continue_event,
                on_step1_complete=on_step1_complete,
                on_progress=on_progress,
                **self.kwargs,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(self.runner.result)
