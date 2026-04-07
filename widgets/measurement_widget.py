from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QThread, QLocale, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from measure_shg import SHGMeasurementRunner
import logging
from measurement_metadata import (
    COMMON_BOXCAR_SENSITIVITIES,
    build_sample_catalog_key,
    format_beam_profile_display,
    format_filter_display,
    format_sample_display,
    load_beam_profile_catalog,
    load_nd_filter_catalog,
    load_sample_catalog,
    normalize_crystal_orientation,
    parse_boxcar_sensitivity,
)

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class SHGMeasurementWidget(QGroupBox):
    def __init__(self, devices_tab=None, parent=None):
        super().__init__("SHG Measurement", parent)
        self.devices_tab = devices_tab

        self.runner = None
        self.thread = None

        # --- UI Elements ---
        self.sample_preset_combo = QComboBox()
        self.sample_preset_combo.currentIndexChanged.connect(self._apply_selected_sample_preset)
        self.reload_samples_btn = QPushButton("Reload Samples")
        self.reload_samples_btn.clicked.connect(self.reload_sample_catalog)

        self.sample_edit = QLineEdit()
        self.sample_edit.setPlaceholderText("<sample id>_<cut axis>_<measured coefficient> ex.) 'BMF44_010_d31'")

        self.material_combo = QComboBox()
        self.material_combo.addItems(CRYSTALS.keys())

        self.beam_profile_combo = QComboBox()
        self.reload_beams_btn = QPushButton("Reload Beams")
        self.reload_beams_btn.clicked.connect(self.reload_beam_profile_catalog)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["rotation", "wedge"])
        self.main_axis_edit = QLineEdit()

        self.channel_combo_1 = QComboBox()
        for ch in range(1, 9):
            self.channel_combo_1.addItem(f"CH{ch}")
        self.channel_combo_2 = QComboBox()
        for ch in range(1, 9):
            self.channel_combo_2.addItem(f"CH{ch}")
        self.channel_combo_2.setCurrentIndex(1)

        self.input_pol_spin = QDoubleSpinBox()
        self.input_pol_spin.setLocale(QLocale.c())
        self.input_pol_spin.setRange(0, 180)
        self.input_pol_spin.setSuffix(" deg")

        self.detected_pol_spin = QDoubleSpinBox()
        self.detected_pol_spin.setLocale(QLocale.c())
        self.detected_pol_spin.setRange(0, 180)
        self.detected_pol_spin.setSuffix(" deg")

        self.start_spin = QDoubleSpinBox()
        self.end_spin = QDoubleSpinBox()
        self.step_spin = QDoubleSpinBox()
        for spin_box in [self.start_spin, self.end_spin, self.step_spin]:
            spin_box.setLocale(QLocale.c())
            spin_box.setDecimals(3)
            spin_box.setRange(-9999, 9999)

        self.run_btn = QPushButton("Start Measurement")
        self.run_btn.clicked.connect(self.start_measurement)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self.abort_measurement)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Experiment notes (optional)...")
        self.notes_edit.setFixedHeight(60)

        self.boxcar_sensitivity_combo = QComboBox()
        self.boxcar_sensitivity_combo.setEditable(True)
        self.boxcar_sensitivity_combo.addItems(COMMON_BOXCAR_SENSITIVITIES)
        self.boxcar_sensitivity_combo.setCurrentText("1 V / 0.1 V")

        self.no_filter_checkbox = QCheckBox("No ND filter")
        self.no_filter_checkbox.setChecked(False)
        self.no_filter_checkbox.toggled.connect(self._toggle_filter_selection)

        self.reload_filters_btn = QPushButton("Reload Filters")
        self.reload_filters_btn.clicked.connect(self.reload_filter_catalog)

        self.filter_list = QListWidget()
        self.filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.filter_list.setFixedHeight(110)
        self.filter_list.itemSelectionChanged.connect(self._sync_filter_checkbox)

        self._filter_catalog_map = {}
        self._sample_catalog_map = {}
        self._beam_profile_catalog_map = {}
        self.reload_reference_catalogs()

        # --- Plot area ---
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(320)
        self.ax = self.figure.add_subplot(111)

        # --- Layout ---
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

        setup_layout = QVBoxLayout()
        method_layout = QHBoxLayout()
        method_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        method_layout.addWidget(QLabel("Method:"))
        method_layout.addWidget(self.method_combo)
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Rotation/Translation axis:"))
        axis_layout.addWidget(self.main_axis_edit)
        setup_layout.addLayout(method_layout)
        setup_layout.addLayout(axis_layout)

        channel_layout = QVBoxLayout()
        channel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        ref_ch_layout = QHBoxLayout()
        ref_ch_layout.addWidget(QLabel("Reference Channel:"))
        ref_ch_layout.addWidget(self.channel_combo_1)
        sig_ch_layout = QHBoxLayout()
        sig_ch_layout.addWidget(QLabel("Measuring Channel:"))
        sig_ch_layout.addWidget(self.channel_combo_2)
        channel_layout.addLayout(ref_ch_layout)
        channel_layout.addLayout(sig_ch_layout)

        h1 = QHBoxLayout()
        h1.addLayout(setup_layout, 1)
        h1.addLayout(channel_layout, 1)
        layout.addLayout(h1)

        pol_layout = QVBoxLayout()
        pol_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        pol1_row = QHBoxLayout()
        pol1_row.addWidget(QLabel("Input Polarization:"))
        pol1_row.addWidget(self.input_pol_spin)
        pol2_row = QHBoxLayout()
        pol2_row.addWidget(QLabel("Detected Polarization:"))
        pol2_row.addWidget(self.detected_pol_spin)
        pol_layout.addLayout(pol1_row)
        pol_layout.addLayout(pol2_row)

        points_layout = QVBoxLayout()
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start (-180, 180):"))
        start_row.addWidget(self.start_spin)
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End (start, 180):"))
        end_row.addWidget(self.end_spin)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        step_row.addWidget(self.step_spin)
        points_layout.addLayout(start_row)
        points_layout.addLayout(end_row)
        points_layout.addLayout(step_row)

        parameter_layout = QHBoxLayout()
        parameter_layout.addLayout(points_layout)
        parameter_layout.addLayout(pol_layout)
        layout.addLayout(parameter_layout)

        sensitivity_row = QHBoxLayout()
        sensitivity_row.addWidget(QLabel("Boxcar sensitivity:"))
        sensitivity_row.addWidget(self.boxcar_sensitivity_combo, 1)
        layout.addLayout(sensitivity_row)

        filter_header = QHBoxLayout()
        filter_header.addWidget(QLabel("ND filters:"))
        filter_header.addStretch(1)
        filter_header.addWidget(self.reload_filters_btn)
        layout.addLayout(filter_header)
        layout.addWidget(self.no_filter_checkbox)
        layout.addWidget(self.filter_list)

        self.dry_run_checkbox = QCheckBox("Dry Run")
        self.dry_run_checkbox.setChecked(False)

        layout.addWidget(self.dry_run_checkbox)
        layout.addWidget(QLabel("Notes:"))
        layout.addWidget(self.notes_edit)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.abort_btn)
        layout.addWidget(self.canvas)

        self.setLayout(root_layout)

    def reload_reference_catalogs(self):
        self.reload_filter_catalog()
        self.reload_sample_catalog()
        self.reload_beam_profile_catalog()

    def reload_filter_catalog(self):
        catalog = load_nd_filter_catalog()
        self._filter_catalog_map = {entry["filter_id"]: entry for entry in catalog["filters"]}
        self.filter_list.clear()
        for entry in catalog["filters"]:
            item = QListWidgetItem(format_filter_display(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry["filter_id"])
            self.filter_list.addItem(item)
        self._toggle_filter_selection(self.no_filter_checkbox.isChecked())

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

        restored_index = 0
        if current_id in self._beam_profile_catalog_map:
            for index in range(self.beam_profile_combo.count()):
                if self.beam_profile_combo.itemData(index) == current_id:
                    restored_index = index
                    break
        elif catalog["beam_profiles"]:
            latest_id = max(
                (entry["id"] for entry in catalog["beam_profiles"] if entry.get("id")),
                default=None,
            )
            if latest_id is not None:
                for index in range(self.beam_profile_combo.count()):
                    if self.beam_profile_combo.itemData(index) == latest_id:
                        restored_index = index
                        break
        self.beam_profile_combo.setCurrentIndex(restored_index)

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
        coefficient = ""
        if current_text.startswith(prefix):
            coefficient = current_text[len(prefix):].strip()
        else:
            parts = current_text.split("_")
            if len(parts) >= 3:
                coefficient = "_".join(parts[2:]).strip()
        self.sample_edit.setText(prefix + coefficient)

    def _toggle_filter_selection(self, checked: bool):
        if checked:
            self.filter_list.clearSelection()
        self.filter_list.setEnabled(not checked)

    def _sync_filter_checkbox(self):
        if self.filter_list.selectedItems():
            self.no_filter_checkbox.blockSignals(True)
            self.no_filter_checkbox.setChecked(False)
            self.no_filter_checkbox.blockSignals(False)
            self.filter_list.setEnabled(True)

    def _selected_filter_entries(self):
        if self.no_filter_checkbox.isChecked():
            return []

        selected_entries = []
        for item in self.filter_list.selectedItems():
            filter_id = item.data(Qt.ItemDataRole.UserRole)
            entry = self._filter_catalog_map.get(filter_id)
            if entry is not None:
                selected_entries.append(entry)
        return selected_entries

    def _parse_measurement_id_text(self, measurement_id: str):
        parts = [part.strip() for part in measurement_id.split("_")]
        if len(parts) < 3:
            return None
        sample_id = parts[0]
        crystal_orientation = normalize_crystal_orientation(parts[1])
        measured_coefficient = "_".join(parts[2:]).strip()
        if not sample_id or crystal_orientation not in {"100", "010", "001"} or not measured_coefficient:
            return None
        return sample_id, crystal_orientation, measured_coefficient

    def connect_controllers(self, laser, stage_lin, stage_rot, boxcar, elliptec=None):
        self.runner = SHGMeasurementRunner(laser, stage_lin, stage_rot, boxcar, elliptec)

    def start_measurement(self):
        dry_run = self.dry_run_checkbox.isChecked()
        if self.runner is None:
            try:
                laser = self.devices_tab.laser_widget.controller if not dry_run else None
                stage_lin = self.devices_tab.stage_lin_widget.controller
                stage_rot = self.devices_tab.stage_rot_widget.controller
                boxcar = self.devices_tab.boxcar_widget.controller if not dry_run else None
                elliptec = self.devices_tab.elliptec_widget.controller
            except AttributeError as exc:
                QMessageBox.warning(self, "Not Ready", "MainWindow does not have controller widgets.")
                logging.error("%s", exc)
                return

            if not dry_run:
                if None in [laser, stage_lin, stage_rot, boxcar, elliptec]:
                    QMessageBox.warning(self, "Not Ready", "One or more controllers are not connected.")
                    return
            else:
                if None in [stage_lin, stage_rot, elliptec]:
                    QMessageBox.warning(self, "Not Ready", "Stage or elliptec controllers are not connected.")
                    return

            self.connect_controllers(laser, stage_lin, stage_rot, boxcar, elliptec)

        if self.runner.is_running:
            QMessageBox.warning(self, "Warning", "Measurement already in progress.")
            return

        measurement_id = self.sample_edit.text().strip()
        selected_sample = self._selected_sample_entry()
        if selected_sample is not None:
            sample_id = selected_sample["sample"]
            crystal_orientation = selected_sample["crystal_orientation"]
            material = selected_sample["material"]
            prefix = self._sample_measurement_prefix(selected_sample)
            if measurement_id.startswith(prefix):
                measured_coefficient = measurement_id[len(prefix):].strip()
            else:
                parsed_measurement = self._parse_measurement_id_text(measurement_id)
                measured_coefficient = "" if parsed_measurement is None else parsed_measurement[2]

            if not measured_coefficient:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Please append a measured coefficient after '{prefix}'.",
                )
                return
        else:
            parsed_measurement = self._parse_measurement_id_text(measurement_id)
            if parsed_measurement is None:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    "Measurement ID must be '<sample id>_<cut axis>_<measured coefficient>' with cut axis 100, 010, or 001.",
                )
                return
            sample_id, crystal_orientation, measured_coefficient = parsed_measurement
            material = self.material_combo.currentText()

        method = self.method_combo.currentText()
        input_polarization = float(self.input_pol_spin.value())
        detected_polarization = float(self.detected_pol_spin.value())
        start = self.start_spin.value()
        end = self.end_spin.value()
        step = self.step_spin.value()
        ref_ch = int(self.channel_combo_1.currentText().replace("CH", ""))
        sig_ch = int(self.channel_combo_2.currentText().replace("CH", ""))
        channels = [ref_ch, sig_ch]
        if not channels:
            QMessageBox.warning(self, "Input Error", "Please select at least one channel.")
            return

        axis = self.main_axis_edit.text()
        notes = self.notes_edit.toPlainText()
        boxcar_sensitivity_text = self.boxcar_sensitivity_combo.currentText().strip()
        try:
            parse_boxcar_sensitivity(boxcar_sensitivity_text)
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
            return

        selected_filters = self._selected_filter_entries()
        selected_beam_profile = self._selected_beam_profile_entry()

        self.ax.clear()
        self.canvas.draw()

        self.thread = SHGPollingThread(
            runner=self.runner,
            sample=sample_id,
            material=material,
            crystal_orientation=crystal_orientation,
            measured_coefficient=measured_coefficient,
            method=method,
            input_polarization=input_polarization,
            detected_polarization=detected_polarization,
            repetition="1000Hz",
            operator="user",
            notes=notes,
            start=start,
            end=end,
            step=step,
            channels=channels,
            axis=axis,
            boxcar_sensitivity_text=boxcar_sensitivity_text,
            nd_filter_entries=selected_filters,
            sample_entry=selected_sample,
            beam_profile_entry=selected_beam_profile,
            dry_run=dry_run,
        )
        self.thread.progress_updated.connect(self.update_plot)
        self.thread.finished.connect(self.finish_measurement)

        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.thread.start()

    def abort_measurement(self):
        if self.runner.is_running:
            self.runner.abort()
            self.abort_btn.setEnabled(False)

    def update_plot(self, pos, signal):
        del pos, signal
        self.ax.clear()
        plot_x = self.runner.positions

        sample_id = self.sample_edit.text()
        for ch_index, ch in enumerate(self.runner.channels):
            ydata = [sample[ch_index] for sample in self.runner.signals]
            if ch_index == 0:
                label = "Reference"
                color = "black"
            elif ch_index == 1:
                label = sample_id if sample_id else f"CH{ch}"
                color = "blue"
            else:
                label = f"CH{ch}"
                color = "gray"

            method = self.method_combo.currentText()
            if method == "rotation":
                self.ax.set_xlabel("Angle (deg)", fontsize=14)
            elif method == "wedge":
                self.ax.set_xlabel("Position (mm)", fontsize=14)
            self.ax.set_ylabel("SHG intensity (a.u.)", fontsize=14)
            self.ax.tick_params(axis="both", labelsize=14)
            self.ax.plot(plot_x, ydata, "-*", label=label, color=color)

        self.ax.legend()
        self.canvas.draw()

    def finish_measurement(self, result_dict):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)

        if result_dict.get("csv_path"):
            from pathlib import Path

            csv_path = Path(result_dict["csv_path"])
            fig_path = csv_path.with_suffix(".png")
            self.figure.savefig(str(fig_path), dpi=300, bbox_inches="tight")
            logging.info("Plot saved to %s", fig_path)

        warnings = result_dict.get("condition_warnings") or []
        if warnings:
            warning_text = "\n".join(f"- {warning}" for warning in warnings)
            QMessageBox.warning(self, "Done with warnings", f"Measurement complete.\n\n{warning_text}")
        else:
            QMessageBox.information(self, "Done", "Measurement complete.")


class SHGPollingThread(QThread):
    progress_updated = pyqtSignal(float, list)
    finished = pyqtSignal(dict)

    def __init__(self, runner, **kwargs):
        super().__init__()
        self.runner = runner
        self.kwargs = kwargs

    def run(self):
        def on_progress(pos, signal):
            self.progress_updated.emit(pos, signal)

        self.runner.run(on_progress=on_progress, **self.kwargs)
        self.finished.emit(self.runner.result)
