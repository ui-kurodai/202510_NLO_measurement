from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QLineEdit, QComboBox, QMessageBox, QCheckBox, QPlainTextEdit, QListWidget, QListWidgetItem,
    QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from measure_shg import SHGMeasurementRunner
import logging
from measurement_metadata import (
    COMMON_BOXCAR_SENSITIVITIES,
    format_filter_display,
    load_nd_filter_catalog,
    parse_boxcar_sensitivity,
)
# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')

class SHGMeasurementWidget(QGroupBox):
    def __init__(self,
                 devices_tab=None,
                 parent=None):
        super().__init__("SHG Measurement", parent)
        self.devices_tab = devices_tab

        self.runner = None
        self.thread = None

        # --- UI Elements ---
        self.sample_edit = QLineEdit()
        self.sample_edit.setPlaceholderText("<sample id>_<cut axis>_<measured coefficient> ex.) 'BMF44_010_d31'")
        self.material_combo = QComboBox()
        self.material_combo.addItems(CRYSTALS.keys())
        self.method_combo = QComboBox()
        self.method_combo.addItems(["rotation", "wedge"])
        self.main_axis_edit = QLineEdit()

        self.channel_combo_1 = QComboBox()
        for ch in range(1, 9): 
            self.channel_combo_1.addItem(f"CH{ch}")
        self.channel_combo_2 = QComboBox()
        for ch in range(1, 9): 
            self.channel_combo_2.addItem(f"CH{ch}")

        self.input_pol_spin = QDoubleSpinBox()
        self.input_pol_spin.setRange(0, 180)
        self.input_pol_spin.setSuffix("°")

        self.detected_pol_spin = QDoubleSpinBox()
        self.detected_pol_spin.setRange(0, 180)
        self.detected_pol_spin.setSuffix("°")

        self.start_spin = QDoubleSpinBox()
        self.end_spin = QDoubleSpinBox()
        self.step_spin = QDoubleSpinBox()
        for s in [self.start_spin, self.end_spin, self.step_spin]:
            s.setDecimals(3)
            s.setRange(-9999, 9999)

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
        self.no_filter_checkbox.setChecked(True)
        self.no_filter_checkbox.toggled.connect(self._toggle_filter_selection)

        self.reload_filters_btn = QPushButton("Reload Catalog")
        self.reload_filters_btn.clicked.connect(self.reload_filter_catalog)

        self.filter_list = QListWidget()
        self.filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.filter_list.setFixedHeight(110)
        self.filter_list.itemSelectionChanged.connect(self._sync_filter_checkbox)
        self._filter_catalog_map = {}
        self.reload_filter_catalog()

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

        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Measurement ID:"))
        sample_layout.addWidget(self.sample_edit)
        sample_layout.addWidget(QLabel("material:"))
        sample_layout.addWidget(self.material_combo)
        layout.addLayout(sample_layout)

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
        sig_ch_layout.addWidget(QLabel("Measureing Channel:"))
        sig_ch_layout.addWidget(self.channel_combo_2)
        channel_layout.addLayout(ref_ch_layout)
        channel_layout.addLayout(sig_ch_layout)

        h1 = QHBoxLayout()
        h1.addLayout(setup_layout, 1)
        h1.addLayout(channel_layout, 1)
        layout.addLayout(h1)

        # setting poralization
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

        # input parameter
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

        # dry-run
        self.dry_run_checkbox = QCheckBox("Dry Run")
        self.dry_run_checkbox.setChecked(False)

        layout.addWidget(self.dry_run_checkbox)
        layout.addWidget(QLabel("Notes:"))
        layout.addWidget(self.notes_edit)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.abort_btn)
        layout.addWidget(self.canvas)

        self.setLayout(root_layout)

    def reload_filter_catalog(self):
        catalog = load_nd_filter_catalog()
        self._filter_catalog_map = {entry["filter_id"]: entry for entry in catalog["filters"]}
        self.filter_list.clear()
        for entry in catalog["filters"]:
            item = QListWidgetItem(format_filter_display(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry["filter_id"])
            self.filter_list.addItem(item)
        self._toggle_filter_selection(self.no_filter_checkbox.isChecked())

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
            except AttributeError as e:
                QMessageBox.warning(self, "Not Ready", "MainWindow does not have controller widgets.")
                logging.error(f"{e}")
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

        sample_info = self.sample_edit.text().split("_")
        if not sample_info:
            QMessageBox.warning(self, "Input Error", "Please enter a sample ID.")
            return
        sample_id = sample_info[0]
        crystal_orientation_str = sample_info[1]
        crystal_orientation = [int(i) for i in crystal_orientation_str]
        measured_coefficient = sample_info[2]
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
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return
        selected_filters = self._selected_filter_entries()


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
            dry_run=dry_run
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
        self.ax.clear()
        plot_x = self.runner.positions

        sample_id = self.sample_edit.text()
        for ch_index, ch in enumerate(self.runner.channels):
            ydata = [s[ch_index] for s in self.runner.signals]
            if ch_index == 0:
                label = "Reference"
                color = "black"
            elif ch_index == 1:
                label = sample_id if sample_id else f"CH{ch}"
                color = "blue"
            else:
                label = f"CH{ch}"  # in case of more than 3 channels
                color = "gray"
            method = self.method_combo.currentText()
            if method == 'rotation':
                self.ax.set_xlabel("Angle (deg)", fontsize=14)
            elif method == "wedge":
                self.ax.set_xlabel("Position (mm)", fontsize=14)
            self.ax.set_ylabel("SHG intensity (a.u.)", fontsize=14)
            self.ax.tick_params(axis='both', labelsize=14)
            self.ax.plot(plot_x, ydata, "-*", label=label, color=color)
        self.ax.legend()
        self.canvas.draw()

    def finish_measurement(self, result_dict):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)

        # saving graph
        if result_dict.get("csv_path"):
            from pathlib import Path
            csv_path = Path(result_dict["csv_path"])
            fig_path = csv_path.with_suffix(".png")
            self.figure.savefig(str(fig_path), dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {fig_path}")
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
