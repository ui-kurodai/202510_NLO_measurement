from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from measurement_metadata import COMMON_BOXCAR_SENSITIVITIES
from widgets.manual_extrema_detection_widget import ManualExtremaDetectionWidget


class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None, width: float = 5, height: float = 3, dpi: int = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ax.grid(True, which="both", alpha=0.25)
        self.figure.tight_layout()

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, which="both", alpha=0.25)


class SavedStrategyListWidget(QListWidget):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.position().toPoint())
            if item is None:
                self.clearSelection()
            elif not item.isSelected():
                self.clearSelection()
                item.setSelected(True)
                self.setCurrentItem(item)
        super().mousePressEvent(event)


class StandardFitWidget(QWidget):
    def __init__(self, slider_steps: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._slider_steps = slider_steps
        self._manual_controls: Dict[str, Dict[str, QDoubleSpinBox | QSlider]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        folder_group = QGroupBox("Folder")
        folder_layout = QVBoxLayout(folder_group)
        self.btn_open = QPushButton("Open Folder…")
        self.lbl_current_folder = QLabel("No folder loaded.")
        self.lbl_current_folder.setWordWrap(True)
        self.lbl_current_folder.setStyleSheet("color: gray;")
        folder_layout.addWidget(self.btn_open)
        folder_layout.addWidget(self.lbl_current_folder)

        reference_group = QGroupBox("Reference Data")
        reference_form = QFormLayout(reference_group)
        sample_row = QHBoxLayout()
        self.sample_preset_combo = QComboBox()
        self.reload_samples_btn = QPushButton("Reload")
        sample_row.addWidget(self.sample_preset_combo, 1)
        sample_row.addWidget(self.reload_samples_btn)
        beam_row = QHBoxLayout()
        self.beam_profile_combo = QComboBox()
        self.reload_beams_btn = QPushButton("Reload")
        beam_row.addWidget(self.beam_profile_combo, 1)
        beam_row.addWidget(self.reload_beams_btn)
        filter_row = QHBoxLayout()
        self.filter_preset_combo = QComboBox()
        self.reload_filters_btn = QPushButton("Reload")
        filter_row.addWidget(self.filter_preset_combo, 1)
        filter_row.addWidget(self.reload_filters_btn)
        reference_form.addRow("Sample preset:", sample_row)
        reference_form.addRow("Beam profile:", beam_row)
        reference_form.addRow("Filter preset:", filter_row)

        meta_edit = QGroupBox("Metadata (editable before fitting)")
        form = QFormLayout(meta_edit)
        self.le_material = QLineEdit()
        self.le_crystal_orientation = QLineEdit()
        self.le_axis = QLineEdit()
        self.sb_t_thin = QDoubleSpinBox()
        self.sb_wedge = QDoubleSpinBox()
        self.sb_beam_rx = QDoubleSpinBox()
        self.sb_beam_ry = QDoubleSpinBox()
        self.sb_input_pol = QDoubleSpinBox()
        self.sb_detected_pol = QDoubleSpinBox()
        for spin_box in [
            self.sb_t_thin,
            self.sb_wedge,
            self.sb_beam_rx,
            self.sb_beam_ry,
            self.sb_input_pol,
            self.sb_detected_pol,
        ]:
            spin_box.setLocale(QLocale.c())
        self.sb_t_thin.setRange(0.0, 1e6)
        self.sb_t_thin.setDecimals(6)
        self.sb_wedge.setRange(-90.0, 90.0)
        self.sb_wedge.setDecimals(6)
        self.sb_beam_rx.setRange(0.0, 1e6)
        self.sb_beam_rx.setDecimals(6)
        self.sb_beam_ry.setRange(0.0, 1e6)
        self.sb_beam_ry.setDecimals(6)
        self.sb_input_pol.setRange(-360.0, 360.0)
        self.sb_input_pol.setDecimals(3)
        self.sb_detected_pol.setRange(-360.0, 360.0)
        self.sb_detected_pol.setDecimals(3)
        self.cmb_boxcar_sensitivity = QComboBox()
        self.cmb_boxcar_sensitivity.setEditable(True)
        self.cmb_boxcar_sensitivity.addItems(COMMON_BOXCAR_SENSITIVITIES)
        self.le_filters = QLineEdit()
        self.le_filters.setPlaceholderText("filter_id=value, ...")
        form.addRow("material:", self.le_material)
        form.addRow("crystal_orientation (e.g. 0,1,1):", self.le_crystal_orientation)
        form.addRow("rotation/translation axis:", self.le_axis)
        form.addRow("input polarization (deg):", self.sb_input_pol)
        form.addRow("detected polarization (deg):", self.sb_detected_pol)
        form.addRow("t_center_mm:", self.sb_t_thin)
        form.addRow("wedge_angle_deg:", self.sb_wedge)
        form.addRow("beam_r_x:", self.sb_beam_rx)
        form.addRow("beam_r_y:", self.sb_beam_ry)
        form.addRow("boxcar_sensitivity:", self.cmb_boxcar_sensitivity)
        form.addRow("filters:", self.le_filters)

        meta_view = QGroupBox("Loaded Metadata (read-only excerpt)")
        meta_view_form = QFormLayout(meta_view)
        self.lbl_sample = QLabel("—")
        self.lbl_method = QLabel("—")
        self.lbl_time = QLabel("—")
        meta_view_form.addRow("Sample:", self.lbl_sample)
        meta_view_form.addRow("Method:", self.lbl_method)
        meta_view_form.addRow("Timestamp:", self.lbl_time)

        left_layout.addWidget(folder_group)
        left_layout.addWidget(reference_group)
        left_layout.addWidget(meta_edit)
        left_layout.addWidget(meta_view)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.plot_tabs = QTabWidget()

        fit_tab = QWidget()
        fit_layout = QVBoxLayout(fit_tab)
        fit_series_row = QHBoxLayout()
        fit_series_row.addWidget(QLabel("Show:"))
        self.chk_fit_show_data = QCheckBox("Data")
        self.chk_fit_show_data.setChecked(True)
        self.chk_fit_show_fitting = QCheckBox("Fitting")
        self.chk_fit_show_fitting.setChecked(True)
        self.chk_fit_show_envelope = QCheckBox("Envelope")
        self.chk_fit_show_envelope.setChecked(True)
        fit_series_row.addWidget(self.chk_fit_show_data)
        fit_series_row.addWidget(self.chk_fit_show_fitting)
        fit_series_row.addWidget(self.chk_fit_show_envelope)
        fit_series_row.addStretch(1)
        fit_layout.addLayout(fit_series_row)
        self.canvas_fit = MplCanvas(fit_tab, width=6.0, height=3.6)
        fit_layout.addWidget(self.canvas_fit)
        fit_layout.addWidget(self._build_manual_fit_group())
        self.plot_tabs.addTab(fit_tab, "Data & Fit")

        resid_tab = QWidget()
        resid_layout = QVBoxLayout(resid_tab)
        self.canvas_resid = MplCanvas(resid_tab, width=6.0, height=2.8)
        resid_layout.addWidget(self.canvas_resid)
        self.plot_tabs.addTab(resid_tab, "Residuals")

        center_tab = QWidget()
        center_layout = QVBoxLayout(center_tab)
        self.canvas_centering = MplCanvas(center_tab, width=6.0, height=2.8)
        center_layout.addWidget(self.canvas_centering)
        self.plot_tabs.addTab(center_tab, "Centering Cost")

        extrema_tab = QWidget()
        extrema_layout = QVBoxLayout(extrema_tab)
        self.extrema_widget = ManualExtremaDetectionWidget(extrema_tab)
        extrema_layout.addWidget(self.extrema_widget)
        self.plot_tabs.addTab(extrema_tab, "Extrema")

        lc_tab = QWidget()
        lc_layout = QVBoxLayout(lc_tab)
        lc_toolbar = QHBoxLayout()
        self.cmb_lc_source = QComboBox()
        self.cmb_lc_source.addItem("Experimental Data", userData="data")
        self.cmb_lc_source.addItem("Fit Curve", userData="fit")
        self.lbl_lc_hint = QLabel("Lc is estimated from adjacent minima pairs.")
        self.lbl_lc_hint.setStyleSheet("color: gray;")
        lc_toolbar.addWidget(QLabel("Source:"))
        lc_toolbar.addWidget(self.cmb_lc_source)
        lc_toolbar.addStretch(1)
        lc_toolbar.addWidget(self.lbl_lc_hint)
        lc_layout.addLayout(lc_toolbar)
        self.canvas_lc = MplCanvas(lc_tab, width=6.0, height=2.8)
        lc_layout.addWidget(self.canvas_lc)
        self.plot_tabs.addTab(lc_tab, "Lc")

        plot_toolbar = QHBoxLayout()
        self.btn_plot_settings = QPushButton("Plot Settings")
        self.btn_save_current_plot = QPushButton("Save Current Plot")
        self.btn_copy_current_plot = QPushButton("Copy Image")
        plot_toolbar.addStretch(1)
        plot_toolbar.addWidget(self.btn_plot_settings)
        plot_toolbar.addWidget(self.btn_save_current_plot)
        plot_toolbar.addWidget(self.btn_copy_current_plot)
        right_layout.addLayout(plot_toolbar)
        right_layout.addWidget(self.plot_tabs, 1)

        self.right_scroll.setWidget(right)
        splitter.addWidget(self.right_scroll)
        splitter.setSizes([360, 720])
        layout.addWidget(splitter, 1)

        self._plot_pages = {
            "fit": fit_tab,
            "resid": resid_tab,
            "centering": center_tab,
            "extrema": extrema_tab,
            "lc": lc_tab,
        }
        self._plot_canvases = {
            "fit": self.canvas_fit,
            "resid": self.canvas_resid,
            "centering": self.canvas_centering,
            "extrema": self.extrema_widget.canvas,
            "lc": self.canvas_lc,
        }

    def _build_manual_fit_group(self) -> QGroupBox:
        group = QGroupBox("Manual Fit Control")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self._create_manual_control_row(
            form=form,
            key="L",
            label="L [mm]:",
            minimum=-1e6,
            maximum=1e6,
            decimals=4,
            step=0.0001,
        )
        self._create_manual_control_row(
            form=form,
            key="peak",
            label="Peak:",
            minimum=0.0,
            maximum=1e9,
            decimals=6,
            step=0.001,
        )
        self.sb_manual_centering = QDoubleSpinBox()
        self.sb_manual_centering.setLocale(QLocale.c())
        self.sb_manual_centering.setRange(-1e6, 1e6)
        self.sb_manual_centering.setDecimals(4)
        self.sb_manual_centering.setSingleStep(0.0001)
        self.sb_manual_centering.setKeyboardTracking(False)
        form.addRow("Centering:", self.sb_manual_centering)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.btn_reset_manual = QPushButton("Reset to Auto Fit")
        self.btn_apply_manual = QPushButton("Overwrite JSON/CSV")
        buttons.addWidget(self.btn_reset_manual)
        buttons.addWidget(self.btn_apply_manual)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.lbl_manual_hint = QLabel(
            "The live overlay uses the current L, Peak, and Centering values. Overwrite updates saved fit values."
        )
        self.lbl_manual_hint.setWordWrap(True)
        self.lbl_manual_hint.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_manual_hint)
        return group

    def _create_manual_control_row(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        minimum: float,
        maximum: float,
        decimals: int,
        step: float,
    ) -> None:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, self._slider_steps)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setMinimumWidth(260)
        value_box = QDoubleSpinBox()
        value_box.setLocale(QLocale.c())
        value_box.setRange(minimum, maximum)
        value_box.setDecimals(decimals)
        value_box.setSingleStep(step)
        value_box.setKeyboardTracking(False)
        value_box.setFixedWidth(88)

        min_box = QDoubleSpinBox()
        min_box.setLocale(QLocale.c())
        min_box.setRange(minimum, maximum)
        min_box.setDecimals(decimals)
        min_box.setSingleStep(step)
        min_box.setKeyboardTracking(False)
        min_box.setFixedWidth(78)

        max_box = QDoubleSpinBox()
        max_box.setLocale(QLocale.c())
        max_box.setRange(minimum, maximum)
        max_box.setDecimals(decimals)
        max_box.setSingleStep(step)
        max_box.setKeyboardTracking(False)
        max_box.setFixedWidth(78)

        lbl_min = QLabel("min")
        lbl_max = QLabel("max")
        lbl_min.setFixedWidth(22)
        lbl_max.setFixedWidth(26)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(value_box, 0)
        row_layout.addWidget(lbl_min, 0)
        row_layout.addWidget(min_box, 0)
        row_layout.addWidget(lbl_max, 0)
        row_layout.addWidget(max_box, 0)
        form.addRow(label, row)

        self._manual_controls[key] = {
            "slider": slider,
            "value": value_box,
            "min": min_box,
            "max": max_box,
        }
