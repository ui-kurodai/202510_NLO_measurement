"""
Fit/Analysis Tab for SHG data (PyQt6)

Features
- Folder-based workflow: user selects a result folder under `results/` that contains exactly one CSV and one JSON.
- Unified reading: UI always refreshes by reading JSON/CSV from disk (works for both past data and just-fitted data).
- Metadata editor: material, crystal_orientation, thickness_info (t_center_mm, wedge_angle_deg), beam_r_x/beam_r_y.
- Strategy picker: dynamically lists strategies in `fitting_strategies/` (excludes `base.py` and `__init__.py`).
- Run fit: executes selected strategy (module autodiscovery), then reloads files.
- Plotting: uses CSV columns; prefers `angle_deg` or `position_mm` for x, shows data vs fitted curve, residuals if available.
- Save: exports current figures (fit & residuals) as PNG into the same folder.

Drop-in usage
    tab_widgets.addTab(FitAnalyzeTab(parent=self), "Analysis")

Notes
- This file assumes your project provides `fitting_strategies/<name>.py` and `shg_analysis.SHgDataAnalysis`.
- Comments are in English (per project preference).
"""
from __future__ import annotations

import json
import sys
import math
import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QLocale, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QGroupBox, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QSizePolicy, QSplitter,
    QDoubleSpinBox, QSlider, QDialog, QDialogButtonBox, QCheckBox,
    QScrollArea
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
    parse_boxcar_sensitivity,
    resolve_selected_filters,
    sample_metadata_from_entry,
)
from fitting_results import extract_fit_payload, merge_fit_payload, upsert_fitting_result
# self made database
# from crystaldatabase import CRYSTALS
# from crystaldatabase import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')


# External analysis entry point (import at top-level for user feedback if missing)
try:
    from shg_analysis import SHGDataAnalysis
except Exception as e:  # pragma: no cover
    SHGDataAnalysis = None  # type: ignore


# ------------------------- Helpers & dataclasses -------------------------
@dataclass
class StrategyInfo:
    module_name: str  # e.g., "jerphagnon1970"
    qualname: str     # e.g., "fitting_strategies.jerphagnon1970"
    class_name: str   # e.g., "Jerphagnon1970Strategy"

    @property
    def display_name(self) -> str:
        short_name = self.class_name[:-8] if self.class_name.endswith("Strategy") else self.class_name
        return f"{self.module_name} / {short_name}"


@dataclass
class PlotSettings:
    font_size: float = 10.0
    show_legend: bool = True
    box_aspect: float = 0.0
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


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


# ------------------------------- Main Tab -------------------------------
class FittingAnalysisWidget(QWidget):
    folderLoaded = pyqtSignal(str)   # emits folder path when a folder has been loaded
    _SLIDER_STEPS = 2000

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("FitAnalyzeTab")

        # State
        self._current_dir: Optional[Path] = None
        self._df: Optional[pd.DataFrame] = None
        self._meta: Dict = {}
        self.csv_path: Optional[Path] = None
        self.json_path: Optional[Path] = None
        self._strategies: List[StrategyInfo] = []
        self._analysis_context: Dict[str, Any] = {}
        self._manual_controls: Dict[str, Dict[str, QDoubleSpinBox | QSlider]] = {}
        self._manual_syncing = False
        self._sample_catalog_map: Dict[str, Dict[str, Any]] = {}
        self._beam_profile_catalog_map: Dict[str, Dict[str, Any]] = {}
        self._filter_catalog_map: Dict[str, Dict[str, Any]] = {}
        self._plot_settings: Dict[str, PlotSettings] = {
            "fit": PlotSettings(font_size=11.0, show_legend=True, box_aspect=0.0),
            "resid": PlotSettings(font_size=10.0, show_legend=False, box_aspect=0.0),
            "centering": PlotSettings(font_size=10.0, show_legend=True, box_aspect=0.0),
            "extrema": PlotSettings(font_size=10.0, show_legend=True, box_aspect=0.0),
            "lc": PlotSettings(font_size=10.0, show_legend=False, box_aspect=0.0),
        }

        # Build UI
        self._build_ui()
        self._connect()
        self._populate_strategy_list()

    # --------------------------- UI construction ---------------------------
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("Open Folder…")
        self.btn_update_json = QPushButton("Update JSON")
        self.btn_fit = QPushButton("Run Fit")
        self.btn_save = QPushButton("Save All Plots")
        self.btn_fit.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_update_json.setEnabled(False)
        toolbar.addWidget(self.btn_open)
        toolbar.addStretch(1)
        toolbar.addWidget(self.btn_update_json)
        toolbar.addWidget(self.btn_fit)
        toolbar.addWidget(self.btn_save)
        main.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Strategy + Metadata
        left = QWidget(); left_layout = QVBoxLayout(left)

        # Strategy group
        strat_group = QGroupBox("Fitting Strategy")
        strat_form = QFormLayout(strat_group)
        self.cmb_strategy = QComboBox()
        self.lbl_strategy_hint = QLabel("Strategy classes found in fitting_strategies/*")
        self.lbl_strategy_hint.setStyleSheet("color: gray;")
        strat_form.addRow("Strategy:", self.cmb_strategy)
        strat_form.addRow("", self.lbl_strategy_hint)

        # Reference presets
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

        # Metadata group (editable)
        meta_edit = QGroupBox("Metadata (editable before fitting)")
        f = QFormLayout(meta_edit)
        self.le_material = QLineEdit()
        self.le_crystal_orientation = QLineEdit()
        self.le_axis = QLineEdit()
        self.sb_t_thin = QDoubleSpinBox(); self.sb_t_thin.setRange(0.0, 1e6); self.sb_t_thin.setDecimals(6)
        self.sb_wedge = QDoubleSpinBox(); self.sb_wedge.setRange(-90.0, 90.0); self.sb_wedge.setDecimals(6)
        self.sb_beam_rx = QDoubleSpinBox(); self.sb_beam_rx.setRange(0.0, 1e6); self.sb_beam_rx.setDecimals(6)
        self.sb_beam_ry = QDoubleSpinBox(); self.sb_beam_ry.setRange(0.0, 1e6); self.sb_beam_ry.setDecimals(6)
        self.sb_input_pol = QDoubleSpinBox(); self.sb_input_pol.setRange(-360.0, 360.0); self.sb_input_pol.setDecimals(3)
        self.sb_detected_pol = QDoubleSpinBox(); self.sb_detected_pol.setRange(-360.0, 360.0); self.sb_detected_pol.setDecimals(3)
        for spin_box in [
            self.sb_t_thin,
            self.sb_wedge,
            self.sb_beam_rx,
            self.sb_beam_ry,
            self.sb_input_pol,
            self.sb_detected_pol,
        ]:
            spin_box.setLocale(QLocale.c())
        self.cmb_boxcar_sensitivity = QComboBox()
        self.cmb_boxcar_sensitivity.setEditable(True)
        self.cmb_boxcar_sensitivity.addItems(COMMON_BOXCAR_SENSITIVITIES)
        self.le_filters = QLineEdit()
        self.le_filters.setPlaceholderText("filter_id=value, ...")
        f.addRow("material:", self.le_material)
        f.addRow("crystal_orientation (e.g. 0,1,1):", self.le_crystal_orientation)
        f.addRow("rotation/translation axis:", self.le_axis)
        f.addRow("input polarization (deg):", self.sb_input_pol)
        f.addRow("detected polarization (deg):", self.sb_detected_pol)
        f.addRow("t_center_mm:", self.sb_t_thin)
        f.addRow("wedge_angle_deg:", self.sb_wedge)
        f.addRow("beam_r_x:", self.sb_beam_rx)
        f.addRow("beam_r_y:", self.sb_beam_ry)
        f.addRow("boxcar_sensitivity:", self.cmb_boxcar_sensitivity)
        f.addRow("filters:", self.le_filters)

        # Metadata (read-only excerpt)
        meta_view = QGroupBox("Loaded Metadata (read-only excerpt)")
        fv = QFormLayout(meta_view)
        self.lbl_sample = QLabel("—")
        self.lbl_method = QLabel("—")
        self.lbl_time = QLabel("—")
        fv.addRow("Sample:", self.lbl_sample)
        fv.addRow("Method:", self.lbl_method)
        fv.addRow("Timestamp:", self.lbl_time)

        left_layout.addWidget(strat_group)
        left_layout.addWidget(reference_group)
        left_layout.addWidget(meta_edit)
        left_layout.addWidget(meta_view)
        left_layout.addStretch(1)

        splitter.addWidget(left)

        # Right panel: Plots + Results table
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right = QWidget(); right_layout = QVBoxLayout(right)

        self.plot_tabs = QTabWidget()
        fit_tab = QWidget(); fit_layout = QVBoxLayout(fit_tab)
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

        resid_tab = QWidget(); resid_layout = QVBoxLayout(resid_tab)
        self.canvas_resid = MplCanvas(resid_tab, width=6.0, height=2.8)
        resid_layout.addWidget(self.canvas_resid)
        self.plot_tabs.addTab(resid_tab, "Residuals")

        center_tab = QWidget(); center_layout = QVBoxLayout(center_tab)
        self.canvas_centering = MplCanvas(center_tab, width=6.0, height=2.8)
        center_layout.addWidget(self.canvas_centering)
        self.plot_tabs.addTab(center_tab, "Centering Cost")

        extrema_tab = QWidget(); extrema_layout = QVBoxLayout(extrema_tab)
        self.canvas_extrema = MplCanvas(extrema_tab, width=6.0, height=2.8)
        extrema_layout.addWidget(self.canvas_extrema)
        self.plot_tabs.addTab(extrema_tab, "Extrema")

        lc_tab = QWidget(); lc_layout = QVBoxLayout(lc_tab)
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
        right_layout.addWidget(self.plot_tabs, 3)

        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Fit Parameter", "Saved Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.tbl.setMinimumHeight(180)
        right_layout.addWidget(self.tbl, 1)

        self.right_scroll.setWidget(right)
        splitter.addWidget(self.right_scroll)
        splitter.setSizes([360, 720])
        main.addWidget(splitter, 1)

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
            "extrema": self.canvas_extrema,
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
            decimals=2,
            step=0.01,
        )
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.btn_reset_manual = QPushButton("Reset to Auto Fit")
        self.btn_apply_manual = QPushButton("Overwrite JSON/CSV")
        buttons.addWidget(self.btn_reset_manual)
        buttons.addWidget(self.btn_apply_manual)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.lbl_manual_hint = QLabel(
            "The live overlay uses the current L and Peak values. Overwrite updates saved fit values."
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
        slider.setRange(0, self._SLIDER_STEPS)
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

    def _connect(self):
        self.btn_open.clicked.connect(self._select_folder)
        self.btn_update_json.clicked.connect(self._update_json_clicked)
        self.btn_fit.clicked.connect(self._run_fit_clicked)
        self.btn_save.clicked.connect(self._save_figures_clicked)
        self.reload_samples_btn.clicked.connect(self.reload_sample_catalog)
        self.reload_beams_btn.clicked.connect(self.reload_beam_profile_catalog)
        self.reload_filters_btn.clicked.connect(self.reload_filter_catalog)
        self.sample_preset_combo.currentIndexChanged.connect(self._apply_selected_sample_preset)
        self.beam_profile_combo.currentIndexChanged.connect(self._apply_selected_beam_profile)
        self.filter_preset_combo.activated.connect(self._filter_preset_activated)
        self.btn_plot_settings.clicked.connect(self._edit_current_plot_settings)
        self.btn_save_current_plot.clicked.connect(self._save_current_plot_clicked)
        self.btn_copy_current_plot.clicked.connect(self._copy_current_plot_clicked)
        self.btn_reset_manual.clicked.connect(self._reset_manual_controls_clicked)
        self.btn_apply_manual.clicked.connect(self._apply_manual_fit_clicked)
        self.cmb_strategy.currentIndexChanged.connect(self._strategy_selection_changed)
        self.cmb_lc_source.currentIndexChanged.connect(lambda *_args: self._render_analysis_plots())
        self.chk_fit_show_data.stateChanged.connect(lambda *_args: self._render_analysis_plots())
        self.chk_fit_show_fitting.stateChanged.connect(lambda *_args: self._render_analysis_plots())
        self.chk_fit_show_envelope.stateChanged.connect(lambda *_args: self._render_analysis_plots())

        for key in self._manual_controls:
            controls = self._manual_controls[key]
            controls["slider"].valueChanged.connect(
                lambda _value, name=key: self._manual_slider_changed(name)
            )
            controls["value"].valueChanged.connect(
                lambda _value, name=key: self._manual_value_changed(name)
            )
            controls["min"].valueChanged.connect(
                lambda _value, name=key: self._manual_range_changed(name)
            )
            controls["max"].valueChanged.connect(
                lambda _value, name=key: self._manual_range_changed(name)
            )
        self.reload_sample_catalog()
        self.reload_beam_profile_catalog()
        self.reload_filter_catalog()

    # ------------------------------- Strategies -------------------------------
    def _populate_strategy_list(self):
        """Scan fitting_strategies package for available strategy classes."""
        self.cmb_strategy.clear()
        self._strategies.clear()
        strategy_dir = Path(__file__).resolve().parent.parent / "fitting_strategies"
        try:
            for path in sorted(strategy_dir.glob("*.py")):
                name = path.stem
                if name in {"__init__", "base"}:
                    continue
                try:
                    module_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                    qual = f"fitting_strategies.{name}"
                    for node in module_ast.body:
                        if isinstance(node, ast.ClassDef) and node.name.endswith("Strategy"):
                            strategy = StrategyInfo(name, qual, node.name)
                            self._strategies.append(strategy)
                except Exception as e:
                    logging.warning("Failed to scan fitting strategy module %s: %s", path, e)
                    continue
            self._strategies.sort(key=lambda s: (s.module_name, s.class_name))
        except Exception as e:
            logging.error(f"Failed to find fitting strategies: {e}")

        for index, strategy in enumerate(self._strategies):
            self.cmb_strategy.addItem(strategy.display_name, userData=index)

        # Fallback text if none found
        if self.cmb_strategy.count() == 0:
            self.cmb_strategy.addItem("(no strategies found)")
            self.lbl_strategy_hint.setText("No strategy classes found.")
        else:
            self.lbl_strategy_hint.setText(
                f"{len(self._strategies)} strategy class(es) found in fitting_strategies/*"
            )
        self._refresh_analysis_views(reset_manual=True)

    def _get_selected_strategy(self) -> Optional[StrategyInfo]:
        idx = self.cmb_strategy.currentIndex()
        if idx < 0:
            return None
        data = self.cmb_strategy.currentData()
        if isinstance(data, int) and 0 <= data < len(self._strategies):
            return self._strategies[data]
        # If no userdata (e.g., no strategies), try a conventional default
        return None

    def _strategy_selection_changed(self, *_args):
        if self._meta:
            self._populate_table_from_json(self._meta)
        self._refresh_analysis_views(reset_manual=True)

    def _fit_payload_for_strategy(
        self,
        meta: Optional[Dict[str, Any]] = None,
        strategy: Optional[StrategyInfo] = None,
    ) -> Dict[str, Any]:
        selected = strategy or self._get_selected_strategy()
        strategy_name = selected.class_name if selected is not None else None
        return extract_fit_payload(meta if meta is not None else self._meta, strategy_name)

    def _meta_with_selected_fit(
        self,
        meta: Optional[Dict[str, Any]] = None,
        strategy: Optional[StrategyInfo] = None,
    ) -> Dict[str, Any]:
        selected = strategy or self._get_selected_strategy()
        strategy_name = selected.class_name if selected is not None else None
        return merge_fit_payload(meta if meta is not None else self._meta, strategy_name)

    def _apply_saved_strategy_selection(self, meta: Dict[str, Any]):
        target_name = str(meta.get("fitting_active_strategy") or "").strip()
        if not target_name:
            payload = extract_fit_payload(meta)
            target_name = str(payload.get("strategy") or "").strip()
        if not target_name:
            return

        for index, strategy in enumerate(self._strategies):
            if strategy.class_name == target_name:
                self.cmb_strategy.setCurrentIndex(index)
                return

    # --------------------------- Reference data ---------------------------
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

    def reload_beam_profile_catalog(self):
        current_id = self.beam_profile_combo.currentData()
        catalog = load_beam_profile_catalog()
        self._beam_profile_catalog_map = {
            entry["id"]: entry for entry in catalog["beam_profiles"]
        }

        self.beam_profile_combo.blockSignals(True)
        self.beam_profile_combo.clear()
        self.beam_profile_combo.addItem("Manual entry", None)
        for entry in catalog["beam_profiles"]:
            self.beam_profile_combo.addItem(format_beam_profile_display(entry), entry["id"])

        restored_index = 0
        if current_id in self._beam_profile_catalog_map:
            for index in range(self.beam_profile_combo.count()):
                if self.beam_profile_combo.itemData(index) == current_id:
                    restored_index = index
                    break
        self.beam_profile_combo.setCurrentIndex(restored_index)
        self.beam_profile_combo.blockSignals(False)

    def reload_filter_catalog(self):
        current_id = self.filter_preset_combo.currentData()
        catalog = load_nd_filter_catalog()
        self._filter_catalog_map = {
            entry["filter_id"]: entry for entry in catalog["filters"]
        }

        self.filter_preset_combo.blockSignals(True)
        self.filter_preset_combo.clear()
        self.filter_preset_combo.addItem("Select filter", None)
        for entry in catalog["filters"]:
            self.filter_preset_combo.addItem(format_filter_display(entry), entry["filter_id"])

        restored_index = 0
        if current_id in self._filter_catalog_map:
            for index in range(self.filter_preset_combo.count()):
                if self.filter_preset_combo.itemData(index) == current_id:
                    restored_index = index
                    break
        self.filter_preset_combo.setCurrentIndex(restored_index)
        self.filter_preset_combo.blockSignals(False)

    def _selected_sample_entry(self) -> Optional[Dict[str, Any]]:
        sample_key = self.sample_preset_combo.currentData()
        if sample_key is None:
            return None
        return self._sample_catalog_map.get(sample_key)

    def _selected_beam_profile_entry(self) -> Optional[Dict[str, Any]]:
        profile_id = self.beam_profile_combo.currentData()
        if profile_id is None:
            return None
        return self._beam_profile_catalog_map.get(profile_id)

    def _selected_filter_entry(self) -> Optional[Dict[str, Any]]:
        filter_id = self.filter_preset_combo.currentData()
        if filter_id is None:
            return None
        return self._filter_catalog_map.get(filter_id)

    def _apply_selected_sample_preset(self, *_args):
        entry = self._selected_sample_entry()
        if entry is None:
            return
        sample_meta = sample_metadata_from_entry(entry)
        self.le_material.setText(str(sample_meta.get("material", "")))
        self.le_crystal_orientation.setText(str(sample_meta.get("crystal_orientation", "")))
        thickness_info = sample_meta.get("thickness_info") or {}
        self.sb_t_thin.setValue(float(thickness_info.get("t_center_mm") or 0.0))
        self.sb_wedge.setValue(float(thickness_info.get("wedge_angle_deg") or 0.0))

    def _apply_selected_beam_profile(self, *_args):
        entry = self._selected_beam_profile_entry()
        if entry is None:
            return
        if entry.get("beam_r_x") is not None:
            self.sb_beam_rx.setValue(float(entry["beam_r_x"]))
        if entry.get("beam_r_y") is not None:
            self.sb_beam_ry.setValue(float(entry["beam_r_y"]))

    def _parse_filters_text(self, text: str) -> Dict[str, float]:
        parsed_filters: Dict[str, float] = {}
        for chunk in str(text).split(","):
            part = chunk.strip()
            if not part or "=" not in part:
                continue
            filter_id, transmission = part.split("=", 1)
            try:
                parsed_filters[filter_id.strip()] = float(transmission.strip())
            except ValueError:
                continue
        return parsed_filters

    def _format_filters_text(self, filters: Dict[str, float]) -> str:
        if not filters:
            return ""
        return ", ".join(f"{filter_id}={transmission:g}" for filter_id, transmission in filters.items())

    def _append_selected_filter_preset(self):
        entry = self._selected_filter_entry()
        if entry is None:
            return

        wavelength_nm = self._safe_float(self._meta.get("wavelength_nm"))
        filters, warnings = resolve_selected_filters(
            selected_filters=[entry],
            fundamental_wavelength_nm=wavelength_nm if np.isfinite(wavelength_nm) else None,
        )
        if not filters:
            warning_text = "\n".join(warnings) if warnings else "Could not derive transmission for the selected filter."
            QMessageBox.warning(self, "Filter not applied", warning_text)
            return

        current = self._parse_filters_text(self.le_filters.text())
        current.update(filters)
        self.le_filters.setText(self._format_filters_text(current))
        if warnings:
            QMessageBox.information(self, "Filter added with warning", "\n".join(warnings))

    def _filter_preset_activated(self, *_args):
        if self.filter_preset_combo.currentData() is None:
            return
        self._append_selected_filter_preset()
        self.filter_preset_combo.blockSignals(True)
        self.filter_preset_combo.setCurrentIndex(0)
        self.filter_preset_combo.blockSignals(False)

    # ------------------------------- File I/O -------------------------------
    def _select_folder(self):
        start_dir = str(self._current_dir) if self._current_dir else "results"
        folder = QFileDialog.getExistingDirectory(self, "Select result folder", start_dir)
        if not folder:
            return
        ok, msg = self._load_folder(Path(folder))
        if not ok:
            QMessageBox.warning(self, "Load failed", msg)
        else:
            self.folderLoaded.emit(folder)

    def _load_folder(self, folder: Path) -> Tuple[bool, str]:
        csv_files = list(folder.glob("*.csv"))
        json_files = list(folder.glob("*.json"))
        if len(csv_files) != 1 or len(json_files) != 1:
            return False, "Folder must contain exactly one .csv and one .json"
        self.csv_path = csv_files[0]
        self.json_path = json_files[0]

        # Read JSON
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            return False, f"Failed to read JSON: {e}"

        # Read CSV
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            return False, f"Failed to read CSV: {e}"

        self._current_dir = folder
        self._meta = meta
        self._df = df

        # Update view fields
        self._apply_saved_strategy_selection(meta)
        self._populate_meta_labels(meta)
        self._prefill_metadata_editors(meta)
        self._populate_table_from_json(meta)
        self._refresh_analysis_views(reset_manual=True)

        self.btn_fit.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_update_json.setEnabled(True)
        return True, "OK"

    def _populate_meta_labels(self, meta: Dict):
        self.lbl_sample.setText(str(meta.get("sample") or meta.get("sample_id") or "—"))
        self.lbl_method.setText(str(meta.get("method") or meta.get("measurement_method") or "—"))
        self.lbl_time.setText(str(meta.get("timestamp") or meta.get("measured_at") or "—"))

    def _prefill_metadata_editors(self, meta: Dict):
        # Basic values if present
        self.le_material.setText(str(meta.get("material", "")))
        # crystal_orientation as list -> "a,b,c"
        ori = meta.get("crystal_orientation")
        if isinstance(ori, (list, tuple)):
            self.le_crystal_orientation.setText(",".join(str(int(x)) for x in ori))
        elif isinstance(ori, str):
            self.le_crystal_orientation.setText(ori)
        else:
            self.le_crystal_orientation.setText("")
        self.le_axis.setText(str(meta.get("rot/trans_axis", "")))
        try:
            self.sb_input_pol.setValue(float(meta.get("input_polarization", 0.0)))
        except Exception:
            self.sb_input_pol.setValue(0.0)
        try:
            self.sb_detected_pol.setValue(float(meta.get("detected_polarization", 0.0)))
        except Exception:
            self.sb_detected_pol.setValue(0.0)
        # thickness_info
        tinfo = meta.get("thickness_info") or {}
        try:
            self.sb_t_thin.setValue(float(tinfo.get("t_center_mm", tinfo.get("t_at_thin_end_mm", 0.0))))
        except Exception:
            self.sb_t_thin.setValue(0.0)
        try:
            self.sb_wedge.setValue(float(tinfo.get("wedge_angle_deg", 0.0)))
        except Exception:
            self.sb_wedge.setValue(0.0)
        # beam sizes
        try:
            self.sb_beam_rx.setValue(float(meta.get("beam_r_x", 0.0)))
        except Exception:
            self.sb_beam_rx.setValue(0.0)
        try:
            self.sb_beam_ry.setValue(float(meta.get("beam_r_y", 0.0)))
        except Exception:
            self.sb_beam_ry.setValue(0.0)
        boxcar = meta.get("boxcar_sensitivity")
        if isinstance(boxcar, dict):
            self.cmb_boxcar_sensitivity.setCurrentText(str(boxcar.get("label") or ""))
        else:
            self.cmb_boxcar_sensitivity.setCurrentText("")

        filters = meta.get("filters")
        if isinstance(filters, dict) and filters:
            normalized_filters = {}
            for key, value in filters.items():
                numeric = self._safe_float(value)
                if np.isfinite(numeric):
                    normalized_filters[str(key)] = numeric
            filter_text = self._format_filters_text(normalized_filters)
        else:
            filter_text = ""
        self.le_filters.setText(filter_text)
        self._sync_reference_presets_from_meta(meta)

    def _sync_reference_presets_from_meta(self, meta: Dict[str, Any]):
        sample_key = build_sample_catalog_key(meta.get("sample"), meta.get("crystal_orientation"))
        self.sample_preset_combo.blockSignals(True)
        sample_index = 0
        for index in range(self.sample_preset_combo.count()):
            if self.sample_preset_combo.itemData(index) == sample_key:
                sample_index = index
                break
        self.sample_preset_combo.setCurrentIndex(sample_index)
        self.sample_preset_combo.blockSignals(False)

        beam_x = self._safe_float(meta.get("beam_r_x"))
        beam_y = self._safe_float(meta.get("beam_r_y"))
        self.beam_profile_combo.blockSignals(True)
        beam_index = 0
        if np.isfinite(beam_x) and np.isfinite(beam_y):
            for index in range(self.beam_profile_combo.count()):
                profile_id = self.beam_profile_combo.itemData(index)
                entry = self._beam_profile_catalog_map.get(profile_id)
                if not entry:
                    continue
                entry_x = self._safe_float(entry.get("beam_r_x"))
                entry_y = self._safe_float(entry.get("beam_r_y"))
                if np.isfinite(entry_x) and np.isfinite(entry_y) and np.isclose(entry_x, beam_x) and np.isclose(entry_y, beam_y):
                    beam_index = index
                    break
        self.beam_profile_combo.setCurrentIndex(beam_index)
        self.beam_profile_combo.blockSignals(False)

        meta_filters = meta.get("filters")
        filter_index = 0
        if isinstance(meta_filters, dict):
            for index in range(self.filter_preset_combo.count()):
                filter_id = self.filter_preset_combo.itemData(index)
                if filter_id and filter_id in meta_filters:
                    filter_index = index
                    break
        self.filter_preset_combo.blockSignals(True)
        self.filter_preset_combo.setCurrentIndex(filter_index)
        self.filter_preset_combo.blockSignals(False)

    # ------------------------------ JSON update ------------------------------
    def _update_json_clicked(self):
        ok, message = self._write_json_metadata(show_message=True)
        if not ok:
            QMessageBox.critical(self, "Update failed", message)

    def _collect_metadata_from_editors(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(meta)
        payload["material"] = self.le_material.text().strip() or payload.get("material")

        ori_txt = self.le_crystal_orientation.text().strip()
        if ori_txt:
            try:
                parts = [int(x) for x in ori_txt.replace(" ", "").split(",") if x != ""]
                if len(parts) == 3:
                    payload["crystal_orientation"] = parts
            except Exception:
                pass
        payload["rot/trans_axis"] = self.le_axis.text().strip()
        payload["input_polarization"] = float(self.sb_input_pol.value())
        payload["detected_polarization"] = float(self.sb_detected_pol.value())

        tinfo = dict(payload.get("thickness_info") or {})
        tinfo["t_center_mm"] = float(self.sb_t_thin.value())
        tinfo.pop("t_at_thin_end_mm", None)
        tinfo["wedge_angle_deg"] = float(self.sb_wedge.value())
        payload["thickness_info"] = tinfo
        payload["beam_r_x"] = float(self.sb_beam_rx.value())
        payload["beam_r_y"] = float(self.sb_beam_ry.value())
        boxcar_text = self.cmb_boxcar_sensitivity.currentText().strip()
        if boxcar_text:
            try:
                payload["boxcar_sensitivity"] = parse_boxcar_sensitivity(boxcar_text)
            except ValueError:
                payload["boxcar_sensitivity"] = {"label": boxcar_text}
        else:
            payload.pop("boxcar_sensitivity", None)

        filters_text = self.le_filters.text().strip()
        if filters_text:
            payload["filters"] = self._parse_filters_text(filters_text)
        else:
            payload["filters"] = {}
        return payload

    def _write_json_metadata(self, show_message: bool = False) -> Tuple[bool, str]:
        if not self.json_path:
            return False, "Load a result folder first."
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            return False, f"Failed to read JSON: {e}"

        meta = self._collect_metadata_from_editors(meta)

        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return False, f"Failed to write JSON: {e}"

        self._meta = meta
        self._populate_table_from_json(meta)
        self._refresh_analysis_views(reset_manual=False)
        if show_message:
            QMessageBox.information(self, "Updated", "JSON metadata updated.")
        return True, "OK"

    # -------------------------------- Run fit --------------------------------
    def _run_fit_clicked(self):
        if not self._current_dir:
            QMessageBox.information(self, "No data", "Load a result folder first.")
            return
        if SHGDataAnalysis is None:
            QMessageBox.critical(self, "Missing module", "shg_analysis is not importable.")
            return
        # Ensure JSON reflects current editor values before fitting
        ok, message = self._write_json_metadata(show_message=False)
        if not ok:
            QMessageBox.critical(self, "Update failed", message)
            return

        # Resolve strategy class
        s = self._get_selected_strategy()
        if s is None:
            QMessageBox.critical(self, "No strategy", "No fitting strategy is available/selected.")
            return
        try:
            mod = importlib.import_module(s.qualname)
            StrategyCls = getattr(mod, s.class_name)
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Failed to import {s.qualname}.{s.class_name}: {e}")
            return

        # Run fit
        try:
            analysis = SHGDataAnalysis(str(self._current_dir))
            results = analysis.run(StrategyCls)
            if results is None:
                QMessageBox.critical(self, "Fitting error", f"{analysis.last_error}")
                return
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            return

        # Reload files from disk and refresh UI
        ok, msg = self._load_folder(self._current_dir)
        if not ok:
            QMessageBox.warning(self, "Reload failed", msg)
            return
        self._refresh_analysis_views(reset_manual=True)

    # ------------------------------ Table/plots ------------------------------
    def _populate_table_from_json(self, meta: Dict):
        """Show saved fit-oriented values without duplicating the metadata panel."""
        rows = []
        fit_payload = self._fit_payload_for_strategy(meta)
        for key, label in [
            ("L_mm", "Corrected L [mm]"),
            ("L_mm_std", "Corrected L std [mm]"),
            ("k_scale", "k scale"),
            ("k_scale_std", "k scale std"),
            ("Pm0", "Peak Pm0"),
            ("Pm0_stderr", "Peak Pm0 std"),
            ("d_factor", "d factor"),
            ("Lc_mean_mm", "Lc mean [mm]"),
            ("Lc_std_mm", "Lc std [mm]"),
            ("residual_rms", "Residual RMS"),
            ("minima_count", "Minima count"),
            ("n_count", "Lc pair count"),
            ("n_peaks", "Peak count"),
        ]:
            if key in fit_payload:
                rows.append((label, fit_payload.get(key)))

        selected = self._get_selected_strategy()
        if selected is not None:
            rows.insert(0, ("strategy", selected.display_name))

        self.tbl.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.tbl.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl.setItem(i, 1, QTableWidgetItem("" if v is None else f"{v}"))

    def _refresh_analysis_views(self, reset_manual: bool = False):
        if self._df is None or self._current_dir is None:
            self._analysis_context = {}
            self._clear_plots()
            return
        self._analysis_context = self._prepare_analysis_context()
        if reset_manual or not self._manual_controls_ready():
            self._initialize_manual_controls_from_context()
        self._render_analysis_plots()

    def _prepare_analysis_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {"error": None}
        if not self._current_dir:
            context["error"] = "Load a result folder first."
            return context
        if SHGDataAnalysis is None:
            context["error"] = "shg_analysis is not importable."
            return context

        selected = self._get_selected_strategy()
        if selected is None:
            context["error"] = "No fitting strategy selected."
            return context
        meta_with_fit = self._meta_with_selected_fit(self._meta, selected)

        try:
            mod = importlib.import_module(selected.qualname)
            strategy_cls = getattr(mod, selected.class_name)
            analysis = SHGDataAnalysis(str(self._current_dir))
            strategy = strategy_cls(analysis)
        except Exception as e:
            context["error"] = f"Failed to initialize strategy: {e}"
            return context

        prepared = analysis.data.copy()
        centering_info = None
        offset_info = {"offset": 0.0}
        notes: List[str] = []

        if hasattr(strategy, "_position_centering"):
            try:
                prepared, centering_info = self._unwrap_data_and_aux(strategy._position_centering(prepared))
            except Exception as e:
                notes.append(f"Centering: {e}")

        if hasattr(strategy, "_subtract_offset"):
            try:
                prepared, offset_info = self._unwrap_data_and_aux(strategy._subtract_offset(prepared))
            except Exception as e:
                notes.append(f"Offset correction: {e}")

        display_x = np.asarray(prepared["position_centered"], dtype=float) if "position_centered" in prepared.columns else np.asarray(prepared["position"], dtype=float)
        if "offset_corrected" in prepared.columns:
            display_y = np.asarray(prepared["offset_corrected"], dtype=float)
        elif "intensity_corrected" in prepared.columns:
            display_y = np.asarray(prepared["intensity_corrected"], dtype=float)
        else:
            display_y = np.asarray(prepared["ch2"], dtype=float)
        raw_y = np.asarray(prepared["intensity_corrected"], dtype=float) if "intensity_corrected" in prepared.columns else np.asarray(display_y, dtype=float)

        context.update(
            {
                "analysis": analysis,
                "strategy": strategy,
                "selected_strategy": selected,
                "prepared_data": prepared,
                "display_x": display_x,
                "display_y": display_y,
                "raw_y": raw_y,
                "offset": float(offset_info.get("offset", 0.0)) if isinstance(offset_info, dict) else 0.0,
                "centering_info": centering_info if isinstance(centering_info, dict) else None,
                "offset_info": offset_info if isinstance(offset_info, dict) else {"offset": 0.0},
                "saved_fit": self._fit_payload_for_strategy(self._meta, selected),
                "auto_L": self._infer_auto_L(meta_with_fit, strategy, prepared),
                "auto_peak": self._infer_auto_peak(meta_with_fit, strategy, prepared),
                "notes": notes,
                "x_label": self._x_axis_label(analysis.meta, prepared),
            }
        )
        return context

    def _unwrap_data_and_aux(self, result: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            data, aux = result
            return data, aux if isinstance(aux, dict) else {}
        return result, {}

    def _unwrap_model_and_aux(self, result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            model, aux = result
            return np.asarray(model, dtype=float), aux if isinstance(aux, dict) else {}
        return np.asarray(result, dtype=float), {}

    def _evaluate_strategy_curves(self, strategy: Any, L_value: float, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        override = {"L": L_value, "theta_deg": x}
        fit_base, fit_aux = self._unwrap_model_and_aux(
            strategy._maker_fringes(override=override, return_aux=True)
        )

        try:
            env_base, env_aux = self._unwrap_model_and_aux(
                strategy._maker_fringes(override=override, envelope=True, return_aux=True)
            )
        except TypeError:
            env_base, env_aux = fit_base, fit_aux

        merged_aux = dict(fit_aux)
        for key, value in env_aux.items():
            merged_aux.setdefault(key, value)
        return fit_base, env_base, merged_aux

    def _x_axis_label(self, meta: Dict[str, Any], data: pd.DataFrame) -> str:
        if "position_centered" in data.columns:
            if str(meta.get("method", "")).lower() == "rotation":
                return "Angle centered (deg)"
            return "Centered position"
        if str(meta.get("method", "")).lower() == "rotation":
            return "Angle (deg)"
        return "Position"

    def _infer_auto_L(self, meta: Dict[str, Any], strategy: Any, data: pd.DataFrame) -> float:
        value = self._safe_float(meta.get("L_mm"))
        if np.isfinite(value):
            return value
        if hasattr(strategy, "_fit_L_small_angle"):
            try:
                fit = strategy._fit_L_small_angle(meta, data)
                value = self._safe_float(fit.get("L_mm"))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
        tinfo = meta.get("thickness_info") or {}
        for key in ("t_center_mm", "t_at_thin_end_mm"):
            value = self._safe_float(tinfo.get(key))
            if np.isfinite(value):
                return value
        return 0.0

    def _infer_auto_peak(self, meta: Dict[str, Any], strategy: Any, data: pd.DataFrame) -> float:
        for key in ("Pm0", "k_scale"):
            value = self._safe_float(meta.get(key))
            if np.isfinite(value):
                return value
        if hasattr(strategy, "_fit_Pm0"):
            try:
                fit, _aux = strategy._fit_Pm0(data)
                value = self._safe_float(fit.get("Pm0"))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
        y = np.asarray(data.get("offset_corrected", data.get("intensity_corrected", [])), dtype=float)
        if y.size:
            value = float(np.nanmax(y))
            if np.isfinite(value):
                return max(value, 0.0)
        return 1.0

    def _safe_float(self, value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _is_wedge_scan(self) -> bool:
        return str(self._meta.get("method", "")).strip().lower() == "wedge"

    def _nominal_thickness_mm(self) -> float:
        tinfo = self._meta.get("thickness_info") or {}
        for key in ("t_center_mm", "t_at_thin_end_mm"):
            value = self._safe_float(tinfo.get(key))
            if np.isfinite(value):
                return value
        return 0.0

    def _current_plot_key(self) -> str:
        current = self.plot_tabs.currentWidget()
        for key, widget in self._plot_pages.items():
            if widget is current:
                return key
        return "fit"

    def _current_plot_canvas(self) -> MplCanvas:
        return self._plot_canvases[self._current_plot_key()]

    def _current_fitting_output_dir(self, create: bool = False) -> Optional[Path]:
        if not self._current_dir:
            return None
        selected = self._get_selected_strategy()
        class_name = selected.class_name if selected is not None else "current"
        safe_class_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in class_name
        ).strip("_") or "current"
        output_dir = self._current_dir / f"fitting_{safe_class_name}"
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_all_canvas_heights()

    def _update_plot_tab_visibility(self):
        if not hasattr(self, "_plot_pages"):
            return
        is_wedge = self._is_wedge_scan()
        for key in ("centering", "lc"):
            index = self.plot_tabs.indexOf(self._plot_pages[key])
            if index >= 0:
                self.plot_tabs.setTabVisible(index, not is_wedge)
        if is_wedge and self._current_plot_key() in {"centering", "lc"}:
            self.plot_tabs.setCurrentWidget(self._plot_pages["fit"])
        if hasattr(self, "chk_fit_show_envelope"):
            self.chk_fit_show_envelope.setEnabled(not is_wedge)
        self._update_all_canvas_heights()

    def _canvas_base_height(self, plot_key: str) -> int:
        return {
            "fit": 360,
            "resid": 260,
            "centering": 260,
            "extrema": 280,
            "lc": 260,
        }.get(plot_key, 280)

    def _canvas_supports_top_axis(self, plot_key: str) -> bool:
        return self._is_wedge_scan() and plot_key in {"fit", "resid", "extrema"}

    def _canvas_target_height(self, plot_key: str) -> int:
        settings = self._plot_settings[plot_key]
        base_height = self._canvas_base_height(plot_key)
        if settings.box_aspect <= 0:
            return base_height

        if hasattr(self, "right_scroll") and self.right_scroll.viewport() is not None:
            width_hint = self.right_scroll.viewport().width() - 80
        else:
            width_hint = self.width() - 520
        width_hint = max(width_hint, 520)

        extra_height = 120 if self._canvas_supports_top_axis(plot_key) else 90
        computed = int(width_hint * settings.box_aspect + extra_height)
        return max(base_height, computed)

    def _update_canvas_height(self, plot_key: str):
        if not hasattr(self, "_plot_canvases"):
            return
        canvas = self._plot_canvases[plot_key]
        height = self._canvas_target_height(plot_key)
        canvas.setFixedHeight(height)

    def _update_all_canvas_heights(self):
        if not hasattr(self, "_plot_canvases"):
            return
        for plot_key in self._plot_canvases:
            self._update_canvas_height(plot_key)

    def _parse_optional_float(self, text: str) -> Optional[float]:
        stripped = str(text).strip()
        if not stripped:
            return None
        return float(stripped)

    def _edit_current_plot_settings(self):
        plot_key = self._current_plot_key()
        settings = self._plot_settings[plot_key]

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Plot Settings: {self.plot_tabs.tabText(self.plot_tabs.currentIndex())}")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        sb_font = QDoubleSpinBox()
        sb_font.setRange(6.0, 48.0)
        sb_font.setDecimals(1)
        sb_font.setSingleStep(0.5)
        sb_font.setValue(settings.font_size)

        cb_legend = QCheckBox("Show legend")
        cb_legend.setChecked(settings.show_legend)

        sb_aspect = QDoubleSpinBox()
        sb_aspect.setRange(0.0, 5.0)
        sb_aspect.setDecimals(2)
        sb_aspect.setSingleStep(0.05)
        sb_aspect.setSpecialValueText("Auto")
        sb_aspect.setValue(settings.box_aspect)

        le_x_min = QLineEdit("" if settings.x_min is None else f"{settings.x_min:g}")
        le_x_max = QLineEdit("" if settings.x_max is None else f"{settings.x_max:g}")
        le_y_min = QLineEdit("" if settings.y_min is None else f"{settings.y_min:g}")
        le_y_max = QLineEdit("" if settings.y_max is None else f"{settings.y_max:g}")

        form.addRow("Font size:", sb_font)
        form.addRow("", cb_legend)
        form.addRow("Box aspect:", sb_aspect)
        form.addRow("X min:", le_x_min)
        form.addRow("X max:", le_x_max)
        form.addRow("Y min:", le_y_min)
        form.addRow("Y max:", le_y_max)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: (
                sb_font.setValue(10.0 if plot_key != "fit" else 11.0),
                cb_legend.setChecked(plot_key != "resid" and plot_key != "lc"),
                sb_aspect.setValue(0.0),
                le_x_min.clear(),
                le_x_max.clear(),
                le_y_min.clear(),
                le_y_max.clear(),
            )
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            self._plot_settings[plot_key] = PlotSettings(
                font_size=float(sb_font.value()),
                show_legend=bool(cb_legend.isChecked()),
                box_aspect=float(sb_aspect.value()),
                x_min=self._parse_optional_float(le_x_min.text()),
                x_max=self._parse_optional_float(le_x_max.text()),
                y_min=self._parse_optional_float(le_y_min.text()),
                y_max=self._parse_optional_float(le_y_max.text()),
            )
        except Exception as e:
            QMessageBox.warning(self, "Invalid setting", str(e))
            return

        self._update_canvas_height(plot_key)
        self._render_analysis_plots()

    def _save_current_plot_clicked(self):
        if not self._current_dir:
            QMessageBox.information(self, "No folder", "Load a result folder first.")
            return
        plot_key = self._current_plot_key()
        output_dir = self._current_fitting_output_dir(create=True) or self._current_dir
        default_name = {
            "fit": "fit_overlay.png",
            "resid": "residuals.png",
            "centering": "centering_cost.png",
            "extrema": "extrema.png",
            "lc": "lc_pairs.png",
        }[plot_key]
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save current plot",
            str(output_dir / default_name),
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            self._current_plot_canvas().figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Saved", f"Saved plot to {path}")

    def _copy_current_plot_clicked(self):
        if not self._current_dir:
            QMessageBox.information(self, "No folder", "Load a result folder first.")
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setPixmap(self._current_plot_canvas().grab())
        QMessageBox.information(self, "Copied", "Current plot image was copied to the clipboard.")

    def _configure_plot_axes(self, canvas: MplCanvas, plot_key: str, y_label: str, top_axis_L_mm: Optional[float] = None):
        settings = self._plot_settings[plot_key]
        ax = canvas.ax
        is_wedge = self._is_wedge_scan()

        bottom_label = "position (mm)" if is_wedge else "Incidence angle (deg.)"
        ax.set_xlabel(bottom_label)
        ax.set_ylabel(y_label)

        font_size = settings.font_size
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        if settings.box_aspect > 0:
            ax.set_box_aspect(settings.box_aspect)

        if settings.x_min is not None or settings.x_max is not None:
            ax.set_xlim(
                left=settings.x_min if settings.x_min is not None else None,
                right=settings.x_max if settings.x_max is not None else None,
            )
        if settings.y_min is not None or settings.y_max is not None:
            ax.set_ylim(
                bottom=settings.y_min if settings.y_min is not None else None,
                top=settings.y_max if settings.y_max is not None else None,
            )

        handles, labels = ax.get_legend_handles_labels()
        if settings.show_legend and handles:
            legend = ax.legend(loc="best", fontsize=font_size)
            if legend is not None:
                legend.set_title(None)

        if not is_wedge or top_axis_L_mm is None:
            return

        strategy = self._analysis_context.get("strategy")
        tinfo = self._meta.get("thickness_info") or {}
        wedge_deg = self._safe_float(tinfo.get("wedge_angle_deg"), 0.0)
        slope = math.tan(math.radians(wedge_deg))
        center_pos = float(getattr(strategy, "center_pos", 18.05))
        if abs(slope) < 1e-12:
            return

        def pos_to_thickness(position):
            arr = np.asarray(position, dtype=float)
            return top_axis_L_mm + (arr - center_pos) * slope

        def thickness_to_pos(thickness):
            arr = np.asarray(thickness, dtype=float)
            return center_pos + (arr - top_axis_L_mm) / slope

        secax = ax.secondary_xaxis("top", functions=(pos_to_thickness, thickness_to_pos))
        secax.set_xlabel("Sample thickness (mm)")
        secax.xaxis.label.set_size(font_size)
        secax.tick_params(axis="x", labelsize=font_size)

    def _manual_controls_ready(self) -> bool:
        if not self._manual_controls:
            return False
        return all(np.isfinite(float(controls["value"].value())) for controls in self._manual_controls.values())

    def _initialize_manual_controls_from_context(self):
        context = self._analysis_context
        if context.get("error"):
            return

        saved_fit = context.get("saved_fit", {})
        auto_L = float(context.get("auto_L", 0.0))
        auto_peak = float(context.get("auto_peak", 1.0))
        y = np.asarray(context.get("display_y", []), dtype=float)
        y_max = float(np.nanmax(y)) if y.size and np.isfinite(np.nanmax(y)) else max(auto_peak, 1.0)

        l_std = self._safe_float(saved_fit.get("L_mm_std"))
        l_span = max(5.0 * l_std, 0.005) if np.isfinite(l_std) and l_std > 0 else max(abs(auto_L) * 0.05, 0.02)
        peak_std = self._safe_float(saved_fit.get("Pm0_stderr"))
        if not np.isfinite(peak_std) or peak_std <= 0:
            peak_std = self._safe_float(saved_fit.get("k_scale_std"))
        if np.isfinite(peak_std) and peak_std > 0:
            peak_span = max(5.0 * peak_std, max(y_max, auto_peak) * 0.1)
        else:
            peak_span = max(abs(auto_peak) * 0.5, y_max * 0.5, 0.5)

        self._set_manual_control("L", auto_L - l_span, auto_L + l_span, auto_L)
        self._set_manual_control("peak", 0.0, max(auto_peak + peak_span, peak_span), max(auto_peak, 0.0))

    def _set_manual_control(self, key: str, minimum: float, maximum: float, value: float):
        controls = self._manual_controls[key]
        if maximum <= minimum:
            maximum = minimum + 1e-9
        value = min(max(value, minimum), maximum)

        self._manual_syncing = True
        try:
            controls["min"].setValue(minimum)
            controls["max"].setValue(maximum)
            controls["value"].setRange(minimum, maximum)
            controls["value"].setValue(value)
            self._sync_slider_to_value(key)
        finally:
            self._manual_syncing = False

    def _sync_slider_to_value(self, key: str):
        controls = self._manual_controls[key]
        minimum = float(controls["min"].value())
        maximum = float(controls["max"].value())
        value = float(controls["value"].value())
        if maximum <= minimum:
            slider_value = 0
        else:
            ratio = (value - minimum) / (maximum - minimum)
            slider_value = int(round(ratio * self._SLIDER_STEPS))
        controls["slider"].blockSignals(True)
        controls["slider"].setValue(max(0, min(self._SLIDER_STEPS, slider_value)))
        controls["slider"].blockSignals(False)

    def _manual_range_changed(self, key: str):
        if self._manual_syncing:
            return
        controls = self._manual_controls[key]
        minimum = float(controls["min"].value())
        maximum = float(controls["max"].value())
        if maximum <= minimum:
            maximum = minimum + 1e-9
            self._manual_syncing = True
            try:
                controls["max"].setValue(maximum)
            finally:
                self._manual_syncing = False
        controls["value"].setRange(minimum, maximum)
        current = min(max(float(controls["value"].value()), minimum), maximum)
        self._manual_syncing = True
        try:
            controls["value"].setValue(current)
            self._sync_slider_to_value(key)
        finally:
            self._manual_syncing = False
        self._render_analysis_plots()

    def _manual_value_changed(self, key: str):
        if self._manual_syncing:
            return
        self._manual_syncing = True
        try:
            self._sync_slider_to_value(key)
        finally:
            self._manual_syncing = False
        self._render_analysis_plots()

    def _manual_slider_changed(self, key: str):
        if self._manual_syncing:
            return
        controls = self._manual_controls[key]
        minimum = float(controls["min"].value())
        maximum = float(controls["max"].value())
        if maximum <= minimum:
            value = minimum
        else:
            value = minimum + (maximum - minimum) * int(controls["slider"].value()) / self._SLIDER_STEPS
        self._manual_syncing = True
        try:
            controls["value"].setValue(value)
        finally:
            self._manual_syncing = False
        self._render_analysis_plots()

    def _reset_manual_controls_clicked(self):
        self._initialize_manual_controls_from_context()
        self._render_analysis_plots()

    def _manual_value(self, key: str) -> float:
        return float(self._manual_controls[key]["value"].value())

    def _compute_live_curves(self) -> Dict[str, Any]:
        context = self._analysis_context
        if context.get("error"):
            return {"error": context["error"]}

        strategy = context["strategy"]
        x = np.asarray(context["display_x"], dtype=float)
        y = np.asarray(context["display_y"], dtype=float)
        L_value = self._manual_value("L")
        peak_value = self._manual_value("peak")

        try:
            fit_base, env_base, fit_aux = self._evaluate_strategy_curves(strategy, L_value, x)
        except Exception as e:
            return {"error": f"Failed to evaluate current fit: {e}"}

        fit_curve = peak_value * fit_base
        envelope_curve = peak_value * env_base
        residual = fit_curve - y

        return {
            "x": x,
            "y": y,
            "fit_curve": fit_curve,
            "envelope_curve": envelope_curve,
            "residual": residual,
            "L_value": L_value,
            "peak_value": peak_value,
            "fit_curve_raw": fit_curve + float(context.get("offset", 0.0)),
            "fit_aux": fit_aux,
            "d_factor": fit_aux.get("d_factor"),
        }

    def _compute_extrema_info(self) -> Dict[str, Any]:
        context = self._analysis_context
        if context.get("error"):
            return {"error": context["error"]}

        strategy = context["strategy"]
        x = np.asarray(context["display_x"], dtype=float)
        y = np.asarray(context["display_y"], dtype=float)
        prepared = context["prepared_data"]
        info: Dict[str, Any] = {"minima_idx": np.array([], dtype=int), "maxima_idx": np.array([], dtype=int)}

        try:
            info["minima_idx"] = np.asarray(strategy.detect_minima(x, y), dtype=int)
        except Exception as e:
            info["minima_error"] = str(e)

        if hasattr(strategy, "_fit_Pm0"):
            try:
                _result, aux = strategy._fit_Pm0(prepared)
                info["maxima_idx"] = np.asarray(aux.get("maxima_idx", []), dtype=int)
            except Exception as e:
                info["maxima_error"] = str(e)

        return info

    def _make_fit_theory_dataframe(self, L_value: float, peak_value: float) -> pd.DataFrame:
        context = self._analysis_context
        strategy = context["strategy"]
        x = np.asarray(context["display_x"], dtype=float)
        point_count = max(4001, len(x) * 5 if len(x) else 4001)
        grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), point_count)
        theory = peak_value * np.asarray(
            strategy._maker_fringes(override={"L": L_value, "theta_deg": grid}),
            dtype=float,
        )
        return pd.DataFrame(
            {
                "position": grid,
                "position_centered": grid,
                "intensity_corrected": theory,
            }
        )

    def _compute_lc_diagnostics(self, L_value: float, peak_value: float) -> Dict[str, Any]:
        context = self._analysis_context
        if context.get("error"):
            return {"error": context["error"]}

        strategy = context["strategy"]
        if not hasattr(strategy, "_calc_Lc_large_angle"):
            return {"error": "The selected strategy does not provide Lc diagnostics."}

        source = self.cmb_lc_source.currentData() or "data"
        try:
            lc_data = self._make_fit_theory_dataframe(L_value, peak_value) if source == "fit" else context["prepared_data"].copy()
            result, aux = strategy._calc_Lc_large_angle(
                context["analysis"].meta,
                lc_data,
                [0, 180],
                L_value,
            )
            return {"result": result, "aux": aux, "source": source, "data": lc_data}
        except Exception as e:
            return {"error": str(e), "source": source}
    
    def _render_analysis_plots(self):
        self._update_plot_tab_visibility()
        self._update_all_canvas_heights()
        if not self._analysis_context or self._analysis_context.get("error"):
            self._clear_plots()
            if self._analysis_context.get("error"):
                message = str(self._analysis_context["error"])
                for canvas in [
                    self.canvas_fit,
                    self.canvas_resid,
                    self.canvas_centering,
                    self.canvas_extrema,
                    self.canvas_lc,
                ]:
                    self._show_plot_message(canvas, message)
            self.btn_apply_manual.setEnabled(False)
            return

        live = self._compute_live_curves()
        extrema = self._compute_extrema_info()
        lc_info = self._compute_lc_diagnostics(
            L_value=float(live.get("L_value", self._manual_value("L"))),
            peak_value=float(live.get("peak_value", self._manual_value("peak"))),
        )

        self._render_fit_plot(live)
        self._render_residual_plot(live)
        self._render_centering_plot()
        self._render_extrema_plot(live, extrema)
        self._render_lc_plot(lc_info)
        self.btn_apply_manual.setEnabled("error" not in live)

        notes = self._analysis_context.get("notes") or []
        self.lbl_manual_hint.setText(
            "The live overlay uses the current L and Peak values. Overwrite updates saved fit values."
            if not notes else
            "The live overlay uses the current L and Peak values. " + " | ".join(str(note) for note in notes)
        )

    def _render_fit_plot(self, live: Dict[str, Any]):
        if "error" in live:
            self._show_plot_message(self.canvas_fit, str(live["error"]))
            return
        self.canvas_fit.clear()
        ax = self.canvas_fit.ax
        sample_label = str(self._meta.get("sample") or self._meta.get("sample_id") or "Data")
        if self.chk_fit_show_data.isChecked():
            ax.plot(live["x"], live["y"], linestyle="none", marker="o", markersize=3, label=sample_label)
        if self.chk_fit_show_fitting.isChecked():
            ax.plot(live["x"], live["fit_curve"], linewidth=1.6, label="Fitting")
        if self.chk_fit_show_envelope.isChecked() and not self._is_wedge_scan():
            ax.plot(live["x"], live["envelope_curve"], linewidth=1.2, linestyle="--", label="Envelope")
        nominal_L = self._nominal_thickness_mm()
        delta_um = (float(live["L_value"]) - nominal_L) * 1000.0
        ax.text(
            0.02,
            0.98,
            f"L = {live['L_value']:.4f} mm (ΔL= {delta_um:+.1f} um)\nPeak = {live['peak_value']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        self._configure_plot_axes(
            self.canvas_fit,
            "fit",
            "Signal (V)",
            top_axis_L_mm=float(live["L_value"]),
        )
        self.canvas_fit.figure.tight_layout()
        self.canvas_fit.draw()

    def _render_residual_plot(self, live: Dict[str, Any]):
        if "error" in live:
            self._show_plot_message(self.canvas_resid, str(live["error"]))
            return
        self.canvas_resid.clear()
        ax = self.canvas_resid.ax
        ax.plot(live["x"], live["residual"], linestyle="none", marker=".", markersize=3)
        ax.axhline(0.0, linewidth=1.0)
        self._configure_plot_axes(
            self.canvas_resid,
            "resid",
            "Residual (V)",
            top_axis_L_mm=float(live["L_value"]),
        )
        self.canvas_resid.figure.tight_layout()
        self.canvas_resid.draw()

    def _render_centering_plot(self):
        if self._is_wedge_scan():
            self._show_plot_message(self.canvas_centering, "No data for wedge scans.")
            return
        self.canvas_centering.clear()
        ax = self.canvas_centering.ax
        centering = self._analysis_context.get("centering_info")
        if not isinstance(centering, dict) or centering.get("c_candidates") is None or centering.get("costs") is None:
            self._show_plot_message(self.canvas_centering, "Centering cost is not available for the current strategy/data.")
            return
        ax.plot(centering["c_candidates"], centering["costs"], label="Coarse cost")
        if centering.get("c_local") is not None and centering.get("costs_local") is not None:
            ax.plot(centering["c_local"], centering["costs_local"], label="Refined cost")
        if centering.get("c_best") is not None:
            ax.axvline(float(centering["c_best"]), color="C3", linewidth=1.2, label="Best center")
        self._configure_plot_axes(self.canvas_centering, "centering", "Cost")
        ax.set_xlabel("Center candidate")
        self.canvas_centering.figure.tight_layout()
        self.canvas_centering.draw()

    def _render_extrema_plot(self, live: Dict[str, Any], extrema: Dict[str, Any]):
        if "error" in live:
            self._show_plot_message(self.canvas_extrema, str(live["error"]))
            return
        self.canvas_extrema.clear()
        ax = self.canvas_extrema.ax
        ax.plot(live["x"], live["y"], linewidth=1.0, label="Data")
        ax.plot(live["x"], live["fit_curve"], linewidth=1.2, alpha=0.7, label="Current fit")

        minima_idx = np.asarray(extrema.get("minima_idx", []), dtype=int)
        maxima_idx = np.asarray(extrema.get("maxima_idx", []), dtype=int)
        minima_x = live["x"][minima_idx] if minima_idx.size else np.array([], dtype=float)
        minima_y = live["y"][minima_idx] if minima_idx.size else np.array([], dtype=float)
        maxima_x = live["x"][maxima_idx] if maxima_idx.size else np.array([], dtype=float)
        maxima_y = live["y"][maxima_idx] if maxima_idx.size else np.array([], dtype=float)
        ax.plot(minima_x, minima_y, linestyle="none", marker="*", ms=9, label="* Minima")
        ax.plot(maxima_x, maxima_y, linestyle="none", marker="o", ms=5, label="o Maxima")

        self._configure_plot_axes(
            self.canvas_extrema,
            "extrema",
            "Signal (V)",
            top_axis_L_mm=float(live["L_value"]),
        )
        self.canvas_extrema.figure.tight_layout()
        self.canvas_extrema.draw()

    def _render_lc_plot(self, lc_info: Dict[str, Any]):
        if self._is_wedge_scan():
            self._show_plot_message(self.canvas_lc, "No data for wedge scans.")
            return
        self.canvas_lc.clear()
        ax = self.canvas_lc.ax
        if "error" in lc_info:
            self._show_plot_message(self.canvas_lc, f"Lc plot unavailable: {lc_info['error']}")
            return

        result = lc_info["result"]
        aux = lc_info["aux"]
        data = lc_info["data"]
        source = "fit curve" if lc_info.get("source") == "fit" else "experimental data"

        position = np.asarray(data.get("position_centered", data["position"]), dtype=float)
        minima_idx = np.asarray(aux.get("minima_idx", []), dtype=int)
        minima_x = position[minima_idx] if minima_idx.size else np.array([], dtype=float)
        minima_pos = np.sort(minima_x[minima_x > 0.0])
        minima_neg = np.sort(minima_x[minima_x < 0.0])
        dL_pos = np.asarray(aux.get("dL_pos", []), dtype=float)
        dL_neg = np.asarray(aux.get("dL_neg", []), dtype=float)

        for i in range(min(len(dL_pos), max(len(minima_pos) - 1, 0))):
            ax.plot([minima_pos[i], minima_pos[i + 1]], [1000.0 * dL_pos[i], 1000.0 * dL_pos[i]], color="C0")
        for i in range(min(len(dL_neg), max(len(minima_neg) - 1, 0))):
            ax.plot([minima_neg[i], minima_neg[i + 1]], [1000.0 * dL_neg[i], 1000.0 * dL_neg[i]], color="C1")

        mean_lc = self._safe_float(result.get("Lc_mean_mm"))
        std_lc = self._safe_float(result.get("Lc_std_mm"))
        if np.isfinite(mean_lc):
            ax.axhline(mean_lc * 1000.0, color="0.3", linestyle="--", linewidth=1.0)
        ax.set_title(f"Lc from adjacent minima pairs ({source})")
        if np.isfinite(mean_lc) and np.isfinite(std_lc):
            text = f"mean = {mean_lc * 1000.0:.3f} um\nstd = {std_lc * 1000.0:.3f} um"
        elif np.isfinite(mean_lc):
            text = f"mean = {mean_lc * 1000.0:.3f} um"
        else:
            text = "Lc summary unavailable"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        self._configure_plot_axes(self.canvas_lc, "lc", "Lc (um)")
        self.canvas_lc.figure.tight_layout()
        self.canvas_lc.draw()

    def _show_plot_message(self, canvas: MplCanvas, message: str):
        canvas.clear()
        canvas.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=canvas.ax.transAxes,
            wrap=True,
        )
        canvas.figure.tight_layout()
        canvas.draw()

    def _apply_manual_fit_clicked(self):
        if not self.json_path or not self.csv_path:
            QMessageBox.information(self, "No data", "Load a result folder first.")
            return
        selected = self._get_selected_strategy()
        if selected is None:
            QMessageBox.critical(self, "No strategy", "No fitting strategy is available/selected.")
            return

        ok, message = self._write_json_metadata(show_message=False)
        if not ok:
            QMessageBox.critical(self, "Update failed", message)
            return

        live = self._compute_live_curves()
        if "error" in live:
            QMessageBox.critical(self, "Manual fit failed", str(live["error"]))
            return

        lc_info = self._compute_lc_diagnostics(
            L_value=float(live["L_value"]),
            peak_value=float(live["peak_value"]),
        )

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Read failed", str(e))
            return

        fit_result = {
            "L_mm": float(live["L_value"]),
            "L_mm_std": 0.0,
            "k_scale": float(live["peak_value"]),
            "k_scale_std": 0.0,
            "Pm0": float(live["peak_value"]),
            "Pm0_stderr": 0.0,
            "residual_rms": float(np.sqrt(np.mean(np.square(live["residual"])))),
        }
        existing_fit = self._fit_payload_for_strategy(meta, selected)
        if existing_fit:
            fit_result = {**existing_fit, **fit_result}
        d_factor = self._safe_float(live.get("d_factor"))
        if np.isfinite(d_factor):
            fit_result["d_factor"] = float(d_factor)

        if "error" not in lc_info:
            result = lc_info["result"]
            for key in ("Lc_mean_mm", "Lc_std_mm", "minima_count", "n_count"):
                if key in result:
                    fit_result[key] = result[key]

        meta = upsert_fitting_result(
            meta,
            selected.class_name,
            fit_result,
            strategy_module=selected.qualname,
            strategy_display_name=selected.display_name,
        )

        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", str(e))
            return

        try:
            csv_df = pd.read_csv(self.csv_path)
            prepared = self._analysis_context.get("prepared_data")
            if isinstance(prepared, pd.DataFrame) and len(prepared) == len(csv_df):
                for column in ("position_centered", "offset_corrected"):
                    if column in prepared.columns:
                        csv_df[column] = prepared[column].to_numpy()
            csv_df["fit"] = np.asarray(live["fit_curve_raw"], dtype=float)
            csv_df["fit_envelope"] = np.asarray(live["envelope_curve"], dtype=float)
            csv_df.to_csv(self.csv_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "CSV update failed", str(e))
            return

        self._meta = meta
        ok, msg = self._load_folder(self._current_dir)
        if not ok:
            QMessageBox.warning(self, "Reload failed", msg)
            return
        self._populate_table_from_json(meta)
        QMessageBox.information(self, "Saved", "Current L and Peak values were written to JSON/CSV.")

    def _clear_plots(self):
        for canvas in [
            self.canvas_fit,
            self.canvas_resid,
            self.canvas_centering,
            self.canvas_extrema,
            self.canvas_lc,
        ]:
            canvas.clear()
            canvas.draw()

    # --------------------------------- Save ---------------------------------
    def _save_figures_clicked(self):
        if not self._current_dir:
            QMessageBox.information(self, "No folder", "Load a result folder first.")
            return
        try:
            output_dir = self._current_fitting_output_dir(create=True) or self._current_dir
            figures = {
                "fit_overlay.png": self.canvas_fit.figure,
                "residuals.png": self.canvas_resid.figure,
                "centering_cost.png": self.canvas_centering.figure,
                "extrema.png": self.canvas_extrema.figure,
                "lc_pairs.png": self.canvas_lc.figure,
            }
            for filename, figure in figures.items():
                figure.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Saved", f"Plots for all analysis tabs were saved to {output_dir}")


# --------------------------- Standalone test hook ---------------------------
# if __name__ == "__main__":  # optional manual test
#     from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
#     app = QApplication(sys.argv)
#     win = QMainWindow(); win.setWindowTitle("Demo — Fit/Analysis Tab")
#     tabs = QTabWidget(); tabs.addTab(FitAnalyzeTab(), "Analysis")
#     win.setCentralWidget(tabs)
#     win.resize(1120, 720); win.show()
#     sys.exit(app.exec())
