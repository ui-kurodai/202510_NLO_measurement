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
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QLocale, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QGroupBox, QMessageBox,
    QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QSlider, QDialog, QDialogButtonBox, QCheckBox,
    QListWidget, QListWidgetItem, QMenu, QProgressBar, QStackedWidget,
)

import logging
from measurement_metadata import (
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
from fitting_results import (
    extract_fit_payload,
    merge_fit_payload,
    normalize_fitting_entries,
    remove_fitting_results,
    upsert_fitting_result,
)
from widgets.refractive_index_global_fit_widget import RefractiveIndexGlobalFitWidget
from widgets.standard_fit_widget import MplCanvas, SavedStrategyListWidget, StandardFitWidget
from windows_dialogs import select_multiple_directories
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
    data_plot_style: str = "markers"
    show_fit_annotation: bool = True
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


class FitWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, folder: str, strategy_module: str, strategy_class: str):
        super().__init__()
        self.folder = folder
        self.strategy_module = strategy_module
        self.strategy_class = strategy_class

    def run(self) -> None:
        try:
            if SHGDataAnalysis is None:
                raise RuntimeError("shg_analysis is not importable.")
            module = importlib.import_module(self.strategy_module)
            strategy_cls = getattr(module, self.strategy_class)
            analysis = SHGDataAnalysis(self.folder)
            results = analysis.run(strategy_cls)
            if results is None:
                raise RuntimeError(str(getattr(analysis, "last_error", "Unknown fitting error.")))
            self.succeeded.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


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
        self._transient_strategy_names: List[str] = []
        self._analysis_context: Dict[str, Any] = {}
        self._live_curve_cache: Dict[str, Any] = {}
        self._nfit_manual_overrides: Dict[str, Dict[str, float]] = {}
        self._nfit_card_views: Dict[str, Dict[str, Any]] = {}
        self._nfit_manual_result_id: Optional[str] = None
        self._fit_worker: Optional[FitWorker] = None
        self._fit_requested_strategy_name: Optional[str] = None
        self._busy = False
        self._use_saved_fit_preview = False
        self._manual_controls: Dict[str, Dict[str, QDoubleSpinBox | QSlider]] = {}
        self._manual_syncing = False
        self._extrema_force_reset = False
        self._sample_catalog_map: Dict[str, Dict[str, Any]] = {}
        self._beam_profile_catalog_map: Dict[str, Dict[str, Any]] = {}
        self._filter_catalog_map: Dict[str, Dict[str, Any]] = {}
        self._plot_settings: Dict[str, PlotSettings] = {
            key: self._default_plot_settings(key)
            for key in ("fit", "resid", "centering", "extrema", "lc")
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

        nav_bar = QHBoxLayout()
        self.btn_nav_home = self._make_mode_button("Menu")
        self.btn_nav_standard = self._make_mode_button("Standard Fit")
        self.btn_nav_nfit = self._make_mode_button("Refractive Index Global Fit")
        self.btn_update_json = QPushButton("Update JSON")
        self.btn_fit = QPushButton("Run Fit")
        self.btn_save = QPushButton("Save All Plots")
        self.btn_fit.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_update_json.setEnabled(False)
        nav_bar.addWidget(self.btn_nav_home)
        nav_bar.addWidget(self.btn_nav_standard)
        nav_bar.addWidget(self.btn_nav_nfit)
        nav_bar.addStretch(1)
        self.lbl_busy = QLabel()
        self.lbl_busy.setStyleSheet("color: #555; font-weight: 600;")
        self.lbl_busy.setVisible(False)
        self.busy_progress = QProgressBar()
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setTextVisible(False)
        self.busy_progress.setFixedWidth(90)
        self.busy_progress.setVisible(False)
        nav_bar.addWidget(self.lbl_busy)
        nav_bar.addWidget(self.busy_progress)
        nav_bar.addWidget(self.btn_update_json)
        nav_bar.addWidget(self.btn_fit)
        nav_bar.addWidget(self.btn_save)
        main.addLayout(nav_bar)

        self.strategy_group = self._build_strategy_group()
        main.addWidget(self.strategy_group)

        self.page_stack = QStackedWidget()
        self.home_page = self._build_home_page()
        self.standard_fit_widget = StandardFitWidget(self._SLIDER_STEPS, self)
        self.nfit_widget = RefractiveIndexGlobalFitWidget(self)
        self.page_stack.addWidget(self.home_page)
        self.page_stack.addWidget(self.standard_fit_widget)
        self.page_stack.addWidget(self.nfit_widget)
        main.addWidget(self.page_stack, 1)

        self.results_group = QGroupBox("Saved Fit Summary")
        results_layout = QVBoxLayout(self.results_group)
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Fit Parameter", "Saved Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setMinimumHeight(180)
        results_layout.addWidget(self.tbl)
        main.addWidget(self.results_group)

        for attr in [
            "btn_open",
            "lbl_current_folder",
            "sample_preset_combo",
            "reload_samples_btn",
            "beam_profile_combo",
            "reload_beams_btn",
            "filter_preset_combo",
            "reload_filters_btn",
            "le_material",
            "le_crystal_orientation",
            "le_axis",
            "sb_t_thin",
            "sb_wedge",
            "sb_beam_rx",
            "sb_beam_ry",
            "sb_input_pol",
            "sb_detected_pol",
            "cmb_boxcar_sensitivity",
            "btn_metadata_edit",
            "no_filter_checkbox",
            "filter_list",
            "lbl_sample",
            "lbl_method",
            "lbl_time",
            "right_scroll",
            "plot_tabs",
            "chk_fit_show_data",
            "chk_fit_show_fitting",
            "chk_fit_show_envelope",
            "canvas_fit",
            "canvas_resid",
            "canvas_centering",
            "extrema_widget",
            "cmb_lc_source",
            "lbl_lc_hint",
            "canvas_lc",
            "btn_plot_settings",
            "btn_save_current_plot",
            "btn_copy_current_plot",
            "sb_manual_centering",
            "btn_reset_manual",
            "btn_apply_manual",
            "lbl_manual_hint",
        ]:
            setattr(self, attr, getattr(self.standard_fit_widget, attr))
        self._plot_pages = self.standard_fit_widget._plot_pages
        self._plot_canvases = self.standard_fit_widget._plot_canvases
        self._manual_controls = self.standard_fit_widget._manual_controls

        for attr in [
            "lbl_nfit_intro",
            "lst_nfit_group_paths",
            "btn_nfit_select_folders",
            "btn_nfit_refresh",
            "lbl_nfit_hint",
            "nfit_measurements_host",
            "nfit_measurements_layout",
        ]:
            setattr(self, attr, getattr(self.nfit_widget, attr))
        self.lbl_nfit_current_folder = self.nfit_widget.lbl_current_folder

        self._page_indexes = {
            "home": self.page_stack.indexOf(self.home_page),
            "standard": self.page_stack.indexOf(self.standard_fit_widget),
            "nfit": self.page_stack.indexOf(self.nfit_widget),
        }
        self._page_buttons = {
            "home": self.btn_nav_home,
            "standard": self.btn_nav_standard,
            "nfit": self.btn_nav_nfit,
        }
        self._current_page_key = "home"
        self._update_folder_status_labels()
        self._set_analysis_page("home")
        return

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
        self.lst_saved_strategies = SavedStrategyListWidget()
        self.lst_saved_strategies.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_saved_strategies.setMinimumHeight(140)
        self.lst_saved_strategies.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cmb_strategy = QComboBox()
        self.btn_add_strategy = QPushButton("+")
        self.btn_add_strategy.setFixedWidth(32)
        add_strategy_row = QHBoxLayout()
        add_strategy_row.setContentsMargins(0, 0, 0, 0)
        add_strategy_row.setSpacing(6)
        add_strategy_row.addWidget(self.cmb_strategy, 1)
        add_strategy_row.addWidget(self.btn_add_strategy, 0)
        self.lbl_strategy_hint = QLabel("Saved strategies are listed here. Use + to open another strategy.")
        self.lbl_strategy_hint.setStyleSheet("color: gray;")
        strat_form.addRow("Saved fits:", self.lst_saved_strategies)
        strat_form.addRow("Add strategy:", add_strategy_row)
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
        self.analysis_pages = QTabWidget()

        standard_page = QWidget(); standard_layout = QVBoxLayout(standard_page)

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
        self.extrema_widget = ManualExtremaDetectionWidget(extrema_tab)
        extrema_layout.addWidget(self.extrema_widget)
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
        standard_layout.addLayout(plot_toolbar)
        standard_layout.addWidget(self.plot_tabs, 3)

        self.nfit_page = self._build_nfit_page()
        self.analysis_pages.addTab(standard_page, "Standard Fit")
        self.analysis_pages.addTab(self.nfit_page, "Refractive Index Fit")
        right_layout.addWidget(self.analysis_pages, 3)

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

    def _build_nfit_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        config_group = QGroupBox("Global n-Fit Setup")
        config_layout = QVBoxLayout(config_group)
        self.lbl_nfit_intro = QLabel(
            "Use one measurement folder per line. Update JSON or Run Fit to save the list into "
            "`n_fit_group_paths`. If the list is empty, only the current folder is used."
        )
        self.lbl_nfit_intro.setWordWrap(True)
        self.lbl_nfit_intro.setStyleSheet("color: gray;")
        self.lst_nfit_group_paths = QListWidget()
        self.lst_nfit_group_paths.setMinimumHeight(120)

        nfit_btn_row = QHBoxLayout()
        self.btn_nfit_select_folders = QPushButton("Select Folders…")
        self.btn_nfit_current_only = QPushButton("Use Current Folder Only")
        self.btn_nfit_refresh = QPushButton("Refresh Preview")
        nfit_btn_row.addWidget(self.btn_nfit_select_folders)
        nfit_btn_row.addWidget(self.btn_nfit_current_only)
        nfit_btn_row.addWidget(self.btn_nfit_refresh)
        nfit_btn_row.addStretch(1)

        self.lbl_nfit_hint = QLabel("Select a global n-fit strategy to preview grouped measurements.")
        self.lbl_nfit_hint.setWordWrap(True)
        self.lbl_nfit_hint.setStyleSheet("color: gray;")

        config_layout.addWidget(self.lbl_nfit_intro)
        config_layout.addWidget(self.lst_nfit_group_paths)
        config_layout.addLayout(nfit_btn_row)
        config_layout.addWidget(self.lbl_nfit_hint)
        layout.addWidget(config_group)

        self.nfit_measurements_host = QWidget()
        self.nfit_measurements_layout = QVBoxLayout(self.nfit_measurements_host)
        self.nfit_measurements_layout.setContentsMargins(0, 0, 0, 0)
        self.nfit_measurements_layout.setSpacing(8)
        layout.addWidget(self.nfit_measurements_host, 1)
        layout.addStretch(1)
        return page

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

    def _build_strategy_group(self) -> QGroupBox:
        strat_group = QGroupBox("Fitting Strategy")
        strat_form = QFormLayout(strat_group)
        self.lst_saved_strategies = SavedStrategyListWidget()
        self.lst_saved_strategies.setSelectionMode(self.lst_saved_strategies.SelectionMode.ExtendedSelection)
        self.lst_saved_strategies.setMinimumHeight(140)
        self.lst_saved_strategies.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cmb_strategy = QComboBox()
        self.btn_add_strategy = QPushButton("+")
        self.btn_add_strategy.setFixedWidth(32)
        add_strategy_row = QHBoxLayout()
        add_strategy_row.setContentsMargins(0, 0, 0, 0)
        add_strategy_row.setSpacing(6)
        add_strategy_row.addWidget(self.cmb_strategy, 1)
        add_strategy_row.addWidget(self.btn_add_strategy, 0)
        self.lbl_strategy_hint = QLabel("Saved strategies are listed here. Use + to open another strategy.")
        self.lbl_strategy_hint.setStyleSheet("color: gray;")
        strat_form.addRow("Saved fits:", self.lst_saved_strategies)
        strat_form.addRow("Add strategy:", add_strategy_row)
        strat_form.addRow("", self.lbl_strategy_hint)
        return strat_group

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        title = QLabel("Choose a fitting workflow")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        subtitle = QLabel(
            "Use Standard Fit for one measurement folder, or open Refractive Index Global Fit to combine multiple measurements."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(16)
        self.btn_home_standard = QPushButton("Standard Fit")
        self.btn_home_standard.setMinimumHeight(120)
        self.btn_home_standard.setStyleSheet("font-size: 18px; font-weight: 600; text-align: left; padding: 18px;")
        self.btn_home_nfit = QPushButton("Refractive Index Global Fit")
        self.btn_home_nfit.setMinimumHeight(120)
        self.btn_home_nfit.setStyleSheet("font-size: 18px; font-weight: 600; text-align: left; padding: 18px;")
        button_row.addWidget(self.btn_home_standard, 1)
        button_row.addWidget(self.btn_home_nfit, 1)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return page

    def _make_mode_button(self, label: str) -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        return button

    def _set_analysis_page(self, page_key: str) -> None:
        if page_key not in self._page_indexes:
            return
        page_changed = page_key != self._current_page_key
        self._current_page_key = page_key
        self.page_stack.setCurrentIndex(self._page_indexes[page_key])
        for key, button in self._page_buttons.items():
            button.blockSignals(True)
            button.setChecked(key == page_key)
            button.blockSignals(False)
        self._sync_page_chrome()
        if page_changed:
            self._populate_strategy_list()
        if page_key != "home" and self._analysis_context:
            self._render_analysis_plots()

    def _sync_page_chrome(self) -> None:
        self.results_group.setVisible(self._current_page_key != "home")
        self.strategy_group.setVisible(self._current_page_key != "home")

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = bool(busy)
        self.lbl_busy.setText(message)
        self.lbl_busy.setVisible(busy)
        self.busy_progress.setVisible(busy)
        for widget in (
            self.btn_nav_home,
            self.btn_nav_standard,
            self.btn_nav_nfit,
            self.btn_update_json,
            self.btn_fit,
            self.btn_save,
            self.strategy_group,
            self.page_stack,
        ):
            widget.setEnabled(not busy)
        if not busy:
            has_data = self._current_dir is not None
            self.btn_fit.setEnabled(has_data)
            self.btn_save.setEnabled(has_data)
            self.btn_update_json.setEnabled(has_data)
        QApplication.processEvents()

    def _nfit_refresh_clicked(self) -> None:
        if self._busy:
            return
        self._set_busy(True, "Loading folders and updating preview...")
        try:
            self._refresh_analysis_views(reset_manual=False)
        finally:
            self._set_busy(False)

    def _update_folder_status_labels(self) -> None:
        folder_text = str(self._current_dir) if self._current_dir else "No folder loaded."
        self.lbl_current_folder.setText(folder_text)
        self.lbl_nfit_current_folder.setText(folder_text)

    def _connect(self):
        self.btn_open.clicked.connect(self._select_folder)
        self.btn_update_json.clicked.connect(self._update_json_clicked)
        self.btn_fit.clicked.connect(self._run_fit_clicked)
        self.btn_save.clicked.connect(self._save_figures_clicked)
        self.btn_nav_home.clicked.connect(lambda: self._set_analysis_page("home"))
        self.btn_nav_standard.clicked.connect(lambda: self._set_analysis_page("standard"))
        self.btn_nav_nfit.clicked.connect(lambda: self._set_analysis_page("nfit"))
        self.btn_home_standard.clicked.connect(lambda: self._set_analysis_page("standard"))
        self.btn_home_nfit.clicked.connect(lambda: self._set_analysis_page("nfit"))
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
        self.extrema_widget.extremaChanged.connect(self._render_analysis_plots)
        self.lst_saved_strategies.itemSelectionChanged.connect(self._strategy_selection_changed)
        self.lst_saved_strategies.customContextMenuRequested.connect(self._show_saved_strategy_context_menu)
        self.cmb_strategy.currentIndexChanged.connect(self._picker_strategy_changed)
        self.btn_add_strategy.clicked.connect(self._add_strategy_from_picker)
        self.cmb_lc_source.currentIndexChanged.connect(lambda *_args: self._render_analysis_plots())
        self.plot_tabs.currentChanged.connect(lambda *_args: self._render_analysis_plots())
        self.chk_fit_show_data.stateChanged.connect(lambda *_args: self._render_analysis_plots())
        self.chk_fit_show_fitting.stateChanged.connect(lambda *_args: self._render_analysis_plots())
        self.chk_fit_show_envelope.stateChanged.connect(lambda *_args: self._render_analysis_plots())
        self.sb_manual_centering.valueChanged.connect(lambda *_args: self._manual_centering_changed())
        self.btn_nfit_select_folders.clicked.connect(self._nfit_select_folders_clicked)
        self.btn_nfit_refresh.clicked.connect(self._nfit_refresh_clicked)
        self.lst_nfit_group_paths.itemChanged.connect(self._nfit_folder_check_changed)

        for key in self._manual_controls:
            controls = self._manual_controls[key]
            controls["slider"].valueChanged.connect(
                lambda _value, name=key: self._manual_slider_changed(name)
            )
            controls["slider"].sliderReleased.connect(
                lambda name=key: self._manual_slider_released(name)
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
        current_picker_name = self._picker_strategy_name()
        current_selected_name = self._selected_strategy_name(allow_picker_fallback=False)

        self.cmb_strategy.blockSignals(True)
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
                        if (
                            isinstance(node, ast.ClassDef)
                            and node.name.endswith("Strategy")
                            and self._strategy_allowed_on_current_page(node.name)
                        ):
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

        if self.cmb_strategy.count() == 0:
            self.cmb_strategy.addItem("(no strategies found)")
        else:
            preferred_picker_name = current_picker_name or current_selected_name
            if preferred_picker_name:
                self._set_picker_strategy(preferred_picker_name)
            else:
                self.cmb_strategy.setCurrentIndex(0)
        self.cmb_strategy.blockSignals(False)

        self._refresh_saved_strategy_list(self._meta, preferred_strategy_name=current_selected_name)
        self._refresh_analysis_views(reset_manual=True)

    def _strategy_allowed_on_current_page(self, class_name: str) -> bool:
        is_global_nfit = class_name.endswith("GlobalNFitStrategy")
        if self._current_page_key == "nfit":
            return is_global_nfit
        return not is_global_nfit

    def _find_strategy_info(self, strategy_name: str | None) -> Optional[StrategyInfo]:
        normalized = str(strategy_name or "").strip()
        if not normalized:
            return None
        for strategy in self._strategies:
            if strategy.class_name == normalized:
                return strategy
        return None

    def _picker_strategy_name(self) -> Optional[str]:
        idx = self.cmb_strategy.currentIndex()
        if idx < 0:
            return None
        data = self.cmb_strategy.currentData()
        if isinstance(data, int) and 0 <= data < len(self._strategies):
            return self._strategies[data].class_name
        return None

    def _set_picker_strategy(self, strategy_name: str | None) -> None:
        normalized = str(strategy_name or "").strip()
        if not normalized:
            return
        for index, strategy in enumerate(self._strategies):
            if strategy.class_name == normalized:
                self.cmb_strategy.setCurrentIndex(index)
                return

    def _current_strategy_item(self) -> Optional[QListWidgetItem]:
        item = self.lst_saved_strategies.currentItem()
        if item is not None and item.isSelected():
            return item
        selected_items = self.lst_saved_strategies.selectedItems()
        if selected_items:
            return selected_items[0]
        if self.lst_saved_strategies.count() == 1:
            return self.lst_saved_strategies.item(0)
        return None

    def _selected_strategy_name(self, *, allow_picker_fallback: bool = True) -> Optional[str]:
        item = self._current_strategy_item()
        if item is not None:
            strategy_name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if strategy_name:
                return strategy_name
        if allow_picker_fallback and self.lst_saved_strategies.count() == 0:
            return self._picker_strategy_name()
        return None

    def _selected_result_id(self) -> Optional[str]:
        item = self._current_strategy_item()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole + 2) or "").strip() or None

    def _is_strategy_saved(self, strategy_name: str) -> bool:
        normalized = str(strategy_name or "").strip()
        if not normalized:
            return False
        for entry in normalize_fitting_entries(self._meta):
            if str(entry.get("strategy") or "").strip() == normalized:
                return True
        return False

    def _csv_fit_matches_selected_strategy(self) -> bool:
        selected = str(self._selected_strategy_name() or "").strip()
        selected_result_id = str(self._selected_result_id() or "").strip()
        active = str((self._meta or {}).get("fitting_active_strategy") or "").strip()
        active_result_id = str((self._meta or {}).get("fitting_active_result_id") or "").strip()
        return bool(
            selected
            and active
            and selected == active
            and (not selected_result_id or selected_result_id == active_result_id)
            and isinstance(self._df, pd.DataFrame)
            and "fit" in self._df.columns
            and self._is_strategy_saved(selected)
        )

    def _display_name_for_strategy(
        self,
        strategy_name: str | None,
        *,
        saved_entry: Optional[Dict[str, Any]] = None,
        saved: bool = True,
    ) -> str:
        normalized = str(strategy_name or "").strip()
        info = self._find_strategy_info(normalized)
        if info is not None:
            label = info.display_name
        elif saved_entry and str(saved_entry.get("strategy_display_name") or "").strip():
            label = str(saved_entry.get("strategy_display_name") or "").strip()
        else:
            label = normalized or "(unnamed)"
        if not saved:
            return f"{label} (new)"
        if normalized and info is None:
            return f"{label} (unavailable)"
        return label

    def _refresh_saved_strategy_list(
        self,
        meta: Optional[Dict[str, Any]],
        *,
        preferred_strategy_name: Optional[str] = None,
    ) -> None:
        saved_entries = [
            entry
            for entry in normalize_fitting_entries(meta)
            if self._strategy_allowed_on_current_page(str(entry.get("strategy") or ""))
        ]
        saved_names: List[str] = []
        for entry in saved_entries:
            strategy_name = str(entry.get("strategy") or "").strip()
            if not strategy_name:
                continue
            saved_names.append(strategy_name)

        self._transient_strategy_names = [
            name for name in self._transient_strategy_names
            if name not in saved_map
        ]
        visible_transient_names = [
            name for name in self._transient_strategy_names
            if name not in saved_names and self._strategy_allowed_on_current_page(name)
        ]

        target_name = str(preferred_strategy_name or "").strip()
        target_result_id = str((meta or {}).get("fitting_active_result_id") or "").strip()
        if not target_name:
            target_name = str((meta or {}).get("fitting_active_strategy") or "").strip()
        if not target_name and saved_names:
            target_name = saved_names[-1]
        if not target_name and visible_transient_names:
            target_name = visible_transient_names[-1]

        self.lst_saved_strategies.blockSignals(True)
        self.lst_saved_strategies.clear()

        for entry in saved_entries:
            strategy_name = str(entry.get("strategy") or "").strip()
            result_id = str(entry.get("result_id") or "").strip()
            result_label = str(entry.get("result_label") or "").strip()
            display_name = self._display_name_for_strategy(strategy_name, saved_entry=entry, saved=True)
            if result_label:
                display_name = f"{display_name} | {result_label}"
            item = QListWidgetItem(
                display_name
            )
            item.setData(Qt.ItemDataRole.UserRole, strategy_name)
            item.setData(Qt.ItemDataRole.UserRole + 1, True)
            item.setData(Qt.ItemDataRole.UserRole + 2, result_id)
            tooltip_lines = [f"strategy: {strategy_name}"]
            strategy_module = str(entry.get("strategy_module") or "").strip()
            if strategy_module:
                tooltip_lines.append(f"module: {strategy_module}")
            item.setToolTip("\n".join(tooltip_lines))
            self.lst_saved_strategies.addItem(item)
            if strategy_name == target_name and (not target_result_id or result_id == target_result_id):
                item.setSelected(True)
                self.lst_saved_strategies.setCurrentItem(item)

        for strategy_name in visible_transient_names:
            item = QListWidgetItem(self._display_name_for_strategy(strategy_name, saved=False))
            item.setData(Qt.ItemDataRole.UserRole, strategy_name)
            item.setData(Qt.ItemDataRole.UserRole + 1, False)
            item.setData(Qt.ItemDataRole.UserRole + 2, "")
            item.setForeground(QColor("gray"))
            item.setToolTip(f"strategy: {strategy_name}\nNot saved to JSON yet.")
            self.lst_saved_strategies.addItem(item)
            if strategy_name == target_name:
                item.setSelected(True)
                self.lst_saved_strategies.setCurrentItem(item)

        if self.lst_saved_strategies.currentItem() is None and self.lst_saved_strategies.count() > 0:
            first_item = self.lst_saved_strategies.item(0)
            first_item.setSelected(True)
            self.lst_saved_strategies.setCurrentItem(first_item)

        self.lst_saved_strategies.blockSignals(False)

        saved_count = len(saved_entries)
        available_count = len(self._strategies)
        if available_count == 0:
            self.lbl_strategy_hint.setText("No strategy classes found.")
        elif saved_count == 0:
            self.lbl_strategy_hint.setText(
                f"No saved fitting result in JSON. Use + to open one of {available_count} strategies."
            )
        else:
            self.lbl_strategy_hint.setText(
                f"{saved_count} saved fitting result(s). Ctrl+click for multi-select, then right-click to delete."
            )

    def _get_selected_strategy(self) -> Optional[StrategyInfo]:
        return self._find_strategy_info(self._selected_strategy_name())

    def _picker_strategy_changed(self, *_args):
        if self.lst_saved_strategies.count() != 0:
            return
        if self._meta:
            self._populate_table_from_json(self._meta)
        self._refresh_analysis_views(reset_manual=True)

    def _add_strategy_from_picker(self):
        strategy_name = self._picker_strategy_name()
        if not strategy_name:
            QMessageBox.information(self, "No strategy", "No fitting strategy is available.")
            return
        if self._is_strategy_saved(strategy_name):
            self._refresh_saved_strategy_list(self._meta, preferred_strategy_name=strategy_name)
            self._strategy_selection_changed()
            return
        if strategy_name not in self._transient_strategy_names:
            self._transient_strategy_names.append(strategy_name)
        self._refresh_saved_strategy_list(self._meta, preferred_strategy_name=strategy_name)
        self._strategy_selection_changed()

    def _show_saved_strategy_context_menu(self, pos):
        if not self.lst_saved_strategies.selectedItems():
            return
        menu = QMenu(self)
        action = menu.addAction("Delete")
        action.triggered.connect(self._delete_selected_strategy_results)
        menu.exec(self.lst_saved_strategies.mapToGlobal(pos))

    def _delete_selected_strategy_results(self):
        if not self.json_path:
            QMessageBox.information(self, "No data", "Load a result folder first.")
            return

        selected_items = self.lst_saved_strategies.selectedItems()
        if not selected_items:
            return

        saved_names: List[str] = []
        saved_result_ids: List[str] = []
        transient_names: List[str] = []
        display_lines: List[str] = []
        for item in selected_items:
            strategy_name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            is_saved = bool(item.data(Qt.ItemDataRole.UserRole + 1))
            if not strategy_name:
                continue
            display_lines.append(f" - {item.text()}")
            if is_saved:
                saved_names.append(strategy_name)
                result_id = str(item.data(Qt.ItemDataRole.UserRole + 2) or "").strip()
                if result_id:
                    saved_result_ids.append(result_id)
            else:
                transient_names.append(strategy_name)

        if not saved_names and not transient_names:
            return

        if saved_names and transient_names:
            message = (
                "Delete the selected fitting results from JSON and remove unsaved entries from the list?\n\n"
                + "\n".join(display_lines)
            )
        elif saved_names:
            message = "Delete the selected fitting results from JSON?\n\n" + "\n".join(display_lines)
        else:
            message = "Remove the selected unsaved entries from the list?\n\n" + "\n".join(display_lines)

        reply = QMessageBox.question(
            self,
            "Delete fitting results",
            message,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return

        if saved_names:
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Read failed", str(e))
                return

            meta = remove_fitting_results(meta, saved_names, saved_result_ids or None)

            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                QMessageBox.critical(self, "Write failed", str(e))
                return

            self._meta = meta

        removed_names = set(saved_names) | set(transient_names)
        self._transient_strategy_names = [
            name for name in self._transient_strategy_names
            if name not in removed_names
        ]
        self._refresh_saved_strategy_list(self._meta)
        if self._meta:
            self._populate_table_from_json(self._meta)
        self._refresh_analysis_views(reset_manual=True)

    def _strategy_selection_changed(self, *_args):
        self._use_saved_fit_preview = False
        self._nfit_manual_overrides.clear()
        self._nfit_card_views.clear()
        self._nfit_manual_result_id = None
        selected_name = self._selected_strategy_name(allow_picker_fallback=False)
        if selected_name:
            self._set_picker_strategy(selected_name)
        selected_result_id = self._selected_result_id()
        if selected_result_id:
            global_result = self._global_result_by_id(selected_result_id)
            if global_result:
                self._set_nfit_group_paths(
                    [str(path) for path in global_result.get("group_source_dirs", [])],
                    [str(path) for path in global_result.get("excluded_source_dirs", [])],
                )
        if self._meta:
            self._populate_table_from_json(self._meta)
        self._refresh_analysis_views(reset_manual=True)

    def _global_result_by_id(self, result_id: str | None) -> Dict[str, Any]:
        normalized = str(result_id or "").strip()
        if not normalized:
            return {}
        for entry in self._meta.get("n_fit_global_results", []) or []:
            if (
                isinstance(entry, dict)
                and str(entry.get("result_id") or "").strip() == normalized
            ):
                return dict(entry)
        return {}

    def _fit_payload_for_strategy(
        self,
        meta: Optional[Dict[str, Any]] = None,
        strategy: Optional[StrategyInfo] = None,
    ) -> Dict[str, Any]:
        strategy_name = strategy.class_name if strategy is not None else self._selected_strategy_name()
        return extract_fit_payload(
            meta if meta is not None else self._meta,
            strategy_name,
            self._selected_result_id(),
        )

    def _meta_with_selected_fit(
        self,
        meta: Optional[Dict[str, Any]] = None,
        strategy: Optional[StrategyInfo] = None,
    ) -> Dict[str, Any]:
        strategy_name = strategy.class_name if strategy is not None else self._selected_strategy_name()
        return merge_fit_payload(
            meta if meta is not None else self._meta,
            strategy_name,
            self._selected_result_id(),
        )

    def _apply_saved_strategy_selection(self, meta: Dict[str, Any]):
        target_name = str(meta.get("fitting_active_strategy") or "").strip()
        if not target_name:
            payload = extract_fit_payload(meta)
            target_name = str(payload.get("strategy") or "").strip()
        self._set_picker_strategy(target_name)
        self._refresh_saved_strategy_list(meta, preferred_strategy_name=target_name)

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
        selected_ids = {
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self.filter_list.selectedItems()
        }
        no_filter_checked = self.no_filter_checkbox.isChecked()
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

        self.filter_list.blockSignals(True)
        try:
            self.filter_list.clear()
            for entry in catalog["filters"]:
                item = QListWidgetItem(format_filter_display(entry))
                item.setData(Qt.ItemDataRole.UserRole, entry["filter_id"])
                item.setSelected(entry["filter_id"] in selected_ids)
                self.filter_list.addItem(item)
        finally:
            self.filter_list.blockSignals(False)
        self.no_filter_checkbox.setChecked(no_filter_checked and not selected_ids)
        self.standard_fit_widget._toggle_filter_selection(self.no_filter_checkbox.isChecked())

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

    def _selected_filter_entries(self) -> List[Dict[str, Any]]:
        if self.no_filter_checkbox.isChecked():
            return []
        selected_entries: List[Dict[str, Any]] = []
        for item in self.filter_list.selectedItems():
            filter_id = item.data(Qt.ItemDataRole.UserRole)
            entry = self._filter_catalog_map.get(filter_id)
            if entry is not None:
                selected_entries.append(entry)
        return selected_entries

    def _selected_filters_dict(self) -> Dict[str, float]:
        wavelength_nm = self._safe_float(self._meta.get("wavelength_nm"))
        filters, _warnings = resolve_selected_filters(
            selected_filters=self._selected_filter_entries(),
            fundamental_wavelength_nm=wavelength_nm if np.isfinite(wavelength_nm) else None,
        )
        return filters

    def _set_selected_filters_from_dict(self, filters: Dict[str, Any]) -> None:
        selected_ids = {str(key) for key, value in filters.items() if np.isfinite(self._safe_float(value))}
        self.filter_list.blockSignals(True)
        try:
            if not selected_ids:
                self.no_filter_checkbox.setChecked(True)
                self.filter_list.clearSelection()
            else:
                self.no_filter_checkbox.setChecked(False)
                for index in range(self.filter_list.count()):
                    item = self.filter_list.item(index)
                    item.setSelected(str(item.data(Qt.ItemDataRole.UserRole)) in selected_ids)
            self.standard_fit_widget._toggle_filter_selection(self.no_filter_checkbox.isChecked())
        finally:
            self.filter_list.blockSignals(False)

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

    def _parse_nfit_group_paths(self) -> List[str]:
        return [
            str(self.lst_nfit_group_paths.item(index).data(Qt.ItemDataRole.UserRole) or "").strip()
            for index in range(self.lst_nfit_group_paths.count())
            if str(self.lst_nfit_group_paths.item(index).data(Qt.ItemDataRole.UserRole) or "").strip()
        ]

    def _parse_nfit_excluded_paths(self) -> List[str]:
        return [
            str(self.lst_nfit_group_paths.item(index).data(Qt.ItemDataRole.UserRole) or "").strip()
            for index in range(self.lst_nfit_group_paths.count())
            if self.lst_nfit_group_paths.item(index).checkState() != Qt.CheckState.Checked
        ]

    def _nfit_folder_check_changed(self, *_args) -> None:
        total_count = self.lst_nfit_group_paths.count()
        excluded_count = len(self._parse_nfit_excluded_paths())
        included_count = total_count - excluded_count
        self.lbl_nfit_hint.setText(
            f"Selected folders: {total_count} | Included in fit: {included_count} | "
            f"Result-only: {excluded_count}\n"
            "Preview is unchanged. Use Refresh Preview to reload the grouped data."
        )

    def _set_nfit_group_paths(
        self,
        paths: List[str],
        excluded_paths: Optional[List[str]] = None,
    ) -> None:
        excluded = {str(Path(path).resolve()) for path in (excluded_paths or []) if str(path).strip()}
        self.lst_nfit_group_paths.blockSignals(True)
        try:
            self.lst_nfit_group_paths.clear()
            for raw_path in paths:
                if not str(raw_path).strip():
                    continue
                path = Path(raw_path).resolve()
                item = QListWidgetItem(path.name or str(path))
                item.setToolTip(str(path))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Unchecked
                    if str(path) in excluded
                    else Qt.CheckState.Checked
                )
                self.lst_nfit_group_paths.addItem(item)
        finally:
            self.lst_nfit_group_paths.blockSignals(False)

    def _select_multiple_directories(self, start_dir: Optional[Path] = None) -> List[str]:
        initial_dir = str(start_dir or self._current_dir or Path.cwd())
        try:
            folders = select_multiple_directories(
                parent_hwnd=int(self.window().winId()) if self.window() is not None else None,
                title="Select measurement folder(s)",
                initial_dir=initial_dir,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Selection failed", f"Failed to open native folder picker: {exc}")
            return []
        return [str(path) for path in folders if str(path).strip()]

    def _nfit_select_folders_clicked(self):
        selected = self._select_multiple_directories()
        if not selected:
            return
        selected_paths = [str(Path(path).resolve()) for path in selected]
        ok, msg = False, "Folder loading failed."
        self._set_busy(True, "Loading selected folders...")
        try:
            ok, msg = self._load_folder(Path(selected_paths[0]))
            if ok:
                self._set_nfit_group_paths(selected_paths)
                self._refresh_analysis_views(reset_manual=False)
        finally:
            self._set_busy(False)
        if not ok:
            QMessageBox.warning(self, "Load failed", msg)
            return

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

        selected_ids = set(filters.keys())
        self.no_filter_checkbox.setChecked(False)
        self.filter_list.blockSignals(True)
        try:
            for index in range(self.filter_list.count()):
                item = self.filter_list.item(index)
                if str(item.data(Qt.ItemDataRole.UserRole)) in selected_ids:
                    item.setSelected(True)
        finally:
            self.filter_list.blockSignals(False)
        self.standard_fit_widget._toggle_filter_selection(False)
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
        if self._busy:
            return
        start_dir = str(self._current_dir) if self._current_dir else "results"
        folder = QFileDialog.getExistingDirectory(self, "Select result folder", start_dir)
        if not folder:
            return
        ok, msg = False, "Folder loading failed."
        self._set_busy(True, "Loading folder and preparing plots...")
        try:
            ok, msg = self._load_folder(Path(folder))
        finally:
            self._set_busy(False)
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
        self._transient_strategy_names.clear()
        self._update_folder_status_labels()

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
        self._set_selected_filters_from_dict(filters if isinstance(filters, dict) else {})
        raw_group_paths = meta.get("n_fit_group_paths")
        if isinstance(raw_group_paths, list):
            raw_excluded_paths = meta.get("n_fit_excluded_paths")
            self._set_nfit_group_paths(
                [str(path) for path in raw_group_paths],
                [str(path) for path in raw_excluded_paths] if isinstance(raw_excluded_paths, list) else [],
            )
        else:
            self._set_nfit_group_paths([])
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

        payload["filters"] = self._selected_filters_dict()
        n_fit_group_paths = self._parse_nfit_group_paths()
        if n_fit_group_paths:
            payload["n_fit_group_paths"] = n_fit_group_paths
            excluded_paths = self._parse_nfit_excluded_paths()
            if excluded_paths:
                payload["n_fit_excluded_paths"] = excluded_paths
            else:
                payload.pop("n_fit_excluded_paths", None)
        else:
            payload.pop("n_fit_group_paths", None)
            payload.pop("n_fit_excluded_paths", None)
        payload = self.extrema_widget.merge_into_metadata(payload)
        return payload

    def _write_json_metadata(self, show_message: bool = False, refresh_views: bool = True) -> Tuple[bool, str]:
        if not self.json_path:
            return False, "Load a result folder first."
        preferred_strategy_name = self._selected_strategy_name()
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
        self.extrema_widget.mark_saved()
        if preferred_strategy_name:
            self._set_picker_strategy(preferred_strategy_name)
            self._refresh_saved_strategy_list(meta, preferred_strategy_name=preferred_strategy_name)
        else:
            self._apply_saved_strategy_selection(meta)
        self._populate_table_from_json(meta)
        if refresh_views:
            self._refresh_analysis_views(reset_manual=False)
        if show_message:
            QMessageBox.information(self, "Updated", "JSON metadata updated.")
        return True, "OK"

    # -------------------------------- Run fit --------------------------------
    def _run_fit_clicked(self):
        if self._busy:
            return
        if not self._current_dir:
            QMessageBox.information(self, "No data", "Load a result folder first.")
            return
        if SHGDataAnalysis is None:
            QMessageBox.critical(self, "Missing module", "shg_analysis is not importable.")
            return
        requested_strategy_name = self._selected_strategy_name()
        # Ensure JSON reflects current editor values before fitting
        ok, message = self._write_json_metadata(show_message=False, refresh_views=False)
        if not ok:
            QMessageBox.critical(self, "Update failed", message)
            return

        # Resolve strategy class
        s = self._find_strategy_info(requested_strategy_name) if requested_strategy_name else None
        if s is None:
            s = self._get_selected_strategy()
        if s is None:
            QMessageBox.critical(self, "No strategy", "No fitting strategy is available/selected.")
            return
        self._fit_requested_strategy_name = requested_strategy_name
        self._fit_worker = FitWorker(str(self._current_dir), s.qualname, s.class_name)
        self._fit_worker.succeeded.connect(self._fit_worker_succeeded)
        self._fit_worker.failed.connect(self._fit_worker_failed)
        self._fit_worker.finished.connect(self._fit_worker_finished)
        self._set_busy(True, f"Running {s.display_name}...")
        self._fit_worker.start()

    def _fit_worker_succeeded(self, _results: object) -> None:
        self.lbl_busy.setText("Loading fitted results and updating plots...")
        QApplication.processEvents()
        if not self._current_dir:
            return
        ok, msg = self._load_folder(self._current_dir)
        if not ok:
            QMessageBox.warning(self, "Reload failed", msg)
            return
        requested_strategy_name = self._fit_requested_strategy_name
        if requested_strategy_name:
            self._set_picker_strategy(requested_strategy_name)
            self._refresh_saved_strategy_list(
                self._meta,
                preferred_strategy_name=requested_strategy_name,
            )
            if self._meta:
                self._populate_table_from_json(self._meta)

    def _fit_worker_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Fit failed", message)

    def _fit_worker_finished(self) -> None:
        worker = self._fit_worker
        self._fit_worker = None
        self._fit_requested_strategy_name = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()

    # ------------------------------ Table/plots ------------------------------
    def _populate_table_from_json(self, meta: Dict):
        """Show saved fit-oriented values without duplicating the metadata panel."""
        rows = []
        fit_payload = self._fit_payload_for_strategy(meta)
        for key, label in [
            ("L_mm", "Corrected L [mm]"),
            ("L_mm_std", "Corrected L std [mm]"),
            ("centering_pos", "Centering position"),
            ("k_scale", "k scale"),
            ("k_scale_std", "k scale std"),
            ("Pm0", "Peak Pm0"),
            ("Pm0_stderr", "Peak Pm0 std"),
            ("d_rel_abs", "|d| relative"),
            ("d_component", "d component"),
            ("d_factor", "d factor"),
            ("Lc_mean_mm", "Lc(0) [mm]"),
            ("Lc_std_mm", "Lc(0) std [mm]"),
            ("Lc_pair_mean_mm", "Lc pair mean [mm]"),
            ("Lc_pair_std_mm", "Lc pair std [mm]"),
            ("lc_extrapolation_order", "Lc extrapolation order"),
            ("lc_order_residual_rms", "Lc fit residual RMS [mm]"),
            ("residual_rms", "Residual RMS"),
            ("minima_count", "Minima count"),
            ("n_count", "Lc pair count"),
            ("n_peaks", "Peak count"),
            ("group_size", "Group size"),
            ("phase_pair_count", "Phase pair count"),
            ("n_fit_cost", "n-fit cost"),
            ("n_fit_success", "n-fit success"),
            ("dn_w_a", "dn w a"),
            ("dn_w_b", "dn w b"),
            ("dn_w_c", "dn w c"),
            ("dn_2w_a", "dn 2w a"),
            ("dn_2w_b", "dn 2w b"),
            ("dn_2w_c", "dn 2w c"),
            ("n_w_a", "n w a"),
            ("n_w_b", "n w b"),
            ("n_w_c", "n w c"),
            ("n_2w_a", "n 2w a"),
            ("n_2w_b", "n 2w b"),
            ("n_2w_c", "n 2w c"),
        ]:
            if key in fit_payload:
                rows.append((label, fit_payload.get(key)))

        selected_name = self._selected_strategy_name()
        if selected_name:
            rows.insert(
                0,
                ("strategy", self._display_name_for_strategy(selected_name, saved=self._is_strategy_saved(selected_name))),
            )

        self.tbl.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.tbl.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl.setItem(i, 1, QTableWidgetItem("" if v is None else f"{v}"))

    def _refresh_analysis_views(self, reset_manual: bool = False):
        if self._df is None or self._current_dir is None:
            self._analysis_context = {}
            self._clear_plots()
            self._clear_nfit_measurements("Load a result folder first.")
            return
        self._analysis_context = self._prepare_analysis_context()
        self._live_curve_cache = {}
        if reset_manual or not self._manual_controls_ready():
            self._initialize_manual_controls_from_context()
            self._use_saved_fit_preview = self._csv_fit_matches_selected_strategy()
        self._extrema_force_reset = bool(reset_manual)
        self._render_analysis_plots()

    def _prepare_analysis_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {"error": None}
        if not self._current_dir:
            context["error"] = "Load a result folder first."
            return context
        if SHGDataAnalysis is None:
            context["error"] = "shg_analysis is not importable."
            return context

        selected_name = self._selected_strategy_name()
        selected = self._get_selected_strategy()
        if selected is None:
            if selected_name:
                context["error"] = f"Selected strategy '{selected_name}' is not available."
            else:
                context["error"] = "No fitting strategy selected."
            return context
        meta_with_fit = self._meta_with_selected_fit(self._meta, selected)

        try:
            mod = importlib.import_module(selected.qualname)
            strategy_cls = getattr(mod, selected.class_name)
            analysis = SHGDataAnalysis(str(self._current_dir))
            n_fit_group_paths = self._parse_nfit_group_paths()
            if n_fit_group_paths:
                analysis.meta["n_fit_group_paths"] = n_fit_group_paths
                excluded_paths = self._parse_nfit_excluded_paths()
                if excluded_paths:
                    analysis.meta["n_fit_excluded_paths"] = excluded_paths
                else:
                    analysis.meta.pop("n_fit_excluded_paths", None)
            else:
                analysis.meta.pop("n_fit_group_paths", None)
                analysis.meta.pop("n_fit_excluded_paths", None)
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
        if hasattr(strategy, "_load_measurement_group"):
            try:
                context["n_fit_group"] = strategy._load_measurement_group()
            except Exception as e:
                context["n_fit_group_error"] = str(e)
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

    def _evaluate_strategy_curves(
        self,
        strategy: Any,
        L_value: float,
        x: np.ndarray,
        dn_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        override = {"L": L_value, "theta_deg": x}
        if dn_override:
            override["dn_override"] = dn_override
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

    def _live_curve_cache_key(
        self,
        strategy: Any,
        L_value: float,
        x: np.ndarray,
        dn_override: Optional[Dict[str, float]],
    ) -> Tuple[Any, ...]:
        x = np.asarray(x, dtype=float)
        if x.size:
            x_key = (
                int(x.size),
                float(np.nanmin(x)),
                float(np.nanmax(x)),
                float(np.nanmean(x)),
                float(np.nanstd(x)),
            )
        else:
            x_key = (0, float("nan"), float("nan"), float("nan"), float("nan"))
        dn_key = tuple(sorted((dn_override or {}).items()))
        return (
            strategy.__class__.__module__,
            strategy.__class__.__name__,
            round(float(L_value), 12),
            x_key,
            dn_key,
        )

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
        d_rel_abs = self._safe_float(meta.get("d_rel_abs"))
        if np.isfinite(d_rel_abs):
            return float(d_rel_abs**2)
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

    def _strategy_uses_d_rel_abs(self, strategy: Any | None = None) -> bool:
        if strategy is None and isinstance(self._analysis_context, dict):
            strategy = self._analysis_context.get("strategy")
        return str(getattr(strategy, "INTENSITY_SCALE_PARAMETER", "")).strip() == "d_rel_abs"

    def _intensity_scale_from_control(self, peak_value: float, strategy: Any | None = None) -> float:
        return float(peak_value)

    def _decimals_for_sigfigs(self, reference: float, sigfigs: int = 3) -> int:
        reference = abs(float(reference))
        if not np.isfinite(reference) or reference <= 0.0:
            return sigfigs
        exponent = math.floor(math.log10(reference))
        return max(0, min(18, int(sigfigs - 1 - exponent)))

    def _step_for_sigfigs(self, reference: float, sigfigs: int = 3) -> float:
        reference = abs(float(reference))
        if not np.isfinite(reference) or reference <= 0.0:
            return 10.0 ** (1 - sigfigs)
        exponent = math.floor(math.log10(reference))
        return 10.0 ** (exponent - sigfigs + 1)

    def _precision_reference(self, *values: float) -> float:
        finite_positive = [
            abs(float(value))
            for value in values
            if np.isfinite(float(value)) and abs(float(value)) > 0.0
        ]
        if finite_positive:
            return min(finite_positive)
        finite_abs = [
            abs(float(value))
            for value in values
            if np.isfinite(float(value))
        ]
        return max(finite_abs) if finite_abs else 1.0

    def _apply_peak_spinbox_precision(self, controls: Dict[str, Any], *values: float) -> None:
        reference = self._precision_reference(*values)
        decimals = self._decimals_for_sigfigs(reference)
        step = self._step_for_sigfigs(reference)
        for name in ("value", "min", "max"):
            controls[name].setDecimals(decimals)
            controls[name].setSingleStep(step)

    def _format_sigfigs(self, value: float, sigfigs: int = 3) -> str:
        value = float(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{sigfigs}g}"

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
        class_name = self._selected_strategy_name() or "current"
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

    def _default_plot_settings(self, plot_key: str) -> PlotSettings:
        return PlotSettings(
            font_size=11.0 if plot_key == "fit" else 10.0,
            show_legend=plot_key not in {"resid", "lc"},
            box_aspect=0.0,
            data_plot_style="markers",
            show_fit_annotation=True,
        )

    def _parse_range_bound(self, text: str) -> Optional[float]:
        stripped = str(text).strip()
        if not stripped:
            return None
        return float(stripped)

    def _parse_optional_range_bounds(
        self,
        min_text: str,
        max_text: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        range_min = self._parse_range_bound(min_text)
        range_max = self._parse_range_bound(max_text)
        if range_min is not None and range_max is not None and range_min > range_max:
            raise ValueError("Range min must be less than or equal to max.")
        return range_min, range_max

    def _fit_data_plot_kwargs(self) -> Dict[str, Any]:
        style = self._plot_settings["fit"].data_plot_style
        kwargs: Dict[str, Any] = {
            "marker": "*",
            "markersize": 5,
        }
        if style == "line+markers":
            kwargs["linestyle"] = "-"
            kwargs["linewidth"] = 1.0
        elif style == "line":
            kwargs["linestyle"] = "-"
            kwargs["linewidth"] = 1.2
            kwargs["marker"] = None
        else:
            kwargs["linestyle"] = "none"
        return kwargs

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

        cmb_data_style = QComboBox()
        cmb_data_style.addItem("Markers only", userData="markers")
        cmb_data_style.addItem("Markers + line", userData="line+markers")
        cmb_data_style.addItem("Line only", userData="line")
        data_style_index = max(cmb_data_style.findData(settings.data_plot_style), 0)
        cmb_data_style.setCurrentIndex(data_style_index)

        cb_fit_annotation = QCheckBox("Show L / Peak annotation")
        cb_fit_annotation.setChecked(settings.show_fit_annotation)

        sb_aspect = QDoubleSpinBox()
        sb_aspect.setRange(0.0, 5.0)
        sb_aspect.setDecimals(2)
        sb_aspect.setSingleStep(0.05)
        sb_aspect.setSpecialValueText("Auto")
        sb_aspect.setValue(settings.box_aspect)

        le_x_min = QLineEdit("" if settings.x_min is None else f"{settings.x_min:g}")
        le_x_min.setPlaceholderText("auto")
        le_x_max = QLineEdit("" if settings.x_max is None else f"{settings.x_max:g}")
        le_x_max.setPlaceholderText("auto")
        le_y_min = QLineEdit("" if settings.y_min is None else f"{settings.y_min:g}")
        le_y_min.setPlaceholderText("auto")
        le_y_max = QLineEdit("" if settings.y_max is None else f"{settings.y_max:g}")
        le_y_max.setPlaceholderText("auto")

        x_range_row = QWidget()
        x_range_layout = QHBoxLayout(x_range_row)
        x_range_layout.setContentsMargins(0, 0, 0, 0)
        x_range_layout.setSpacing(6)
        x_range_layout.addWidget(le_x_min)
        x_range_layout.addWidget(QLabel("-"))
        x_range_layout.addWidget(le_x_max)

        y_range_row = QWidget()
        y_range_layout = QHBoxLayout(y_range_row)
        y_range_layout.setContentsMargins(0, 0, 0, 0)
        y_range_layout.setSpacing(6)
        y_range_layout.addWidget(le_y_min)
        y_range_layout.addWidget(QLabel("-"))
        y_range_layout.addWidget(le_y_max)

        form.addRow("Font size:", sb_font)
        form.addRow("", cb_legend)
        if plot_key == "fit":
            form.addRow("Data style:", cmb_data_style)
            form.addRow("", cb_fit_annotation)
        form.addRow("Box aspect:", sb_aspect)
        form.addRow("X range:", x_range_row)
        form.addRow("Y range:", y_range_row)
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
            lambda: self._restore_plot_settings_dialog_defaults(
                plot_key=plot_key,
                sb_font=sb_font,
                cb_legend=cb_legend,
                sb_aspect=sb_aspect,
                le_x_min=le_x_min,
                le_x_max=le_x_max,
                le_y_min=le_y_min,
                le_y_max=le_y_max,
                cmb_data_style=cmb_data_style,
                cb_fit_annotation=cb_fit_annotation,
            )
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            x_min, x_max = self._parse_optional_range_bounds(le_x_min.text(), le_x_max.text())
            y_min, y_max = self._parse_optional_range_bounds(le_y_min.text(), le_y_max.text())
            self._plot_settings[plot_key] = PlotSettings(
                font_size=float(sb_font.value()),
                show_legend=bool(cb_legend.isChecked()),
                box_aspect=float(sb_aspect.value()),
                data_plot_style=str(cmb_data_style.currentData() or "markers"),
                show_fit_annotation=bool(cb_fit_annotation.isChecked()),
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        except Exception as e:
            QMessageBox.warning(self, "Invalid setting", str(e))
            return

        self._update_canvas_height(plot_key)
        self._render_analysis_plots()

    def _restore_plot_settings_dialog_defaults(
        self,
        plot_key: str,
        sb_font: QDoubleSpinBox,
        cb_legend: QCheckBox,
        sb_aspect: QDoubleSpinBox,
        le_x_min: QLineEdit,
        le_x_max: QLineEdit,
        le_y_min: QLineEdit,
        le_y_max: QLineEdit,
        cmb_data_style: QComboBox,
        cb_fit_annotation: QCheckBox,
    ) -> None:
        defaults = self._default_plot_settings(plot_key)
        sb_font.setValue(defaults.font_size)
        cb_legend.setChecked(defaults.show_legend)
        sb_aspect.setValue(defaults.box_aspect)
        le_x_min.clear()
        le_x_max.clear()
        le_y_min.clear()
        le_y_max.clear()
        cmb_data_style.setCurrentIndex(max(cmb_data_style.findData(defaults.data_plot_style), 0))
        cb_fit_annotation.setChecked(defaults.show_fit_annotation)

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
        return all(np.isfinite(float(controls["value"].value())) for controls in self._manual_controls.values()) and np.isfinite(float(self.sb_manual_centering.value()))

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
        centering_value = self._safe_float(saved_fit.get("centering_pos"))
        if not np.isfinite(centering_value):
            centering_info = context.get("centering_info")
            if isinstance(centering_info, dict):
                centering_value = self._safe_float(centering_info.get("c_best"))
        if not np.isfinite(centering_value):
            centering_value = 0.0

        self._set_manual_control("L", auto_L - l_span, auto_L + l_span, auto_L)
        self._set_manual_control("peak", 0.0, max(auto_peak + peak_span, peak_span), max(auto_peak, 0.0))
        self.sb_manual_centering.setValue(float(centering_value))

    def _set_manual_control(self, key: str, minimum: float, maximum: float, value: float):
        controls = self._manual_controls[key]
        if maximum <= minimum:
            maximum = minimum + 1e-9
        value = min(max(value, minimum), maximum)
        if key == "peak":
            self._apply_peak_spinbox_precision(controls, minimum, maximum, value)

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
        self._use_saved_fit_preview = False
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
        current_before = float(controls["value"].value())
        if key == "peak":
            self._apply_peak_spinbox_precision(controls, minimum, maximum, current_before)
        controls["value"].setRange(minimum, maximum)
        current = min(max(current_before, minimum), maximum)
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
        self._use_saved_fit_preview = False
        self._manual_syncing = True
        try:
            if key == "peak":
                controls = self._manual_controls[key]
                self._apply_peak_spinbox_precision(
                    controls,
                    float(controls["min"].value()),
                    float(controls["max"].value()),
                    float(controls["value"].value()),
                )
            self._sync_slider_to_value(key)
        finally:
            self._manual_syncing = False
        self._render_analysis_plots()

    def _manual_slider_changed(self, key: str):
        if self._manual_syncing:
            return
        self._use_saved_fit_preview = False
        controls = self._manual_controls[key]
        minimum = float(controls["min"].value())
        maximum = float(controls["max"].value())
        if maximum <= minimum:
            value = minimum
        else:
            value = minimum + (maximum - minimum) * int(controls["slider"].value()) / self._SLIDER_STEPS
        self._manual_syncing = True
        try:
            if key == "peak":
                self._apply_peak_spinbox_precision(controls, minimum, maximum, value)
            controls["value"].setValue(value)
        finally:
            self._manual_syncing = False
        strategy = self._analysis_context.get("strategy") if isinstance(self._analysis_context, dict) else None
        if (
            strategy is not None
            and getattr(strategy, "LIVE_UPDATE_ON_SLIDER", True) is False
            and key != "peak"
        ):
            return
        self._render_analysis_plots()

    def _manual_slider_released(self, key: str):
        if self._manual_syncing:
            return
        self._use_saved_fit_preview = False
        self._render_analysis_plots()

    def _manual_centering_changed(self):
        if self._manual_syncing:
            return
        self._use_saved_fit_preview = False
        self._render_analysis_plots()

    def _reset_manual_controls_clicked(self):
        self._initialize_manual_controls_from_context()
        self._use_saved_fit_preview = self._csv_fit_matches_selected_strategy()
        self._render_analysis_plots()

    def _manual_value(self, key: str) -> float:
        return float(self._manual_controls[key]["value"].value())

    def _manual_centering_value(self) -> float:
        return float(self.sb_manual_centering.value())

    def _current_prepared_data(self) -> pd.DataFrame:
        context = self._analysis_context
        prepared = context.get("prepared_data")
        if not isinstance(prepared, pd.DataFrame):
            return pd.DataFrame()
        current = prepared.copy()
        if "position" in current.columns:
            current["position_centered"] = np.asarray(current["position"], dtype=float) - self._manual_centering_value()
        return current

    def _current_display_xy(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        current = self._current_prepared_data()
        if current.empty:
            return np.array([], dtype=float), np.array([], dtype=float), current
        x = np.asarray(current.get("position_centered", current["position"]), dtype=float)
        if "offset_corrected" in current.columns:
            y = np.asarray(current["offset_corrected"], dtype=float)
        elif "intensity_corrected" in current.columns:
            y = np.asarray(current["intensity_corrected"], dtype=float)
        else:
            y = np.asarray(current["ch2"], dtype=float)
        return x, y, current

    def _compute_auto_extrema_info(self) -> Dict[str, Any]:
        context = self._analysis_context
        if context.get("error"):
            return {"error": context["error"]}

        strategy = context["strategy"]
        x, y, prepared = self._current_display_xy()
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

    def _compute_live_curves(self) -> Dict[str, Any]:
        context = self._analysis_context
        if context.get("error"):
            return {"error": context["error"]}

        strategy = context["strategy"]
        x, y, _prepared = self._current_display_xy()
        L_value = self._manual_value("L")
        peak_value = self._manual_value("peak")
        dn_override = self._dn_override_from_saved_fit(
            context.get("saved_fit", {}) if isinstance(context.get("saved_fit"), dict) else {}
        )

        try:
            cache_key = self._live_curve_cache_key(strategy, L_value, x, dn_override or None)
            cached = self._live_curve_cache.get("model")
            if isinstance(cached, dict) and cached.get("key") == cache_key:
                fit_base = cached["fit_base"]
                env_base = cached["env_base"]
                fit_aux = cached["fit_aux"]
            else:
                fit_base, env_base, fit_aux = self._evaluate_strategy_curves(
                    strategy,
                    L_value,
                    x,
                    dn_override=dn_override or None,
                )
                self._live_curve_cache["model"] = {
                    "key": cache_key,
                    "fit_base": fit_base,
                    "env_base": env_base,
                    "fit_aux": fit_aux,
                }
        except Exception as e:
            return {"error": f"Failed to evaluate current fit: {e}"}

        intensity_scale = self._intensity_scale_from_control(peak_value, strategy)
        fit_curve = intensity_scale * fit_base
        envelope_curve = intensity_scale * env_base
        residual = fit_curve - y

        return {
            "x": x,
            "y": y,
            "strategy": strategy,
            "fit_curve": fit_curve,
            "envelope_curve": envelope_curve,
            "residual": residual,
            "L_value": L_value,
            "peak_value": peak_value,
            "intensity_scale": intensity_scale,
            "centering_value": self._manual_centering_value(),
            "fit_curve_raw": fit_curve + float(context.get("offset", 0.0)),
            "fit_aux": fit_aux,
            "d_factor": fit_aux.get("d_factor"),
        }

    def _compute_saved_fit_curves(self) -> Dict[str, Any]:
        context = self._analysis_context
        if context.get("error"):
            return {"error": context["error"]}
        if not isinstance(self._df, pd.DataFrame) or "fit" not in self._df.columns:
            return {"error": "No saved fit curve is available in the CSV."}

        x, y, _prepared = self._current_display_xy()
        fit_raw = np.asarray(self._df["fit"], dtype=float)
        if fit_raw.shape != y.shape:
            return {"error": "Saved fit curve length does not match the displayed data."}

        offset = float(context.get("offset", 0.0))
        fit_curve = fit_raw - offset if "offset_corrected" in self._current_prepared_data().columns else fit_raw
        if "fit_envelope" in self._df.columns and len(self._df["fit_envelope"]) == len(fit_raw):
            envelope_curve = np.asarray(self._df["fit_envelope"], dtype=float)
        else:
            envelope_curve = fit_curve

        L_value = self._safe_float((context.get("saved_fit") or {}).get("L_mm"))
        if not np.isfinite(L_value):
            L_value = self._manual_value("L")
        peak_value = self._manual_value("peak")

        return {
            "x": x,
            "y": y,
            "strategy": context.get("strategy"),
            "fit_curve": fit_curve,
            "envelope_curve": envelope_curve,
            "residual": fit_curve - y,
            "L_value": L_value,
            "peak_value": peak_value,
            "intensity_scale": peak_value,
            "centering_value": self._manual_centering_value(),
            "fit_curve_raw": fit_raw,
            "fit_aux": {},
            "d_factor": None,
            "saved_preview": True,
        }

    def _make_fit_theory_dataframe(self, L_value: float, peak_value: float) -> pd.DataFrame:
        context = self._analysis_context
        strategy = context["strategy"]
        x, _y, _prepared = self._current_display_xy()
        point_count = max(4001, len(x) * 5 if len(x) else 4001)
        grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), point_count)
        dn_override = self._dn_override_from_saved_fit(
            context.get("saved_fit", {}) if isinstance(context.get("saved_fit"), dict) else {}
        )
        intensity_scale = self._intensity_scale_from_control(peak_value, strategy)
        theory = intensity_scale * np.asarray(
            strategy._maker_fringes(
                override={
                    "L": L_value,
                    "theta_deg": grid,
                    **({"dn_override": dn_override} if dn_override else {}),
                }
            ),
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
            lc_data = self._make_fit_theory_dataframe(L_value, peak_value) if source == "fit" else self._current_prepared_data()
            minima_override = None
            if source == "data":
                minima_override = self.extrema_widget.saved_minima_indices()
            result, aux = strategy._calc_Lc_large_angle(
                context["analysis"].meta,
                lc_data,
                [0, 180],
                L_value,
                minima_idx_override=minima_override,
            )
            return {"result": result, "aux": aux, "source": source, "data": lc_data}
        except Exception as e:
            message = str(e)
            if "No valid adjacent-minima pairs to compute Lc." in message or "No valid adjacent-minima pairs after filtering." in message:
                return {"skipped": True, "message": message, "source": source}
            return {"error": message, "source": source}

    def _dn_override_from_saved_fit(self, fit_payload: Dict[str, Any]) -> Dict[str, float]:
        dn_override: Dict[str, float] = {}
        for key in ("dn_w_a", "dn_w_b", "dn_w_c", "dn_2w_a", "dn_2w_b", "dn_2w_c"):
            value = self._safe_float(fit_payload.get(key))
            if np.isfinite(value):
                dn_override[key] = float(value)
        return dn_override

    def _nfit_group_result_by_source(self) -> Dict[str, Dict[str, Any]]:
        selected_global_result = self._global_result_by_id(self._selected_result_id())
        raw = selected_global_result.get("group_results") if selected_global_result else None
        if not isinstance(raw, list):
            raw = self._meta.get("n_fit_group_results")
        if not isinstance(raw, list):
            return {}
        mapping: Dict[str, Dict[str, Any]] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("source_dir") or "").strip()
            if key:
                mapping[key] = entry
        return mapping

    def _render_nfit_page(self):
        context = self._analysis_context
        if not context:
            self._clear_nfit_measurements("Load a result folder first.")
            return
        if context.get("error"):
            self._clear_nfit_measurements(str(context["error"]))
            return

        strategy = context.get("strategy")
        if not hasattr(strategy, "_load_measurement_group"):
            self._clear_nfit_measurements("Select a global refractive-index fitting strategy to use this page.")
            return

        group_error = context.get("n_fit_group_error")
        if group_error:
            self._clear_nfit_measurements(f"Grouped measurement preview is unavailable: {group_error}")
            return

        measurements = context.get("n_fit_group")
        if not isinstance(measurements, list) or len(measurements) == 0:
            self._clear_nfit_measurements("No grouped measurements are available.")
            return

        fit_payload = context.get("saved_fit", {}) if isinstance(context.get("saved_fit"), dict) else {}
        dn_override = self._dn_override_from_saved_fit(fit_payload)
        group_results = self._nfit_group_result_by_source()
        selected_result_id = self._selected_result_id()
        if self._nfit_manual_result_id != selected_result_id:
            self._nfit_manual_overrides.clear()
            self._nfit_manual_result_id = selected_result_id
        self._nfit_card_views.clear()
        self._clear_nfit_measurements()

        has_saved_global_fit = bool(dn_override)
        summary_lines = [
            f"Measurements in group: {len(measurements)}",
            f"Included in global fit: {sum(bool(item.get('included_in_fit', True)) for item in measurements)}",
        ]
        if has_saved_global_fit:
            summary_lines.append(
                "Saved dn: "
                + ", ".join(f"{key}={value:+.5f}" for key, value in dn_override.items())
            )
        else:
            summary_lines.append("No saved dn values yet. Run the global n-fit strategy to overlay fitted curves.")
        self.lbl_nfit_hint.setText("\n".join(summary_lines))

        for measurement in measurements:
            meta = measurement["meta"]
            prepared = measurement["data"]
            x = np.asarray(prepared.get("position_centered", prepared["position"]), dtype=float)
            y = np.asarray(prepared.get("offset_corrected", prepared["intensity_corrected"]), dtype=float)
            minima_idx = np.asarray(measurement.get("minima_idx", []), dtype=int)

            title = (
                f"{meta.get('sample') or meta.get('sample_id') or 'measurement'} | "
                f"in {meta.get('input_polarization')} / out {meta.get('detected_polarization')}"
            )
            group_box = QGroupBox(title)
            box_layout = QVBoxLayout(group_box)
            card_header = QHBoxLayout()
            card_header.addStretch(1)
            edit_button = QPushButton("Edit")
            edit_button.setCheckable(True)
            edit_button.setEnabled(has_saved_global_fit)
            card_header.addWidget(edit_button)
            box_layout.addLayout(card_header)

            info_lines = [
                f"cut={meta.get('crystal_orientation')}, axis={meta.get('rot/trans_axis')}, material={meta.get('material')}",
                "global fit: included" if measurement.get("included_in_fit", True) else "global fit: excluded (result applied)",
            ]
            result_entry = group_results.get(str(measurement.get("source_dir") or ""))
            fitted_L = None
            if isinstance(result_entry, dict):
                fitted_L = self._safe_float(result_entry.get("L_mm"))
                dL_mm = self._safe_float(result_entry.get("dL_mm"))
                if np.isfinite(fitted_L):
                    if np.isfinite(dL_mm):
                        info_lines.append(f"L = {fitted_L:.6f} mm (dL = {1000.0 * dL_mm:+.2f} um)")
                    else:
                        info_lines.append(f"L = {fitted_L:.6f} mm")
            if not np.isfinite(self._safe_float(fitted_L)):
                info_lines.append(f"L0 = {float(measurement['L0_mm']):.6f} mm")
            info_lines.append(f"minima={len(minima_idx)}, phase pairs={len(measurement.get('phase_pairs_deg', []))}")

            info_label = QLabel("\n".join(info_lines))
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: gray;")
            box_layout.addWidget(info_label)

            canvas = MplCanvas(group_box, width=6.2, height=2.8)
            canvas.setFixedHeight(260)
            box_layout.addWidget(canvas)

            source_dir = str(measurement.get("source_dir") or "")
            initial_L = fitted_L if np.isfinite(self._safe_float(fitted_L)) else float(measurement["L0_mm"])
            initial_peak = self._safe_float(
                result_entry.get("k_scale_small_angle") if isinstance(result_entry, dict) else None
            )
            if not np.isfinite(initial_peak):
                initial_peak = max(float(np.nanmax(y)), 0.0) if y.size else 1.0
            initial_centering = self._safe_float(
                result_entry.get("centering_pos") if isinstance(result_entry, dict) else None
            )
            if not np.isfinite(initial_centering):
                centering_info = measurement.get("centering_info")
                initial_centering = self._safe_float(
                    centering_info.get("c_best") if isinstance(centering_info, dict) else None,
                    0.0,
                )
            override = self._nfit_manual_overrides.setdefault(
                source_dir,
                {
                    "L_mm": float(initial_L),
                    "peak": float(initial_peak),
                    "centering_pos": float(initial_centering),
                },
            )

            editor = QWidget(group_box)
            editor_layout = QVBoxLayout(editor)
            editor_layout.setContentsMargins(0, 0, 0, 0)
            editor_controls: Dict[str, Dict[str, Any]] = {}
            self._add_nfit_editor_row(
                editor_layout,
                editor_controls,
                "L_mm",
                "L [mm]",
                float(override["L_mm"]),
                max(abs(float(override["L_mm"])) * 0.02, 0.01),
            )
            self._add_nfit_editor_row(
                editor_layout,
                editor_controls,
                "peak",
                "Peak",
                float(override["peak"]),
                abs(float(override["peak"])) * 0.5
                if float(override["peak"]) != 0.0
                else 1e-12,
                nonnegative=True,
            )
            self._add_nfit_editor_row(
                editor_layout,
                editor_controls,
                "centering_pos",
                "Centering",
                float(override["centering_pos"]),
                2.0,
            )
            editor_buttons = QHBoxLayout()
            reset_button = QPushButton("Reset")
            save_button = QPushButton("Save Changes")
            editor_buttons.addWidget(reset_button)
            editor_buttons.addWidget(save_button)
            editor_buttons.addStretch(1)
            editor_layout.addLayout(editor_buttons)
            linked_label = QLabel()
            linked_label.setStyleSheet("color: gray;")
            editor_layout.addWidget(linked_label)
            editor.setVisible(False)
            box_layout.addWidget(editor)

            view = {
                "measurement": measurement,
                "result_entry": result_entry if isinstance(result_entry, dict) else {},
                "canvas": canvas,
                "info_label": info_label,
                "controls": editor_controls,
                "dn_override": dn_override,
                "initial": {
                    "L_mm": float(initial_L),
                    "peak": float(initial_peak),
                    "centering_pos": float(initial_centering),
                },
                "thickness_group_key": str(
                    (result_entry or {}).get("thickness_group_key") or ""
                ) if isinstance(result_entry, dict) else "",
            }
            self._nfit_card_views[source_dir] = view
            linked_count = self._nfit_linked_source_dirs(source_dir)
            linked_label.setText(
                f"L is linked to {len(linked_count)} folder(s) in this thickness group."
                if len(linked_count) > 1
                else "L applies only to this folder."
            )
            edit_button.toggled.connect(editor.setVisible)
            reset_button.clicked.connect(
                lambda _checked=False, source=source_dir: self._reset_nfit_manual_card(source)
            )
            save_button.clicked.connect(
                lambda _checked=False, source=source_dir: self._save_nfit_manual_changes(source)
            )
            for key, controls in editor_controls.items():
                controls["slider"].valueChanged.connect(
                    lambda value, source=source_dir, name=key: self._nfit_editor_slider_changed(
                        source, name, value
                    )
                )
                controls["value"].valueChanged.connect(
                    lambda value, source=source_dir, name=key: self._nfit_editor_value_changed(
                        source, name, value
                    )
                )
            self._draw_nfit_card(source_dir)
            self.nfit_measurements_layout.addWidget(group_box)

        self.nfit_measurements_layout.addStretch(1)

    def _add_nfit_editor_row(
        self,
        layout: QVBoxLayout,
        controls: Dict[str, Dict[str, Any]],
        key: str,
        label: str,
        value: float,
        span: float,
        *,
        nonnegative: bool = False,
    ) -> None:
        minimum = max(0.0, value - span) if nonnegative else value - span
        maximum = max(value + span, minimum + 1e-9)
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, self._SLIDER_STEPS)
        value_box = QDoubleSpinBox()
        value_box.setLocale(QLocale.c())
        if key == "peak":
            value_box.setDecimals(self._decimals_for_sigfigs(value))
            value_box.setSingleStep(self._step_for_sigfigs(value))
        else:
            value_box.setDecimals(6)
        if nonnegative:
            value_box.setRange(0.0, 1e12)
        else:
            value_box.setRange(-1e12, 1e12)
        value_box.setValue(value)
        value_box.setKeyboardTracking(False)
        value_box.setFixedWidth(110)
        slider.setValue(
            int(round((value - minimum) / (maximum - minimum) * self._SLIDER_STEPS))
        )
        row.addWidget(slider, 1)
        row.addWidget(value_box)
        layout.addLayout(row)
        controls[key] = {
            "slider": slider,
            "value": value_box,
            "minimum": minimum,
            "maximum": maximum,
            "syncing": False,
        }

    def _nfit_linked_source_dirs(self, source_dir: str) -> List[str]:
        view = self._nfit_card_views.get(source_dir) or {}
        group_key = str(view.get("thickness_group_key") or "")
        if not group_key:
            return [source_dir]
        return [
            candidate
            for candidate, candidate_view in self._nfit_card_views.items()
            if str(candidate_view.get("thickness_group_key") or "") == group_key
        ]

    def _nfit_editor_slider_changed(self, source_dir: str, key: str, slider_value: int) -> None:
        view = self._nfit_card_views.get(source_dir)
        if not view:
            return
        controls = view["controls"][key]
        if controls["syncing"]:
            return
        fraction = float(slider_value) / float(self._SLIDER_STEPS)
        value = controls["minimum"] + fraction * (controls["maximum"] - controls["minimum"])
        controls["syncing"] = True
        controls["value"].setValue(value)
        controls["syncing"] = False
        self._set_nfit_override_value(source_dir, key, value)

    def _nfit_editor_value_changed(self, source_dir: str, key: str, value: float) -> None:
        view = self._nfit_card_views.get(source_dir)
        if not view:
            return
        controls = view["controls"][key]
        if controls["syncing"]:
            return
        value = float(value)
        if key == "L_mm":
            span = 0.03
            minimum = value - span
            maximum = value + span
        elif key == "peak":
            span = abs(value) * 0.2 if value != 0.0 else 1e-12
            minimum = max(0.0, value - span)
            maximum = value + span
            controls["value"].setDecimals(self._decimals_for_sigfigs(value))
            controls["value"].setSingleStep(self._step_for_sigfigs(value))
        else:
            span = max(abs(value) * 0.2, 1e-6)
            minimum = value - span
            maximum = value + span
        controls["minimum"] = minimum
        controls["maximum"] = maximum
        fraction = (value - minimum) / (maximum - minimum)
        controls["syncing"] = True
        controls["slider"].setValue(
            int(round(min(max(fraction, 0.0), 1.0) * self._SLIDER_STEPS))
        )
        controls["syncing"] = False
        self._set_nfit_override_value(source_dir, key, value)

    def _set_nfit_override_value(self, source_dir: str, key: str, value: float) -> None:
        targets = self._nfit_linked_source_dirs(source_dir) if key == "L_mm" else [source_dir]
        for target in targets:
            self._nfit_manual_overrides.setdefault(target, {})[key] = float(value)
            target_view = self._nfit_card_views.get(target)
            if target_view and key in target_view["controls"]:
                controls = target_view["controls"][key]
                controls["syncing"] = True
                controls["value"].setValue(float(value))
                fraction = (float(value) - controls["minimum"]) / (
                    controls["maximum"] - controls["minimum"]
                )
                controls["slider"].setValue(
                    int(round(min(max(fraction, 0.0), 1.0) * self._SLIDER_STEPS))
                )
                controls["syncing"] = False
            self._draw_nfit_card(target)

    def _reset_nfit_manual_card(self, source_dir: str) -> None:
        view = self._nfit_card_views.get(source_dir)
        if view:
            self._set_nfit_override_value(source_dir, "L_mm", view["initial"]["L_mm"])
            self._set_nfit_override_value(source_dir, "peak", view["initial"]["peak"])
            self._set_nfit_override_value(
                source_dir, "centering_pos", view["initial"]["centering_pos"]
            )

    def _draw_nfit_card(self, source_dir: str) -> None:
        view = self._nfit_card_views.get(source_dir)
        override = self._nfit_manual_overrides.get(source_dir)
        if not view or not override:
            return
        measurement = view["measurement"]
        raw_data = measurement["analysis"].data
        raw_x = np.asarray(raw_data["position"], dtype=float)
        x = raw_x - float(override["centering_pos"])
        prepared = measurement["data"]
        y = np.asarray(
            prepared.get("offset_corrected", prepared["intensity_corrected"]),
            dtype=float,
        )
        canvas = view["canvas"]
        canvas.clear()
        ax = canvas.ax
        ax.plot(x, y, linestyle="none", marker="o", markersize=2.5, label="Data")
        try:
            model = np.asarray(
                measurement["strategy"]._maker_fringes(
                    override={
                        "meta": measurement["meta"],
                        "data": prepared,
                        "theta_deg": x,
                        "L": float(override["L_mm"]),
                        "dn_override": view["dn_override"],
                    }
                ),
                dtype=float,
            )
            ax.plot(
                x,
                float(override["peak"]) * model,
                linewidth=1.4,
                label="Global n-fit",
            )
        except Exception as exc:
            view["info_label"].setText(view["info_label"].text() + f"\nfit overlay error: {exc}")
        ax.set_xlabel("Incidence angle (deg.)")
        ax.set_ylabel("Signal (V)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        canvas.figure.tight_layout()
        canvas.draw()

    def _save_nfit_manual_changes(self, _source_dir: str) -> None:
        result_id = self._selected_result_id()
        selected = self._get_selected_strategy()
        global_result = self._global_result_by_id(result_id)
        if not result_id or selected is None or not global_result:
            QMessageBox.warning(
                self,
                "No saved global fit",
                "Run or select a saved global n-fit before saving manual changes.",
            )
            return

        group_results = [
            dict(entry)
            for entry in global_result.get("group_results", [])
            if isinstance(entry, dict)
        ]
        result_by_source = {
            str(entry.get("source_dir") or ""): entry
            for entry in group_results
        }
        for source_dir, override in self._nfit_manual_overrides.items():
            entry = result_by_source.get(source_dir)
            if entry is None:
                continue
            entry["L_mm"] = float(override["L_mm"])
            entry["dL_mm"] = float(override["L_mm"]) - float(entry.get("L0_mm") or 0.0)
            entry["k_scale_small_angle"] = float(override["peak"])
            entry["centering_pos"] = float(override["centering_pos"])
            entry["manual_adjusted"] = True

        thickness_groups = [
            dict(entry)
            for entry in global_result.get("thickness_groups", [])
            if isinstance(entry, dict)
        ]
        for thickness_group in thickness_groups:
            key = str(thickness_group.get("key") or "")
            linked_entries = [
                entry
                for entry in group_results
                if str(entry.get("thickness_group_key") or "") == key
            ]
            if not linked_entries:
                continue
            L_mm = float(linked_entries[0]["L_mm"])
            thickness_group["L_mm"] = L_mm
            thickness_group["dL_mm"] = L_mm - float(thickness_group.get("L0_mm") or 0.0)

        global_result["group_results"] = group_results
        global_result["thickness_groups"] = thickness_groups
        global_result["manual_adjusted_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        failures: List[str] = []
        for source_dir in global_result.get("group_source_dirs", []):
            source_path = Path(str(source_dir))
            local_result = result_by_source.get(str(source_path.resolve()))
            view = self._nfit_card_views.get(str(source_path.resolve()))
            if local_result is None or view is None:
                failures.append(f"{source_path.name}: measurement is not loaded")
                continue
            json_files = list(source_path.glob("*.json"))
            csv_files = list(source_path.glob("*.csv"))
            if len(json_files) != 1 or len(csv_files) != 1:
                failures.append(f"{source_path.name}: expected one JSON and one CSV")
                continue
            try:
                meta = json.loads(json_files[0].read_text(encoding="utf-8"))
                history = [
                    dict(entry)
                    for entry in meta.get("n_fit_global_results", [])
                    if isinstance(entry, dict)
                ]
                for index, entry in enumerate(history):
                    if str(entry.get("result_id") or "") == result_id:
                        history[index] = dict(global_result)
                        break
                else:
                    history.append(dict(global_result))
                meta["n_fit_global_results"] = history
                meta["n_fit_active_result_id"] = result_id
                meta["n_fit_global_result"] = dict(global_result)
                meta["n_fit_group_results"] = [dict(entry) for entry in group_results]
                meta["n_fit_thickness_group_results"] = [
                    dict(entry) for entry in thickness_groups
                ]
                meta["n_fit_local_result"] = dict(local_result)

                fit_payload = extract_fit_payload(meta, selected.class_name, result_id)
                fit_payload.update(
                    {
                        "L_mm": float(local_result["L_mm"]),
                        "k_scale": float(local_result["k_scale_small_angle"]),
                        "Pm0": float(local_result["k_scale_small_angle"]),
                        "centering_pos": float(local_result["centering_pos"]),
                    }
                )
                meta = upsert_fitting_result(
                    meta,
                    selected.class_name,
                    fit_payload,
                    strategy_module=selected.qualname,
                    strategy_display_name=selected.display_name,
                    result_id=result_id,
                    result_label=str(global_result.get("result_label") or ""),
                )
                json_files[0].write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                measurement = view["measurement"]
                override = self._nfit_manual_overrides[str(source_path.resolve())]
                csv_df = pd.read_csv(csv_files[0])
                raw_x = np.asarray(csv_df["position"], dtype=float)
                centered_x = raw_x - float(override["centering_pos"])
                model = np.asarray(
                    measurement["strategy"]._maker_fringes(
                        override={
                            "meta": measurement["meta"],
                            "data": measurement["data"],
                            "theta_deg": centered_x,
                            "L": float(override["L_mm"]),
                            "dn_override": view["dn_override"],
                        }
                    ),
                    dtype=float,
                )
                offset = float((measurement.get("offset_info") or {}).get("offset", 0.0))
                csv_df["position_centered"] = centered_x
                csv_df["fit"] = float(override["peak"]) * model + offset
                csv_df.to_csv(csv_files[0], index=False)
            except Exception as exc:
                failures.append(f"{source_path.name}: {exc}")

        if failures:
            QMessageBox.warning(
                self,
                "Partially saved",
                "Some folders could not be updated:\n" + "\n".join(failures),
            )
            return

        ok, message = self._load_folder(self._current_dir)
        if not ok:
            QMessageBox.warning(self, "Reload failed", message)
            return
        self._refresh_saved_strategy_list(
            self._meta,
            preferred_strategy_name=selected.class_name,
        )
        QMessageBox.information(
            self,
            "Saved",
            "Manual L, Peak, and Centering values were saved for the global fit group.",
        )

    def _clear_nfit_measurements(self, message: Optional[str] = None):
        while self.nfit_measurements_layout.count():
            item = self.nfit_measurements_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                while child_layout.count():
                    sub_item = child_layout.takeAt(0)
                    sub_widget = sub_item.widget()
                    if sub_widget is not None:
                        sub_widget.deleteLater()
        if message:
            label = QLabel(message)
            label.setWordWrap(True)
            label.setStyleSheet("color: gray;")
            self.nfit_measurements_layout.addWidget(label)
            self.lbl_nfit_hint.setText(message)
    
    def _render_analysis_plots(self):
        self._update_plot_tab_visibility()
        self._update_all_canvas_heights()
        if not self._analysis_context:
            self._clear_plots()
            self._clear_nfit_measurements("Load a result folder first.")
            self.btn_apply_manual.setEnabled(False)
            return
        if self._analysis_context.get("error"):
            message = str(self._analysis_context["error"])
            self._render_fit_data_only_plot(message)
            for canvas in [self.canvas_resid, self.canvas_centering, self.canvas_lc]:
                self._show_plot_message(canvas, message)
            self.extrema_widget.show_message(message)
            self._clear_nfit_measurements(message)
            self.btn_apply_manual.setEnabled(False)
            self.lbl_manual_hint.setText(f"Fitting unavailable. Showing data only. {message}")
            return

        live = self._compute_saved_fit_curves() if self._use_saved_fit_preview else self._compute_live_curves()
        if "error" in live:
            message = str(live["error"])
            self._render_fit_data_only_plot(message)
            for canvas in [self.canvas_resid, self.canvas_centering, self.canvas_lc]:
                self._show_plot_message(canvas, message)
            self.extrema_widget.show_message(message)
            self._clear_nfit_measurements(message)
            self.btn_apply_manual.setEnabled(False)
            notes = self._analysis_context.get("notes") or []
            note_text = "" if not notes else " | " + " | ".join(str(note) for note in notes)
            self.lbl_manual_hint.setText(f"Fitting unavailable. Showing data only. {message}{note_text}")
            return

        current_plot = self._current_plot_key()
        extrema = self._compute_auto_extrema_info() if current_plot == "extrema" else {
            "minima_idx": np.array([], dtype=int),
            "maxima_idx": np.array([], dtype=int),
        }
        lc_info = (
            self._compute_lc_diagnostics(
                L_value=float(live.get("L_value", self._manual_value("L"))),
                peak_value=float(live.get("peak_value", self._manual_value("peak"))),
            )
            if current_plot == "lc"
            else {"error": "Open the Lc tab to compute Lc diagnostics."}
        )

        self._render_fit_plot(live)
        self._render_residual_plot(live)
        self._render_centering_plot()
        self._render_extrema_plot(live, extrema)
        if current_plot == "lc":
            self._render_lc_plot(lc_info)
        if self._current_page_key == "nfit":
            self._render_nfit_page()
        self.btn_apply_manual.setEnabled("error" not in live)

        notes = self._analysis_context.get("notes") or []
        self.lbl_manual_hint.setText(
            "The live overlay uses the current L, Peak, and Centering values. Overwrite updates saved fit values."
            if not notes else
            "The live overlay uses the current L, Peak, and Centering values. " + " | ".join(str(note) for note in notes)
        )

    def _render_fit_data_only_plot(self, error_message: str):
        self.canvas_fit.clear()
        ax = self.canvas_fit.ax
        context = self._analysis_context or {}
        x, y, current = self._current_display_xy()
        data = self._df
        if x.size and y.size:
            pass
        elif data is None or data.empty:
            self._show_plot_message(self.canvas_fit, error_message)
            return
        elif "position_centered" in data.columns:
            x = np.asarray(data["position_centered"], dtype=float)
        elif "position" in data.columns:
            x = np.asarray(data["position"], dtype=float)
        elif "angle_deg" in data.columns:
            x = np.asarray(data["angle_deg"], dtype=float)
        elif "position_mm" in data.columns:
            x = np.asarray(data["position_mm"], dtype=float)
        else:
            x = np.arange(len(data), dtype=float)

        if not (x.size and y.size):
            if "offset_corrected" in data.columns:
                y = np.asarray(data["offset_corrected"], dtype=float)
            elif "intensity_corrected" in data.columns:
                y = np.asarray(data["intensity_corrected"], dtype=float)
            elif "ch2" in data.columns:
                y = np.asarray(data["ch2"], dtype=float)
            else:
                numeric_columns = [
                    column for column in data.columns
                    if column not in {"position", "position_centered", "angle_deg", "position_mm", "fit", "fit_envelope"}
                    and np.issubdtype(data[column].dtype, np.number)
                ]
                if not numeric_columns:
                    self._show_plot_message(self.canvas_fit, error_message)
                    return
                y = np.asarray(data[numeric_columns[-1]], dtype=float)

        sample_label = str(self._meta.get("sample") or self._meta.get("sample_id") or "Data")
        ax.plot(x, y, label=sample_label, **self._fit_data_plot_kwargs())
        self._configure_plot_axes(self.canvas_fit, "fit", "Signal (V)")
        self.canvas_fit.figure.tight_layout()
        self.canvas_fit.draw()

    def _render_fit_plot(self, live: Dict[str, Any]):
        if "error" in live:
            self._show_plot_message(self.canvas_fit, str(live["error"]))
            return
        self.canvas_fit.clear()
        ax = self.canvas_fit.ax
        settings = self._plot_settings["fit"]
        sample_label = str(self._meta.get("sample") or self._meta.get("sample_id") or "Data")
        if self.chk_fit_show_data.isChecked():
            ax.plot(live["x"], live["y"], label=sample_label, **self._fit_data_plot_kwargs())
        if self.chk_fit_show_fitting.isChecked():
            ax.plot(live["x"], live["fit_curve"], linewidth=1.6, label="Fitting")
        if self.chk_fit_show_envelope.isChecked() and not self._is_wedge_scan():
            ax.plot(live["x"], live["envelope_curve"], linewidth=1.2, linestyle="--", label="Envelope")
        nominal_L = self._nominal_thickness_mm()
        delta_um = (float(live["L_value"]) - nominal_L) * 1000.0
        ax.text(
            0.02,
            0.98,
            f"L = {live['L_value']:.4f} mm (ΔL= {delta_um:+.1f} um)\n"
            f"Peak = {self._format_sigfigs(live['peak_value'], 3)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        if not settings.show_fit_annotation and ax.texts:
            ax.texts[-1].remove()
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
            self.extrema_widget.show_message(str(live["error"]))
            return
        context = self._analysis_context
        self.extrema_widget.set_plot_data(
            meta=self._meta,
            prepared_data=context["prepared_data"],
            display_x=np.asarray(live["x"], dtype=float),
            display_y=np.asarray(live["y"], dtype=float),
            fit_curve=np.asarray(live["fit_curve"], dtype=float),
            L_value=float(live["L_value"]),
            auto_minima_idx=np.asarray(extrema.get("minima_idx", []), dtype=int),
            auto_maxima_idx=np.asarray(extrema.get("maxima_idx", []), dtype=int),
            configure_axes=lambda canvas, top_axis_L_mm: self._configure_plot_axes(
                canvas,
                "extrema",
                "Signal (V)",
                top_axis_L_mm=top_axis_L_mm,
            ),
            force_reset=self._extrema_force_reset,
        )
        self._extrema_force_reset = False

    def _render_lc_plot(self, lc_info: Dict[str, Any]):
        if self._is_wedge_scan():
            self._show_plot_message(self.canvas_lc, "No data for wedge scans.")
            return
        self.canvas_lc.clear()
        ax = self.canvas_lc.ax
        if lc_info.get("skipped"):
            message = str(lc_info.get("message") or "Lc diagnostics were skipped.")
            self._show_plot_message(self.canvas_lc, f"Lc plot skipped: {message}")
            return
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
        pair_center_deg = np.asarray(aux.get("pair_center_deg", []), dtype=float)
        pair_lc_mm = np.asarray(aux.get("pair_lc_mm", []), dtype=float)

        for i in range(min(len(dL_pos), max(len(minima_pos) - 1, 0))):
            ax.plot([minima_pos[i], minima_pos[i + 1]], [1000.0 * dL_pos[i], 1000.0 * dL_pos[i]], color="C0")
        for i in range(min(len(dL_neg), max(len(minima_neg) - 1, 0))):
            ax.plot([minima_neg[i], minima_neg[i + 1]], [1000.0 * dL_neg[i], 1000.0 * dL_neg[i]], color="C1")
        finite_pairs = np.isfinite(pair_center_deg) & np.isfinite(pair_lc_mm)
        if np.any(finite_pairs):
            ax.scatter(
                pair_center_deg[finite_pairs],
                1000.0 * pair_lc_mm[finite_pairs],
                color="0.2",
                marker="o",
                s=18,
                zorder=3,
            )

        mean_lc = self._safe_float(result.get("Lc_mean_mm"))
        std_lc = self._safe_float(result.get("Lc_std_mm"))
        pair_mean_lc = self._safe_float(result.get("Lc_pair_mean_mm"))
        fit_theta_deg = np.asarray(aux.get("fit_theta_deg", []), dtype=float)
        fit_lc_mm = np.asarray(aux.get("fit_lc_mm", []), dtype=float)
        finite_fit = np.isfinite(fit_theta_deg) & np.isfinite(fit_lc_mm)
        if np.any(finite_fit):
            theta_fit = fit_theta_deg[finite_fit]
            lc_fit_um = 1000.0 * fit_lc_mm[finite_fit]
            ax.plot(theta_fit, lc_fit_um, color="C3", linewidth=1.6)
            ax.plot(-theta_fit, lc_fit_um, color="C3", linewidth=1.6)
        if np.isfinite(mean_lc):
            ax.scatter([0.0], [mean_lc * 1000.0], color="C3", marker="o", zorder=4)
            ax.axhline(mean_lc * 1000.0, color="0.3", linestyle="--", linewidth=1.0)
        ax.set_title(f"Empirical Lc(0) extrapolation from minima pairs ({source})")
        if np.isfinite(mean_lc) and np.isfinite(std_lc):
            text = f"Lc(0) = {mean_lc * 1000.0:.3f} +/- {std_lc * 1000.0:.3f} um"
        elif np.isfinite(mean_lc):
            text = f"Lc(0) = {mean_lc * 1000.0:.3f} um"
        else:
            text = "Lc summary unavailable"
        if np.isfinite(pair_mean_lc):
            text += f"\npair mean = {pair_mean_lc * 1000.0:.3f} um"
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

        ok, message = self._write_json_metadata(show_message=False, refresh_views=False)
        if not ok:
            QMessageBox.critical(self, "Update failed", message)
            return

        live = self._compute_saved_fit_curves() if self._use_saved_fit_preview else self._compute_live_curves()
        if "error" in live:
            QMessageBox.critical(self, "Manual fit failed", str(live["error"]))
            return

        if self._strategy_uses_d_rel_abs(live.get("strategy")):
            lc_info = {"error": "Lc diagnostics skipped for Braun1997 manual save."}
        else:
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

        if self._strategy_uses_d_rel_abs(live.get("strategy")):
            linear_coeff = max(float(live["peak_value"]), 0.0)
            fit_result = {
                "L_mm": float(live["L_value"]),
                "d_rel_abs": float(np.sqrt(linear_coeff)),
                "d_component": str((self._analysis_context.get("saved_fit") or {}).get("d_component") or meta.get("d_component", "")),
            }
        else:
            fit_result = {
                "L_mm": float(live["L_value"]),
                "L_mm_std": 0.0,
                "centering_pos": self._manual_centering_value(),
                "k_scale": float(live["peak_value"]),
                "k_scale_std": 0.0,
                "Pm0": float(live["peak_value"]),
                "Pm0_stderr": 0.0,
                "residual_rms": float(np.sqrt(np.mean(np.square(live["residual"])))),
            }
        existing_fit = self._fit_payload_for_strategy(meta, selected)
        if existing_fit and not self._strategy_uses_d_rel_abs(live.get("strategy")):
            fit_result = {**existing_fit, **fit_result}
        d_factor = self._safe_float(live.get("d_factor"))
        if np.isfinite(d_factor) and not self._strategy_uses_d_rel_abs(live.get("strategy")):
            fit_result["d_factor"] = float(d_factor)

        if (
            "error" not in lc_info
            and not lc_info.get("skipped")
            and not self._strategy_uses_d_rel_abs(live.get("strategy"))
        ):
            result = lc_info["result"]
            for key in (
                "Lc_mean_mm",
                "Lc_std_mm",
                "Lc_pair_mean_mm",
                "Lc_pair_std_mm",
                "lc_extrapolation_order",
                "lc_order_residual_rms",
                "minima_count",
                "n_count",
            ):
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
            prepared = self._current_prepared_data()
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
        self._df = csv_df
        self._use_saved_fit_preview = True
        self.extrema_widget.mark_saved()
        self._refresh_saved_strategy_list(meta)
        self._populate_table_from_json(meta)
        QMessageBox.information(self, "Saved", "Current L, Peak, and Centering values were written to JSON/CSV.")

    def _clear_plots(self):
        for canvas in [self.canvas_fit, self.canvas_resid, self.canvas_centering, self.canvas_lc]:
            canvas.clear()
            canvas.draw()
        self.extrema_widget.clear_plot()

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
                "extrema.png": self.extrema_widget.canvas.figure,
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
