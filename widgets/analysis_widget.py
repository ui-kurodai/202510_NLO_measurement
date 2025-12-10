"""
Fit/Analysis Tab for SHG data (PyQt6)

Features
- Folder-based workflow: user selects a result folder under `results/` that contains exactly one CSV and one JSON.
- Unified reading: UI always refreshes by reading JSON/CSV from disk (works for both past data and just-fitted data).
- Metadata editor: material, crystal_orientation, thickness_info (t_at_thin_end_mm, wedge_angle_deg), beam_r_x/beam_r_y.
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
import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QGroupBox, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QSizePolicy, QSplitter,
    QDoubleSpinBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import logging
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


class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None, width: float = 5, height: float = 3, dpi: int = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.ax.grid(True, which="both", alpha=0.25)
        self.figure.tight_layout()

    def clear(self):
        self.ax.cla()
        self.ax.grid(True, which="both", alpha=0.25)


# ------------------------------- Main Tab -------------------------------
class FittingAnalysisWidget(QWidget):
    folderLoaded = pyqtSignal(str)   # emits folder path when a folder has been loaded
    # fitFinished = pyqtSignal(dict)   # emits result dict when a fit has finished

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
        self._last_fit_result: Optional[Dict] = None

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
        self.btn_save = QPushButton("Save Figures")
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
        self.lbl_strategy_hint = QLabel("Modules found in fitting_strategies/*")
        self.lbl_strategy_hint.setStyleSheet("color: gray;")
        strat_form.addRow("Strategy:", self.cmb_strategy)
        strat_form.addRow("", self.lbl_strategy_hint)

        # Metadata group (editable)
        meta_edit = QGroupBox("Metadata (editable before fitting)")
        f = QFormLayout(meta_edit)
        self.le_material = QLineEdit()
        self.le_crystal_orientation = QLineEdit()
        self.sb_t_thin = QDoubleSpinBox(); self.sb_t_thin.setRange(0.0, 1e6); self.sb_t_thin.setDecimals(6)
        self.sb_wedge = QDoubleSpinBox(); self.sb_wedge.setRange(-90.0, 90.0); self.sb_wedge.setDecimals(6)
        self.sb_beam_rx = QDoubleSpinBox(); self.sb_beam_rx.setRange(0.0, 1e6); self.sb_beam_rx.setDecimals(6)
        self.sb_beam_ry = QDoubleSpinBox(); self.sb_beam_ry.setRange(0.0, 1e6); self.sb_beam_ry.setDecimals(6)
        f.addRow("material:", self.le_material)
        f.addRow("crystal_orientation (e.g. 0,1,1):", self.le_crystal_orientation)
        f.addRow("t_at_thin_end_mm:", self.sb_t_thin)
        f.addRow("wedge_angle_deg:", self.sb_wedge)
        f.addRow("beam_r_x:", self.sb_beam_rx)
        f.addRow("beam_r_y:", self.sb_beam_ry)

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
        left_layout.addWidget(meta_edit)
        left_layout.addWidget(meta_view)
        left_layout.addStretch(1)

        splitter.addWidget(left)

        # Right panel: Plots + Results table
        right = QWidget(); right_layout = QVBoxLayout(right)

        self.plot_tabs = QTabWidget()
        fit_tab = QWidget(); fit_layout = QVBoxLayout(fit_tab)
        self.canvas_fit = MplCanvas(fit_tab, width=6.0, height=3.6)
        fit_layout.addWidget(self.canvas_fit)
        self.plot_tabs.addTab(fit_tab, "Data & Fit")

        resid_tab = QWidget(); resid_layout = QVBoxLayout(resid_tab)
        self.canvas_resid = MplCanvas(resid_tab, width=6.0, height=2.8)
        resid_layout.addWidget(self.canvas_resid)
        self.plot_tabs.addTab(resid_tab, "Residuals")

        right_layout.addWidget(self.plot_tabs, 3)

        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_layout.addWidget(self.tbl, 1)

        splitter.addWidget(right)
        splitter.setSizes([360, 720])
        main.addWidget(splitter, 1)

    def _connect(self):
        self.btn_open.clicked.connect(self._select_folder)
        self.btn_update_json.clicked.connect(self._update_json_clicked)
        self.btn_fit.clicked.connect(self._run_fit_clicked)
        self.btn_save.clicked.connect(self._save_figures_clicked)

    # ------------------------------- Strategies -------------------------------
    def _populate_strategy_list(self):
        """Scan fitting_strategies package for available strategies."""
        self.cmb_strategy.clear()
        self._strategies.clear()
        try:
            # pkg = importlib.import_module("fitting_strategies")
            # pkg_path = Path(pkg.__file__).parent
            # print("fitting_strategies path =", pkg_path)
            for m in pkgutil.iter_modules(["fitting_strategies"]):
                print("candidate module:", m.name, "ispkg=", m.ispkg)
                name = m.name
                if name in {"__init__", "base"}:
                    continue
                qual = f"fitting_strategies.{name}"
                # Probe for a class ending with "Strategy"
                try:
                    mod = importlib.import_module(qual)
                    cls_name = None
                    for obj_name, obj in inspect.getmembers(mod, inspect.isclass):
                        print(obj_name, obj.__module__)
                        if obj.__module__ == qual and obj_name.endswith("Strategy"):
                            cls_name = obj_name
                            break
                    if cls_name:
                        self._strategies.append(StrategyInfo(name, qual, cls_name))
                        self.cmb_strategy.addItem(f"{name}", userData=len(self._strategies)-1)
                except Exception:
                    continue
        except Exception as e:
            logging.error(f"Failed to find fitting strategies: {e}")
        # Fallback text if none found
        if self.cmb_strategy.count() == 0:
            self.cmb_strategy.addItem("(no strategies found)")

    def _get_selected_strategy(self) -> Optional[StrategyInfo]:
        idx = self.cmb_strategy.currentIndex()
        if idx < 0:
            return None
        data = self.cmb_strategy.currentData()
        if isinstance(data, int) and 0 <= data < len(self._strategies):
            return self._strategies[data]
        # If no userdata (e.g., no strategies), try a conventional default
        return None

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
        self._populate_meta_labels(meta)
        self._prefill_metadata_editors(meta)
        self._clear_plots()
        self._populate_table_from_json(meta)

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
        # thickness_info
        tinfo = meta.get("thickness_info") or {}
        try:
            self.sb_t_thin.setValue(float(tinfo.get("t_at_thin_end_mm", 0.0)))
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

    # ------------------------------ JSON update ------------------------------
    def _update_json_clicked(self):
        if not self.json_path:
            QMessageBox.information(self, "No JSON", "Load a result folder first.")
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Read failed", str(e))
            return

        # Collect values from editors
        meta["material"] = self.le_material.text().strip() or meta.get("material")
        # crystal_orientation: parse "a,b,c" -> [a,b,c]
        ori_txt = self.le_crystal_orientation.text().strip()
        if ori_txt:
            try:
                parts = [int(x) for x in ori_txt.replace(" ", "").split(",") if x != ""]
                if len(parts) == 3:
                    meta["crystal_orientation"] = parts
            except Exception:
                pass
        # thickness_info
        tinfo = meta.get("thickness_info") or {}
        tinfo["t_at_thin_end_mm"] = float(self.sb_t_thin.value())
        tinfo["wedge_angle_deg"] = float(self.sb_wedge.value())
        meta["thickness_info"] = tinfo
        # beam radii
        meta["beam_r_x"] = float(self.sb_beam_rx.value())
        meta["beam_r_y"] = float(self.sb_beam_ry.value())

        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", str(e))
            return

        self._meta = meta
        QMessageBox.information(self, "Updated", "JSON metadata updated.")
        self._populate_table_from_json(meta)

    # -------------------------------- Run fit --------------------------------
    def _run_fit_clicked(self):
        if not self._current_dir:
            QMessageBox.information(self, "No data", "Load a result folder first.")
            return
        if SHGDataAnalysis is None:
            QMessageBox.critical(self, "Missing module", "shg_analysis is not importable.")
            return
        # Ensure JSON reflects current editor values before fitting
        self._update_json_clicked()

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
            # self._last_fit_result = dict(results) if isinstance(results, dict) else {"result": str(results)}
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            return

        # Reload files from disk and refresh UI
        ok, msg = self._load_folder(self._current_dir)
        if not ok:
            QMessageBox.warning(self, "Reload failed", msg)
        self._render_from_csv()
        # self.fitFinished.emit(self._last_fit_result or {})

    # ------------------------------ Table/plots ------------------------------
    def _populate_table_from_json(self, meta: Dict):
        """Fill the result table prioritizing important fit keys if present."""
        rows = []
        # Priority metrics if present (after fitting)
        for key, label in [
            ("L_mm", "Corrected L [mm]"),
            ("Lc_mean_mm", "Lc mean [mm]"),
            ("Pm0", "Peak Pm0"),
            ("residual_rms", "Residual RMS"),
        ]:
            if key in meta:
                rows.append((label, meta.get(key)))

        # Add more from meta if needed (std, counts, etc.)
        for k in [
            "L_mm_std", "k_scale", "k_scale_std", "Lc_std_mm", "minima_count",
            "n_count", "Pm0_stderr", "n_peaks"
        ]:
            if k in meta:
                rows.append((k, meta.get(k)))

        # Basic info for context
        for k in ["sample", "material", "crystal_orientation", "method", "timestamp"]:
            if k in meta:
                rows.append((k, meta.get(k)))

        self.tbl.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.tbl.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl.setItem(i, 1, QTableWidgetItem("" if v is None else f"{v}"))

    def _render_from_csv(self):
        if self._df is None:
            self._clear_plots(); return
        df = self._df

        if all(col in df.columns for col in ("position", "ch2")):
            x = df["position"].values
            y = df["ch2"].values
            yfit = df["fit"].values if "fit" in df.columns else None

            # Data & Fit plot
            self.canvas_fit.clear()
            sample_label = str(self._meta.get("sample") or self._meta.get("sample_id") or "ch2")
            self.canvas_fit.ax.plot(x, y, linestyle="none", marker="o", markersize=3, label=sample_label)
            if yfit is not None:
                self.canvas_fit.ax.plot(x, yfit, linewidth=1.6, label="fitting")
            self.canvas_fit.ax.set_xlabel("position")
            self.canvas_fit.ax.set_ylabel("Signal")
            self.canvas_fit.ax.legend(loc="best")
            self.canvas_fit.figure.tight_layout(); self.canvas_fit.draw()


            # Residuals plot: fit - ch2
            self.canvas_resid.clear()
            if yfit is not None:
                r = yfit - y
                self.canvas_resid.ax.plot(x, r, linestyle="none", marker=".", markersize=3)
            self.canvas_resid.ax.axhline(0.0, linewidth=1.0)
            self.canvas_resid.ax.set_xlabel("position")
            self.canvas_resid.ax.set_ylabel("Residual (fit - ch2)")
            self.canvas_resid.figure.tight_layout(); self.canvas_resid.draw()
            return
    
    def _clear_plots(self):
        self.canvas_fit.clear(); self.canvas_fit.draw()
        self.canvas_resid.clear(); self.canvas_resid.draw()

    # --------------------------------- Save ---------------------------------
    def _save_figures_clicked(self):
        if not self._current_dir:
            QMessageBox.information(self, "No folder", "Load a result folder first.")
            return
        try:
            fit_png = self._current_dir / "fit_overlay.png"
            resid_png = self._current_dir / "residuals.png"
            self.canvas_fit.figure.savefig(fit_png, dpi=200, bbox_inches="tight")
            self.canvas_resid.figure.savefig(resid_png, dpi=200, bbox_inches="tight")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Saved", "Figures saved as fit_overlay.png and residuals.png.")


# --------------------------- Standalone test hook ---------------------------
# if __name__ == "__main__":  # optional manual test
#     from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
#     app = QApplication(sys.argv)
#     win = QMainWindow(); win.setWindowTitle("Demo — Fit/Analysis Tab")
#     tabs = QTabWidget(); tabs.addTab(FitAnalyzeTab(), "Analysis")
#     win.setCentralWidget(tabs)
#     win.resize(1120, 720); win.show()
#     sys.exit(app.exec())
