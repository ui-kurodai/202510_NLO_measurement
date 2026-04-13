from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QMessageBox,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector


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


class ManualExtremaDetectionWidget(QWidget):
    extremaChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.canvas = MplCanvas(self, width=6.0, height=2.8)
        self._configure_axes: Optional[Callable[[MplCanvas, Optional[float]], None]] = None
        self._meta: Dict[str, Any] = {}
        self._prepared_data: Optional[pd.DataFrame] = None
        self._display_x = np.array([], dtype=float)
        self._display_y = np.array([], dtype=float)
        self._fit_curve = np.array([], dtype=float)
        self._L_value: Optional[float] = None
        self._auto_minima_idx = np.array([], dtype=int)
        self._auto_maxima_idx = np.array([], dtype=int)
        self._saved_minima_idx: Optional[np.ndarray] = None
        self._saved_maxima_idx: Optional[np.ndarray] = None
        self._editor: Dict[str, Any] = {
            "initialized": False,
            "data_length": 0,
            "minima_idx": np.array([], dtype=int),
            "maxima_idx": np.array([], dtype=int),
            "selected_kind": None,
            "selected_index": None,
            "zoom_limits": None,
            "dirty": False,
            "source": "auto",
        }
        self._pick_targets: Dict[Any, tuple[str, np.ndarray]] = {}
        self._selector: Optional[RectangleSelector] = None

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.btn_add_min = QPushButton("Add Min in View")
        self.btn_add_max = QPushButton("Add Max in View")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_reset = QPushButton("Reset to Detected")
        self.btn_reset_zoom = QPushButton("Reset Zoom")
        toolbar.addWidget(self.btn_add_min)
        toolbar.addWidget(self.btn_add_max)
        toolbar.addWidget(self.btn_remove)
        toolbar.addWidget(self.btn_reset)
        toolbar.addWidget(self.btn_reset_zoom)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        layout.addWidget(self.canvas)
        self.lbl_hint = QLabel(
            "Left-click a marker to select it. Left-drag to zoom. "
            "Use Update JSON to save minima/maxima."
        )
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_hint)

        self.btn_add_min.clicked.connect(lambda: self._add_extremum_from_view("minima"))
        self.btn_add_max.clicked.connect(lambda: self._add_extremum_from_view("maxima"))
        self.btn_remove.clicked.connect(self._remove_selected_extremum)
        self.btn_reset.clicked.connect(self._reset_to_detected)
        self.btn_reset_zoom.clicked.connect(self._reset_zoom)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self._set_status()

    def clear_plot(self):
        self.canvas.clear()
        self.canvas.draw()
        self._set_status("Load valid data to inspect extrema.")

    def show_message(self, message: str):
        self.canvas.clear()
        self.canvas.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.canvas.ax.transAxes,
            wrap=True,
        )
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self._set_status(message)

    def set_plot_data(
        self,
        *,
        meta: Dict[str, Any],
        prepared_data: pd.DataFrame,
        display_x: np.ndarray,
        display_y: np.ndarray,
        fit_curve: np.ndarray,
        L_value: Optional[float],
        auto_minima_idx: np.ndarray,
        auto_maxima_idx: np.ndarray,
        configure_axes: Callable[[MplCanvas, Optional[float]], None],
        force_reset: bool = False,
    ):
        self._meta = dict(meta)
        self._prepared_data = prepared_data.copy()
        self._display_x = np.asarray(display_x, dtype=float)
        self._display_y = np.asarray(display_y, dtype=float)
        self._fit_curve = np.asarray(fit_curve, dtype=float)
        self._L_value = None if L_value is None else float(L_value)
        self._auto_minima_idx = self._normalize_indices(auto_minima_idx, len(self._display_x))
        self._auto_maxima_idx = self._normalize_indices(auto_maxima_idx, len(self._display_x))
        self._configure_axes = configure_axes
        self._prepare_state(auto_minima_idx, auto_maxima_idx, force_reset=force_reset)
        self._render()

    def merge_into_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(meta)
        if not self._editor.get("initialized"):
            return payload
        payload["minima"] = [int(index) for index in np.asarray(self._editor["minima_idx"], dtype=int)]
        payload["maxima"] = [int(index) for index in np.asarray(self._editor["maxima_idx"], dtype=int)]
        return payload

    def mark_saved(self):
        if not self._editor.get("initialized"):
            return
        self._saved_minima_idx = np.asarray(self._editor["minima_idx"], dtype=int).copy()
        self._saved_maxima_idx = np.asarray(self._editor["maxima_idx"], dtype=int).copy()
        self._editor["dirty"] = False
        self._editor["source"] = "saved"
        self._set_status()

    def saved_minima_indices(self) -> Optional[np.ndarray]:
        if self._saved_minima_idx is None:
            return None
        return np.asarray(self._saved_minima_idx, dtype=int).copy()

    def _prepare_state(self, auto_minima_idx: np.ndarray, auto_maxima_idx: np.ndarray, *, force_reset: bool):
        length = len(self._display_x)
        self._saved_minima_idx = self._load_saved_extrema_indices("minima", length)
        self._saved_maxima_idx = self._load_saved_extrema_indices("maxima", length)
        needs_init = (
            force_reset
            or not bool(self._editor.get("initialized"))
            or int(self._editor.get("data_length", -1)) != length
        )
        if not needs_init:
            return

        minima_idx = self._saved_minima_idx if self._saved_minima_idx is not None else self._normalize_indices(auto_minima_idx, length)
        maxima_idx = self._saved_maxima_idx if self._saved_maxima_idx is not None else self._normalize_indices(auto_maxima_idx, length)
        self._editor.update(
            {
                "initialized": True,
                "data_length": length,
                "minima_idx": minima_idx,
                "maxima_idx": maxima_idx,
                "selected_kind": None,
                "selected_index": None,
                "zoom_limits": None,
                "dirty": False,
                "source": "saved" if self._saved_minima_idx is not None or self._saved_maxima_idx is not None else "auto",
            }
        )

    def _raw_x(self) -> np.ndarray:
        if isinstance(self._prepared_data, pd.DataFrame):
            if "position" in self._prepared_data.columns:
                return np.asarray(self._prepared_data["position"], dtype=float)
            if "angle_deg" in self._prepared_data.columns:
                return np.asarray(self._prepared_data["angle_deg"], dtype=float)
            if "position_mm" in self._prepared_data.columns:
                return np.asarray(self._prepared_data["position_mm"], dtype=float)
        return np.asarray(self._display_x, dtype=float)

    def _load_saved_extrema_indices(self, kind: str, length: int) -> Optional[np.ndarray]:
        raw = self._meta.get(kind)
        if not isinstance(raw, list):
            return None

        raw_x = self._raw_x()
        centered_x = np.asarray(self._display_x, dtype=float)
        loaded: List[int] = []
        for item in raw:
            if isinstance(item, dict):
                index_value = item.get("index")
                try:
                    index = int(index_value)
                except Exception:
                    index = None
                if index is None:
                    for key, axis in (("position", raw_x), ("position_centered", centered_x)):
                        try:
                            target = float(item.get(key))
                        except Exception:
                            continue
                        if axis.size == 0 or not np.isfinite(target):
                            continue
                        index = int(np.argmin(np.abs(axis - target)))
                        break
                if index is not None:
                    loaded.append(index)
                continue
            try:
                loaded.append(int(item))
            except Exception:
                continue
        return self._normalize_indices(loaded, length)

    def _normalize_indices(self, indices: Any, length: int) -> np.ndarray:
        if indices is None:
            iterable: List[Any] = []
        elif isinstance(indices, np.ndarray):
            iterable = np.asarray(indices).reshape(-1).tolist()
        elif isinstance(indices, (list, tuple, set)):
            iterable = list(indices)
        else:
            iterable = [indices]

        normalized: List[int] = []
        for item in iterable:
            try:
                index = int(item)
            except Exception:
                continue
            if 0 <= index < length:
                normalized.append(index)
        if not normalized:
            return np.array([], dtype=int)
        return np.asarray(sorted(set(normalized)), dtype=int)

    def _set_status(self, message: Optional[str] = None):
        has_data = bool(self._editor.get("initialized")) and len(self._display_x) > 0
        self.btn_add_min.setEnabled(has_data)
        self.btn_add_max.setEnabled(has_data)
        self.btn_reset.setEnabled(has_data)
        self.btn_reset_zoom.setEnabled(has_data and bool(self._editor.get("zoom_limits")))
        selected_kind = self._editor.get("selected_kind")
        selected_index = self._editor.get("selected_index")
        self.btn_remove.setEnabled(has_data and selected_kind in {"minima", "maxima"} and selected_index is not None)

        if message is not None:
            self.lbl_hint.setText(message)
            return

        if not has_data:
            self.lbl_hint.setText("Load valid data to inspect extrema.")
            return

        minima_count = len(np.asarray(self._editor["minima_idx"], dtype=int))
        maxima_count = len(np.asarray(self._editor["maxima_idx"], dtype=int))
        source = str(self._editor.get("source") or "auto")
        dirty = bool(self._editor.get("dirty"))
        selected_text = ""
        if selected_kind in {"minima", "maxima"} and selected_index is not None:
            selected_text = f" Selected: {selected_kind[:-1]} @ index {int(selected_index)}."
        save_text = " Unsaved edits. Use Update JSON to store them." if dirty else " Update JSON writes minima/maxima."
        self.lbl_hint.setText(
            f"Left-click a marker to select it. Left-drag to zoom. "
            f"Minima={minima_count}, Maxima={maxima_count} ({source}).{selected_text}{save_text}"
        )

    def _current_view_indices(self) -> np.ndarray:
        if self._display_x.size == 0:
            return np.array([], dtype=int)
        x0, x1 = self.canvas.ax.get_xlim()
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        mask = np.isfinite(self._display_x) & (self._display_x >= lo) & (self._display_x <= hi)
        return np.flatnonzero(mask)

    def _add_extremum(self, kind: str, index: int):
        if kind not in {"minima", "maxima"}:
            return
        length = len(self._display_x)
        if not (0 <= int(index) < length):
            return

        other_kind = "maxima" if kind == "minima" else "minima"
        target_key = f"{kind}_idx"
        other_key = f"{other_kind}_idx"
        updated = list(np.asarray(self._editor.get(target_key, []), dtype=int))
        updated.append(int(index))
        self._editor[target_key] = self._normalize_indices(updated, length)
        other = [value for value in np.asarray(self._editor.get(other_key, []), dtype=int) if int(value) != int(index)]
        self._editor[other_key] = self._normalize_indices(other, length)
        self._editor["selected_kind"] = kind
        self._editor["selected_index"] = int(index)
        self._editor["dirty"] = True
        self._editor["source"] = "manual"
        self._render()
        self.extremaChanged.emit()

    def _add_extremum_from_view(self, kind: str):
        view_indices = self._current_view_indices()
        if view_indices.size == 0:
            QMessageBox.information(self, "No points", "No data points are visible in the current extrema view.")
            return

        visible_y = self._display_y[view_indices]
        finite = np.isfinite(visible_y)
        if not np.any(finite):
            QMessageBox.information(self, "No finite points", "Visible points do not contain finite intensity values.")
            return

        finite_indices = view_indices[finite]
        finite_y = visible_y[finite]
        chosen = finite_indices[np.argmin(finite_y)] if kind == "minima" else finite_indices[np.argmax(finite_y)]
        self._add_extremum(kind, int(chosen))

    def _remove_selected_extremum(self):
        kind = self._editor.get("selected_kind")
        selected_index = self._editor.get("selected_index")
        if kind not in {"minima", "maxima"} or selected_index is None:
            return

        key = f"{kind}_idx"
        remaining = [
            int(value)
            for value in np.asarray(self._editor.get(key, []), dtype=int)
            if int(value) != int(selected_index)
        ]
        self._editor[key] = self._normalize_indices(remaining, len(self._display_x))
        self._editor["selected_kind"] = None
        self._editor["selected_index"] = None
        self._editor["dirty"] = True
        self._editor["source"] = "manual"
        self._render()
        self.extremaChanged.emit()

    def _reset_to_detected(self):
        if self._display_x.size == 0:
            return
        self._editor.update(
            {
                "minima_idx": np.asarray(self._auto_minima_idx, dtype=int).copy(),
                "maxima_idx": np.asarray(self._auto_maxima_idx, dtype=int).copy(),
                "selected_kind": None,
                "selected_index": None,
                "zoom_limits": None,
                "dirty": False,
                "source": "auto",
            }
        )
        self._render()
        self.extremaChanged.emit()

    def _reset_zoom(self):
        if not self._editor.get("zoom_limits"):
            return
        self._editor["zoom_limits"] = None
        self._render()

    def _on_pick(self, event):
        target = self._pick_targets.get(event.artist)
        if target is None:
            return
        kind, indices = target
        picked_candidates = getattr(event, "ind", None)
        if picked_candidates is None or len(picked_candidates) == 0:
            return
        picked = int(np.asarray(indices, dtype=int)[int(picked_candidates[0])])
        self._editor["selected_kind"] = kind
        self._editor["selected_index"] = picked
        self._render()

    def _on_zoom_selected(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None or eclick.ydata is None or erelease.ydata is None:
            return
        if abs(float(erelease.xdata) - float(eclick.xdata)) < 1e-12:
            return
        if abs(float(erelease.ydata) - float(eclick.ydata)) < 1e-12:
            return
        self._editor["zoom_limits"] = {
            "xlim": tuple(sorted((float(eclick.xdata), float(erelease.xdata)))),
            "ylim": tuple(sorted((float(eclick.ydata), float(erelease.ydata)))),
        }
        self._render()

    def _install_selector(self):
        if self._selector is not None:
            try:
                self._selector.set_active(False)
                self._selector.disconnect_events()
            except Exception:
                pass
        self._selector = RectangleSelector(
            self.canvas.ax,
            self._on_zoom_selected,
            useblit=False,
            button=[1],
            spancoords="pixels",
            minspanx=6,
            minspany=6,
            interactive=False,
        )

    def _render(self):
        self.canvas.clear()
        self._pick_targets = {}
        ax = self.canvas.ax
        ax.plot(self._display_x, self._display_y, linewidth=1.0, label="Data")
        ax.plot(self._display_x, self._fit_curve, linewidth=1.2, alpha=0.7, label="Current fit")

        minima_idx = np.asarray(self._editor.get("minima_idx", []), dtype=int)
        maxima_idx = np.asarray(self._editor.get("maxima_idx", []), dtype=int)
        minima_artist = ax.plot(
            self._display_x[minima_idx] if minima_idx.size else np.array([], dtype=float),
            self._display_y[minima_idx] if minima_idx.size else np.array([], dtype=float),
            linestyle="none",
            marker="*",
            ms=9,
            label="* Minima",
            picker=8,
        )[0]
        maxima_artist = ax.plot(
            self._display_x[maxima_idx] if maxima_idx.size else np.array([], dtype=float),
            self._display_y[maxima_idx] if maxima_idx.size else np.array([], dtype=float),
            linestyle="none",
            marker="o",
            ms=5,
            label="o Maxima",
            picker=8,
        )[0]
        self._pick_targets[minima_artist] = ("minima", minima_idx)
        self._pick_targets[maxima_artist] = ("maxima", maxima_idx)

        selected_kind = self._editor.get("selected_kind")
        selected_index = self._editor.get("selected_index")
        if selected_kind in {"minima", "maxima"} and selected_index is not None:
            selected_index = int(selected_index)
            if 0 <= selected_index < len(self._display_x):
                ax.plot(
                    [self._display_x[selected_index]],
                    [self._display_y[selected_index]],
                    linestyle="none",
                    marker="s",
                    ms=11,
                    markerfacecolor="none",
                    markeredgecolor="C3",
                    markeredgewidth=1.5,
                    label="Selected",
                )

        if callable(self._configure_axes):
            self._configure_axes(self.canvas, self._L_value)

        zoom_limits = self._editor.get("zoom_limits")
        if isinstance(zoom_limits, dict):
            xlim = zoom_limits.get("xlim")
            ylim = zoom_limits.get("ylim")
            if isinstance(xlim, tuple) and len(xlim) == 2:
                ax.set_xlim(xlim)
            if isinstance(ylim, tuple) and len(ylim) == 2:
                ax.set_ylim(ylim)

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self._install_selector()
        self._set_status()
