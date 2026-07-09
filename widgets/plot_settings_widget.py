from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class SeriesPlotSettings:
    label: str
    color: str = "C0"
    style: str = "*"
    visible: bool = True
    legend_visible: bool = True


@dataclass
class ExtraAxisPlotSettings:
    key: str
    name: str
    visible: bool = True
    label: str = ""
    label_font_size: float = 10.0
    tick_font_size: float = 10.0
    axis_min: Optional[float] = None
    axis_max: Optional[float] = None
    log_scale: bool = False
    digit_count: int = -1
    scientific: bool = False
    tick_count: int = 0
    ticks_text: str = ""


@dataclass
class SharedPlotSettings:
    figure_width: float = 6.0
    figure_height: float = 2.8
    show_legend: bool = True
    show_grid: bool = True
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    font_family: str = "Arial"
    label_font_size: float = 10.0
    legend_font_size: float = 10.0
    tick_font_size: float = 10.0
    show_annotation: bool = True
    marker_size: float = 5.0
    line_width: float = 1.2
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_log: bool = False
    y_log: bool = False
    digit_count: int = -1
    x_digit_count: int = -1
    y_digit_count: int = -1
    x_scientific: bool = False
    y_scientific: bool = False
    x_tick_count: int = 0
    y_tick_count: int = 0
    x_ticks_text: str = ""
    y_ticks_text: str = ""
    colormap: str = "viridis"
    series: Dict[str, SeriesPlotSettings] = field(default_factory=dict)
    series_order: List[str] = field(default_factory=list)
    extra_axes: Dict[str, ExtraAxisPlotSettings] = field(default_factory=dict)


class PlotSettingsDialog(QDialog):
    applied = pyqtSignal()

    STYLE_OPTIONS = ["*", "*-", "^", "s", ".", "-", "--", ":", "x", "o"]
    COLOR_OPTIONS = [
        ("Blue", "C0"),
        ("Orange", "C1"),
        ("Green", "C2"),
        ("Red", "C3"),
        ("Purple", "C4"),
        ("Brown", "C5"),
        ("Gray", "0.3"),
        ("Black", "black"),
        ("White", "white"),
    ]

    def __init__(
        self,
        settings: SharedPlotSettings,
        series_defaults: List[SeriesPlotSettings],
        *,
        title: str = "Plot Settings",
        heatmap: bool = False,
        extra_axis_defaults: Optional[List[ExtraAxisPlotSettings]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 560)
        self.settings = settings
        self.series_defaults = series_defaults
        self.heatmap = heatmap
        self.extra_axis_defaults = extra_axis_defaults or []
        self._ensure_series()
        self._ensure_extra_axes()

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_text_tab(), "Text")
        tabs.addTab(self._build_data_tab(), "Data")
        tabs.addTab(self._build_axis_tab(), "Axis")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        if apply_button is not None:
            apply_button.clicked.connect(self.apply)
        layout.addWidget(buttons)

    def _ensure_series(self) -> None:
        for item in self.series_defaults:
            self.settings.series.setdefault(item.label, SeriesPlotSettings(**item.__dict__))
        known = {item.label for item in self.series_defaults}
        self.settings.series = {
            label: value for label, value in self.settings.series.items() if label in known
        }
        if not self.settings.series_order:
            self.settings.series_order = [item.label for item in self.series_defaults]
        self.settings.series_order = [
            label for label in self.settings.series_order if label in self.settings.series
        ]
        for item in self.series_defaults:
            if item.label not in self.settings.series_order:
                self.settings.series_order.append(item.label)

    def _ensure_extra_axes(self) -> None:
        for item in self.extra_axis_defaults:
            self.settings.extra_axes.setdefault(item.key, ExtraAxisPlotSettings(**item.__dict__))
        known = {item.key for item in self.extra_axis_defaults}
        self.settings.extra_axes = {
            key: value for key, value in self.settings.extra_axes.items() if key in known
        }

    def _build_general_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.figure_width = QDoubleSpinBox()
        self.figure_width.setRange(2.0, 24.0)
        self.figure_width.setSingleStep(0.2)
        self.figure_width.setValue(self.settings.figure_width)
        self.figure_height = QDoubleSpinBox()
        self.figure_height.setRange(1.5, 18.0)
        self.figure_height.setSingleStep(0.2)
        self.figure_height.setValue(self.settings.figure_height)
        self.legend = QCheckBox()
        self.legend.setChecked(self.settings.show_legend)
        self.grid = QCheckBox()
        self.grid.setChecked(self.settings.show_grid)
        form.addRow("Width", self.figure_width)
        form.addRow("Height", self.figure_height)
        form.addRow("Legend", self.legend)
        form.addRow("Grid", self.grid)
        return page

    def _build_text_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.title_edit = QLineEdit(self.settings.title)
        self.font_family = QComboBox()
        for font_name in ["Arial", "DejaVu Sans", "Times New Roman", "Calibri", "Cambria"]:
            self.font_family.addItem(font_name)
        self.font_family.setEditable(True)
        index = self.font_family.findText(self.settings.font_family)
        self.font_family.setCurrentIndex(index if index >= 0 else 0)
        self.label_size = self._font_spin(self.settings.label_font_size)
        self.legend_size = self._font_spin(self.settings.legend_font_size)
        self.tick_size = self._font_spin(self.settings.tick_font_size)
        self.annotation = QCheckBox()
        self.annotation.setChecked(self.settings.show_annotation)
        form.addRow("Title", self.title_edit)
        form.addRow("Font", self.font_family)
        form.addRow("Label size", self.label_size)
        form.addRow("Legend size", self.legend_size)
        form.addRow("Ticks size", self.tick_size)
        form.addRow("Additional text", self.annotation)

        self.extra_axis_text_widgets: Dict[str, Dict[str, object]] = {}
        form.addRow(QLabel("Axis labels"))
        self.x_label_edit = QLineEdit(self.settings.x_label)
        self.y_label_edit = QLineEdit(self.settings.y_label)
        form.addRow("X axis", self.x_label_edit)
        form.addRow("Y axis", self.y_label_edit)
        if self.extra_axis_defaults:
            form.addRow(QLabel("Extra axis labels"))
        for axis in self.extra_axis_defaults:
            current = self.settings.extra_axes[axis.key]
            enabled = QCheckBox(current.name)
            enabled.setChecked(current.visible)
            label = QLineEdit(current.label)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(enabled)
            row_layout.addWidget(QLabel("Label"))
            row_layout.addWidget(label, 1)
            form.addRow(row)
            self.extra_axis_text_widgets[axis.key] = {
                "enabled": enabled,
                "label": label,
            }
        return page

    def _font_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(5.0, 48.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setValue(value)
        return spin

    def _build_data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        if self.heatmap:
            cmap_row = QHBoxLayout()
            cmap_row.addWidget(QLabel("Colormap"))
            self.colormap = QComboBox()
            for cmap in ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "Spectral"]:
                self.colormap.addItem(cmap)
            index = self.colormap.findText(self.settings.colormap)
            self.colormap.setCurrentIndex(index if index >= 0 else 0)
            cmap_row.addWidget(self.colormap)
            cmap_row.addStretch(1)
            layout.addLayout(cmap_row)
        else:
            self.colormap = QComboBox()

        self.series_table = QTableWidget(len(self.settings.series_order), 5)
        self.series_table.setHorizontalHeaderLabels(["Data", "Color", "Style", "Show", "Legend"])
        self.series_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.series_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.series_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.series_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.series_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        for row, label in enumerate(self.settings.series_order):
            self._populate_series_row(row, label)
        layout.addWidget(self.series_table)

        order_row = QHBoxLayout()
        up_button = QPushButton("Up")
        down_button = QPushButton("Down")
        up_button.clicked.connect(lambda: self._move_selected_row(-1))
        down_button.clicked.connect(lambda: self._move_selected_row(1))
        order_row.addStretch(1)
        order_row.addWidget(up_button)
        order_row.addWidget(down_button)
        layout.addLayout(order_row)

        numeric_row = QHBoxLayout()
        numeric_row.addWidget(QLabel("Mark size"))
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0.0, 40.0)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setValue(self.settings.marker_size)
        numeric_row.addWidget(self.marker_size)
        numeric_row.addWidget(QLabel("Line width"))
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.0, 20.0)
        self.line_width.setSingleStep(0.2)
        self.line_width.setValue(self.settings.line_width)
        numeric_row.addWidget(self.line_width)
        numeric_row.addStretch(1)
        layout.addLayout(numeric_row)
        return page

    def _populate_series_row(self, row: int, label: str) -> None:
        series = self.settings.series[label]
        label_item = QTableWidgetItem(series.label)
        label_item.setData(Qt.ItemDataRole.UserRole, label)
        self.series_table.setItem(row, 0, label_item)

        color_button = QPushButton(series.color)
        color_button.clicked.connect(lambda _checked=False, r=row: self._choose_color(r))
        self._paint_color_button(color_button, series.color)
        self.series_table.setCellWidget(row, 1, color_button)

        style = QComboBox()
        style.addItems(self.STYLE_OPTIONS)
        style.setCurrentText(series.style if series.style in self.STYLE_OPTIONS else "*")
        self.series_table.setCellWidget(row, 2, style)

        show = QCheckBox()
        show.setChecked(series.visible)
        self.series_table.setCellWidget(row, 3, show)

        legend = QCheckBox()
        legend.setChecked(series.legend_visible)
        self.series_table.setCellWidget(row, 4, legend)

    def _paint_color_button(self, button: QPushButton, color: str) -> None:
        button.setText(color)
        if color.startswith("C") or color.startswith("0."):
            button.setStyleSheet("")
            return
        button.setStyleSheet(f"background-color: {QColor(color).name()};")

    def _choose_color(self, row: int) -> None:
        button = self.series_table.cellWidget(row, 1)
        if not isinstance(button, QPushButton):
            return
        current = button.text() or "black"
        initial = QColor(current) if not current.startswith("C") else QColor("black")
        color = QColorDialog.getColor(initial, self, "Choose curve color")
        if not color.isValid():
            return
        self._paint_color_button(button, color.name())

    def _move_selected_row(self, delta: int) -> None:
        row = self.series_table.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.series_table.rowCount():
            return
        self._read_series_table()
        labels = list(self.settings.series_order)
        labels[row], labels[target] = labels[target], labels[row]
        self.settings.series_order = labels
        self._rebuild_series_table()
        self.series_table.selectRow(target)

    def _rebuild_series_table(self) -> None:
        self.series_table.setRowCount(len(self.settings.series_order))
        for row, label in enumerate(self.settings.series_order):
            self._populate_series_row(row, label)

    def _build_axis_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.x_axis_widgets = self._make_axis_group(
            "X axis",
            axis_min=self.settings.x_min,
            axis_max=self.settings.x_max,
            log_scale=self.settings.x_log,
            digit_count=self.settings.x_digit_count,
            scientific=self.settings.x_scientific,
            tick_count=self.settings.x_tick_count,
            ticks_text=self.settings.x_ticks_text,
        )
        self.y_axis_widgets = self._make_axis_group(
            "Y axis",
            axis_min=self.settings.y_min,
            axis_max=self.settings.y_max,
            log_scale=self.settings.y_log,
            digit_count=self.settings.y_digit_count,
            scientific=self.settings.y_scientific,
            tick_count=self.settings.y_tick_count,
            ticks_text=self.settings.y_ticks_text,
        )
        layout.addWidget(self.x_axis_widgets["group"])
        layout.addWidget(self.y_axis_widgets["group"])
        self.extra_axis_axis_widgets: Dict[str, Dict[str, object]] = {}
        for axis in self.extra_axis_defaults:
            current = self.settings.extra_axes[axis.key]
            widgets = self._make_axis_group(
                current.name,
                axis_min=current.axis_min,
                axis_max=current.axis_max,
                log_scale=current.log_scale,
                digit_count=current.digit_count,
                scientific=current.scientific,
                tick_count=current.tick_count,
                ticks_text=current.ticks_text,
            )
            self.extra_axis_axis_widgets[axis.key] = widgets
            layout.addWidget(widgets["group"])
        layout.addStretch(1)
        return page

    def _make_axis_group(
        self,
        title: str,
        *,
        axis_min: Optional[float],
        axis_max: Optional[float],
        log_scale: bool,
        digit_count: int,
        scientific: bool,
        tick_count: int,
        ticks_text: str,
    ) -> Dict[str, object]:
        group = QGroupBox(title)
        layout = QGridLayout(group)
        min_edit = QLineEdit(self._format_bound(axis_min))
        max_edit = QLineEdit(self._format_bound(axis_max))
        log_box = QCheckBox("log")
        log_box.setChecked(log_scale)
        digits = QSpinBox()
        digits.setRange(-1, 12)
        digits.setSpecialValueText("Auto")
        digits.setValue(digit_count)
        scientific_box = QCheckBox("scientific")
        scientific_box.setChecked(scientific)
        tick_count_box = QSpinBox()
        tick_count_box.setRange(0, 30)
        tick_count_box.setSpecialValueText("Auto")
        tick_count_box.setValue(tick_count)
        ticks_edit = QLineEdit(ticks_text)
        ticks_edit.setPlaceholderText("optional: 0, 1, 2")

        layout.addWidget(QLabel("Range"), 0, 0)
        layout.addWidget(min_edit, 0, 1)
        layout.addWidget(QLabel("-"), 0, 2)
        layout.addWidget(max_edit, 0, 3)
        layout.addWidget(log_box, 0, 4)
        layout.addWidget(QLabel("Digits"), 1, 0)
        layout.addWidget(digits, 1, 1)
        layout.addWidget(scientific_box, 1, 2, 1, 2)
        layout.addWidget(QLabel("Ticks count"), 2, 0)
        layout.addWidget(tick_count_box, 2, 1)
        layout.addWidget(QLabel("Ticks list"), 3, 0)
        layout.addWidget(ticks_edit, 3, 1, 1, 4)
        layout.setColumnStretch(3, 1)
        return {
            "group": group,
            "min": min_edit,
            "max": max_edit,
            "log": log_box,
            "digits": digits,
            "scientific": scientific_box,
            "tick_count": tick_count_box,
            "ticks_text": ticks_edit,
        }

    def _format_bound(self, value: Optional[float]) -> str:
        return "" if value is None else f"{value:g}"

    def _parse_bound(self, text: str) -> Optional[float]:
        stripped = text.strip()
        return None if not stripped else float(stripped)

    def _read_series_table(self) -> None:
        order: List[str] = []
        for row in range(self.series_table.rowCount()):
            item = self.series_table.item(row, 0)
            if item is None:
                continue
            key = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            legend_label = item.text().strip() or key
            order.append(key)
            color_button = self.series_table.cellWidget(row, 1)
            style_combo = self.series_table.cellWidget(row, 2)
            show_box = self.series_table.cellWidget(row, 3)
            legend_box = self.series_table.cellWidget(row, 4)
            self.settings.series[key] = SeriesPlotSettings(
                label=legend_label,
                color=color_button.text() if isinstance(color_button, QPushButton) else "C0",
                style=style_combo.currentText() if isinstance(style_combo, QComboBox) else "*",
                visible=show_box.isChecked() if isinstance(show_box, QCheckBox) else True,
                legend_visible=legend_box.isChecked() if isinstance(legend_box, QCheckBox) else True,
            )
        self.settings.series_order = order

    def _read_form(self) -> bool:
        try:
            self._read_series_table()
            self.settings.figure_width = float(self.figure_width.value())
            self.settings.figure_height = float(self.figure_height.value())
            self.settings.show_legend = self.legend.isChecked()
            self.settings.show_grid = self.grid.isChecked()
            self.settings.title = self.title_edit.text()
            self.settings.x_label = self.x_label_edit.text()
            self.settings.y_label = self.y_label_edit.text()
            self.settings.font_family = self.font_family.currentText() or "Arial"
            self.settings.label_font_size = float(self.label_size.value())
            self.settings.legend_font_size = float(self.legend_size.value())
            self.settings.tick_font_size = float(self.tick_size.value())
            self.settings.show_annotation = self.annotation.isChecked()
            self.settings.marker_size = float(self.marker_size.value())
            self.settings.line_width = float(self.line_width.value())
            self._read_main_axis_widgets()
            if self.heatmap:
                self.settings.colormap = self.colormap.currentText() or "viridis"
            for key, widgets in self.extra_axis_text_widgets.items():
                axis = self.settings.extra_axes[key]
                enabled = widgets["enabled"]
                label = widgets["label"]
                axis.visible = enabled.isChecked() if isinstance(enabled, QCheckBox) else axis.visible
                axis.label = label.text() if isinstance(label, QLineEdit) else axis.label
                axis.label_font_size = self.settings.label_font_size
                axis.tick_font_size = self.settings.tick_font_size
            for key, widgets in self.extra_axis_axis_widgets.items():
                axis = self.settings.extra_axes[key]
                self._read_extra_axis_widgets(axis, widgets)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Invalid setting", str(exc))
            return False
        return True

    def _read_main_axis_widgets(self) -> None:
        x_axis = self._read_axis_widgets(self.x_axis_widgets)
        y_axis = self._read_axis_widgets(self.y_axis_widgets)
        self.settings.x_min = x_axis["min"]
        self.settings.x_max = x_axis["max"]
        self.settings.x_log = x_axis["log"]
        self.settings.x_digit_count = x_axis["digits"]
        self.settings.x_scientific = x_axis["scientific"]
        self.settings.x_tick_count = x_axis["tick_count"]
        self.settings.x_ticks_text = x_axis["ticks_text"]
        self.settings.y_min = y_axis["min"]
        self.settings.y_max = y_axis["max"]
        self.settings.y_log = y_axis["log"]
        self.settings.y_digit_count = y_axis["digits"]
        self.settings.y_scientific = y_axis["scientific"]
        self.settings.y_tick_count = y_axis["tick_count"]
        self.settings.y_ticks_text = y_axis["ticks_text"]
        self.settings.digit_count = max(self.settings.x_digit_count, self.settings.y_digit_count)

    def _read_extra_axis_widgets(self, axis: ExtraAxisPlotSettings, widgets: Dict[str, object]) -> None:
        values = self._read_axis_widgets(widgets)
        axis.axis_min = values["min"]
        axis.axis_max = values["max"]
        axis.log_scale = values["log"]
        axis.digit_count = values["digits"]
        axis.scientific = values["scientific"]
        axis.tick_count = values["tick_count"]
        axis.ticks_text = values["ticks_text"]

    def _read_axis_widgets(self, widgets: Dict[str, object]) -> Dict[str, object]:
        min_edit = widgets["min"]
        max_edit = widgets["max"]
        log_box = widgets["log"]
        digits = widgets["digits"]
        scientific = widgets["scientific"]
        tick_count = widgets["tick_count"]
        ticks_text = widgets["ticks_text"]
        return {
            "min": self._parse_bound(min_edit.text()) if isinstance(min_edit, QLineEdit) else None,
            "max": self._parse_bound(max_edit.text()) if isinstance(max_edit, QLineEdit) else None,
            "log": log_box.isChecked() if isinstance(log_box, QCheckBox) else False,
            "digits": int(digits.value()) if isinstance(digits, QSpinBox) else 4,
            "scientific": scientific.isChecked() if isinstance(scientific, QCheckBox) else False,
            "tick_count": int(tick_count.value()) if isinstance(tick_count, QSpinBox) else 0,
            "ticks_text": ticks_text.text() if isinstance(ticks_text, QLineEdit) else "",
        }

    def apply(self) -> None:
        if self._read_form():
            self.applied.emit()

    def accept(self) -> None:
        if not self._read_form():
            return
        super().accept()
