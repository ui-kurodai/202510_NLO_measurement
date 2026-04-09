from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from comparison_utils import (
    ComparisonResult,
    compare_experiment_folders,
    extract_measurement_summary,
    load_single_measurement_json,
    write_comparison_results,
)
from fitting_results import normalize_fitting_entries
from windows_dialogs import select_multiple_directories


class ComparisonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reference_root: Path | None = None
        self.target_roots: list[Path] = []
        self._results: list[ComparisonResult] = []
        self._row_enabled: dict[str, bool] = {}
        self._syncing_selection = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.btn_select_reference = QPushButton("Select Reference Folder...")
        self.btn_select_target = QPushButton("Select Target Folders...")
        self.btn_compare = QPushButton("Compare")
        self.btn_write_json = QPushButton("Write to Target JSON")
        self.btn_write_json.setEnabled(False)
        button_row.addWidget(self.btn_select_reference)
        button_row.addWidget(self.btn_select_target)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_compare)
        button_row.addWidget(self.btn_write_json)
        layout.addLayout(button_row)

        config_group = QGroupBox("Comparison Setup")
        config_form = QFormLayout(config_group)
        self.lbl_reference_folder = QLabel("(not selected)")
        self.lbl_target_folders = QLabel("(not selected)")
        self.lbl_reference_sample = QLabel("(not loaded)")
        self.lbl_reference_thickness = QLabel("(not loaded)")
        self.lbl_reference_in_pol = QLabel("(not loaded)")
        self.lbl_reference_out_pol = QLabel("(not loaded)")
        self.lbl_reference_strategy = QLabel("(not loaded)")
        self.lbl_reference_folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_target_folders.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_target_folders.setWordWrap(True)
        self.sb_reference_d = QDoubleSpinBox()
        self.sb_reference_d.setRange(-1e6, 1e6)
        self.sb_reference_d.setDecimals(6)
        self.sb_reference_d.setSingleStep(0.01)
        self.sb_reference_d.setValue(0.3)
        config_form.addRow("Reference folder:", self.lbl_reference_folder)
        config_form.addRow("Reference sample:", self.lbl_reference_sample)
        config_form.addRow("Reference thickness:", self.lbl_reference_thickness)
        config_form.addRow("Reference input pol.:", self.lbl_reference_in_pol)
        config_form.addRow("Reference detected pol.:", self.lbl_reference_out_pol)
        config_form.addRow("Reference fitting:", self.lbl_reference_strategy)
        config_form.addRow("Target folder(s):", self.lbl_target_folders)
        config_form.addRow("Reference d coefficient:", self.sb_reference_d)
        layout.addWidget(config_group)

        splitter = QSplitter(Qt.Orientation.Vertical)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_summary = QLabel("Load two experiment folders and run comparison.")
        table_layout.addWidget(self.lbl_summary)
        table_split = QHBoxLayout()
        table_split.setContentsMargins(0, 0, 0, 0)
        table_split.setSpacing(0)

        self.fixed_table = QTableWidget(0, 3)
        self.fixed_table.setHorizontalHeaderLabels(["Show", "Target holder", "Strategy"])
        self.fixed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.fixed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fixed_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fixed_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fixed_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.fixed_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.fixed_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.fixed_table.setMinimumWidth(420)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Target sample",
                "Peak type",
                "Peak ref",
                "Peak target",
                "d_factor ref",
                "d_factor target",
                "Filter diff",
                "I_target/I_ref",
                "d_target/d_ref",
                "calculated_d",
                "Status",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        self.fixed_table.verticalScrollBar().valueChanged.connect(self.table.verticalScrollBar().setValue)
        self.table.verticalScrollBar().valueChanged.connect(self.fixed_table.verticalScrollBar().setValue)
        self.fixed_table.currentCellChanged.connect(self._sync_table_selection_from_fixed)
        self.table.currentCellChanged.connect(self._sync_table_selection_from_scroll)

        table_split.addWidget(self.fixed_table, 0)
        table_split.addWidget(self.table, 1)
        table_layout.addLayout(table_split, 1)
        splitter.addWidget(table_panel)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Log"))
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output, 1)
        splitter.addWidget(log_panel)
        splitter.setSizes([420, 180])

        layout.addWidget(splitter, 1)

        self.btn_select_reference.clicked.connect(self._select_reference_folder)
        self.btn_select_target.clicked.connect(self._select_target_folder)
        self.btn_compare.clicked.connect(self._run_comparison)
        self.btn_write_json.clicked.connect(self._write_results)

    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    def _select_reference_folder(self) -> None:
        start_dir = str(self.reference_root) if self.reference_root else "results"
        folder = QFileDialog.getExistingDirectory(self, "Select reference experiment folder", start_dir)
        if not folder:
            return
        self.reference_root = Path(folder)
        self.lbl_reference_folder.setText(str(self.reference_root))
        self._refresh_reference_info()

    def _select_target_folder(self) -> None:
        start_dir = str(self.target_roots[0]) if self.target_roots else "results"
        folders = self._select_target_directories_native(start_dir)
        if not folders:
            return
        self.target_roots = folders
        self.lbl_target_folders.setText("\n".join(str(path) for path in self.target_roots))

    def _run_comparison(self) -> None:
        self.log_output.clear()
        self.table.setRowCount(0)
        self._results = []
        self._row_enabled.clear()
        self.btn_write_json.setEnabled(False)

        if self.reference_root is None or not self.target_roots:
            QMessageBox.information(self, "Missing target", "Select one reference folder and at least one target folder first.")
            return

        reference_d = float(self.sb_reference_d.value())
        all_results: list[ComparisonResult] = []
        all_warnings: list[str] = []
        for target_root in self.target_roots:
            results, warnings = compare_experiment_folders(
                reference_root=self.reference_root,
                target_root=target_root,
                reference_d_value=reference_d,
            )
            all_results.extend(results)
            for warning in warnings:
                all_warnings.append(f"{target_root.name}: {warning}")
        self._results = all_results
        self._populate_table(all_results)

        ready_count = sum(1 for result in all_results if result.error is None)
        self.lbl_summary.setText(
            f"Compared {len(self.target_roots)} target folder(s) | results {len(all_results)} | ready {ready_count} | reference d = {reference_d:g}"
        )

        for warning in all_warnings:
            self._append_log(f"WARNING: {warning}")
        for result in all_results:
            strategy_text = self._strategy_text(result)
            self._append_log(f"{result.target_json_path.parent.name} [{strategy_text}]: {result.status_text}")

        if all_results:
            self.btn_write_json.setEnabled(any(result.error is None for result in all_results))
        elif all_warnings:
            QMessageBox.information(self, "No results", "\n".join(all_warnings))

    def _populate_table(self, results: list[ComparisonResult]) -> None:
        for result in results:
            self._row_enabled.setdefault(result.key, True)

        ordered_results = sorted(
            results,
            key=lambda result: (
                0 if self._row_enabled.get(result.key, True) else 1,
                result.target_json_path.parent.name.lower(),
                result.target_sample.lower(),
                result.target_strategy.lower(),
            ),
        )

        self.fixed_table.setRowCount(len(ordered_results))
        self.table.setRowCount(len(ordered_results))

        group_colors = [
            QColor(255, 255, 255),
            QColor(247, 250, 255),
        ]
        group_index = -1
        previous_group = None

        for row_index, result in enumerate(ordered_results):
            current_group = str(result.target_json_path)
            if current_group != previous_group:
                group_index += 1
                previous_group = current_group
            enabled = self._row_enabled.get(result.key, True)
            base_color = group_colors[group_index % len(group_colors)]
            if not enabled:
                base_color = QColor(242, 242, 242)

            checkbox = QCheckBox()
            checkbox.setChecked(enabled)
            checkbox.stateChanged.connect(
                lambda state, key=result.key: self._toggle_result_visibility(key, state == Qt.CheckState.Checked.value)
            )
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.addWidget(checkbox)
            self.fixed_table.setCellWidget(row_index, 0, checkbox_widget)

            fixed_values = [
                result.target_json_path.parent.name,
                self._strategy_text(result),
            ]
            for column_index, value in enumerate(fixed_values, start=1):
                item = QTableWidgetItem(value)
                self._style_item(item, base_color, enabled)
                self.fixed_table.setItem(row_index, column_index, item)

            values = [
                result.target_sample,
                result.peak_label,
                self._fmt(result.peak_ref),
                self._fmt(result.peak_target),
                self._fmt(result.d_factor_ref),
                self._fmt(result.d_factor_target),
                result.differing_filters_text,
                self._fmt(result.intensity_ratio),
                self._fmt(result.d_ratio),
                self._fmt(result.calculated_d),
                self._status_text(result, enabled),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                self._style_item(item, base_color, enabled)
                self.table.setItem(row_index, column_index, item)

            row_height = max(self.fixed_table.verticalHeader().defaultSectionSize(), 28)
            self.fixed_table.setRowHeight(row_index, row_height)
            self.table.setRowHeight(row_index, row_height)

        self.fixed_table.resizeColumnsToContents()
        self.table.resizeColumnsToContents()
        self.fixed_table.setColumnWidth(1, max(self.fixed_table.columnWidth(1), 180))
        self.fixed_table.setColumnWidth(2, max(self.fixed_table.columnWidth(2), 220))

    def _write_results(self) -> None:
        if self.reference_root is None:
            QMessageBox.information(self, "Missing reference", "Select a reference folder first.")
            return
        if not self._results:
            QMessageBox.information(self, "No results", "Run comparison first.")
            return

        written, skipped, warnings = write_comparison_results(self.reference_root, self._results)
        for warning in warnings:
            self._append_log(f"WARNING: {warning}")
        self._append_log(f"WRITE DONE | updated={written} | skipped={skipped}")
        QMessageBox.information(
            self,
            "Completed",
            f"Updated: {written}\nSkipped: {skipped}",
        )

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.6g}"

    def _strategy_text(self, result: ComparisonResult) -> str:
        return result.target_strategy or "(legacy)"

    def _refresh_reference_info(self) -> None:
        self.lbl_reference_sample.setText("(unavailable)")
        self.lbl_reference_thickness.setText("(unavailable)")
        self.lbl_reference_in_pol.setText("(unavailable)")
        self.lbl_reference_out_pol.setText("(unavailable)")
        self.lbl_reference_strategy.setText("(unavailable)")
        if self.reference_root is None:
            return

        meta, _json_path, warnings = load_single_measurement_json(self.reference_root)
        if warnings:
            self._append_log(f"WARNING: {self.reference_root.name}: " + " | ".join(warnings))
        if meta is None:
            return

        summary = extract_measurement_summary(meta)
        self.lbl_reference_sample.setText(summary["sample"] or "(empty)")
        self.lbl_reference_thickness.setText(summary["thickness"] or "(empty)")
        self.lbl_reference_in_pol.setText(summary["input_polarization"] or "(empty)")
        self.lbl_reference_out_pol.setText(summary["detected_polarization"] or "(empty)")
        fitting_entries = normalize_fitting_entries(meta)
        if fitting_entries:
            strategy_names = [str(entry.get("strategy") or "(legacy)") for entry in fitting_entries]
            self.lbl_reference_strategy.setText(", ".join(strategy_names))
        else:
            self.lbl_reference_strategy.setText("(no saved fit)")

    def _select_target_directories_native(self, start_dir: str) -> list[Path]:
        try:
            return select_multiple_directories(
                parent_hwnd=int(self.window().winId()) if self.window() is not None else None,
                title="Select target experiment folders",
                initial_dir=start_dir,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Selection failed", f"Failed to open native folder picker: {exc}")
            return []

    def _toggle_result_visibility(self, key: str, enabled: bool) -> None:
        self._row_enabled[key] = enabled
        self._populate_table(self._results)

    def _style_item(self, item: QTableWidgetItem, background: QColor, enabled: bool) -> None:
        item.setBackground(QBrush(background))
        if enabled:
            item.setForeground(QBrush(QColor(20, 20, 20)))
        else:
            item.setForeground(QBrush(QColor(150, 150, 150)))

    def _status_text(self, result: ComparisonResult, enabled: bool) -> str:
        prefix = "" if enabled else "[hidden] "
        return prefix + result.status_text

    def _sync_table_selection_from_fixed(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        del current_column, previous_row, previous_column
        if self._syncing_selection or current_row < 0:
            return
        self._syncing_selection = True
        try:
            self.table.setCurrentCell(current_row, 0)
        finally:
            self._syncing_selection = False

    def _sync_table_selection_from_scroll(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        del current_column, previous_row, previous_column
        if self._syncing_selection or current_row < 0:
            return
        self._syncing_selection = True
        try:
            self.fixed_table.setCurrentCell(current_row, 1)
        finally:
            self._syncing_selection = False
