from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
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


@dataclass
class ComparisonFilterSettings:
    allowed_strategy_labels: set[str] = field(default_factory=set)
    exclude_missing_filter_csv: bool = False


class ComparisonFilterDialog(QDialog):
    applied = pyqtSignal()

    def __init__(
        self,
        settings: ComparisonFilterSettings,
        available_strategy_labels: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filtering")
        self.resize(520, 420)
        self._settings = ComparisonFilterSettings(
            allowed_strategy_labels=set(settings.allowed_strategy_labels),
            exclude_missing_filter_csv=settings.exclude_missing_filter_csv,
        )
        self._available_strategy_labels = list(available_strategy_labels)
        self._strategy_checkboxes: dict[str, QCheckBox] = {}
        self._build_ui()

    @property
    def filter_settings(self) -> ComparisonFilterSettings:
        return ComparisonFilterSettings(
            allowed_strategy_labels=set(self._settings.allowed_strategy_labels),
            exclude_missing_filter_csv=self._settings.exclude_missing_filter_csv,
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_strategy_tab(), "Strategy")
        tabs.addTab(self._build_filter_diff_tab(), "Filter diff")
        layout.addWidget(tabs)

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

    def _build_strategy_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Choose strategies to keep near the top. If none are checked, all strategies are allowed."))

        action_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_clear_all = QPushButton("Clear All")
        btn_select_all.clicked.connect(self._select_all_strategies)
        btn_clear_all.clicked.connect(self._clear_all_strategies)
        action_row.addWidget(btn_select_all)
        action_row.addWidget(btn_clear_all)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        strategy_labels = list(dict.fromkeys(self._available_strategy_labels))
        if not strategy_labels:
            container_layout.addWidget(QLabel("No compared strategies yet. Run Compare once to populate this list."))
        for strategy_label in strategy_labels:
            checkbox = QCheckBox(strategy_label)
            checkbox.setChecked(strategy_label in self._settings.allowed_strategy_labels)
            self._strategy_checkboxes[strategy_label] = checkbox
            container_layout.addWidget(checkbox)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        return page

    def _build_filter_diff_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.chk_exclude_missing_filter_csv = QCheckBox(
            "Demote rows whose filter diff contains filters without transmission CSV"
        )
        self.chk_exclude_missing_filter_csv.setChecked(self._settings.exclude_missing_filter_csv)
        layout.addWidget(self.chk_exclude_missing_filter_csv)
        layout.addWidget(
            QLabel(
                "Rows that fail this condition stay visible, but move to the gray lower section just like unchecked rows."
            )
        )
        layout.addStretch(1)
        return page

    def _select_all_strategies(self) -> None:
        for checkbox in self._strategy_checkboxes.values():
            checkbox.setChecked(True)

    def _clear_all_strategies(self) -> None:
        for checkbox in self._strategy_checkboxes.values():
            checkbox.setChecked(False)

    def _apply_to_internal_settings(self) -> None:
        self._settings.allowed_strategy_labels = {
            label for label, checkbox in self._strategy_checkboxes.items() if checkbox.isChecked()
        }
        self._settings.exclude_missing_filter_csv = self.chk_exclude_missing_filter_csv.isChecked()

    def apply(self) -> None:
        self._apply_to_internal_settings()
        self.applied.emit()

    def accept(self) -> None:
        self.apply()
        super().accept()


class DraggableTableWidget(QTableWidget):
    rowMoveRequested = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_start_row: int | None = None
        self._drag_target_row: int | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_row = self.rowAt(int(event.position().y()))
            self._drag_target_row = self._drag_start_row
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_start_row is not None:
            hovered_row = self.rowAt(int(event.position().y()))
            if hovered_row >= 0:
                self._drag_target_row = hovered_row
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if (
                event.button() == Qt.MouseButton.LeftButton
                and self._drag_start_row is not None
                and self._drag_target_row is not None
                and self._drag_start_row >= 0
                and self._drag_target_row >= 0
                and self._drag_start_row != self._drag_target_row
            ):
                self.rowMoveRequested.emit(self._drag_start_row, self._drag_target_row)
        finally:
            self._drag_start_row = None
            self._drag_target_row = None
        super().mouseReleaseEvent(event)


class ReferenceSelectionGroup(QGroupBox):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.reference_root: Path | None = None
        self.target_roots: list[Path] = []
        self._index = index
        self._build_ui()
        self.set_index(index)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        button_row = QHBoxLayout()
        self.btn_select_reference = QPushButton("Select Reference Folder...")
        self.btn_select_target = QPushButton("Select Target Folders...")
        self.btn_remove = QPushButton("Remove Set")
        button_row.addWidget(self.btn_select_reference)
        button_row.addWidget(self.btn_select_target)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_remove)
        layout.addLayout(button_row)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.lbl_reference_folder = QLabel("(not selected)")
        self.lbl_reference_folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_reference_info = QLabel("(not loaded)")
        self.lbl_reference_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_target_folders = QLabel("(not selected)")
        self.lbl_target_folders.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.sb_reference_d = QDoubleSpinBox()
        self.sb_reference_d.setRange(-1e6, 1e6)
        self.sb_reference_d.setDecimals(6)
        self.sb_reference_d.setSingleStep(0.01)
        self.sb_reference_d.setValue(0.3)

        form.addRow("Reference:", self.lbl_reference_folder)
        form.addRow("Info:", self.lbl_reference_info)
        form.addRow("Targets:", self.lbl_target_folders)
        form.addRow("Reference d:", self.sb_reference_d)
        layout.addLayout(form)

    def set_index(self, index: int) -> None:
        self._index = index
        title = f"Reference Set {index}"
        if self.reference_root is not None:
            title += f" | {self.reference_root.name}"
        self.setTitle(title)

    def is_empty(self) -> bool:
        return self.reference_root is None and not self.target_roots

    def is_complete(self) -> bool:
        return self.reference_root is not None and bool(self.target_roots)

    def reference_d_value(self) -> float:
        return float(self.sb_reference_d.value())

    def reference_name(self) -> str:
        return self.reference_root.name if self.reference_root is not None else f"Reference {self._index}"

    def set_reference_root(self, root: Path | None) -> None:
        self.reference_root = root
        if root is None:
            self.lbl_reference_folder.setText("(not selected)")
            self.lbl_reference_folder.setToolTip("")
            self.lbl_reference_info.setText("(not loaded)")
            self.lbl_reference_info.setToolTip("")
        else:
            self.lbl_reference_folder.setText(root.name)
            self.lbl_reference_folder.setToolTip(str(root))
        self.set_index(self._index)

    def set_target_roots(self, roots: list[Path]) -> None:
        self.target_roots = list(roots)
        if not self.target_roots:
            self.lbl_target_folders.setText("(not selected)")
            self.lbl_target_folders.setToolTip("")
            return

        self.lbl_target_folders.setText(_summarize_path_names(self.target_roots))
        self.lbl_target_folders.setToolTip("\n".join(str(path) for path in self.target_roots))

    def refresh_reference_info(self) -> list[str]:
        self.lbl_reference_info.setText("(not loaded)")
        self.lbl_reference_info.setToolTip("")
        if self.reference_root is None:
            return []

        meta, _json_path, warnings = load_single_measurement_json(self.reference_root)
        if meta is None:
            self.lbl_reference_info.setText("(unavailable)")
            return warnings

        summary = extract_measurement_summary(meta)
        fitting_entries = normalize_fitting_entries(meta)
        strategy_names = [str(entry.get("strategy") or "(legacy)") for entry in fitting_entries] or ["(no saved fit)"]
        info_parts = [
            summary["sample"] or "(empty sample)",
            summary["thickness"] or "thickness ?",
            _compact_polarization_text(summary["input_polarization"], summary["detected_polarization"]),
            ", ".join(strategy_names),
        ]
        compact_text = " | ".join(part for part in info_parts if part)
        self.lbl_reference_info.setText(compact_text or "(empty)")

        tooltip_lines = [
            f"Sample: {summary['sample'] or '(empty)'}",
            f"Thickness: {summary['thickness'] or '(empty)'}",
            f"Input pol.: {summary['input_polarization'] or '(empty)'}",
            f"Detected pol.: {summary['detected_polarization'] or '(empty)'}",
            f"Fitting: {', '.join(strategy_names)}",
        ]
        self.lbl_reference_info.setToolTip("\n".join(tooltip_lines))
        return warnings


class ComparisonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._reference_groups: list[ReferenceSelectionGroup] = []
        self._results: list[ComparisonResult] = []
        self._filter_settings = ComparisonFilterSettings()
        self._row_enabled: dict[str, bool] = {}
        self._manual_row_order: list[str] = []
        self._displayed_result_keys: list[str] = []
        self._syncing_selection = False
        self._syncing_row_move = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.btn_add_reference = QPushButton("Add Reference...")
        self.btn_compare = QPushButton("Compare")
        self.btn_write_json = QPushButton("Write to Target JSON")
        self.btn_write_json.setEnabled(False)
        button_row.addWidget(self.btn_add_reference)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_compare)
        button_row.addWidget(self.btn_write_json)
        layout.addLayout(button_row)

        setup_group = QGroupBox("Comparison Setup")
        setup_layout = QVBoxLayout(setup_group)
        setup_layout.setContentsMargins(8, 8, 8, 8)
        setup_layout.setSpacing(6)

        self.reference_scroll = QScrollArea()
        self.reference_scroll.setWidgetResizable(True)
        self.reference_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.reference_scroll.setMaximumHeight(240)

        self.reference_groups_container = QWidget()
        self.reference_groups_layout = QVBoxLayout(self.reference_groups_container)
        self.reference_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_groups_layout.setSpacing(8)
        self.reference_groups_layout.addStretch(1)
        self.reference_scroll.setWidget(self.reference_groups_container)
        setup_layout.addWidget(self.reference_scroll)
        layout.addWidget(setup_group)

        splitter = QSplitter(Qt.Orientation.Vertical)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_summary = QLabel("Add at least one reference set, choose target folders, and run comparison.")
        table_layout.addWidget(self.lbl_summary)
        filter_row = QHBoxLayout()
        self.btn_filtering = QPushButton("Filtering...")
        self.lbl_filter_summary = QLabel(self._filter_summary_text())
        self.lbl_filter_summary.setStyleSheet("color: #555;")
        filter_row.addWidget(self.btn_filtering)
        filter_row.addWidget(self.lbl_filter_summary, 1)
        table_layout.addLayout(filter_row)
        table_split = QHBoxLayout()
        table_split.setContentsMargins(0, 0, 0, 0)
        table_split.setSpacing(0)

        self.fixed_table = DraggableTableWidget(0, 3)
        self.fixed_table.setHorizontalHeaderLabels(["Show", "Target holder", "Strategy"])
        self.fixed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.fixed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fixed_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fixed_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fixed_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.fixed_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.fixed_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.fixed_table.setMinimumWidth(420)
        self.fixed_table.verticalHeader().setSectionsMovable(True)
        self.fixed_table.verticalHeader().sectionMoved.connect(self._sync_row_move_from_fixed)

        self.table = DraggableTableWidget(0, 13)
        self.table.setHorizontalHeaderLabels(
            [
                "Reference",
                "Target sample",
                "Filter diff",
                "Fit scale ref",
                "Fit scale target",
                "d_factor ref",
                "d_factor target",
                "Corrected d scale ref",
                "Corrected d scale target",
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
        self.table.verticalHeader().setSectionsMovable(True)

        self.fixed_table.verticalScrollBar().valueChanged.connect(self.table.verticalScrollBar().setValue)
        self.table.verticalScrollBar().valueChanged.connect(self.fixed_table.verticalScrollBar().setValue)
        self.fixed_table.currentCellChanged.connect(self._sync_table_selection_from_fixed)
        self.table.currentCellChanged.connect(self._sync_table_selection_from_scroll)
        self.fixed_table.rowMoveRequested.connect(self._move_result_row)
        self.table.rowMoveRequested.connect(self._move_result_row)

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

        self.btn_add_reference.clicked.connect(self._handle_add_reference)
        self.btn_compare.clicked.connect(self._run_comparison)
        self.btn_write_json.clicked.connect(self._write_results)
        self.btn_filtering.clicked.connect(self._open_filter_dialog)

        self._add_reference_group()

    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    def _open_filter_dialog(self) -> None:
        strategy_labels = self._available_strategy_labels()
        dialog = ComparisonFilterDialog(self._filter_settings, strategy_labels, parent=self)
        dialog.applied.connect(lambda current_dialog=dialog: self._apply_filter_settings_from_dialog(current_dialog))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_filter_settings_from_dialog(dialog)

    def _apply_filter_settings(self) -> None:
        self.lbl_filter_summary.setText(self._filter_summary_text())
        if self._results:
            self._populate_table(self._results)

    def _apply_filter_settings_from_dialog(self, dialog: ComparisonFilterDialog) -> None:
        self._filter_settings = dialog.filter_settings
        self._apply_filter_settings()

    def _available_strategy_labels(self) -> list[str]:
        labels = [self._strategy_text(result) for result in self._results]
        labels.extend(sorted(self._filter_settings.allowed_strategy_labels))
        unique_labels: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            unique_labels.append(label)
        return unique_labels

    def _filter_summary_text(self) -> str:
        parts: list[str] = []
        if self._filter_settings.allowed_strategy_labels:
            parts.append(
                "Strategy: " + ", ".join(sorted(self._filter_settings.allowed_strategy_labels))
            )
        else:
            parts.append("Strategy: all")
        if self._filter_settings.exclude_missing_filter_csv:
            parts.append("Filter diff: hide missing CSV")
        else:
            parts.append("Filter diff: allow missing CSV")
        return " | ".join(parts)

    def _add_reference_group(self) -> ReferenceSelectionGroup:
        group = ReferenceSelectionGroup(len(self._reference_groups) + 1, self)
        group.btn_select_reference.clicked.connect(lambda _checked=False, current=group: self._select_reference_for_group(current))
        group.btn_select_target.clicked.connect(lambda _checked=False, current=group: self._select_targets_for_group(current))
        group.btn_remove.clicked.connect(lambda _checked=False, current=group: self._remove_reference_group(current))
        self._reference_groups.append(group)
        self.reference_groups_layout.insertWidget(self.reference_groups_layout.count() - 1, group)
        return group

    def _remove_reference_group(self, group: ReferenceSelectionGroup) -> None:
        if group not in self._reference_groups:
            return
        self._invalidate_results("Reference set removed. Run comparison again.")
        self._reference_groups.remove(group)
        self.reference_groups_layout.removeWidget(group)
        group.deleteLater()
        for index, current_group in enumerate(self._reference_groups, start=1):
            current_group.set_index(index)
        if not self._reference_groups:
            self._add_reference_group()

    def _handle_add_reference(self) -> None:
        empty_group = next((group for group in self._reference_groups if group.is_empty()), None)
        if empty_group is not None:
            self._select_reference_for_group(empty_group)
            return

        group = self._add_reference_group()
        if not self._select_reference_for_group(group) and group.is_empty():
            self._remove_reference_group(group)

    def _select_reference_for_group(self, group: ReferenceSelectionGroup) -> bool:
        start_dir = str(group.reference_root) if group.reference_root else "results"
        folder = QFileDialog.getExistingDirectory(self, "Select reference experiment folder", start_dir)
        if not folder:
            return False

        group.set_reference_root(Path(folder))
        self._invalidate_results("Reference changed. Run comparison again.")
        warnings = group.refresh_reference_info()
        for warning in warnings:
            self._append_log(f"WARNING: {group.reference_name()}: {warning}")
        return True

    def _select_targets_for_group(self, group: ReferenceSelectionGroup) -> bool:
        if group.target_roots:
            start_dir = str(group.target_roots[0])
        elif group.reference_root is not None:
            start_dir = str(group.reference_root.parent)
        else:
            start_dir = "results"

        folders = self._select_target_directories_native(start_dir)
        if not folders:
            return False

        group.set_target_roots(folders)
        self._invalidate_results("Target folders changed. Run comparison again.")
        return True

    def _invalidate_results(self, summary_text: str | None = None) -> None:
        self._results = []
        self._row_enabled.clear()
        self._manual_row_order.clear()
        self._displayed_result_keys.clear()
        self.fixed_table.setRowCount(0)
        self.table.setRowCount(0)
        self.btn_write_json.setEnabled(False)
        if summary_text:
            self.lbl_summary.setText(summary_text)

    def _run_comparison(self) -> None:
        self.log_output.clear()
        self.fixed_table.setRowCount(0)
        self.table.setRowCount(0)
        self._results = []
        self._row_enabled.clear()
        self._manual_row_order.clear()
        self._displayed_result_keys.clear()
        self.btn_write_json.setEnabled(False)

        ready_groups: list[ReferenceSelectionGroup] = []
        incomplete_groups: list[str] = []
        for group in self._reference_groups:
            if group.is_empty():
                continue
            if group.is_complete():
                ready_groups.append(group)
            else:
                incomplete_groups.append(group.title())

        if incomplete_groups:
            QMessageBox.information(
                self,
                "Incomplete setup",
                "Complete both reference and target selection for:\n" + "\n".join(incomplete_groups),
            )
            return

        if not ready_groups:
            QMessageBox.information(
                self,
                "Missing setup",
                "Add at least one reference set with a reference folder and one or more target folders first.",
            )
            return

        all_results: list[ComparisonResult] = []
        all_warnings: list[str] = []
        total_targets = 0
        for group in ready_groups:
            reference_root = group.reference_root
            if reference_root is None:
                continue

            reference_d = group.reference_d_value()
            total_targets += len(group.target_roots)
            for target_root in group.target_roots:
                results, warnings = compare_experiment_folders(
                    reference_root=reference_root,
                    target_root=target_root,
                    reference_d_value=reference_d,
                )
                all_results.extend(results)
                for warning in warnings:
                    all_warnings.append(f"{reference_root.name} -> {target_root.name}: {warning}")

        self._results = all_results
        self._populate_table(all_results)

        ready_count = sum(1 for result in all_results if result.error is None)
        self.lbl_summary.setText(
            f"Reference sets {len(ready_groups)} | target folders {total_targets} | results {len(all_results)} | ready {ready_count}"
        )

        for warning in all_warnings:
            self._append_log(f"WARNING: {warning}")
        for result in all_results:
            strategy_text = self._strategy_text(result)
            self._append_log(
                f"{result.reference_json_path.parent.name} -> {result.target_json_path.parent.name} [{strategy_text}]: {result.status_text}"
            )

        if all_results:
            self.btn_write_json.setEnabled(any(result.error is None for result in all_results))
        elif all_warnings:
            QMessageBox.information(self, "No results", "\n".join(all_warnings))

    def _populate_table(self, results: list[ComparisonResult]) -> None:
        for result in results:
            self._row_enabled.setdefault(result.key, True)
        self._reconcile_manual_row_order(results)

        ordered_results = sorted(
            results,
            key=lambda result: (
                self._display_bucket(result),
                self._manual_order_index(result.key),
            ),
        )
        self._displayed_result_keys = [result.key for result in ordered_results]

        self.fixed_table.setRowCount(len(ordered_results))
        self.table.setRowCount(len(ordered_results))

        group_colors = [
            QColor(255, 255, 255),
            QColor(247, 250, 255),
        ]
        group_index = -1
        previous_group = None

        for row_index, result in enumerate(ordered_results):
            current_group = f"{result.reference_json_path.parent}::{result.target_json_path}"
            if current_group != previous_group:
                group_index += 1
                previous_group = current_group
            enabled = self._is_row_active(result)
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
                result.reference_json_path.parent.name,
                result.target_sample,
                self._fmt(result.peak_ref),
                self._fmt(result.peak_target),
                self._fmt(result.d_factor_ref),
                self._fmt(result.d_factor_target),
                self._fmt(result.d_scale_ref),
                self._fmt(result.d_scale_target),
                self._fmt(result.intensity_ratio),
                self._fmt(result.d_ratio),
                self._fmt(result.calculated_d),
                self._status_text(result, enabled),
            ]
            for column_index, value in enumerate(values):
                actual_column_index = column_index if column_index < 2 else column_index + 1
                item = QTableWidgetItem(value)
                self._style_item(item, base_color, enabled)
                self.table.setItem(row_index, actual_column_index, item)

            self.table.setCellWidget(
                row_index,
                2,
                self._build_filter_diff_widget(result, base_color, enabled),
            )

            row_height = max(self.fixed_table.verticalHeader().defaultSectionSize(), 28)
            self.fixed_table.setRowHeight(row_index, row_height)
            self.table.setRowHeight(row_index, row_height)

        self.fixed_table.resizeColumnsToContents()
        self.table.resizeColumnsToContents()
        self.fixed_table.setColumnWidth(1, max(self.fixed_table.columnWidth(1), 180))
        self.fixed_table.setColumnWidth(2, max(self.fixed_table.columnWidth(2), 220))
        self.table.setColumnWidth(0, max(self.table.columnWidth(0), 180))
        self.table.setColumnWidth(2, max(self.table.columnWidth(2), 400))

    def _write_results(self) -> None:
        selected_results = [result for result in self._results if self._is_row_active(result)]
        if not selected_results:
            message = "Run comparison first." if not self._results else "No rows currently pass the checkbox/filter conditions for saving."
            QMessageBox.information(self, "No results", message)
            return

        conflicting_targets = self._find_conflicting_target_writes(selected_results)
        if conflicting_targets:
            QMessageBox.warning(
                self,
                "Conflicting references",
                "The same target JSON is assigned to multiple references, so writing would overwrite results:\n"
                + "\n".join(conflicting_targets),
            )
            return

        grouped_results: dict[Path, list[ComparisonResult]] = {}
        for result in selected_results:
            grouped_results.setdefault(result.reference_json_path.parent, []).append(result)

        written_total = 0
        skipped_total = 0
        warnings_total: list[str] = []
        for reference_root, grouped in grouped_results.items():
            written, skipped, warnings = write_comparison_results(reference_root, grouped)
            written_total += written
            skipped_total += skipped
            warnings_total.extend(warnings)

        for warning in warnings_total:
            self._append_log(f"WARNING: {warning}")
        self._append_log(f"WRITE DONE | updated={written_total} | skipped={skipped_total}")
        QMessageBox.information(
            self,
            "Completed",
            f"Updated: {written_total}\nSkipped: {skipped_total}",
        )

    def _find_conflicting_target_writes(self, results: list[ComparisonResult]) -> list[str]:
        references_by_target: dict[Path, set[Path]] = {}
        for result in results:
            if result.error:
                continue
            references_by_target.setdefault(result.target_json_path, set()).add(result.reference_json_path.parent)

        conflicts: list[str] = []
        for target_json_path, reference_roots in sorted(references_by_target.items(), key=lambda item: str(item[0])):
            if len(reference_roots) <= 1:
                continue
            reference_names = ", ".join(sorted(reference_root.name for reference_root in reference_roots))
            conflicts.append(f"{target_json_path.parent.name} ({target_json_path.name}) <- {reference_names}")
        return conflicts

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.6g}"

    def _strategy_text(self, result: ComparisonResult) -> str:
        strategy = result.target_strategy or "(legacy)"
        label = result.target_result_label.strip()
        if label and label != strategy:
            return f"{strategy} | {label}"
        return strategy

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

    def _passes_filters(self, result: ComparisonResult) -> bool:
        allowed_strategy_labels = self._filter_settings.allowed_strategy_labels
        if allowed_strategy_labels and self._strategy_text(result) not in allowed_strategy_labels:
            return False
        if self._filter_settings.exclude_missing_filter_csv and result.differing_filters_missing_csv:
            return False
        return True

    def _is_row_active(self, result: ComparisonResult) -> bool:
        return self._row_enabled.get(result.key, True) and self._passes_filters(result)

    def _display_bucket(self, result: ComparisonResult) -> int:
        return 0 if self._is_row_active(result) else 1

    def _style_item(self, item: QTableWidgetItem, background: QColor, enabled: bool) -> None:
        item.setBackground(QBrush(background))
        if enabled:
            item.setForeground(QBrush(QColor(20, 20, 20)))
        else:
            item.setForeground(QBrush(QColor(150, 150, 150)))

    def _build_filter_diff_widget(self, result: ComparisonResult, background: QColor, enabled: bool) -> QWidget:
        label = QLabel()
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setMargin(4)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setText(self._format_filter_diff_html(result, enabled))

        widget = QWidget()
        widget.setAutoFillBackground(True)
        widget.setStyleSheet(f"background-color: {background.name()};")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        return widget

    def _format_filter_diff_html(self, result: ComparisonResult, enabled: bool) -> str:
        text = result.differing_filters_text or ""
        missing_ids = sorted(set(result.differing_filters_missing_csv))
        if not missing_ids:
            color = "#141414" if enabled else "#969696"
            return f"<span style=\"color: {color};\">{escape(text)}</span>"

        highlighted_text = escape(text)
        for filter_id in sorted(missing_ids, key=len, reverse=True):
            escaped_filter_id = escape(filter_id)
            highlighted_text = highlighted_text.replace(
                escaped_filter_id,
                f"<span style=\"color: #c62828; font-weight: 700;\">{escaped_filter_id}</span>",
            )

        default_color = "#141414" if enabled else "#969696"
        return f"<span style=\"color: {default_color};\">{highlighted_text}</span>"

    def _status_text(self, result: ComparisonResult, enabled: bool) -> str:
        prefixes: list[str] = []
        if not self._row_enabled.get(result.key, True):
            prefixes.append("[hidden]")
        if not self._passes_filters(result):
            prefixes.append("[filtered]")
        mode = "Braun pseudo d" if result.calculation_mode == "braun_pseudo_d" else ""
        status = result.status_text
        prefix = " ".join(prefixes)
        suffix = " | ".join(part for part in (mode, status) if part)
        if prefix and suffix:
            return f"{prefix} {suffix}"
        return prefix or suffix

    def _reconcile_manual_row_order(self, results: list[ComparisonResult]) -> None:
        seen_keys = {result.key for result in results}
        self._manual_row_order = [key for key in self._manual_row_order if key in seen_keys]
        for result in results:
            if result.key not in self._manual_row_order:
                self._manual_row_order.append(result.key)

    def _manual_order_index(self, key: str) -> int:
        try:
            return self._manual_row_order.index(key)
        except ValueError:
            return len(self._manual_row_order)

    def _sync_row_move_from_fixed(self, logical_index: int, old_visual_index: int, new_visual_index: int) -> None:
        del logical_index
        if self._syncing_row_move:
            return
        self._syncing_row_move = True
        try:
            self.table.verticalHeader().moveSection(old_visual_index, new_visual_index)
            self._apply_manual_order_from_view()
        finally:
            self._syncing_row_move = False

    def _apply_manual_order_from_view(self) -> None:
        header = self.fixed_table.verticalHeader()
        self._manual_row_order = [
            self._displayed_result_keys[header.logicalIndex(visual_index)]
            for visual_index in range(header.count())
            if 0 <= header.logicalIndex(visual_index) < len(self._displayed_result_keys)
        ]
        self._populate_table(self._results)

    def _move_result_row(self, source_row: int, target_row: int) -> None:
        if self._syncing_row_move:
            return
        if not (0 <= source_row < len(self._displayed_result_keys) and 0 <= target_row < len(self._displayed_result_keys)):
            return

        source_key = self._displayed_result_keys[source_row]
        target_key = self._displayed_result_keys[target_row]
        source_result = next((result for result in self._results if result.key == source_key), None)
        target_result = next((result for result in self._results if result.key == target_key), None)
        if source_result is None or target_result is None:
            return
        if self._display_bucket(source_result) != self._display_bucket(target_result):
            return

        self._reconcile_manual_row_order(self._results)
        try:
            self._manual_row_order.remove(source_key)
        except ValueError:
            return

        target_index = self._manual_order_index(target_key)
        self._manual_row_order.insert(target_index, source_key)
        self._populate_table(self._results)

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


def _summarize_path_names(paths: list[Path], max_items: int = 3) -> str:
    names = [path.name for path in paths]
    if not names:
        return "(not selected)"
    if len(names) <= max_items:
        return ", ".join(names)
    return ", ".join(names[:max_items]) + f", +{len(names) - max_items} more"


def _compact_polarization_text(input_pol: str, detected_pol: str) -> str:
    input_text = input_pol or "?"
    detected_text = detected_pol or "?"
    return f"in {input_text} / out {detected_text}"
