from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_metadata import (  # noqa: E402
    COMMON_BOXCAR_SENSITIVITIES,
    ND_FILTER_CATALOG_PATH,
    apply_condition_metadata,
    format_filter_display,
    load_nd_filter_catalog,
)


class FilterBoxcarMetadataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add ND Filter / Boxcar Metadata")
        self.resize(760, 620)

        self.selected_folder: Path | None = None
        self.filter_map: dict[str, dict] = {}

        self._build_ui()
        self._reload_filter_catalog()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        button_row = QHBoxLayout()
        self.btn_select_folder = QPushButton("Select Experiment Folder")
        self.btn_reload_catalog = QPushButton("Reload Filter Catalog")
        self.btn_apply = QPushButton("Apply to JSON")
        button_row.addWidget(self.btn_select_folder)
        button_row.addWidget(self.btn_reload_catalog)
        button_row.addWidget(self.btn_apply)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.lbl_folder = QLabel("Selected folder: (not selected)")
        self.lbl_catalog = QLabel(f"Catalog: {ND_FILTER_CATALOG_PATH}")
        self.lbl_catalog.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lbl_folder)
        layout.addWidget(self.lbl_catalog)

        sensitivity_row = QHBoxLayout()
        sensitivity_row.addWidget(QLabel("Boxcar sensitivity:"))
        self.cmb_sensitivity = QComboBox()
        self.cmb_sensitivity.setEditable(True)
        self.cmb_sensitivity.addItems(COMMON_BOXCAR_SENSITIVITIES)
        self.cmb_sensitivity.setCurrentText("1 V / 0.1 V")
        sensitivity_row.addWidget(self.cmb_sensitivity, 1)
        layout.addLayout(sensitivity_row)

        wavelength_row = QHBoxLayout()
        wavelength_row.addWidget(QLabel("Fallback laser wavelength [nm]:"))
        self.sb_fallback_wavelength = QDoubleSpinBox()
        self.sb_fallback_wavelength.setRange(1.0, 100000.0)
        self.sb_fallback_wavelength.setDecimals(3)
        self.sb_fallback_wavelength.setValue(1064.0)
        wavelength_row.addWidget(self.sb_fallback_wavelength)
        wavelength_row.addStretch(1)
        layout.addLayout(wavelength_row)

        self.chk_no_filter = QCheckBox("No ND filter")
        self.chk_no_filter.setChecked(True)
        layout.addWidget(self.chk_no_filter)

        filter_header = QHBoxLayout()
        filter_header.addWidget(QLabel("ND filters:"))
        filter_header.addStretch(1)
        layout.addLayout(filter_header)

        self.filter_list = QListWidget()
        self.filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.filter_list.setMinimumHeight(180)
        self.filter_list.setEnabled(False)
        layout.addWidget(self.filter_list)

        layout.addWidget(QLabel("Log:"))
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output, 1)

        self.btn_select_folder.clicked.connect(self._choose_folder)
        self.btn_reload_catalog.clicked.connect(self._reload_filter_catalog)
        self.btn_apply.clicked.connect(self._apply_updates)
        self.chk_no_filter.toggled.connect(self._toggle_filter_selection)
        self.filter_list.itemSelectionChanged.connect(self._sync_filter_checkbox)

    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    def _choose_folder(self) -> None:
        start_dir = str(self.selected_folder) if self.selected_folder else "results"
        folder = QFileDialog.getExistingDirectory(self, "Select experiment folder", start_dir)
        if not folder:
            return
        self.selected_folder = Path(folder)
        self.lbl_folder.setText(f"Selected folder: {self.selected_folder}")

    def _reload_filter_catalog(self) -> None:
        catalog = load_nd_filter_catalog()
        self.filter_map = {entry["filter_id"]: entry for entry in catalog["filters"]}
        self.filter_list.clear()
        for entry in catalog["filters"]:
            item = QListWidgetItem(format_filter_display(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry["filter_id"])
            self.filter_list.addItem(item)
        self._toggle_filter_selection(self.chk_no_filter.isChecked())

    def _toggle_filter_selection(self, checked: bool) -> None:
        if checked:
            self.filter_list.clearSelection()
        self.filter_list.setEnabled(not checked)

    def _sync_filter_checkbox(self) -> None:
        if self.filter_list.selectedItems():
            self.chk_no_filter.blockSignals(True)
            self.chk_no_filter.setChecked(False)
            self.chk_no_filter.blockSignals(False)
            self.filter_list.setEnabled(True)

    def _selected_filter_entries(self) -> list[dict]:
        if self.chk_no_filter.isChecked():
            return []
        selected = []
        for item in self.filter_list.selectedItems():
            filter_id = item.data(Qt.ItemDataRole.UserRole)
            entry = self.filter_map.get(filter_id)
            if entry is not None:
                selected.append(entry)
        return selected

    def _find_target_json_files(self, root_dir: Path) -> list[Path]:
        files = []
        for path in sorted(root_dir.rglob("*.json")):
            if path == ND_FILTER_CATALOG_PATH:
                continue
            if path.name.endswith(".bak"):
                continue
            files.append(path)
        return files

    def _apply_updates(self) -> None:
        self.log_output.clear()
        if self.selected_folder is None:
            QMessageBox.information(self, "No folder", "Select an experiment folder first.")
            return

        target_files = self._find_target_json_files(self.selected_folder)
        if not target_files:
            QMessageBox.information(self, "No JSON", "No JSON files found under the selected folder.")
            return

        selected_filters = self._selected_filter_entries()
        updated = 0
        errors = 0
        warning_count = 0

        for json_path in target_files:
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                wavelength_nm = data.get("wavelength_nm")
                if wavelength_nm in (None, ""):
                    wavelength_nm = float(self.sb_fallback_wavelength.value())
                else:
                    wavelength_nm = float(wavelength_nm)

                updated_data, warnings = apply_condition_metadata(
                    metadata=data,
                    boxcar_sensitivity_text=self.cmb_sensitivity.currentText().strip(),
                    selected_filters=selected_filters,
                    fundamental_wavelength_nm=wavelength_nm,
                )

                backup_path = json_path.with_suffix(json_path.suffix + ".bak")
                # if not backup_path.exists():
                #     shutil.copy2(json_path, backup_path)

                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=2)

                self._append_log(f"UPDATED: {json_path}")
                for warning in warnings:
                    self._append_log(f"  WARNING: {warning}")
                    warning_count += 1
                updated += 1
            except Exception as exc:
                self._append_log(f"ERROR: {json_path} | {exc}")
                errors += 1

        self._append_log("-" * 60)
        self._append_log(f"DONE | updated={updated} | warnings={warning_count} | errors={errors}")
        QMessageBox.information(
            self,
            "Completed",
            f"Updated: {updated}\nWarnings: {warning_count}\nErrors: {errors}",
        )


def launch_filter_boxcar_metadata_dialog() -> FilterBoxcarMetadataDialog:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    dialog = FilterBoxcarMetadataDialog()
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
