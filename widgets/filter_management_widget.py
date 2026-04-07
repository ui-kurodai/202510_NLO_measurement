from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from measurement_metadata import (
    ND_FILTER_CATALOG_PATH,
    build_filter_id,
    format_filter_display,
    load_nd_filter_catalog,
    normalize_filter_entry,
    resolve_catalog_path,
    save_nd_filter_catalog,
)


class FilterCatalogWidget(QWidget):
    catalog_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog = {"version": 1, "filters": []}
        self._current_filter_id: str | None = None

        self._build_ui()
        self.reload_catalog()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        self.catalog_path_label = QLabel(f"Catalog: {ND_FILTER_CATALOG_PATH}")
        self.catalog_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_reload = QPushButton("Reload")
        self.btn_new = QPushButton("New Filter")
        self.btn_save = QPushButton("Save Filter")
        self.btn_delete = QPushButton("Delete Filter")
        header.addWidget(self.catalog_path_label, 1)
        header.addWidget(self.btn_reload)
        header.addWidget(self.btn_new)
        header.addWidget(self.btn_save)
        header.addWidget(self.btn_delete)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Registered ND filters"))
        self.filter_list = QListWidget()
        left_layout.addWidget(self.filter_list, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        form_group = QGroupBox("Filter editor")
        form = QFormLayout(form_group)
        self.lbl_filter_id = QLabel("(auto)")
        self.le_product_name = QLineEdit()
        self.le_product_name.setPlaceholderText("e.g. thorlabs_NE10A-A")
        self.le_instance_id = QLineEdit()
        self.le_instance_id.setPlaceholderText("e.g. #1")
        self.le_nominal_od = QLineEdit()
        self.le_nominal_od.setPlaceholderText("e.g. 1.0")
        self.le_csv_path = QLineEdit()
        self.le_csv_path.setPlaceholderText("Optional transmission spectrum CSV")
        self.btn_browse_csv = QPushButton("Browse...")
        self.te_notes = QPlainTextEdit()
        self.te_notes.setPlaceholderText("Optional notes, calibration memo, etc.")
        self.te_notes.setFixedHeight(100)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.le_csv_path, 1)
        csv_row.addWidget(self.btn_browse_csv)

        form.addRow("Filter ID:", self.lbl_filter_id)
        form.addRow("Product Name:", self.le_product_name)
        form.addRow("Instance ID:", self.le_instance_id)
        form.addRow("Nominal OD:", self.le_nominal_od)
        form.addRow("Transmission CSV:", csv_row)
        form.addRow("Notes:", self.te_notes)
        right_layout.addWidget(form_group)

        hint = QLabel(
            "Use Product Name for the filter type and Instance ID for the individual piece. "
            "Filter ID is auto-generated as <product_name>_<instance_id> when possible. "
            "If a transmission CSV is registered, the app reads the SHG wavelength point from that file. "
            "If it cannot be used, the app falls back to the nominal OD."
        )
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([360, 540])

        self.btn_reload.clicked.connect(self.reload_catalog)
        self.btn_new.clicked.connect(self._start_new_filter)
        self.btn_save.clicked.connect(self._save_current_filter)
        self.btn_delete.clicked.connect(self._delete_current_filter)
        self.btn_browse_csv.clicked.connect(self._browse_transmission_csv)
        self.filter_list.currentItemChanged.connect(self._load_selected_filter)
        self.le_product_name.textChanged.connect(self._refresh_filter_id_preview)
        self.le_instance_id.textChanged.connect(self._refresh_filter_id_preview)

    def reload_catalog(self) -> None:
        self._catalog = load_nd_filter_catalog()
        self.filter_list.clear()
        for entry in self._catalog["filters"]:
            item = QListWidgetItem(format_filter_display(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry["filter_id"])
            self.filter_list.addItem(item)

        if self.filter_list.count() > 0:
            self.filter_list.setCurrentRow(0)
        else:
            self._start_new_filter()

    def _entry_by_id(self, filter_id: str) -> dict | None:
        for entry in self._catalog["filters"]:
            if entry["filter_id"] == filter_id:
                return entry
        return None

    def _load_selected_filter(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        filter_id = current.data(Qt.ItemDataRole.UserRole)
        entry = self._entry_by_id(filter_id)
        if entry is None:
            return

        self._current_filter_id = entry["filter_id"]
        self.lbl_filter_id.setText(entry["filter_id"])
        self.le_product_name.setText(entry.get("product_name") or "")
        self.le_instance_id.setText(entry.get("instance_id") or "")
        self.le_nominal_od.setText("" if entry.get("nominal_od") is None else f"{entry['nominal_od']:g}")
        self.le_csv_path.setText(entry.get("transmission_csv_path") or "")
        self.te_notes.setPlainText(entry.get("notes") or "")

    def _start_new_filter(self) -> None:
        self.filter_list.clearSelection()
        self._current_filter_id = None
        self.le_product_name.clear()
        self.le_instance_id.clear()
        self.le_nominal_od.clear()
        self.le_csv_path.clear()
        self.te_notes.clear()
        self._refresh_filter_id_preview()

    def _refresh_filter_id_preview(self) -> None:
        if self._current_filter_id is not None:
            return
        candidate_filter_id = build_filter_id(
            self.le_product_name.text().strip(),
            self.le_instance_id.text().strip(),
        )
        if not self.le_product_name.text().strip() and not self.le_instance_id.text().strip():
            self.lbl_filter_id.setText("(auto)")
        else:
            self.lbl_filter_id.setText(candidate_filter_id)

    def _get_form_values(self) -> dict[str, str]:
        return {
            "product_name": self.le_product_name.text(),
            "instance_id": self.le_instance_id.text(),
            "nominal_od": self.le_nominal_od.text(),
            "transmission_csv_path": self.le_csv_path.text(),
            "notes": self.te_notes.toPlainText(),
        }

    def _set_form_values(self, values: dict[str, str]) -> None:
        self.le_product_name.setText(values.get("product_name", ""))
        self.le_instance_id.setText(values.get("instance_id", ""))
        self.le_nominal_od.setText(values.get("nominal_od", ""))
        self.le_csv_path.setText(values.get("transmission_csv_path", ""))
        self.te_notes.setPlainText(values.get("notes", ""))

    def _enter_new_mode_keep_values(self) -> None:
        self.filter_list.blockSignals(True)
        self.filter_list.clearSelection()
        self.filter_list.blockSignals(False)
        self._current_filter_id = None
        self._refresh_filter_id_preview()

    def _browse_transmission_csv(self) -> None:
        current_path = resolve_catalog_path(self.le_csv_path.text().strip())
        if current_path is not None and current_path.exists():
            start_dir = str(current_path.parent if current_path.is_file() else current_path)
        else:
            start_dir = str(ND_FILTER_CATALOG_PATH.parent)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select transmission spectrum CSV",
            start_dir,
            "CSV files (*.csv);;All files (*)",
        )
        if selected:
            self.le_csv_path.setText(selected)

    def _parse_optional_float(self, field_name: str, text: str) -> float | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number.") from exc

    def _save_current_filter(self) -> None:
        try:
            nominal_od = self._parse_optional_float("Nominal OD", self.le_nominal_od.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
            return
        csv_path_text = self.le_csv_path.text().strip()

        if not any(
            text.strip()
            for text in [
                self.le_product_name.text(),
                self.le_instance_id.text(),
            ]
        ):
            QMessageBox.warning(self, "Input Error", "Please enter at least Product Name or Instance ID.")
            return

        resolved_csv_path = resolve_catalog_path(csv_path_text) if csv_path_text else None
        if csv_path_text and (resolved_csv_path is None or not resolved_csv_path.exists()):
            QMessageBox.warning(self, "Input Error", "Transmission CSV path does not exist.")
            return

        candidate_filter_id = build_filter_id(
            self.le_product_name.text().strip(),
            self.le_instance_id.text().strip(),
        )

        entry = normalize_filter_entry(
            {
                "filter_id": self._current_filter_id or "",
                "product_name": self.le_product_name.text().strip(),
                "instance_id": self.le_instance_id.text().strip(),
                "nominal_od": nominal_od,
                "transmission_csv_path": csv_path_text,
                "notes": self.te_notes.toPlainText().strip(),
            }
        )

        existing_ids = {item["filter_id"] for item in self._catalog["filters"]}
        overwrite_target_id = None
        if self._current_filter_id is None:
            entry["filter_id"] = candidate_filter_id
            self.lbl_filter_id.setText(entry["filter_id"])
            if candidate_filter_id in existing_ids:
                confirm = QMessageBox.question(
                    self,
                    "Filter Already Exists",
                    f"'{candidate_filter_id}' is already registered. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return
                overwrite_target_id = candidate_filter_id
        else:
            self.lbl_filter_id.setText(entry["filter_id"])

        replaced = False
        for index, existing in enumerate(self._catalog["filters"]):
            target_id = overwrite_target_id or self._current_filter_id
            if existing["filter_id"] == target_id:
                self._catalog["filters"][index] = entry
                replaced = True
                break
        if not replaced:
            self._catalog["filters"].append(entry)

        form_values = self._get_form_values()
        save_nd_filter_catalog(self._catalog)
        self.reload_catalog()
        self._set_form_values(form_values)
        self._enter_new_mode_keep_values()
        self.catalog_updated.emit()
        QMessageBox.information(self, "Saved", "Filter catalog updated.")

    def _select_filter(self, filter_id: str) -> None:
        for index in range(self.filter_list.count()):
            item = self.filter_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == filter_id:
                self.filter_list.setCurrentRow(index)
                return

    def _delete_current_filter(self) -> None:
        if self._current_filter_id is None:
            QMessageBox.information(self, "No Filter", "Select a saved filter first.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Filter",
            f"Delete '{self._current_filter_id}' from the catalog?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._catalog["filters"] = [
            entry for entry in self._catalog["filters"] if entry["filter_id"] != self._current_filter_id
        ]
        save_nd_filter_catalog(self._catalog)
        self.reload_catalog()
        self.catalog_updated.emit()
        QMessageBox.information(self, "Deleted", "Filter removed from the catalog.")
