from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from crystaldatabase import CRYSTALS
from measurement_metadata import (
    BEAM_PROFILE_CATALOG_PATH,
    SAMPLE_CATALOG_PATH,
    build_sample_catalog_key,
    format_beam_profile_display,
    format_sample_display,
    load_beam_profile_catalog,
    load_sample_catalog,
    normalize_beam_profile_entry,
    normalize_crystal_orientation,
    normalize_sample_entry,
    save_beam_profile_catalog,
    save_sample_catalog,
)
from widgets.filter_management_widget import FilterCatalogWidget


class SampleCatalogWidget(QWidget):
    catalog_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog = {"version": 1, "samples": []}
        self._current_sample_key: str | None = None

        self._build_ui()
        self.reload_catalog()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        self.catalog_path_label = QLabel(f"Catalog: {SAMPLE_CATALOG_PATH}")
        self.catalog_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_reload = QPushButton("Reload")
        self.btn_new = QPushButton("New Sample")
        self.btn_save = QPushButton("Save Sample")
        self.btn_delete = QPushButton("Delete Sample")
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
        left_layout.addWidget(QLabel("Registered samples"))
        self.sample_list = QListWidget()
        left_layout.addWidget(self.sample_list, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        form_group = QGroupBox("Sample editor")
        form = QFormLayout(form_group)
        self.le_sample = QLineEdit()
        self.le_sample.setPlaceholderText("e.g. quartz-1")
        self.material_combo = QComboBox()
        self.material_combo.addItems(CRYSTALS.keys())
        self.orientation_combo = QComboBox()
        self.orientation_combo.setEditable(True)
        self.orientation_combo.addItems(["100", "010", "001"])
        self.le_t_center = QLineEdit()
        self.le_t_center.setPlaceholderText("e.g. 0.511")
        self.le_wedge = QLineEdit()
        self.le_wedge.setPlaceholderText("e.g. 0.0")

        form.addRow("Sample:", self.le_sample)
        form.addRow("Material:", self.material_combo)
        form.addRow("Cut axis:", self.orientation_combo)
        form.addRow("t_center_mm:", self.le_t_center)
        form.addRow("wedge_angle_deg:", self.le_wedge)
        right_layout.addWidget(form_group)

        hint = QLabel(
            "Sample presets are stored in the same shape used in measurement JSON. "
            "Selecting one during measurement fills the Measurement ID prefix, fixes the material, "
            "and injects thickness_info into the saved metadata."
        )
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([360, 540])

        self.btn_reload.clicked.connect(self.reload_catalog)
        self.btn_new.clicked.connect(self._start_new_sample)
        self.btn_save.clicked.connect(self._save_current_sample)
        self.btn_delete.clicked.connect(self._delete_current_sample)
        self.sample_list.currentItemChanged.connect(self._load_selected_sample)

    def reload_catalog(self) -> None:
        self._catalog = load_sample_catalog()
        self.sample_list.clear()
        for entry in self._catalog["samples"]:
            item = QListWidgetItem(format_sample_display(entry))
            item.setData(
                Qt.ItemDataRole.UserRole,
                build_sample_catalog_key(entry["sample"], entry["crystal_orientation"]),
            )
            self.sample_list.addItem(item)

        if self.sample_list.count() > 0:
            self.sample_list.setCurrentRow(0)
        else:
            self._start_new_sample()

    def _entry_by_key(self, sample_key: str) -> dict | None:
        for entry in self._catalog["samples"]:
            if build_sample_catalog_key(entry["sample"], entry["crystal_orientation"]) == sample_key:
                return entry
        return None

    def _load_selected_sample(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        sample_key = current.data(Qt.ItemDataRole.UserRole)
        entry = self._entry_by_key(sample_key)
        if entry is None:
            return

        thickness_info = entry.get("thickness_info") or {}
        self._current_sample_key = sample_key
        self.le_sample.setText(entry.get("sample") or "")
        self._set_material(entry.get("material") or "")
        self.orientation_combo.setCurrentText(entry.get("crystal_orientation") or "")
        self.le_t_center.setText("" if thickness_info.get("t_center_mm") is None else f"{thickness_info['t_center_mm']:g}")
        self.le_wedge.setText("" if thickness_info.get("wedge_angle_deg") is None else f"{thickness_info['wedge_angle_deg']:g}")

    def _start_new_sample(self) -> None:
        self.sample_list.clearSelection()
        self._current_sample_key = None
        self.le_sample.clear()
        self.material_combo.setCurrentIndex(0)
        self.orientation_combo.setCurrentText("010")
        self.le_t_center.clear()
        self.le_wedge.clear()

    def _get_form_values(self) -> dict[str, str]:
        return {
            "sample": self.le_sample.text(),
            "material": self.material_combo.currentText(),
            "crystal_orientation": self.orientation_combo.currentText(),
            "t_center_mm": self.le_t_center.text(),
            "wedge_angle_deg": self.le_wedge.text(),
        }

    def _set_form_values(self, values: dict[str, str]) -> None:
        self.le_sample.setText(values.get("sample", ""))
        self._set_material(values.get("material", ""))
        self.orientation_combo.setCurrentText(values.get("crystal_orientation", "010"))
        self.le_t_center.setText(values.get("t_center_mm", ""))
        self.le_wedge.setText(values.get("wedge_angle_deg", ""))

    def _enter_new_mode_keep_values(self) -> None:
        self.sample_list.blockSignals(True)
        self.sample_list.clearSelection()
        self.sample_list.blockSignals(False)
        self._current_sample_key = None

    def _set_material(self, material: str) -> None:
        index = self.material_combo.findText(material)
        if index >= 0:
            self.material_combo.setCurrentIndex(index)
        elif material:
            self.material_combo.setCurrentText(material)

    def _parse_required_float(self, field_name: str, text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError(f"{field_name} is required.")
        try:
            return float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number.") from exc

    def _save_current_sample(self) -> None:
        try:
            t_center_mm = self._parse_required_float("t_center_mm", self.le_t_center.text())
            wedge_angle_deg = self._parse_required_float("wedge_angle_deg", self.le_wedge.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
            return

        sample_name = self.le_sample.text().strip()
        material = self.material_combo.currentText().strip()
        crystal_orientation = normalize_crystal_orientation(self.orientation_combo.currentText())

        if not sample_name:
            QMessageBox.warning(self, "Input Error", "Please enter a sample name.")
            return
        if crystal_orientation not in {"100", "010", "001"}:
            QMessageBox.warning(self, "Input Error", "Cut axis must be one of 100, 010, or 001.")
            return
        if not material:
            QMessageBox.warning(self, "Input Error", "Please select a material.")
            return

        entry = normalize_sample_entry(
            {
                "sample": sample_name,
                "material": material,
                "crystal_orientation": crystal_orientation,
                "thickness_info": {
                    "t_center_mm": t_center_mm,
                    "wedge_angle_deg": wedge_angle_deg,
                },
            }
        )

        candidate_key = build_sample_catalog_key(entry["sample"], entry["crystal_orientation"])
        existing_keys = {
            build_sample_catalog_key(item["sample"], item["crystal_orientation"])
            for item in self._catalog["samples"]
        }
        overwrite_target_key = None

        if self._current_sample_key is None:
            if candidate_key in existing_keys:
                confirm = QMessageBox.question(
                    self,
                    "Sample Already Exists",
                    f"'{candidate_key}' is already registered. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return
                overwrite_target_key = candidate_key
        elif candidate_key != self._current_sample_key and candidate_key in existing_keys:
            confirm = QMessageBox.question(
                self,
                "Sample Already Exists",
                f"'{candidate_key}' is already registered. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            overwrite_target_key = candidate_key

        replaced = False
        target_key = overwrite_target_key or self._current_sample_key
        for index, existing in enumerate(self._catalog["samples"]):
            existing_key = build_sample_catalog_key(existing["sample"], existing["crystal_orientation"])
            if existing_key == target_key:
                self._catalog["samples"][index] = entry
                replaced = True
                break

        if self._current_sample_key and overwrite_target_key and overwrite_target_key != self._current_sample_key:
            self._catalog["samples"] = [
                existing
                for existing in self._catalog["samples"]
                if build_sample_catalog_key(existing["sample"], existing["crystal_orientation"]) != self._current_sample_key
            ]

        if not replaced:
            self._catalog["samples"].append(entry)

        form_values = self._get_form_values()
        save_sample_catalog(self._catalog)
        self.reload_catalog()
        self._set_form_values(form_values)
        self._enter_new_mode_keep_values()
        self.catalog_updated.emit()
        QMessageBox.information(self, "Saved", "Sample catalog updated.")

    def _select_sample(self, sample_key: str) -> None:
        for index in range(self.sample_list.count()):
            item = self.sample_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == sample_key:
                self.sample_list.setCurrentRow(index)
                return

    def _delete_current_sample(self) -> None:
        if self._current_sample_key is None:
            QMessageBox.information(self, "No Sample", "Select a saved sample first.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Sample",
            f"Delete '{self._current_sample_key}' from the catalog?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._catalog["samples"] = [
            entry
            for entry in self._catalog["samples"]
            if build_sample_catalog_key(entry["sample"], entry["crystal_orientation"]) != self._current_sample_key
        ]
        save_sample_catalog(self._catalog)
        self.reload_catalog()
        self.catalog_updated.emit()
        QMessageBox.information(self, "Deleted", "Sample removed from the catalog.")


class BeamProfileCatalogWidget(QWidget):
    catalog_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog = {"version": 1, "beam_profiles": []}
        self._current_profile_id: str | None = None

        self._build_ui()
        self.reload_catalog()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        self.catalog_path_label = QLabel(f"Catalog: {BEAM_PROFILE_CATALOG_PATH}")
        self.catalog_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_reload = QPushButton("Reload")
        self.btn_new = QPushButton("New Beam Profile")
        self.btn_save = QPushButton("Save Beam Profile")
        self.btn_delete = QPushButton("Delete Beam Profile")
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
        left_layout.addWidget(QLabel("Registered beam profiles"))
        self.profile_list = QListWidget()
        left_layout.addWidget(self.profile_list, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        form_group = QGroupBox("Beam profile editor")
        form = QFormLayout(form_group)
        self.le_id = QLineEdit()
        self.le_id.setPlaceholderText("e.g. 20251017_1940")
        self.le_beam_rx = QLineEdit()
        self.le_beam_rx.setPlaceholderText("e.g. 360.0")
        self.le_beam_ry = QLineEdit()
        self.le_beam_ry.setPlaceholderText("e.g. 423.0")
        self.le_fitting_type = QLineEdit()
        self.le_fitting_type.setPlaceholderText("e.g. Gaussian")

        form.addRow("ID:", self.le_id)
        form.addRow("beam_r_x:", self.le_beam_rx)
        form.addRow("beam_r_y:", self.le_beam_ry)
        form.addRow("fitting_type:", self.le_fitting_type)
        right_layout.addWidget(form_group)

        hint = QLabel(
            "Beam profiles are stored per measurement campaign. "
            "Selecting one during measurement writes beam_r_x and beam_r_y into the saved JSON."
        )
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([360, 540])

        self.btn_reload.clicked.connect(self.reload_catalog)
        self.btn_new.clicked.connect(self._start_new_profile)
        self.btn_save.clicked.connect(self._save_current_profile)
        self.btn_delete.clicked.connect(self._delete_current_profile)
        self.profile_list.currentItemChanged.connect(self._load_selected_profile)

    def reload_catalog(self) -> None:
        self._catalog = load_beam_profile_catalog()
        self.profile_list.clear()
        for entry in self._catalog["beam_profiles"]:
            item = QListWidgetItem(format_beam_profile_display(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            self.profile_list.addItem(item)

        if self.profile_list.count() > 0:
            self.profile_list.setCurrentRow(0)
        else:
            self._start_new_profile()

    def _entry_by_id(self, profile_id: str) -> dict | None:
        for entry in self._catalog["beam_profiles"]:
            if entry["id"] == profile_id:
                return entry
        return None

    def _load_selected_profile(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        profile_id = current.data(Qt.ItemDataRole.UserRole)
        entry = self._entry_by_id(profile_id)
        if entry is None:
            return

        self._current_profile_id = entry["id"]
        self.le_id.setText(entry.get("id") or "")
        self.le_beam_rx.setText("" if entry.get("beam_r_x") is None else f"{entry['beam_r_x']:g}")
        self.le_beam_ry.setText("" if entry.get("beam_r_y") is None else f"{entry['beam_r_y']:g}")
        self.le_fitting_type.setText(entry.get("fitting_type") or "")

    def _start_new_profile(self) -> None:
        self.profile_list.clearSelection()
        self._current_profile_id = None
        self.le_id.clear()
        self.le_beam_rx.clear()
        self.le_beam_ry.clear()
        self.le_fitting_type.clear()

    def _parse_required_float(self, field_name: str, text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError(f"{field_name} is required.")
        try:
            return float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number.") from exc

    def _save_current_profile(self) -> None:
        try:
            beam_r_x = self._parse_required_float("beam_r_x", self.le_beam_rx.text())
            beam_r_y = self._parse_required_float("beam_r_y", self.le_beam_ry.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
            return

        profile_id = self.le_id.text().strip()
        if not profile_id:
            QMessageBox.warning(self, "Input Error", "Please enter an ID.")
            return

        entry = normalize_beam_profile_entry(
            {
                "id": profile_id,
                "beam_r_x": beam_r_x,
                "beam_r_y": beam_r_y,
                "fitting_type": self.le_fitting_type.text().strip(),
            }
        )

        existing_ids = {item["id"] for item in self._catalog["beam_profiles"]}
        overwrite_target_id = None
        if self._current_profile_id is None:
            if profile_id in existing_ids:
                confirm = QMessageBox.question(
                    self,
                    "Beam Profile Already Exists",
                    f"'{profile_id}' is already registered. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return
                overwrite_target_id = profile_id
        elif profile_id != self._current_profile_id and profile_id in existing_ids:
            confirm = QMessageBox.question(
                self,
                "Beam Profile Already Exists",
                f"'{profile_id}' is already registered. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            overwrite_target_id = profile_id

        replaced = False
        target_id = overwrite_target_id or self._current_profile_id
        for index, existing in enumerate(self._catalog["beam_profiles"]):
            if existing["id"] == target_id:
                self._catalog["beam_profiles"][index] = entry
                replaced = True
                break

        if self._current_profile_id and overwrite_target_id and overwrite_target_id != self._current_profile_id:
            self._catalog["beam_profiles"] = [
                existing
                for existing in self._catalog["beam_profiles"]
                if existing["id"] != self._current_profile_id
            ]

        if not replaced:
            self._catalog["beam_profiles"].append(entry)

        save_beam_profile_catalog(self._catalog)
        self.reload_catalog()
        self._select_profile(profile_id)
        self.catalog_updated.emit()
        QMessageBox.information(self, "Saved", "Beam profile catalog updated.")

    def _select_profile(self, profile_id: str) -> None:
        for index in range(self.profile_list.count()):
            item = self.profile_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == profile_id:
                self.profile_list.setCurrentRow(index)
                return

    def _delete_current_profile(self) -> None:
        if self._current_profile_id is None:
            QMessageBox.information(self, "No Beam Profile", "Select a saved beam profile first.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Beam Profile",
            f"Delete '{self._current_profile_id}' from the catalog?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._catalog["beam_profiles"] = [
            entry for entry in self._catalog["beam_profiles"] if entry["id"] != self._current_profile_id
        ]
        save_beam_profile_catalog(self._catalog)
        self.reload_catalog()
        self.catalog_updated.emit()
        QMessageBox.information(self, "Deleted", "Beam profile removed from the catalog.")


class DataManagementWidget(QWidget):
    catalog_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Manage reusable reference data for measurements. "
            "These presets can be pulled into the measurement form so you do not have to type them every time."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.filter_catalog_widget = FilterCatalogWidget()
        self.sample_catalog_widget = SampleCatalogWidget()
        self.beam_profile_catalog_widget = BeamProfileCatalogWidget()

        self.tabs.addTab(self.filter_catalog_widget, "ND Filters")
        self.tabs.addTab(self.sample_catalog_widget, "Samples")
        self.tabs.addTab(self.beam_profile_catalog_widget, "Beam Profiles")

        self.filter_catalog_widget.catalog_updated.connect(self.catalog_updated.emit)
        self.sample_catalog_widget.catalog_updated.connect(self.catalog_updated.emit)
        self.beam_profile_catalog_widget.catalog_updated.connect(self.catalog_updated.emit)
