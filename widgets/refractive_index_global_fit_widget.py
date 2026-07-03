from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class RefractiveIndexGlobalFitWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        folder_group = QGroupBox("Folder & Global n-Fit Setup")
        folder_layout = QVBoxLayout(folder_group)
        self.btn_nfit_select_folders = QPushButton("Select Measurement Folder(s)…")
        self.lbl_current_folder = QLabel("No folder loaded.")
        self.lbl_current_folder.setWordWrap(True)
        self.lbl_current_folder.setStyleSheet("color: gray;")
        self.lbl_nfit_intro = QLabel(
            "Select one measurement folder for a single-folder fit, or select multiple folders "
            "for one global n-fit. Uncheck a folder to exclude it from fitting while still applying "
            "the fitted refractive-index result to that folder."
        )
        self.lbl_nfit_intro.setWordWrap(True)
        self.lbl_nfit_intro.setStyleSheet("color: gray;")
        self.lst_nfit_group_paths = QListWidget()
        self.lst_nfit_group_paths.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.lst_nfit_group_paths.setMinimumHeight(120)

        button_row = QHBoxLayout()
        self.btn_nfit_refresh = QPushButton("Refresh Preview")
        button_row.addWidget(self.btn_nfit_refresh)

        self.lbl_nfit_hint = QLabel("Select a global n-fit strategy to preview grouped measurements.")
        self.lbl_nfit_hint.setWordWrap(True)
        self.lbl_nfit_hint.setStyleSheet("color: gray;")

        folder_layout.addWidget(self.btn_nfit_select_folders)
        folder_layout.addWidget(self.lbl_current_folder)
        folder_layout.addWidget(self.lbl_nfit_intro)
        folder_layout.addWidget(self.lst_nfit_group_paths)
        folder_layout.addLayout(button_row)
        folder_layout.addWidget(self.lbl_nfit_hint)

        left_layout.addWidget(folder_group)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        self.nfit_scroll = QScrollArea()
        self.nfit_scroll.setWidgetResizable(True)
        self.nfit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nfit_measurements_host = QWidget()
        self.nfit_measurements_layout = QVBoxLayout(self.nfit_measurements_host)
        self.nfit_measurements_layout.setContentsMargins(0, 0, 0, 0)
        self.nfit_measurements_layout.setSpacing(8)
        self.nfit_scroll.setWidget(self.nfit_measurements_host)
        splitter.addWidget(self.nfit_scroll)
        splitter.setSizes([360, 720])

        layout.addWidget(splitter, 1)
