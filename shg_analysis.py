import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import math

class SHGDataAnalysis:
    def __init__(self, base_path):
        """Initialize with paths and crystal database"""
        self.base_path = base_path
        json_path = [file for file in os.listdir(base_path) if file.endswith(".json")]
        if len(json_path) == 1:
            self.json_path = os.path.join(base_path, json_path[0])
        else:
            raise ValueError("Multiple json file detected")
        csv_path = [file for file in os.listdir(base_path) if file.endswith(".csv")]
        if len(csv_path) == 1:
            self.csv_path = os.path.join(base_path, csv_path[0])
        else:
            raise ValueError("Multiple csv file detected")
        self.meta = None
        self.data = None
        self._load_metadata()
        self._load_data()

    def _load_metadata(self):
        """Load measurement metadata from JSON"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def _load_data(self):
        """Load measured CSV data and compute corrected intensity (main/BBO)"""
        # df = pd.read_csv(self.csv_path)
        # self.data = {
        #     "position": df["position"].to_numpy(),
        #     "BBO_ref": df.iloc[:, 1].to_numpy(),
        #     "main_signal": df.iloc[:, 2].to_numpy()
        # }
        self.data = pd.read_csv(self.csv_path)
        self.data["intensity_corrected"] = self.data.iloc[:, 2] / self.data.iloc[:, 1]

    def calc_thickness_array(self):
        """Calculate crystal thickness array from metadata"""
        t_thin = self.meta["thickness_info"]["t_at_thin_end_mm"]
        wedge_angle_deg = self.meta["thickness_info"]["wedge_angle_deg"]

        if self.meta["method"] == "rotation":
            return np.full_like(self.data["position"], t_thin, dtype=float)
        elif self.meta["method"] == "wedge":
            return t_thin + self.data["position"] * np.tan(np.radians(wedge_angle_deg))
        else:
            raise ValueError(f"Unknown method: {self.meta['method']}")

    def run(self, strategy):
        """Run a fitting strategy that implements fit_all(meta, data, crystal_db)."""
        return strategy.fit_all(strategy, self)
