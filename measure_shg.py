import os
import csv
import json
import time
from datetime import datetime
import logging

# from devices.laser_control import CrylasQLaserController, CrylasQLaserDecoder
# from devices.osms2035_control import OSMS2035Controller
# from devices.osms60yaw_control import OSMS60YAWController
# from devices.boxcar_control import BoxcarInterfaceController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - in %(filename)s - %(message)s')

class SHGMeasurementRunner:
    def __init__(self, laser, stage_lin, stage_rot, boxcar, elliptec=None):
        self.laser = laser
        self.stage_lin = stage_lin
        self.stage_rot = stage_rot
        self.boxcar = boxcar
        self.elliptec = elliptec

        self.positions = []
        self.signals = []
        self.channels = []

        self.result = None
        self.is_running = False
        self._abort = False

    def run(self,
        sample: str,
        material: str,
        crystal_orientation,
        measured_coefficient: str,
        method: str,
        input_polarization: float,
        detected_polarization: float,
        repetition: str,
        operator: str,
        notes: str,
        start: float,
        end: float,
        step: float,
        channels: list[int],
        axis: str,
        dry_run: bool = False,
        on_progress=None    # emit the latest signal for realtime GUI display
    ) -> dict:
        """
        Run SHG measurement. If skip_laser=False, laser is started/stopped internally.
        Returns measurement result for plotting or saving.
        """
        self.positions = []
        self.signals = []
        self.channels = channels
        self._abort = False
        self.is_running = True
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = os.path.join("results", f"{timestamp}_{sample}_{measured_coefficient}_{method}")
        os.makedirs(base_dir, exist_ok=True)
        base_filename = f"in{input_polarization}_out{detected_polarization}"
        csv_path = os.path.join(base_dir, base_filename + ".csv")
        meta_path = os.path.join(base_dir, base_filename + ".json")

        class DummyDevice:
            def __getattr__(self, name):
                return None  # or some default dummy value
            
        if dry_run:
            self.laser = DummyDevice()

        metadata = {
            # sample data
            "sample": sample,
            "material": material,
            "crystal_orientation": crystal_orientation,
            # "thickness_info": {
            #     "t_at_thin_end_mm": t_thin,
            #     "wedge_angle_deg": wedge,
            #     },

            # # preparation
            # "beam_r_x": r_x,
            # "beam_r_y": r_y,

            # measurement data
            "method": method,
            "rot/trans_axis": axis,
            "wavelength_nm": self.laser.wavelength_nm,
            "input_polarization": input_polarization,
            "detected_polarization": detected_polarization,
            "ref_ch":channels[0],
            "sig_ch":channels[1],
            "repetition": repetition,
            "operator": operator,
            "notes": notes,
            "start": start,
            "end": end,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }        

        if not dry_run:
            with open(meta_path, "w") as meta_file:
                json.dump(metadata, meta_file, indent=2)

        if not dry_run:
            print("[measure_shg] Turning laser ON...")
            self.laser.start()

        scan_values = self._make_scan_points(start, end, step)
        csv_file = None
        writer = None
        if not dry_run:
            csv_file = open(csv_path, mode="w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(["position"] + [f"ch{ch}" for ch in channels])

        try:
            self.stage_lin.reset()
            self.stage_rot.reset()
            if method == "rotation":
                # move to the center
                center = 18.05
                self.stage_lin.millimeter = center
            for pos in scan_values:
                if self._abort:
                    logging.warning("[SHG] Measurement aborted.")
                    break
                logging.info(f"[SHG] Moving to {pos:.3f} ({method})")

                if method == "rotation":
                    self.stage_rot.move_to_angle(pos, "ccw")

                elif method == "wedge":
                    self.stage_lin.millimeter = pos
                else:
                    logging.error(f"Unknown method: {method}")

                time.sleep(0.3)

                if dry_run:
                    signals = [0.0 for _ in channels]
                else:
                    signals = [self.boxcar.read_analog(ch) for ch in channels]

                self.positions.append(pos)
                self.signals.append(signals)

                if writer:
                    writer.writerow([pos] + signals)

                if on_progress:
                    on_progress(pos, signals)


        finally:
            if writer:
                csv_file.close()
            if not dry_run:
                print("[measure_shg] Turning laser OFF...")
                self.laser.stop()

            self.is_running = False

            self.result = {
                "positions": self.positions,
                "signals": self.signals,
                "channels": self.channels,
                "csv_path": csv_path if not dry_run else None,
                "meta_path": meta_path if not dry_run else None
            }
    
    def abort(self):
        """Stop the measurement early (from GUI or user action)."""
        self._abort = True

    def _make_scan_points(self, start: float, end: float, step: float) -> list:
        if start > end:
            logging.error("The start angle should be smaller than the end")
        if not (-180 <= start < 180 and -180 <= end < 180):
            logging.error("Invalid target angle: -180 <= target < 180")        
        vals = []
        val = start
        while val <= end:
            vals.append(val)
            val += step
        return vals
