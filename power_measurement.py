from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
from datetime import datetime

from measurement_metadata import beam_metadata_from_entry, sample_metadata_from_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class PowerMeasurementRunner:
    def __init__(self, stage_rot, powermeter):
        self.stage_rot = stage_rot
        self.powermeter = powermeter
        self.scans = []
        self.fundamental_power = None
        self.result = None
        self.is_running = False
        self._abort = False

    def run(
        self,
        sample: str,
        material: str,
        crystal_orientation: str,
        measurement_id: str,
        estimated_angles: list[float],
        scan_range: float,
        step: float,
        axis: str,
        fundamental_wavelength_nm: float,
        shg_wavelength_nm: float,
        operator: str,
        notes: str,
        sample_entry: dict | None = None,
        beam_profile_entry: dict | None = None,
        dry_run: bool = False,
        on_step1_complete=None,
        on_progress=None,
    ) -> dict:
        self._abort = False
        self.is_running = True
        self.scans = []
        self.fundamental_power = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = os.path.join("results", f"{timestamp}_{sample}_power_{measurement_id}")
        os.makedirs(base_dir, exist_ok=True)
        meta_path = os.path.join(base_dir, "power_measurement.json")

        metadata = {
            "sample": sample,
            "material": material,
            "crystal_orientation": crystal_orientation,
            "measurement_id": measurement_id,
            "method": "power_phase_matching_scan",
            "rot/trans_axis": axis,
            "fundamental_wavelength_nm": fundamental_wavelength_nm,
            "shg_wavelength_nm": shg_wavelength_nm,
            "estimated_pm_angles_deg": estimated_angles,
            "scan_range_deg": scan_range,
            "step_deg": step,
            "operator": operator,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }
        if sample_entry is not None:
            metadata.update(sample_metadata_from_entry(sample_entry))
        if beam_profile_entry is not None:
            metadata.update(beam_metadata_from_entry(beam_profile_entry))

        csv_paths = []
        try:
            self.fundamental_power = self._measure_fundamental_power(
                wavelength_nm=fundamental_wavelength_nm,
                dry_run=dry_run,
            )
            metadata["fundamental_power"] = self.fundamental_power
            self._write_metadata(meta_path, metadata, dry_run=dry_run)

            if on_step1_complete:
                on_step1_complete(self.fundamental_power)

            if self._abort:
                return self._finish_result(base_dir, meta_path, csv_paths, aborted=True)

            csv_paths = self._measure_shg_power_scan(
                base_dir=base_dir,
                estimated_angles=estimated_angles,
                scan_range=scan_range,
                step=step,
                shg_wavelength_nm=shg_wavelength_nm,
                dry_run=dry_run,
                on_progress=on_progress,
            )
        finally:
            self.is_running = False
            self.result = self._finish_result(base_dir, meta_path, csv_paths, aborted=self._abort)

        return self.result

    def run_fundamental_power(
        self,
        sample: str,
        material: str,
        crystal_orientation: str,
        measurement_id: str,
        axis: str,
        fundamental_wavelength_nm: float,
        shg_wavelength_nm: float,
        estimated_angles: list[float],
        scan_range: float,
        step: float,
        operator: str,
        notes: str,
        sample_entry: dict | None = None,
        beam_profile_entry: dict | None = None,
        dry_run: bool = False,
    ) -> dict:
        self._abort = False
        self.is_running = True
        self.scans = []
        self.fundamental_power = None
        base_dir, meta_path, metadata = self._prepare_run_context(
            sample=sample,
            material=material,
            crystal_orientation=crystal_orientation,
            measurement_id=measurement_id,
            axis=axis,
            fundamental_wavelength_nm=fundamental_wavelength_nm,
            shg_wavelength_nm=shg_wavelength_nm,
            estimated_angles=estimated_angles,
            scan_range=scan_range,
            step=step,
            operator=operator,
            notes=notes,
            sample_entry=sample_entry,
            beam_profile_entry=beam_profile_entry,
            measurement_kind="fundamental_power",
        )
        try:
            self.fundamental_power = self._measure_fundamental_power(fundamental_wavelength_nm, dry_run=dry_run)
            metadata["fundamental_power"] = self.fundamental_power
            self._write_metadata(meta_path, metadata, dry_run=dry_run)
        finally:
            self.is_running = False
            self.result = self._finish_result(base_dir, meta_path, [], aborted=self._abort)
        return self.result

    def run_shg_power_scan(
        self,
        sample: str,
        material: str,
        crystal_orientation: str,
        measurement_id: str,
        estimated_angles: list[float],
        scan_range: float,
        step: float,
        axis: str,
        fundamental_wavelength_nm: float,
        shg_wavelength_nm: float,
        operator: str,
        notes: str,
        sample_entry: dict | None = None,
        beam_profile_entry: dict | None = None,
        dry_run: bool = False,
        on_progress=None,
    ) -> dict:
        self._abort = False
        self.is_running = True
        self.scans = []
        base_dir, meta_path, metadata = self._prepare_run_context(
            sample=sample,
            material=material,
            crystal_orientation=crystal_orientation,
            measurement_id=measurement_id,
            axis=axis,
            fundamental_wavelength_nm=fundamental_wavelength_nm,
            shg_wavelength_nm=shg_wavelength_nm,
            estimated_angles=estimated_angles,
            scan_range=scan_range,
            step=step,
            operator=operator,
            notes=notes,
            sample_entry=sample_entry,
            beam_profile_entry=beam_profile_entry,
            measurement_kind="shg_power_scan",
        )
        csv_paths = []
        try:
            self._write_metadata(meta_path, metadata, dry_run=dry_run)
            csv_paths = self._measure_shg_power_scan(
                base_dir=base_dir,
                estimated_angles=estimated_angles,
                scan_range=scan_range,
                step=step,
                shg_wavelength_nm=shg_wavelength_nm,
                dry_run=dry_run,
                on_progress=on_progress,
            )
        finally:
            self.is_running = False
            self.result = self._finish_result(base_dir, meta_path, csv_paths, aborted=self._abort)
        return self.result

    def abort(self):
        self._abort = True

    def _prepare_run_context(
        self,
        sample: str,
        material: str,
        crystal_orientation: str,
        measurement_id: str,
        axis: str,
        fundamental_wavelength_nm: float,
        shg_wavelength_nm: float,
        estimated_angles: list[float],
        scan_range: float,
        step: float,
        operator: str,
        notes: str,
        measurement_kind: str,
        sample_entry: dict | None,
        beam_profile_entry: dict | None,
    ) -> tuple[str, str, dict]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = os.path.join("results", f"{timestamp}_{sample}_power_{measurement_id}_{measurement_kind}")
        os.makedirs(base_dir, exist_ok=True)
        meta_path = os.path.join(base_dir, "power_measurement.json")
        metadata = {
            "sample": sample,
            "material": material,
            "crystal_orientation": crystal_orientation,
            "measurement_id": measurement_id,
            "measurement_kind": measurement_kind,
            "method": "power_phase_matching_scan",
            "rot/trans_axis": axis,
            "fundamental_wavelength_nm": fundamental_wavelength_nm,
            "shg_wavelength_nm": shg_wavelength_nm,
            "estimated_pm_angles_deg": estimated_angles,
            "scan_range_deg": scan_range,
            "step_deg": step,
            "operator": operator,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }
        if sample_entry is not None:
            metadata.update(sample_metadata_from_entry(sample_entry))
        if beam_profile_entry is not None:
            metadata.update(beam_metadata_from_entry(beam_profile_entry))
        return base_dir, meta_path, metadata

    def _measure_shg_power_scan(
        self,
        base_dir: str,
        estimated_angles: list[float],
        scan_range: float,
        step: float,
        shg_wavelength_nm: float,
        dry_run: bool,
        on_progress=None,
    ) -> list[str]:
        csv_paths = []
        if not dry_run:
            self.powermeter.set_power_mode()
            self.powermeter.set_wavelength_nm(shg_wavelength_nm)

        for scan_index, angle in enumerate(estimated_angles, start=1):
            if self._abort:
                break
            scan_label = f"theta{scan_index}"
            scan_points = self._make_positive_scan_points(angle, scan_range, step)
            csv_path = os.path.join(base_dir, f"{scan_label}.csv")
            csv_paths.append(csv_path)
            scan_record = {"label": scan_label, "estimated_angle": angle, "positions": [], "powers": []}
            self.scans.append(scan_record)

            with self._open_csv(csv_path, dry_run=dry_run) as writer:
                if writer is not None:
                    writer.writerow(["angle_deg", "power_w", "min_w", "max_w", "std_w", "n"])
                for pos in scan_points:
                    if self._abort:
                        break
                    logging.info("[Power] Moving to %.4f deg for %s", pos, scan_label)
                    if not dry_run:
                        self.stage_rot.move_to_angle(pos, "ccw")
                        time.sleep(2.0)
                        stats = self._average_power_tail(total_wait_s=4.0, tail_s=2.0)
                    else:
                        time.sleep(0.02)
                        stats = self._dry_run_power(pos, angle)

                    power = float(stats["mean_w"])
                    scan_record["positions"].append(pos)
                    scan_record["powers"].append(power)
                    if writer is not None:
                        writer.writerow([pos, power, stats["min_w"], stats["max_w"], stats["std_w"], stats["n"]])
                    if on_progress:
                        on_progress(scan_index - 1, pos, power)
        return csv_paths

    def _measure_fundamental_power(self, wavelength_nm: float, dry_run: bool) -> dict:
        if dry_run:
            values = [1.0002 + random.uniform(-0.000002, 0.000002) for _ in range(30)]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            return {
                "mean_w": mean,
                "min_w": min(values),
                "max_w": max(values),
                "std_w": variance ** 0.5,
                "n": len(values),
                "zero_check_passed": True,
                "zero_mean_w": 0.0,
                "zero_abs_limit_w": 1e-6,
            }

        self.powermeter.set_power_mode()
        self.powermeter.set_wavelength_nm(wavelength_nm)
        self.powermeter.zero(wait_s=30.0)
        zero_stats = self.powermeter.average_power(duration_s=3.0)
        zero_mean = float(zero_stats["mean_w"])
        if abs(zero_mean) > 1e-6:
            raise RuntimeError(f"Zero check failed: mean power {zero_mean:.6g} W exceeds +/- 1 uW.")

        stats = dict(self.powermeter.average_power(duration_s=3.0))
        stats["zero_check_passed"] = True
        stats["zero_mean_w"] = zero_mean
        stats["zero_abs_limit_w"] = 1e-6
        stats.pop("values_w", None)
        return stats

    def _average_power_tail(self, total_wait_s: float, tail_s: float) -> dict:
        values = []
        deadline = time.monotonic() + total_wait_s
        tail_start = deadline - tail_s
        self.powermeter.start_stream()
        try:
            while time.monotonic() < deadline:
                try:
                    value = self.powermeter.read_power()
                    if time.monotonic() >= tail_start:
                        values.append(value)
                except RuntimeError:
                    pass
                time.sleep(0.1)
        finally:
            self.powermeter.stop_stream()
        if not values:
            raise RuntimeError("No Ophir readings were collected during scan averaging.")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean_w": mean, "min_w": min(values), "max_w": max(values), "std_w": variance ** 0.5, "n": len(values)}

    def _dry_run_power(self, pos: float, center: float) -> dict:
        width = 0.35
        value = 2e-6 + 4e-5 * math.exp(-((pos - center) ** 2) / (2 * width**2)) + random.uniform(-2e-7, 2e-7)
        return {"mean_w": value, "min_w": value * 0.98, "max_w": value * 1.02, "std_w": abs(value) * 0.01, "n": 20}

    def _make_positive_scan_points(self, center: float, scan_range: float, step: float) -> list[float]:
        if step <= 0:
            raise ValueError("Step must be positive.")
        start = center - abs(scan_range)
        end = center + abs(scan_range)
        vals = []
        val = start
        while val <= end + step * 1e-9:
            wrapped = (val + 180) % 360 - 180
            vals.append(wrapped)
            val += step
        return vals

    def _write_metadata(self, meta_path: str, metadata: dict, dry_run: bool):
        if dry_run:
            return
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

    def _open_csv(self, csv_path: str, dry_run: bool):
        class NullCsv:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        if dry_run:
            return NullCsv()

        class CsvContext:
            def __enter__(self_inner):
                self_inner.file = open(csv_path, mode="w", newline="", encoding="utf-8")
                return csv.writer(self_inner.file)

            def __exit__(self_inner, exc_type, exc, tb):
                self_inner.file.close()
                return False

        return CsvContext()

    def _finish_result(self, base_dir: str, meta_path: str, csv_paths: list[str], aborted: bool) -> dict:
        return {
            "base_dir": base_dir,
            "meta_path": meta_path,
            "csv_paths": csv_paths,
            "fundamental_power": self.fundamental_power,
            "scans": self.scans,
            "aborted": aborted,
        }
