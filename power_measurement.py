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
    def __init__(self, stage_rot, powermeter, laser=None):
        self.stage_rot = stage_rot
        self.powermeter = powermeter
        self.laser = laser
        self.scans = []
        self.fundamental_power = None
        self.result = None
        self.is_running = False
        self._abort = False
        self._last_zero_stats = None

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
        fundamental_range_index: int | None = None,
        shg_range_index: int | None = None,
        fundamental_range_label: str | None = None,
        shg_range_label: str | None = None,
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
        self._last_zero_stats = None

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
            "fundamental_measurement_range": fundamental_range_label or "Auto",
            "shg_measurement_range": shg_range_label or "Auto",
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
            self._prepare_optical_measurement(
                wavelength_nm=fundamental_wavelength_nm,
                range_index=fundamental_range_index,
                dry_run=dry_run,
            )
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

            self._prepare_optical_measurement(
                wavelength_nm=shg_wavelength_nm,
                range_index=shg_range_index,
                dry_run=dry_run,
            )
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
        fundamental_range_index: int | None = None,
        shg_range_index: int | None = None,
        fundamental_range_label: str | None = None,
        shg_range_label: str | None = None,
        sample_entry: dict | None = None,
        beam_profile_entry: dict | None = None,
        dry_run: bool = False,
    ) -> dict:
        self._abort = False
        self.is_running = True
        self.scans = []
        self.fundamental_power = None
        self._last_zero_stats = None
        base_dir, meta_path, metadata = self._prepare_run_context(
            sample=sample,
            material=material,
            crystal_orientation=crystal_orientation,
            measurement_id=measurement_id,
            axis=axis,
            fundamental_wavelength_nm=fundamental_wavelength_nm,
            shg_wavelength_nm=shg_wavelength_nm,
            fundamental_range_label=fundamental_range_label,
            shg_range_label=shg_range_label,
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
            self._prepare_optical_measurement(
                wavelength_nm=fundamental_wavelength_nm,
                range_index=fundamental_range_index,
                dry_run=dry_run,
            )
            self.fundamental_power = self._measure_fundamental_power(fundamental_wavelength_nm, dry_run=dry_run)
            metadata["fundamental_power"] = self.fundamental_power
        finally:
            self.is_running = False
            self.result = self._finish_result(None, None, [], aborted=self._abort, metadata=metadata)
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
        fundamental_range_index: int | None = None,
        shg_range_index: int | None = None,
        fundamental_range_label: str | None = None,
        shg_range_label: str | None = None,
        sample_entry: dict | None = None,
        beam_profile_entry: dict | None = None,
        dry_run: bool = False,
        on_progress=None,
    ) -> dict:
        self._abort = False
        self.is_running = True
        self.scans = []
        self._last_zero_stats = None
        base_dir, meta_path, metadata = self._prepare_run_context(
            sample=sample,
            material=material,
            crystal_orientation=crystal_orientation,
            measurement_id=measurement_id,
            axis=axis,
            fundamental_wavelength_nm=fundamental_wavelength_nm,
            shg_wavelength_nm=shg_wavelength_nm,
            fundamental_range_label=fundamental_range_label,
            shg_range_label=shg_range_label,
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
            self._prepare_optical_measurement(
                wavelength_nm=shg_wavelength_nm,
                range_index=shg_range_index,
                dry_run=dry_run,
            )
            csv_paths = self._measure_shg_power_scan(
                base_dir=None,
                estimated_angles=estimated_angles,
                scan_range=scan_range,
                step=step,
                shg_wavelength_nm=shg_wavelength_nm,
                dry_run=dry_run,
                on_progress=on_progress,
            )
        finally:
            self.is_running = False
            self.result = self._finish_result(None, None, csv_paths, aborted=self._abort, metadata=metadata)
        return self.result

    def abort(self):
        self._abort = True

    def _prepare_optical_measurement(self, wavelength_nm: float, range_index: int | None, dry_run: bool) -> None:
        if dry_run:
            return
        self._ensure_laser_off()
        self.powermeter.set_power_mode()
        self.powermeter.set_wavelength_nm(wavelength_nm)
        self.powermeter.set_range_index_or_auto(range_index)
        self.powermeter.zero(wait_s=float(getattr(self.powermeter, "zero_wait_s", 30.0)))
        self._last_zero_stats = self._read_zero_after_zeroing(
            duration_s=float(getattr(self.powermeter, "zero_check_duration_s", 3.0))
        )
        self._ensure_laser_on()

    def _ensure_laser_off(self) -> None:
        if self.laser is None:
            return
        try:
            is_on = bool(self.laser.is_emission_on)
        except Exception as exc:
            logging.warning("Could not read laser emission state before zeroing: %s", exc)
            is_on = True
        if not is_on:
            logging.info("[Power] Laser emission already OFF for zeroing.")
            return

        logging.info("[Power] Turning laser OFF before power meter zeroing.")
        self.laser.stop()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                if not self.laser.is_emission_on:
                    logging.info("[Power] Laser emission confirmed OFF for zeroing.")
                    return
            except Exception as exc:
                logging.warning("Could not verify laser emission OFF state: %s", exc)
            time.sleep(0.5)
        raise RuntimeError("Laser stop command was sent, but emission did not turn OFF within 8 s.")

    def _read_zero_after_zeroing(self, duration_s: float = 3.0, abs_limit_w: float = 1e-6) -> dict:
        zero_stats = dict(self.powermeter.average_power(duration_s=duration_s))
        zero_stats.pop("values_w", None)
        zero_mean = float(zero_stats["mean_w"])
        zero_stats["zero_check_passed"] = abs(zero_mean) <= abs_limit_w
        zero_stats["zero_mean_w"] = zero_mean
        zero_stats["zero_abs_limit_w"] = abs_limit_w
        return zero_stats

    def _ensure_laser_on(self) -> None:
        if self.laser is None:
            return
        try:
            is_on = bool(self.laser.is_emission_on)
        except Exception as exc:
            logging.warning("Could not read laser emission state before power measurement: %s", exc)
            is_on = False
        if not is_on:
            logging.info("[Power] Turning laser ON before power measurement.")
            self.laser.start()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                if self.laser.is_emission_on:
                    logging.info("[Power] Laser emission confirmed ON.")
                    return
            except Exception as exc:
                logging.warning("Could not verify laser emission state: %s", exc)
            time.sleep(0.5)

        try:
            error_message = self.laser.last_error_message
        except Exception:
            error_message = "unavailable"
        raise RuntimeError(
            "Laser start command was sent, but emission did not turn ON within 8 s. "
            f"Check interlock, fatal error, laser readiness, and controller status. Last error: {error_message}"
        )

    def _prepare_run_context(
        self,
        sample: str,
        material: str,
        crystal_orientation: str,
        measurement_id: str,
        axis: str,
        fundamental_wavelength_nm: float,
        shg_wavelength_nm: float,
        fundamental_range_label: str | None,
        shg_range_label: str | None,
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
        base_dir = os.path.join("PM_power_results", f"{timestamp}_{sample}_power_{measurement_id}_{measurement_kind}")
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
            "fundamental_measurement_range": fundamental_range_label or "Auto",
            "shg_measurement_range": shg_range_label or "Auto",
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
        base_dir: str | None,
        estimated_angles: list[float],
        scan_range: float,
        step: float,
        shg_wavelength_nm: float,
        dry_run: bool,
        on_progress=None,
    ) -> list[str]:
        csv_paths = []
        if not dry_run:
            self.powermeter.set_wavelength_nm(shg_wavelength_nm)
            self._return_rotation_stage_origin()
            self.powermeter.start_stream()

        try:
            for scan_index, angle in enumerate(estimated_angles, start=1):
                if self._abort:
                    break
                scan_label = f"theta{scan_index}"
                scan_points = self._make_positive_scan_points(angle, scan_range, step)
                csv_path = None if base_dir is None else os.path.join(base_dir, f"{scan_label}.csv")
                if csv_path is not None:
                    csv_paths.append(csv_path)
                scan_record = {
                    "label": scan_label,
                    "estimated_angle": angle,
                    "positions": [],
                    "powers": [],
                    "stats": [],
                }
                self.scans.append(scan_record)

                with self._open_csv(csv_path, dry_run=dry_run or csv_path is None) as writer:
                    if writer is not None:
                        writer.writerow(["angle_deg", "power_w", "std_w", "n"])
                    for pos in scan_points:
                        if self._abort:
                            break
                        logging.info("[Power] Moving to %.4f deg for %s", pos, scan_label)
                        if not dry_run:
                            self.stage_rot.move_to_angle(pos, "ccw")
                            stats = self._average_power_after_settle_tail(
                                total_wait_s=float(getattr(self.powermeter, "scan_average_total_s", 4.0)),
                                tail_s=float(getattr(self.powermeter, "scan_average_tail_s", 2.0)),
                            )
                        else:
                            time.sleep(0.02)
                            stats = self._dry_run_power(pos, angle)

                        power = float(stats["mean_w"])
                        scan_record["positions"].append(pos)
                        scan_record["powers"].append(power)
                        scan_record["stats"].append(dict(stats))
                        if writer is not None:
                            writer.writerow([pos, power, stats["std_w"], stats["n"]])
                        if on_progress:
                            on_progress(scan_index - 1, pos, power)
        finally:
            if not dry_run:
                self.powermeter.stop_stream()
        return csv_paths

    def _measure_fundamental_power(self, wavelength_nm: float, dry_run: bool) -> dict:
        if dry_run:
            time.sleep(0.02)
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

        self._wait_for_laser_settling(duration_s=20.0)
        stats = dict(self.powermeter.average_power(duration_s=3.0))
        zero_stats = self._last_zero_stats or {}
        stats["zero_check_passed"] = bool(zero_stats.get("zero_check_passed", False))
        stats["zero_mean_w"] = float(zero_stats.get("zero_mean_w", 0.0))
        stats["zero_abs_limit_w"] = float(zero_stats.get("zero_abs_limit_w", 1e-6))
        stats.pop("values_w", None)
        return stats

    def _wait_for_laser_settling(self, duration_s: float) -> None:
        logging.info("[Power] Waiting %.1f s for laser power to settle before fundamental measurement.", duration_s)
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            if self._abort:
                return
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    def _return_rotation_stage_origin(self) -> None:
        if self.stage_rot is None:
            return
        logging.info("[Power] Returning rotation stage to origin before SHG scan.")
        try:
            self.stage_rot.reset()
        except TypeError:
            self.stage_rot.reset("-")

    def _average_power_after_settle_tail(self, total_wait_s: float, tail_s: float) -> dict:
        self._drain_powermeter_stream()
        deadline = time.monotonic() + max(0.0, total_wait_s)
        tail_start = deadline - max(0.0, tail_s)
        values = []
        while time.monotonic() < deadline:
            readings = self.powermeter.read_available_data()
            if time.monotonic() >= tail_start:
                values.extend(
                    reading.power_w
                    for reading in readings
                    if self.powermeter._is_valid_reading(reading)
                )
            time.sleep(0.1)
        if not values:
            raise RuntimeError("No power meter readings were collected during SHG averaging.")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean_w": mean, "min_w": min(values), "max_w": max(values), "std_w": variance ** 0.5, "n": len(values)}

    def _drain_powermeter_stream(self) -> None:
        try:
            while self.powermeter.read_available_data():
                pass
        except RuntimeError:
            pass

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
            raise RuntimeError("No power meter readings were collected during scan averaging.")
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

    def _open_csv(self, csv_path: str | None, dry_run: bool):
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

    def _finish_result(
        self,
        base_dir: str | None,
        meta_path: str | None,
        csv_paths: list[str],
        aborted: bool,
        metadata: dict | None = None,
    ) -> dict:
        return {
            "base_dir": base_dir,
            "meta_path": meta_path,
            "csv_paths": csv_paths,
            "fundamental_power": self.fundamental_power,
            "scans": self.scans,
            "aborted": aborted,
            "metadata": metadata or {},
        }
