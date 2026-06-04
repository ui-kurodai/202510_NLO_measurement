from __future__ import annotations

import importlib.util
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


@dataclass(frozen=True)
class ThorlabsPowerReading:
    power_w: float
    timestamp: float | None = None
    status: Any | None = None


@dataclass(frozen=True)
class ThorlabsOptionSet:
    current_index: int | None
    options: list[str]


class ThorlabsS120CPowerMeterController:
    """
    Adapter for Tinyblack's ThorlabsPowerMeter class.

    S120C is the photodiode sensor; the USB resource is typically a PM100/PM200/PM400
    console. This wrapper keeps the same app-facing API as the Ophir controller.
    """

    zero_wait_s = 3.0
    zero_check_duration_s = 2.0
    scan_average_total_s = 1.0
    scan_average_tail_s = 1.0

    def __init__(self, library_path: str | None = None, sample_interval_s: float = 0.05):
        self.library_path = library_path
        self.sample_interval_s = float(sample_interval_s)
        self._driver_list = None
        self._device = None
        self._driver_class = None
        self._zero_offset_w = 0.0
        self._last_scan_resources: list[str] = []

    @property
    def is_connected(self) -> bool:
        return self._device is not None and bool(getattr(self._device, "isConnected", False))

    @property
    def device_serial(self) -> str | None:
        if self._device is None:
            return None
        serial = getattr(self._device, "serialNumber", None)
        return str(serial) if serial is not None else None

    @property
    def zero_offset_w(self) -> float:
        return self._zero_offset_w

    def _load_driver_class(self):
        if self._driver_class is not None:
            return self._driver_class

        root = Path(__file__).resolve().parents[1]
        candidates = [
            root / "ThorlabsPowerMeter.py",
            root / "external" / "Python-Driver-for-Thorlabs-power-meter" / "ThorlabsPowerMeter.py",
        ]
        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("ThorlabsPowerMeter", path)
            if spec is None or spec.loader is None:
                continue
            module_dir = str(path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self._driver_class = module.ThorlabsPowerMeter
            return self._driver_class

        try:
            from ThorlabsPowerMeter import ThorlabsPowerMeter
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import Tinyblack's ThorlabsPowerMeter class. "
                "Clone https://github.com/Tinyblack/Python-Driver-for-Thorlabs-power-meter "
                "into external/Python-Driver-for-Thorlabs-power-meter, place ThorlabsPowerMeter.py "
                "in the project root, or install it on PYTHONPATH. The Tinyblack driver's GlobalLogger "
                "module must be importable too."
            ) from exc
        self._driver_class = ThorlabsPowerMeter
        return self._driver_class

    def scan_usb(self) -> list[str]:
        driver = self._load_driver_class()
        try:
            self._reset_driver_lists(driver)
            if self.library_path:
                self._driver_list = driver.listDevices(self.library_path)
            else:
                self._driver_list = driver.listDevices()
        except Exception as exc:
            raise RuntimeError(
                "Failed to scan Thorlabs power meters. Install Thorlabs Optical Power Monitor, "
                "make Thorlabs.TLPM_64.Interop.dll available, and install pythonnet."
            ) from exc
        resources = [str(item) for item in getattr(self._driver_list, "resourceName", [])]
        self._last_scan_resources = resources
        return resources

    @staticmethod
    def _reset_driver_lists(driver) -> None:
        for attr in ("resourceName", "modelName", "serialNumber", "manufacturer"):
            setattr(driver, attr, [])
        driver.resourceCount = 0

    def connect(self, resource_name: str | None = None) -> None:
        if self._driver_list is None:
            self.scan_usb()
        resources = list(getattr(self._driver_list, "resourceName", []) or self._last_scan_resources)
        if not resources:
            raise RuntimeError("No Thorlabs power meter resource was detected.")
        resource = resource_name or resources[0]
        if resource not in resources:
            raise RuntimeError(f"Thorlabs resource '{resource}' was not found. Available: {', '.join(resources)}")
        device = self._driver_list.connect(resource)
        if device is None:
            raise RuntimeError(f"Thorlabs resource '{resource}' was detected but could not be opened.")
        self._device = device
        self.set_power_mode()
        self.set_average_time(0.001)
        self.set_timeout_ms(1000)

    def disconnect(self) -> None:
        if self._device is not None:
            try:
                self._device.disconnect()
            except Exception as exc:
                logging.warning("Failed to disconnect Thorlabs power meter: %s", exc)
        self._device = None

    def shutdown(self) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Thorlabs power meter is not connected.")

    def get_sensor_info(self) -> str:
        self._require_connected()
        try:
            self._device.getSensorInfo()
        except Exception as exc:
            return f"unavailable ({exc})"
        fields = [
            getattr(self._device, "sensorName", None),
            getattr(self._device, "sensorSerialNumber", None),
            getattr(self._device, "sensorType", None),
        ]
        return " / ".join(str(field) for field in fields if field)

    def get_version(self) -> str | None:
        driver = self._load_driver_class()
        return str(getattr(driver, "driverVersion", "") or "") or None

    def set_power_mode(self) -> None:
        self._require_connected()

    def set_wavelength_nm(self, wavelength_nm: float, tolerance_nm: float = 0.0) -> str:
        del tolerance_nm
        self._require_connected()
        self._device.setWaveLength(float(wavelength_nm))
        return f"{float(wavelength_nm):g} nm"

    def set_average_time(self, average_time_s: float) -> None:
        self._require_connected()
        self._device.setAverageTime(float(average_time_s))

    def set_timeout_ms(self, timeout_ms: int) -> None:
        self._require_connected()
        self._device.setTimeoutValue(int(timeout_ms))

    def get_range_options(self) -> ThorlabsOptionSet:
        return ThorlabsOptionSet(current_index=None, options=[])

    def set_range_auto(self) -> str:
        self._require_connected()
        self._device.setPowerAutoRange(True)
        return "Auto"

    def set_range_index_or_auto(self, index: int | None) -> str:
        if index is None:
            return self.set_range_auto()
        return self.set_range_index(index)

    def set_range_index(self, index: int) -> str:
        del index
        return self.set_range_auto()

    def get_measurement_modes(self) -> ThorlabsOptionSet:
        return ThorlabsOptionSet(current_index=0, options=["Power"])

    def get_wavelength_options(self) -> ThorlabsOptionSet:
        if self._device is None or getattr(self._device, "wavelengthSet", None) is None:
            return ThorlabsOptionSet(current_index=None, options=[])
        return ThorlabsOptionSet(current_index=0, options=[f"{float(self._device.wavelengthSet):g} nm"])

    def get_pulse_length_options(self) -> ThorlabsOptionSet:
        return ThorlabsOptionSet(current_index=None, options=[])

    def start_stream(self) -> None:
        self._require_connected()

    def stop_stream(self) -> None:
        pass

    def zero(self, wait_s: float = 3.0, poll_interval_s: float = 0.05) -> None:
        self._require_connected()
        stats = self._average_raw_power(duration_s=wait_s, sample_interval_s=poll_interval_s)
        self._zero_offset_w = float(stats["mean_w"])
        logging.info("Thorlabs S120C zero offset set to %.6g W", self._zero_offset_w)

    def read_power(self) -> float:
        return self.read_power_with_metadata().power_w

    def read_power_with_metadata(self) -> ThorlabsPowerReading:
        self._require_connected()
        try:
            self._device.updatePowerReading(0.0)
        except TypeError:
            self._device.updatePowerReading()
        value = self._reading_to_watts(self._device.meterPowerReading, self._device.meterPowerUnit)
        return ThorlabsPowerReading(power_w=value - self._zero_offset_w, timestamp=time.monotonic())

    def read_available_data(self) -> list[ThorlabsPowerReading]:
        return [self.read_power_with_metadata()]

    @staticmethod
    def _is_valid_reading(reading: ThorlabsPowerReading) -> bool:
        return math.isfinite(reading.power_w)

    def average_power(
        self,
        duration_s: float,
        sample_interval_s: float | None = None,
        warmup_s: float = 0.0,
    ) -> dict[str, float | int | list[float]]:
        if warmup_s > 0:
            time.sleep(warmup_s)
        interval = self.sample_interval_s if sample_interval_s is None else float(sample_interval_s)
        values = []
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            values.append(self.read_power())
            time.sleep(max(0.0, interval))
        if not values:
            values.append(self.read_power())
        return self._stats(values)

    def _average_raw_power(self, duration_s: float, sample_interval_s: float) -> dict[str, float | int | list[float]]:
        values = []
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            self._device.updatePowerReading(0.0)
            values.append(self._reading_to_watts(self._device.meterPowerReading, self._device.meterPowerUnit))
            time.sleep(max(0.0, sample_interval_s))
        if not values:
            self._device.updatePowerReading(0.0)
            values.append(self._reading_to_watts(self._device.meterPowerReading, self._device.meterPowerUnit))
        return self._stats(values)

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | int | list[float]]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {
            "mean_w": mean,
            "min_w": min(values),
            "max_w": max(values),
            "std_w": variance ** 0.5,
            "n": len(values),
            "values_w": values,
        }

    @staticmethod
    def _reading_to_watts(value: Any, unit: Any) -> float:
        power = float(value)
        unit_text = str(unit or "W").strip().lower()
        if unit_text == "w":
            return power
        if unit_text == "dbm":
            return 1e-3 * 10 ** (power / 10.0)
        raise RuntimeError(f"Unsupported Thorlabs power unit: {unit}")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
