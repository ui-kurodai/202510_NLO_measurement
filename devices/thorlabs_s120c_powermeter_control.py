from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from thorlabs_power_meter_controller import ThorlabsPowerMeterController as TLPMController

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
    App-facing adapter around ui-kurodai/thorlabs-power-meter-controller.

    S120C is the sensor; the USB resource is a TLPM-compatible Thorlabs meter
    such as PM100/PM200/PM400. This adapter keeps the same small API surface
    used by PowerMeasurementRunner and the device widget.
    """

    zero_wait_s = 3.0
    zero_check_duration_s = 2.0
    scan_average_total_s = 1.0
    scan_average_tail_s = 1.0

    def __init__(self, library_path: str | None = None, sample_interval_s: float = 0.05):
        self.library_path = library_path
        self.sample_interval_s = float(sample_interval_s)
        self._meter = TLPMController(library_path=library_path)
        self._devices_by_resource = {}

    @property
    def is_connected(self) -> bool:
        return bool(self._meter.is_connected)

    @property
    def device_serial(self) -> str | None:
        info = self._meter.device_info
        return None if info is None else info.serial_number

    @property
    def zero_offset_w(self) -> float:
        return float(self._meter.zero_offset_w)

    def scan_usb(self) -> list[str]:
        devices = TLPMController.list_devices(library_path=self.library_path)
        self._devices_by_resource = {device.resource_name: device for device in devices}
        return list(self._devices_by_resource)

    def connect(self, resource_name: str | None = None) -> None:
        self._meter.connect(resource_name=resource_name)
        self.set_power_mode()
        self.set_average_time(0.001)
        self.set_timeout_ms(1000)
        self.set_range_auto()

    def disconnect(self) -> None:
        self._meter.disconnect()

    def shutdown(self) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Thorlabs power meter is not connected.")

    def get_sensor_info(self) -> str:
        self._require_connected()
        try:
            sensor = self._meter.get_sensor_info()
        except Exception as exc:
            return f"unavailable ({exc})"
        fields = [
            sensor.get("name"),
            sensor.get("serial_number"),
            sensor.get("calibration_message"),
        ]
        return " / ".join(str(field) for field in fields if field)

    def get_version(self) -> str | None:
        return None

    def set_power_mode(self) -> None:
        self._require_connected()

    def set_wavelength_nm(self, wavelength_nm: float, tolerance_nm: float = 0.0) -> str:
        del tolerance_nm
        self._require_connected()
        selected = self._meter.set_wavelength_nm(float(wavelength_nm))
        return f"{selected:g} nm"

    def set_average_time(self, average_time_s: float) -> None:
        self._require_connected()
        self._meter.set_average_time_s(float(average_time_s))

    def set_timeout_ms(self, timeout_ms: int) -> None:
        self._require_connected()
        self._meter.set_timeout_ms(int(timeout_ms))

    def get_range_options(self) -> ThorlabsOptionSet:
        return ThorlabsOptionSet(current_index=None, options=[])

    def set_range_auto(self) -> str:
        self._require_connected()
        self._meter.set_power_auto_range(True)
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
        if not self.is_connected:
            return ThorlabsOptionSet(current_index=None, options=[])
        try:
            wavelength_nm = self._meter.get_wavelength_nm()
        except Exception:
            return ThorlabsOptionSet(current_index=None, options=[])
        return ThorlabsOptionSet(current_index=0, options=[f"{wavelength_nm:g} nm"])

    def get_pulse_length_options(self) -> ThorlabsOptionSet:
        return ThorlabsOptionSet(current_index=None, options=[])

    def start_stream(self) -> None:
        self._require_connected()

    def stop_stream(self) -> None:
        pass

    def zero(self, wait_s: float = 3.0, poll_interval_s: float = 0.05) -> None:
        self._require_connected()
        self._meter.zero_offset(duration_s=float(wait_s), sample_interval_s=float(poll_interval_s))
        logging.info("Thorlabs S120C zero offset set to %.6g W", self.zero_offset_w)

    def read_power(self) -> float:
        return self.read_power_with_metadata().power_w

    def read_power_with_metadata(self) -> ThorlabsPowerReading:
        self._require_connected()
        reading = self._meter.read_power()
        return ThorlabsPowerReading(
            power_w=float(reading.power_w),
            timestamp=float(reading.timestamp),
            status=reading.unit,
        )

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
        return self._meter.average_power_w(duration_s=float(duration_s), sample_interval_s=interval)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
