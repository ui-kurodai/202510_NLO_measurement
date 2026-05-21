from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


@dataclass(frozen=True)
class OphirPowerReading:
    power_w: float
    timestamp: float | None = None
    status: Any | None = None


class Ophir3APowerMeterController:
    """Thin wrapper around Ophir's StarLab/LMMeasurement COM object."""

    def __init__(self, channel: int = 0, timeout_s: float = 5.0):
        self.channel = int(channel)
        self.timeout_s = float(timeout_s)
        self._lm = None
        self._device_handle = None
        self._device_serial = None

    @property
    def is_connected(self) -> bool:
        return self._lm is not None and self._device_handle is not None

    @property
    def device_serial(self) -> str | None:
        return self._device_serial

    def _ensure_com_initialized(self) -> None:
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except Exception:
            pass

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Ophir power meter is not connected.")

    def start_application(self) -> None:
        self._ensure_com_initialized()
        if self._lm is None:
            try:
                import win32com.client
            except ImportError as exc:
                raise RuntimeError("pywin32 is required for Ophir LMMeasurement COM control.") from exc

            try:
                self._lm = win32com.client.Dispatch("OphirLMMeasurement.CoLMMeasurement")
            except Exception as exc:
                raise RuntimeError(
                    "Ophir LMMeasurement COM class is not registered. "
                    "Install StarLab with the LMMeasurement/Automation components, "
                    "then make sure this Python process has the same 32/64-bit architecture "
                    "as the registered Ophir COM component."
                ) from exc
            try:
                self._lm.StartApplication()
            except Exception as exc:
                logging.info("Ophir COM object has no StartApplication method or does not require it: %s", exc)

    def scan_usb(self) -> list[str]:
        self.start_application()
        devices = self._lm.ScanUSB()
        if devices is None:
            return []
        return [str(device) for device in devices]

    def connect(self, device_serial: str | None = None) -> None:
        self.start_application()
        devices = self.scan_usb()
        if not devices:
            raise RuntimeError("No Ophir USB devices were detected.")

        serial = device_serial or devices[0]
        if serial not in devices:
            raise RuntimeError(f"Ophir device '{serial}' was not found. Available: {', '.join(devices)}")

        self._device_handle = self._lm.OpenUSBDevice(serial)
        self._device_serial = serial
        logging.info("Connected Ophir power meter %s", serial)

    def disconnect(self) -> None:
        if self._lm is not None and self._device_handle is not None:
            try:
                self.stop_stream()
            except Exception:
                pass
            try:
                self._lm.Close(self._device_handle)
            except Exception as exc:
                logging.warning("Failed to close Ophir device: %s", exc)
        self._device_handle = None
        self._device_serial = None

    def shutdown(self) -> None:
        self.disconnect()
        if self._lm is not None:
            try:
                self._lm.StopApplication()
            except Exception as exc:
                logging.info("Ophir COM object has no StopApplication method or does not require it: %s", exc)
        self._lm = None

    def get_sensor_info(self) -> Any:
        self._require_connected()
        return self._lm.GetSensorInfo(self._device_handle, self.channel)

    def set_power_mode(self) -> None:
        self._require_connected()
        try:
            self._lm.SetMeasurementMode(self._device_handle, self.channel, "Power")
        except Exception as exc:
            logging.info("Could not explicitly set Ophir measurement mode to Power: %s", exc)

    def get_wavelengths(self) -> tuple[int | None, list[str]]:
        self._require_connected()
        result = self._lm.GetWavelengths(self._device_handle, self.channel)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            current = int(result[0]) if result[0] is not None else None
            options = [str(item) for item in result[1]]
            return current, options
        return None, []

    def set_wavelength_nm(self, wavelength_nm: float, tolerance_nm: float = 1.0) -> str | None:
        self._require_connected()
        target = float(wavelength_nm)
        current_index, options = self.get_wavelengths()

        best_index = None
        best_error = None
        for index, label in enumerate(options):
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", label)
            if match is None:
                continue
            error = abs(float(match.group(1)) - target)
            if best_error is None or error < best_error:
                best_index = index
                best_error = error

        if best_index is None or best_error is None or best_error > tolerance_nm:
            if current_index is not None:
                try:
                    self._lm.ModifyWavelength(self._device_handle, self.channel, current_index, target)
                    self._lm.SetWavelength(self._device_handle, self.channel, current_index)
                    return f"{target:g} nm"
                except Exception as exc:
                    logging.warning("Failed to modify Ophir wavelength slot: %s", exc)
            raise RuntimeError(
                f"No Ophir wavelength entry near {target:g} nm was found. "
                "Add this wavelength in StarLab or use a sensor/meter configuration with editable wavelengths."
            )

        self._lm.SetWavelength(self._device_handle, self.channel, best_index)
        return options[best_index]

    def zero(self, wait_s: float = 30.0, poll_interval_s: float = 0.5) -> None:
        self._require_connected()
        self.stop_stream()
        sent = False
        for command in ("$ZE", "ZE"):
            try:
                self._lm.Write(self._device_handle, command)
                sent = True
                break
            except Exception as exc:
                logging.debug("Ophir zero command %s failed: %s", command, exc)
        if not sent:
            raise RuntimeError("Failed to send zero command to Ophir power meter.")

        deadline = time.monotonic() + max(0.0, wait_s)
        while time.monotonic() < deadline:
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))

    def start_stream(self) -> None:
        self._require_connected()
        self._lm.StartStream(self._device_handle, self.channel)

    def stop_stream(self) -> None:
        if self.is_connected:
            try:
                self._lm.StopStream(self._device_handle, self.channel)
            except Exception:
                pass

    def read_power(self) -> float:
        return self.read_power_with_metadata().power_w

    def read_power_with_metadata(self) -> OphirPowerReading:
        self._require_connected()
        data = self._lm.GetData(self._device_handle, self.channel)
        values, timestamps, statuses = self._normalize_data_response(data)
        if not values:
            raise RuntimeError("No Ophir data point is available yet.")
        return OphirPowerReading(
            power_w=float(values[-1]),
            timestamp=float(timestamps[-1]) if timestamps else None,
            status=statuses[-1] if statuses else None,
        )

    def average_power(self, duration_s: float, sample_interval_s: float = 0.1) -> dict[str, float | int | list[float]]:
        values: list[float] = []
        deadline = time.monotonic() + max(0.0, duration_s)
        self.start_stream()
        try:
            while time.monotonic() < deadline:
                try:
                    values.append(self.read_power())
                except RuntimeError:
                    pass
                time.sleep(sample_interval_s)
        finally:
            self.stop_stream()

        if not values:
            raise RuntimeError("No Ophir readings were collected.")
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
    def _normalize_data_response(data: Any) -> tuple[list[float], list[float], list[Any]]:
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            values = list(data[0] or [])
            timestamps = list(data[1] or [])
            statuses = list(data[2] or [])
            return values, timestamps, statuses
        if isinstance(data, (list, tuple)):
            return [float(value) for value in data], [], []
        return [float(data)], [], []

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
