from __future__ import annotations

import logging
import math
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


@dataclass(frozen=True)
class OphirOptionSet:
    current_index: int | None
    options: list[str]


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

    @staticmethod
    def _option_set_from_response(result: Any) -> OphirOptionSet:
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            current = int(result[0]) if result[0] is not None else None
            options = [str(item) for item in result[1]]
            return OphirOptionSet(current_index=current, options=options)
        return OphirOptionSet(current_index=None, options=[])

    def _set_indexed_option(self, setter_name: str, index: int, options: list[str]) -> str | None:
        self._require_connected()
        if index < 0 or index >= len(options):
            raise ValueError(f"Invalid {setter_name} index: {index}")
        getattr(self._lm, setter_name)(self._device_handle, self.channel, index)
        return options[index]

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
        if self._device_handle in (None, 0, ""):
            raise RuntimeError(
                f"Ophir device '{serial}' was detected but could not be opened. "
                "Close StarLab or any other software using the device and try again."
            )
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

    def get_version(self) -> str | None:
        self.start_application()
        try:
            return str(self._lm.GetVersion())
        except Exception as exc:
            logging.info("Failed to read Ophir COM version: %s", exc)
            return None

    def reset_device(self) -> None:
        self._require_connected()
        self.stop_stream()
        self._lm.ResetDevice(self._device_handle)

    def write_legacy_command(self, command: str) -> None:
        self._require_connected()
        self._lm.Write(self._device_handle, command)
        time.sleep(0.01)

    def ask_legacy_command(self, command: str) -> str:
        self.write_legacy_command(command)
        try:
            return str(self._lm.Read(self._device_handle)).strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to read Ophir response for command {command!r}.") from exc

    @staticmethod
    def _legacy_response_failed(response: str) -> bool:
        text = response.strip().upper()
        return text.startswith("?") or any(token in text for token in ("ERROR", "FAILED", "INVALID", "UNKNOWN"))

    def get_measurement_modes(self) -> OphirOptionSet:
        self._require_connected()
        return self._option_set_from_response(self._lm.GetMeasurementMode(self._device_handle, self.channel))

    def set_measurement_mode_index(self, index: int) -> str | None:
        modes = self.get_measurement_modes().options
        return self._set_indexed_option("SetMeasurementMode", index, modes)

    def set_power_mode(self) -> None:
        self._require_connected()
        try:
            modes = self.get_measurement_modes().options
            power_index = next((index for index, label in enumerate(modes) if "power" in label.lower()), 0)
            self.set_measurement_mode_index(power_index)
        except Exception as exc:
            logging.info("Could not explicitly set Ophir measurement mode to Power: %s", exc)

    def get_wavelengths(self) -> tuple[int | None, list[str]]:
        option_set = self.get_wavelength_options()
        return option_set.current_index, option_set.options

    def get_wavelength_options(self) -> OphirOptionSet:
        self._require_connected()
        return self._option_set_from_response(self._lm.GetWavelengths(self._device_handle, self.channel))

    def set_wavelength_index(self, index: int) -> str | None:
        wavelengths = self.get_wavelength_options().options
        return self._set_indexed_option("SetWavelength", index, wavelengths)

    def modify_wavelength_index(self, index: int, wavelength_nm: float) -> str:
        wavelengths = self.get_wavelength_options().options
        if index < 0 or index >= len(wavelengths):
            raise ValueError(f"Invalid wavelength index: {index}")
        target = float(wavelength_nm)
        self._lm.ModifyWavelength(self._device_handle, self.channel, index, target)
        self._lm.SetWavelength(self._device_handle, self.channel, index)
        return f"{target:g} nm"

    def delete_wavelength_index(self, index: int) -> None:
        wavelengths = self.get_wavelength_options().options
        if index < 0 or index >= len(wavelengths):
            raise ValueError(f"Invalid wavelength index: {index}")
        try:
            delete_method = getattr(self._lm, "DeleteWavelength")
        except AttributeError as exc:
            raise NotImplementedError(
                "This Ophir COM API does not expose wavelength deletion. "
                "Use Add/Set Wavelength to overwrite an editable wavelength slot."
            ) from exc
        delete_method(self._device_handle, self.channel, index)

    def add_or_select_wavelength_nm(self, wavelength_nm: float, tolerance_nm: float = 1.0) -> str:
        self._require_connected()
        target = float(wavelength_nm)
        current_index, options = self.get_wavelengths()

        best_index = None
        best_error = None
        editable_index = None
        for index, label in enumerate(options):
            label_lower = label.lower()
            if editable_index is None and any(token in label_lower for token in ("user", "modify", "custom")):
                editable_index = index
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", label)
            if match is None:
                continue
            error = abs(float(match.group(1)) - target)
            if best_error is None or error < best_error:
                best_index = index
                best_error = error

        if best_index is not None and best_error is not None and best_error <= tolerance_nm:
            self.set_wavelength_index(best_index)
            return options[best_index]

        slot_index = editable_index
        if slot_index is None:
            slot_index = current_index if current_index is not None else 0
        return self.modify_wavelength_index(slot_index, target)

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

        self.set_wavelength_index(best_index)
        return options[best_index]

    def get_range_options(self) -> OphirOptionSet:
        self._require_connected()
        return self._option_set_from_response(self._lm.GetRanges(self._device_handle, self.channel))

    def set_range_index(self, index: int) -> str | None:
        ranges = self.get_range_options().options
        return self._set_indexed_option("SetRange", index, ranges)

    def set_range_auto(self) -> str | None:
        ranges = self.get_range_options().options
        auto_index = next((index for index, label in enumerate(ranges) if "auto" in label.lower()), None)
        if auto_index is None:
            logging.info("Ophir range list does not expose an Auto range; leaving current range unchanged.")
            return None
        return self.set_range_index(auto_index)

    def set_range_index_or_auto(self, index: int | None) -> str | None:
        if index is None:
            return self.set_range_auto()
        return self.set_range_index(int(index))

    def get_pulse_length_options(self) -> OphirOptionSet:
        self._require_connected()
        try:
            return self._option_set_from_response(self._lm.GetPulseLengths(self._device_handle, self.channel))
        except Exception as exc:
            logging.info("Pulse length settings are not available for this Ophir device/sensor: %s", exc)
            return OphirOptionSet(current_index=None, options=[])

    def set_pulse_length_index(self, index: int) -> str | None:
        pulse_lengths = self.get_pulse_length_options().options
        return self._set_indexed_option("SetPulseLength", index, pulse_lengths)

    def configure_default_stream_mode(self) -> None:
        self._require_connected()
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 0, 0)
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 2, 0)

    def configure_immediate_stream_mode(self) -> None:
        self._require_connected()
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 0, 0)
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 2, 1)

    def configure_turbo_stream_mode(self, frequency_hz: float) -> None:
        self._require_connected()
        if frequency_hz <= 0:
            raise ValueError("Turbo frequency must be positive.")
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 2, 0)
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 1, float(frequency_hz))
        self._lm.ConfigureStreamMode(self._device_handle, self.channel, 0, 1)

    def zero(self, wait_s: float = 30.0, poll_interval_s: float = 0.5) -> None:
        self._require_connected()
        self.stop_stream()
        zero_command = None
        first_response = None
        for command in ("ZE", "$ZE"):
            try:
                response = self.ask_legacy_command(command)
                if self._legacy_response_failed(response):
                    logging.debug("Ophir zero command %s returned %s", command, response)
                    continue
                zero_command = command
                first_response = response
                break
            except Exception as exc:
                logging.debug("Ophir zero command %s failed: %s", command, exc)
        if zero_command is None:
            raise RuntimeError("Failed to send zero command to Ophir power meter.")

        deadline = time.monotonic() + max(0.0, wait_s)
        last_status = first_response
        query_command = "ZQ" if zero_command == "ZE" else "$ZQ"
        while time.monotonic() < deadline:
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))
            try:
                last_status = self.ask_legacy_command(query_command)
            except Exception as exc:
                logging.debug("Ophir zero status query %s failed: %s", query_command, exc)
                continue

            status_text = str(last_status).upper()
            if "COMPLETED" in status_text:
                return
            if "FAILED" in status_text or "ABORTED" in status_text:
                raise RuntimeError(f"Ophir zeroing failed: {last_status}")

        raise RuntimeError(f"Ophir zeroing did not complete within {wait_s:g} s. Last status: {last_status}")

    def start_stream(self) -> None:
        self._ensure_com_initialized()
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

    def read_available_data(self) -> list[OphirPowerReading]:
        self._ensure_com_initialized()
        self._require_connected()
        data = self._lm.GetData(self._device_handle, self.channel)
        values, timestamps, statuses = self._normalize_data_response(data)
        readings = []
        for index, value in enumerate(values):
            timestamp = None
            status = None
            if index < len(timestamps):
                try:
                    timestamp = float(timestamps[index])
                except (TypeError, ValueError):
                    timestamp = None
            if index < len(statuses):
                status = statuses[index]
            readings.append(OphirPowerReading(power_w=float(value), timestamp=timestamp, status=status))
        return readings

    @staticmethod
    def _is_valid_reading(reading: OphirPowerReading) -> bool:
        if not math.isfinite(reading.power_w):
            return False
        if reading.status is None:
            return True
        status_text = str(reading.status).strip().lower()
        if not status_text:
            return True
        invalid_tokens = ("error", "invalid", "over", "under", "missing", "saturated")
        return not any(token in status_text for token in invalid_tokens)

    def average_power(
        self,
        duration_s: float,
        sample_interval_s: float = 0.1,
        warmup_s: float = 0.0,
    ) -> dict[str, float | int | list[float]]:
        values: list[float] = []
        self.start_stream()
        try:
            try:
                self._lm.GetData(self._device_handle, self.channel)
            except Exception:
                pass
            if warmup_s > 0:
                time.sleep(warmup_s)
                try:
                    self._lm.GetData(self._device_handle, self.channel)
                except Exception:
                    pass

            deadline = time.monotonic() + max(0.0, duration_s)
            while time.monotonic() < deadline:
                readings = self.read_available_data()
                values.extend(reading.power_w for reading in readings if self._is_valid_reading(reading))
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
