from __future__ import annotations

import logging
from threading import Lock

import pyvisa
from pyvisa.errors import InvalidSession, LibraryError, VisaIOError


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


class ThorlabsPowerMeterController:
    def __init__(self, timeout_ms: int = 5000) -> None:
        self._timeout_ms = timeout_ms
        self._lock = Lock()
        self._resource_manager = self._make_resource_manager()
        self.inst = None

    def _make_resource_manager(self):
        try:
            return pyvisa.ResourceManager()
        except Exception as exc:
            logging.error(f"Failed to initialize VISA resource manager: {exc}")
            raise

    @property
    def is_connected(self) -> bool:
        return self.inst is not None

    def list_resources(self) -> tuple[str, ...]:
        return self._resource_manager.list_resources()

    def connect(self, resource: str) -> None:
        if self.inst is not None:
            self.disconnect()
        try:
            inst = self._resource_manager.open_resource(resource)
            inst.timeout = self._timeout_ms
            self.inst = inst
        except (VisaIOError, InvalidSession, LibraryError, Exception) as exc:
            logging.error(f"Failed to connect power meter: {exc}")
            raise

    def disconnect(self) -> None:
        if self.inst is None:
            return
        try:
            self.inst.close()
        except (VisaIOError, InvalidSession, LibraryError, Exception) as exc:
            logging.error(f"Failed to disconnect power meter: {exc}")
        finally:
            self.inst = None

    def close(self) -> None:
        self.disconnect()

    def query(self, command: str) -> str:
        if not self.is_connected:
            raise RuntimeError("Power meter is not connected.")
        with self._lock:
            try:
                return str(self.inst.query(command)).strip()
            except (VisaIOError, InvalidSession, LibraryError, Exception) as exc:
                logging.error(f"Power meter query failed for {command}: {exc}")
                raise

    @property
    def idn(self) -> str:
        return self.query("*IDN?")

    def read_power_watts(self) -> float:
        commands = ("MEAS:POW?", "READ?")
        last_error = None
        for command in commands:
            try:
                return float(self.query(command))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to read power meter value. Last error: {last_error}")
