from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import elliptec


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "results" / "polarizer_calibration.json"


def normalize_angle(angle_deg: float) -> float:
    normalized = angle_deg % 360.0
    if normalized < 0:
        normalized += 360.0
    return round(normalized, 4)


def logical_polarizer_angle(raw_angle_deg: float, zero_offset_deg: float) -> float:
    logical = normalize_angle(raw_angle_deg - zero_offset_deg)
    if logical >= 180.0:
        logical -= 180.0
    return round(logical, 4)


class PolarizerCalibrationStore:
    @classmethod
    def load_all(cls) -> dict[str, Any]:
        if not CALIBRATION_FILE.exists():
            return {}
        try:
            return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning(f"Failed to load polarizer calibration file: {exc}")
            return {}

    @classmethod
    def load_entry(cls, key: str) -> dict[str, Any]:
        data = cls.load_all()
        entry = data.get(key)
        return entry if isinstance(entry, dict) else {}

    @classmethod
    def save_entry(cls, key: str, entry: dict[str, Any]) -> None:
        data = cls.load_all()
        data[key] = entry
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class _SharedElliptecConnection:
    controller: Any
    lock: Lock
    ref_count: int = 0


class ElliptecControllerHub:
    _connections: dict[str, _SharedElliptecConnection] = {}
    _hub_lock = Lock()

    @classmethod
    def acquire(cls, port: str, debug: bool = False) -> _SharedElliptecConnection:
        with cls._hub_lock:
            connection = cls._connections.get(port)
            if connection is None:
                controller = elliptec.Controller(port, debug=debug)
                connection = _SharedElliptecConnection(controller=controller, lock=Lock(), ref_count=0)
                cls._connections[port] = connection
            connection.ref_count += 1
            return connection

    @classmethod
    def release(cls, port: str) -> None:
        with cls._hub_lock:
            connection = cls._connections.get(port)
            if connection is None:
                return
            connection.ref_count -= 1
            if connection.ref_count <= 0:
                try:
                    connection.controller.close_connection()
                except Exception as exc:
                    logging.warning(f"Failed to close Elliptec connection on {port}: {exc}")
                finally:
                    cls._connections.pop(port, None)


class ElliptecPolarizerController:
    def __init__(self, name: str, calibration_key: str, default_address: str = "0") -> None:
        self.name = name
        self.calibration_key = calibration_key
        self.default_address = str(default_address).upper()
        self.port: str | None = None
        self.address = self.default_address
        self._shared_connection: _SharedElliptecConnection | None = None
        self.rotator = None
        self._zero_offset_deg = 0.0
        self._calibration_metadata = PolarizerCalibrationStore.load_entry(self.calibration_key)
        self._zero_offset_deg = float(self._calibration_metadata.get("zero_offset_deg", 0.0))

    @property
    def is_connected(self) -> bool:
        return self.rotator is not None and self._shared_connection is not None

    @property
    def zero_offset_deg(self) -> float:
        return round(self._zero_offset_deg, 4)

    @property
    def calibration_metadata(self) -> dict[str, Any]:
        return dict(self._calibration_metadata)

    def connect(self, port: str, address: str | None = None, debug: bool = False) -> None:
        if self.is_connected:
            self.disconnect()

        address_to_use = str(address or self.default_address).upper()
        shared_connection = ElliptecControllerHub.acquire(port, debug=debug)
        try:
            with shared_connection.lock:
                rotator = elliptec.Rotator(shared_connection.controller, address=address_to_use, debug=debug)
        except Exception:
            ElliptecControllerHub.release(port)
            raise

        self.port = port
        self.address = address_to_use
        self._shared_connection = shared_connection
        self.rotator = rotator
        logging.info(f"{self.name} connected on {port} (address {self.address})")

    def disconnect(self) -> None:
        if self.port is not None:
            ElliptecControllerHub.release(self.port)
        logging.info(f"{self.name} disconnected")
        self.port = None
        self.rotator = None
        self._shared_connection = None

    def _run_locked(self, func, *args, **kwargs):
        if not self.is_connected or self._shared_connection is None:
            raise RuntimeError(f"{self.name} is not connected.")
        with self._shared_connection.lock:
            return func(*args, **kwargs)

    def get_raw_angle(self) -> float:
        angle = self._run_locked(self.rotator.get_angle)
        if angle is None:
            raise RuntimeError(f"{self.name} did not return an angle.")
        return normalize_angle(float(angle))

    def get_logical_angle(self) -> float:
        return logical_polarizer_angle(self.get_raw_angle(), self.zero_offset_deg)

    def home(self) -> float:
        self._run_locked(self.rotator.home)
        return self.get_raw_angle()

    def move_raw(self, raw_angle_deg: float) -> float:
        target = normalize_angle(raw_angle_deg)
        result = self._run_locked(self.rotator.set_angle, target)
        if result is None:
            raise RuntimeError(f"{self.name} failed to move to {target:.2f} deg.")
        return normalize_angle(float(result))

    def move_to_polarization(self, logical_angle_deg: float) -> float:
        target_raw = normalize_angle(self.zero_offset_deg + logical_angle_deg)
        return self.move_raw(target_raw)

    def set_zero_offset(self, zero_offset_deg: float, persist: bool = True, extra_metadata: dict[str, Any] | None = None) -> None:
        metadata = self.calibration_metadata
        metadata["zero_offset_deg"] = round(normalize_angle(zero_offset_deg), 4)
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if extra_metadata:
            metadata.update(extra_metadata)
        self._calibration_metadata = metadata
        self._zero_offset_deg = float(metadata["zero_offset_deg"])
        if persist:
            PolarizerCalibrationStore.save_entry(self.calibration_key, metadata)

