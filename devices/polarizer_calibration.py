from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np
from scipy.optimize import curve_fit

from devices.polarization_control import normalize_angle


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - in %(filename)s - %(message)s")


def polarizer_response_model(angle_deg, baseline, amplitude, zero_offset_deg):
    radians = np.radians(np.asarray(angle_deg, dtype=float) - zero_offset_deg)
    return baseline + amplitude * np.cos(radians) ** 2


@dataclass
class PolarizerCalibrationResult:
    angles_deg: list[float]
    signals: list[float]
    fit_angles_deg: list[float]
    fit_signals: list[float]
    zero_offset_deg: float
    baseline: float
    amplitude: float
    fit_success: bool


def fit_polarizer_calibration(angles_deg: list[float], signals: list[float]) -> PolarizerCalibrationResult:
    x = np.asarray(angles_deg, dtype=float)
    y = np.asarray(signals, dtype=float)
    if x.size < 3:
        raise ValueError("At least three data points are required for calibration.")

    baseline_guess = float(np.min(y))
    amplitude_guess = float(max(np.max(y) - baseline_guess, 1e-9))
    zero_guess = float(x[np.argmax(y)])

    popt, _ = curve_fit(
        polarizer_response_model,
        x,
        y,
        p0=[baseline_guess, amplitude_guess, zero_guess],
        bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=10000,
    )

    baseline, amplitude, zero_offset_deg = [float(v) for v in popt]
    fit_angles = np.linspace(float(np.min(x)), float(np.max(x)), 361)
    fit_signals = polarizer_response_model(fit_angles, baseline, amplitude, zero_offset_deg)
    return PolarizerCalibrationResult(
        angles_deg=list(x),
        signals=list(y),
        fit_angles_deg=list(fit_angles),
        fit_signals=list(np.asarray(fit_signals, dtype=float)),
        zero_offset_deg=normalize_angle(zero_offset_deg),
        baseline=baseline,
        amplitude=amplitude,
        fit_success=True,
    )


def run_polarizer_calibration(
    polarizer_controller,
    read_signal,
    start_deg: float,
    end_deg: float,
    step_deg: float,
    settle_time_s: float = 0.3,
    sample_count: int = 3,
    progress_callback=None,
) -> PolarizerCalibrationResult:
    if step_deg <= 0:
        raise ValueError("Calibration step must be positive.")
    if end_deg < start_deg:
        raise ValueError("Calibration end angle must be larger than start angle.")
    if sample_count <= 0:
        raise ValueError("Sample count must be positive.")

    angles_deg = []
    signals = []
    current = start_deg
    while current <= end_deg + 1e-9:
        raw_angle = polarizer_controller.move_raw(current)
        time.sleep(settle_time_s)
        samples = [float(read_signal()) for _ in range(sample_count)]
        signal_value = float(np.mean(samples))
        angles_deg.append(raw_angle)
        signals.append(signal_value)
        if progress_callback is not None:
            progress_callback(raw_angle, signal_value)
        current += step_deg

    logging.info(f"Calibration scan collected {len(angles_deg)} points for {polarizer_controller.name}")
    return fit_polarizer_calibration(angles_deg, signals)
