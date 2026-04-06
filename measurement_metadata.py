from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


APP_DATA_DIR = Path(__file__).resolve().parent / "app_data"
ND_FILTER_CATALOG_PATH = APP_DATA_DIR / "nd_filter_catalog.json"
COMMON_BOXCAR_SENSITIVITIES = [
    "1 V / 1 V",
    "1 V / 0.5 V",
    "1 V / 0.2 V",
    "1 V / 0.1 V",
    "1 V / 0.05 V",
    "1 V / 0.02 V",
    "1 V / 0.01 V",
    "1 V / 0.005 V",
]
_BOXCAR_SENSITIVITY_PATTERN = re.compile(
    r"^\s*([0-9]*\.?[0-9]+)\s*V\s*/\s*([0-9]*\.?[0-9]+)\s*V\s*$",
    re.IGNORECASE,
)


def ensure_nd_filter_catalog_exists() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ND_FILTER_CATALOG_PATH.exists():
        save_nd_filter_catalog({"version": 1, "filters": []})


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def build_filter_id(product_name: str, instance_id: str) -> str:
    product = str(product_name or "").strip()
    instance = str(instance_id or "").strip()
    if product and instance:
        return f"{product}_{instance}"
    if product:
        return product
    if instance:
        return instance
    return "filter"


def normalize_filter_entry(entry: dict[str, Any]) -> dict[str, Any]:
    product_name = str(entry.get("product_name") or "").strip()
    instance_id = str(entry.get("instance_id") or "").strip()
    normalized = {
        "filter_id": str(entry.get("filter_id") or "").strip(),
        "product_name": product_name,
        "instance_id": instance_id,
        "nominal_od": _to_optional_float(entry.get("nominal_od")),
        "transmission_csv_path": _to_optional_path(entry.get("transmission_csv_path")),
        "notes": str(entry.get("notes") or "").strip(),
    }
    if not normalized["filter_id"]:
        normalized["filter_id"] = build_filter_id(
            normalized["product_name"],
            normalized["instance_id"],
        )
    return normalized


def load_nd_filter_catalog() -> dict[str, Any]:
    ensure_nd_filter_catalog_exists()
    with ND_FILTER_CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    filters = catalog.get("filters", [])
    if not isinstance(filters, list):
        filters = []

    normalized_filters = [normalize_filter_entry(entry) for entry in filters]
    normalized_filters.sort(key=lambda entry: format_filter_display(entry).lower())

    return {
        "version": int(catalog.get("version", 1)),
        "filters": normalized_filters,
    }


def save_nd_filter_catalog(catalog: dict[str, Any]) -> None:
    ensure_nd_filter_catalog_exists()
    filters = catalog.get("filters", [])
    normalized_filters = [normalize_filter_entry(entry) for entry in filters]
    normalized_filters.sort(key=lambda entry: format_filter_display(entry).lower())
    payload = {
        "version": int(catalog.get("version", 1)),
        "filters": normalized_filters,
    }
    with ND_FILTER_CATALOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_filter_display(entry: dict[str, Any]) -> str:
    label = entry.get("product_name") or entry.get("filter_id") or "Unnamed filter"
    details = []
    if entry.get("instance_id"):
        details.append(str(entry["instance_id"]))
    if entry.get("nominal_od") is not None:
        details.append(f"OD {entry['nominal_od']:g}")
    if entry.get("transmission_csv_path"):
        details.append("CSV")
    return " | ".join([label] + details) if details else label


def parse_boxcar_sensitivity(text: str) -> dict[str, float | str]:
    cleaned = text.strip()
    match = _BOXCAR_SENSITIVITY_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError("Use the format '1 V / 0.1 V'.")

    output_full_scale_v = float(match.group(1))
    input_full_scale_v = float(match.group(2))
    if output_full_scale_v <= 0 or input_full_scale_v <= 0:
        raise ValueError("Sensitivity values must be positive.")

    label = f"{output_full_scale_v:g} V / {input_full_scale_v:g} V"
    return {
        "label": label,
        "magnification": output_full_scale_v / input_full_scale_v,
    }


def _transmission_from_nominal_od(nominal_od: float | None) -> float | None:
    if nominal_od is None:
        return None
    return 10 ** (-nominal_od)


def _safe_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _load_transmission_spectrum(csv_path: Path) -> tuple[list[float], list[float], str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        wavelength_key = None
        transmission_key = None
        for field in reader.fieldnames:
            lowered = field.strip().lower()
            if wavelength_key is None and "nm" in lowered:
                wavelength_key = field
            if transmission_key is None and ("%t" in lowered or "trans" in lowered or lowered in {"t", "% t"}):
                transmission_key = field

        if wavelength_key is None or transmission_key is None:
            raise ValueError("CSV must contain wavelength and transmission columns.")

        wavelengths: list[float] = []
        transmissions: list[float] = []
        for row in reader:
            wavelength = _safe_float(row.get(wavelength_key, ""))
            transmission_raw = _safe_float(row.get(transmission_key, ""))
            if wavelength is None or transmission_raw is None:
                continue
            transmission = transmission_raw / 100.0 if "%" in transmission_key else transmission_raw
            if transmission > 1.0 and "%" not in transmission_key:
                transmission = transmission / 100.0
            wavelengths.append(wavelength)
            transmissions.append(transmission)

    if len(wavelengths) < 2:
        raise ValueError("CSV must contain at least two valid rows.")

    pairs = sorted(zip(wavelengths, transmissions), key=lambda item: item[0])
    wavelengths_sorted = [item[0] for item in pairs]
    transmissions_sorted = [item[1] for item in pairs]
    return wavelengths_sorted, transmissions_sorted, transmission_key


def _interpolate_transmission(wavelengths: list[float], transmissions: list[float], wavelength_nm: float) -> float | None:
    if wavelength_nm < wavelengths[0] or wavelength_nm > wavelengths[-1]:
        return None
    for index, current_wavelength in enumerate(wavelengths):
        if math.isclose(current_wavelength, wavelength_nm, rel_tol=0.0, abs_tol=1e-9):
            return transmissions[index]
        if current_wavelength > wavelength_nm:
            x0 = wavelengths[index - 1]
            x1 = current_wavelength
            y0 = transmissions[index - 1]
            y1 = transmissions[index]
            ratio = (wavelength_nm - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return transmissions[-1]


def resolve_filter_transmission(
    entry: dict[str, Any],
    shg_wavelength_nm: float | None,
) -> dict[str, Any]:
    normalized = normalize_filter_entry(entry)
    warnings: list[str] = []
    fallback_transmission = _transmission_from_nominal_od(normalized["nominal_od"])
    csv_path_text = normalized.get("transmission_csv_path")

    if csv_path_text and shg_wavelength_nm is not None:
        csv_path = Path(csv_path_text)
        try:
            wavelengths, transmissions, transmission_column = _load_transmission_spectrum(csv_path)
            interpolated = _interpolate_transmission(wavelengths, transmissions, shg_wavelength_nm)
            if interpolated is not None:
                return {
                    "applied_transmission": interpolated,
                    "warnings": warnings,
                    "fallback_used": False,
                }
            warnings.append(
                f"{normalized['filter_id']}: SHG wavelength {shg_wavelength_nm:g} nm is outside the transmission CSV range "
                f"({wavelengths[0]:g}-{wavelengths[-1]:g} nm)."
            )
            dataset_range_nm = [wavelengths[0], wavelengths[-1]]
            dataset_column = transmission_column
        except Exception as exc:
            warnings.append(f"{normalized['filter_id']}: failed to read transmission CSV ({exc}).")
            dataset_range_nm = None
            dataset_column = None
    elif csv_path_text and shg_wavelength_nm is None:
        warnings.append(f"{normalized['filter_id']}: laser wavelength is unavailable, so CSV transmission was not used.")
        dataset_range_nm = None
        dataset_column = None
    elif not csv_path_text:
        warnings.append(f"{normalized['filter_id']}: transmission CSV is not registered.")
        dataset_range_nm = None
        dataset_column = None

    if fallback_transmission is not None:
        return {
            "applied_transmission": fallback_transmission,
            "warnings": warnings,
            "fallback_used": True,
        }

    warnings.append(f"{normalized['filter_id']}: no nominal OD is registered, so no transmission could be derived.")
    return {
        "applied_transmission": None,
        "warnings": warnings,
        "fallback_used": False,
    }


def resolve_selected_filters(
    selected_filters: list[dict[str, Any]],
    fundamental_wavelength_nm: float | None,
) -> tuple[dict[str, float], list[str]]:
    shg_wavelength_nm = None if fundamental_wavelength_nm is None else fundamental_wavelength_nm / 2.0
    filters_dict: dict[str, float] = {}
    warnings: list[str] = []

    for entry in selected_filters:
        normalized = normalize_filter_entry(entry)
        resolved = resolve_filter_transmission(normalized, shg_wavelength_nm)
        warnings.extend(resolved["warnings"])
        applied_transmission = resolved["applied_transmission"]
        if applied_transmission is not None:
            filters_dict[normalized["filter_id"]] = applied_transmission

    return filters_dict, warnings


def apply_condition_metadata(
    metadata: dict[str, Any],
    boxcar_sensitivity_text: str,
    selected_filters: list[dict[str, Any]],
    fundamental_wavelength_nm: float | None,
) -> tuple[dict[str, Any], list[str]]:
    sensitivity = parse_boxcar_sensitivity(boxcar_sensitivity_text)
    filters_dict, warnings = resolve_selected_filters(
        selected_filters=selected_filters,
        fundamental_wavelength_nm=fundamental_wavelength_nm,
    )

    for legacy_key in [
        "boxcar_gain",
        "boxcar_settings",
        "nd_filters",
        "nd_filter_ids",
        "nd_filter_labels",
        "condition_warnings",
    ]:
        metadata.pop(legacy_key, None)

    metadata["boxcar_sensitivity"] = sensitivity
    metadata["filters"] = filters_dict
    return metadata, warnings
