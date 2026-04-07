from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


APP_DATA_DIR = Path(__file__).resolve().parent / "app_data"
ND_FILTER_CATALOG_PATH = APP_DATA_DIR / "nd_filter_catalog.json"
SAMPLE_CATALOG_PATH = APP_DATA_DIR / "sample_catalog.json"
BEAM_PROFILE_CATALOG_PATH = APP_DATA_DIR / "beam_profile_catalog.json"
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
        with ND_FILTER_CATALOG_PATH.open("w", encoding="utf-8") as f:
            json.dump({"version": 1, "filters": []}, f, ensure_ascii=False, indent=2)


def ensure_sample_catalog_exists() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_CATALOG_PATH.exists():
        with SAMPLE_CATALOG_PATH.open("w", encoding="utf-8") as f:
            json.dump({"version": 1, "samples": []}, f, ensure_ascii=False, indent=2)


def ensure_beam_profile_catalog_exists() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not BEAM_PROFILE_CATALOG_PATH.exists():
        with BEAM_PROFILE_CATALOG_PATH.open("w", encoding="utf-8") as f:
            json.dump({"version": 1, "beam_profiles": []}, f, ensure_ascii=False, indent=2)


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def normalize_crystal_orientation(value: Any) -> str:
    if value in (None, ""):
        return ""

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned in {"100", "010", "001"}:
            return cleaned

        digits = re.sub(r"[^01]", "", cleaned)
        if digits in {"100", "010", "001"}:
            return digits
        return cleaned

    try:
        items = list(value)
    except TypeError:
        return str(value).strip()

    if len(items) != 3:
        return "".join(str(item).strip() for item in items)

    normalized_items: list[str] = []
    for item in items:
        try:
            number = float(item)
        except (TypeError, ValueError):
            normalized_items.append(str(item).strip())
            continue

        if math.isclose(number, 0.0, abs_tol=1e-9):
            normalized_items.append("0")
        elif math.isclose(number, 1.0, abs_tol=1e-9):
            normalized_items.append("1")
        else:
            normalized_items.append(f"{number:g}")
    return "".join(normalized_items)


def _normalize_thickness_info(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        value = {}
    return {
        "wedge_angle_deg": _to_optional_float(value.get("wedge_angle_deg")),
        "t_center_mm": _to_optional_float(value.get("t_center_mm")),
    }


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


def build_sample_catalog_key(sample: str, crystal_orientation: Any) -> str:
    sample_text = str(sample or "").strip()
    orientation_text = normalize_crystal_orientation(crystal_orientation)
    if sample_text and orientation_text:
        return f"{sample_text}_{orientation_text}"
    return sample_text or orientation_text or "sample"


def normalize_sample_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": str(entry.get("sample") or "").strip(),
        "material": str(entry.get("material") or "").strip(),
        "crystal_orientation": normalize_crystal_orientation(entry.get("crystal_orientation")),
        "thickness_info": _normalize_thickness_info(entry.get("thickness_info")),
    }


def load_sample_catalog() -> dict[str, Any]:
    ensure_sample_catalog_exists()
    with SAMPLE_CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    samples = catalog.get("samples", [])
    if not isinstance(samples, list):
        samples = []

    normalized_samples = [normalize_sample_entry(entry) for entry in samples]
    normalized_samples.sort(
        key=lambda entry: (
            str(entry.get("sample") or "").lower(),
            str(entry.get("crystal_orientation") or "").lower(),
        )
    )

    return {
        "version": int(catalog.get("version", 1)),
        "samples": normalized_samples,
    }


def save_sample_catalog(catalog: dict[str, Any]) -> None:
    ensure_sample_catalog_exists()
    samples = catalog.get("samples", [])
    normalized_samples = [normalize_sample_entry(entry) for entry in samples]
    normalized_samples.sort(
        key=lambda entry: (
            str(entry.get("sample") or "").lower(),
            str(entry.get("crystal_orientation") or "").lower(),
        )
    )
    payload = {
        "version": int(catalog.get("version", 1)),
        "samples": normalized_samples,
    }
    with SAMPLE_CATALOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_sample_display(entry: dict[str, Any]) -> str:
    label = entry.get("sample") or "Unnamed sample"
    details = []
    if entry.get("material"):
        details.append(str(entry["material"]))
    if entry.get("crystal_orientation"):
        details.append(str(entry["crystal_orientation"]))
    thickness_info = entry.get("thickness_info") or {}
    if thickness_info.get("t_center_mm") is not None:
        details.append(f"t {thickness_info['t_center_mm']:g} mm")
    if thickness_info.get("wedge_angle_deg") is not None:
        details.append(f"wedge {thickness_info['wedge_angle_deg']:g} deg")
    return " | ".join([label] + details) if details else label


def sample_metadata_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_sample_entry(entry)
    return {
        "sample": normalized["sample"],
        "material": normalized["material"],
        "crystal_orientation": normalized["crystal_orientation"],
        "thickness_info": dict(normalized["thickness_info"]),
    }


def normalize_beam_profile_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(entry.get("id") or "").strip(),
        "beam_r_x": _to_optional_float(entry.get("beam_r_x")),
        "beam_r_y": _to_optional_float(entry.get("beam_r_y")),
        "fitting_type": str(entry.get("fitting_type") or "").strip(),
    }


def load_beam_profile_catalog() -> dict[str, Any]:
    ensure_beam_profile_catalog_exists()
    with BEAM_PROFILE_CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    beam_profiles = catalog.get("beam_profiles", [])
    if not isinstance(beam_profiles, list):
        beam_profiles = []

    normalized_profiles = [normalize_beam_profile_entry(entry) for entry in beam_profiles]
    normalized_profiles.sort(key=lambda entry: str(entry.get("id") or "").lower())

    return {
        "version": int(catalog.get("version", 1)),
        "beam_profiles": normalized_profiles,
    }


def save_beam_profile_catalog(catalog: dict[str, Any]) -> None:
    ensure_beam_profile_catalog_exists()
    beam_profiles = catalog.get("beam_profiles", [])
    normalized_profiles = [normalize_beam_profile_entry(entry) for entry in beam_profiles]
    normalized_profiles.sort(key=lambda entry: str(entry.get("id") or "").lower())
    payload = {
        "version": int(catalog.get("version", 1)),
        "beam_profiles": normalized_profiles,
    }
    with BEAM_PROFILE_CATALOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_beam_profile_display(entry: dict[str, Any]) -> str:
    label = entry.get("id") or "Unnamed beam profile"
    details = []
    if entry.get("beam_r_x") is not None and entry.get("beam_r_y") is not None:
        details.append(f"{entry['beam_r_x']:g} x {entry['beam_r_y']:g} um")
    if entry.get("fitting_type"):
        details.append(str(entry["fitting_type"]))
    return " | ".join([label] + details) if details else label


def beam_metadata_from_entry(entry: dict[str, Any]) -> dict[str, float]:
    normalized = normalize_beam_profile_entry(entry)
    metadata: dict[str, float] = {}
    if normalized["beam_r_x"] is not None:
        metadata["beam_r_x"] = float(normalized["beam_r_x"])
    if normalized["beam_r_y"] is not None:
        metadata["beam_r_y"] = float(normalized["beam_r_y"])
    return metadata


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
