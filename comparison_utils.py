from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from measurement_metadata import parse_boxcar_sensitivity


@dataclass
class ComparisonResult:
    key: str
    relative_path: str
    reference_json_path: Path
    target_json_path: Path
    reference_sample: str
    target_sample: str
    reference_method: str
    target_method: str
    peak_label: str
    peak_ref: float | None = None
    peak_target: float | None = None
    d_factor_ref: float | None = None
    d_factor_target: float | None = None
    boxcar_label_ref: str = ""
    boxcar_label_target: str = ""
    boxcar_input_scale_ref: float | None = None
    boxcar_input_scale_target: float | None = None
    filter_ratio: float | None = None
    corrected_intensity_ref: float | None = None
    corrected_intensity_target: float | None = None
    intensity_ratio: float | None = None
    d_scale_ref: float | None = None
    d_scale_target: float | None = None
    d_ratio: float | None = None
    calculated_d: float | None = None
    differing_filters_text: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def status_text(self) -> str:
        if self.error:
            return self.error
        if self.warnings:
            return "; ".join(self.warnings)
        return "OK"


def load_single_measurement_json(root_dir: Path) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    candidates = find_direct_measurement_jsons(root_dir)
    if len(candidates) != 1:
        return None, None, [
            "Comparison expects exactly one JSON directly under the selected folder. "
            f"candidates={len(candidates)}"
        ]

    payload = _load_json(candidates[0])
    if payload is None:
        return None, candidates[0], [f"Failed to read JSON: {candidates[0]}"]
    return payload, candidates[0], []


def find_direct_measurement_jsons(root_dir: Path) -> list[Path]:
    return [
        json_path
        for json_path in sorted(root_dir.glob("*.json"))
        if not json_path.name.endswith(".bak")
    ]


def compare_experiment_folders(
    reference_root: Path,
    target_root: Path,
    reference_d_value: float,
) -> tuple[list[ComparisonResult], list[str]]:
    warnings: list[str] = []
    reference_meta, reference_json_path, reference_warnings = load_single_measurement_json(reference_root)
    target_meta, target_json_path, target_warnings = load_single_measurement_json(target_root)
    warnings.extend(reference_warnings)
    warnings.extend(target_warnings)

    if reference_meta is None or target_meta is None or reference_json_path is None or target_json_path is None:
        return [], warnings

    result = compare_measurement_pair(
        key=target_json_path.name,
        reference_root=reference_root,
        target_root=target_root,
        reference_json_path=reference_json_path,
        target_json_path=target_json_path,
        reference_d_value=reference_d_value,
    )
    return [result], warnings


def compare_reference_folder_to_target_json(
    reference_root: Path,
    target_json_path: Path,
    reference_d_value: float,
) -> tuple[ComparisonResult | None, list[str]]:
    warnings: list[str] = []
    reference_meta, reference_json_path, reference_warnings = load_single_measurement_json(reference_root)
    warnings.extend(reference_warnings)

    target_meta = _load_json(target_json_path)
    if target_meta is None:
        warnings.append(f"Failed to read target JSON: {target_json_path}")
        return None, warnings

    if reference_meta is None or reference_json_path is None:
        return None, warnings

    result = compare_measurement_pair(
        key=target_json_path.name,
        reference_root=reference_root,
        target_root=target_json_path.parent,
        reference_json_path=reference_json_path,
        target_json_path=target_json_path,
        reference_d_value=reference_d_value,
    )
    return result, warnings


def compare_measurement_pair(
    key: str,
    reference_root: Path,
    target_root: Path,
    reference_json_path: Path,
    target_json_path: Path,
    reference_d_value: float,
) -> ComparisonResult:
    del reference_root, target_root
    reference_meta = _load_json(reference_json_path)
    target_meta = _load_json(target_json_path)

    result = ComparisonResult(
        key=key,
        relative_path=key,
        reference_json_path=reference_json_path,
        target_json_path=target_json_path,
        reference_sample=_measurement_label(reference_meta),
        target_sample=_measurement_label(target_meta),
        reference_method=str(reference_meta.get("method") or ""),
        target_method=str(target_meta.get("method") or ""),
        peak_label=_peak_label(target_meta or reference_meta),
    )

    if reference_meta is None:
        result.error = f"Failed to read reference JSON: {reference_json_path}"
        return result
    if target_meta is None:
        result.error = f"Failed to read target JSON: {target_json_path}"
        return result

    peak_ref = _extract_peak(reference_meta)
    peak_target = _extract_peak(target_meta)
    d_factor_ref = _extract_positive_float(reference_meta.get("d_factor"))
    d_factor_target = _extract_positive_float(target_meta.get("d_factor"))
    boxcar_ref, label_ref = _extract_boxcar_input_scale(reference_meta)
    boxcar_target, label_target = _extract_boxcar_input_scale(target_meta)
    filters_ref, filters_ref_warning = _extract_filters(reference_meta)
    filters_target, filters_target_warning = _extract_filters(target_meta)

    result.peak_ref = peak_ref
    result.peak_target = peak_target
    result.d_factor_ref = d_factor_ref
    result.d_factor_target = d_factor_target
    result.boxcar_label_ref = label_ref
    result.boxcar_label_target = label_target
    result.boxcar_input_scale_ref = boxcar_ref
    result.boxcar_input_scale_target = boxcar_target

    if filters_ref_warning:
        result.warnings.append(filters_ref_warning)
    if filters_target_warning:
        result.warnings.append(filters_target_warning)

    missing_items: list[str] = []
    if peak_ref is None:
        missing_items.append("reference peak")
    if peak_target is None:
        missing_items.append("target peak")
    if d_factor_ref is None:
        missing_items.append("reference d_factor")
    if d_factor_target is None:
        missing_items.append("target d_factor")
    if boxcar_ref is None:
        missing_items.append("reference boxcar sensitivity")
    if boxcar_target is None:
        missing_items.append("target boxcar sensitivity")
    if missing_items:
        result.error = "Missing fitted/measurement metadata: " + ", ".join(missing_items)
        return result

    filter_ratio, differing_filters_text = _compute_filter_ratio(filters_ref, filters_target)
    result.filter_ratio = filter_ratio
    result.differing_filters_text = differing_filters_text

    transmission_ref = _transmission_product(filters_ref)
    transmission_target = _transmission_product(filters_target)

    result.corrected_intensity_ref = peak_ref * boxcar_ref / transmission_ref
    result.corrected_intensity_target = peak_target * boxcar_target / transmission_target
    result.intensity_ratio = result.corrected_intensity_target / result.corrected_intensity_ref

    result.d_scale_ref = math.sqrt(result.corrected_intensity_ref / d_factor_ref)
    result.d_scale_target = math.sqrt(result.corrected_intensity_target / d_factor_target)
    result.d_ratio = result.d_scale_target / result.d_scale_ref
    result.calculated_d = reference_d_value * result.d_ratio

    if result.reference_method.lower() != result.target_method.lower():
        result.warnings.append(
            f"Method differs: ref={result.reference_method or '?'} / target={result.target_method or '?'}"
        )
    return result


def write_comparison_results(reference_root: Path, results: list[ComparisonResult]) -> tuple[int, int, list[str]]:
    written = 0
    skipped = 0
    warnings: list[str] = []

    for result in results:
        if result.error or result.calculated_d is None or result.intensity_ratio is None or result.d_ratio is None:
            skipped += 1
            continue

        try:
            with result.target_json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            payload["reference_measurement"] = reference_root.name
            payload["I_target/I_ref"] = round(float(result.intensity_ratio), 6)
            payload["d_target/d_ref"] = round(float(result.d_ratio), 6)
            payload["calculated_d"] = round(float(result.calculated_d), 6)

            with result.target_json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as exc:
            warnings.append(f"Failed to update {result.target_json_path.name}: {exc}")
            skipped += 1

    return written, skipped, warnings


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _measurement_label(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return "?"
    sample = str(meta.get("sample") or meta.get("sample_id") or "?")
    input_pol = meta.get("input_polarization")
    detected_pol = meta.get("detected_polarization")
    if input_pol is None or detected_pol is None:
        return sample
    return f"{sample} | in{input_pol}_out{detected_pol}"


def extract_measurement_summary(meta: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(meta, dict):
        return {
            "sample": "",
            "thickness": "",
            "input_polarization": "",
            "detected_polarization": "",
        }

    tinfo = meta.get("thickness_info") or {}
    thickness_value = tinfo.get("t_center_mm", tinfo.get("t_at_thin_end_mm"))
    thickness_text = ""
    if thickness_value not in (None, ""):
        try:
            thickness_text = f"{float(thickness_value):g} mm"
        except (TypeError, ValueError):
            thickness_text = str(thickness_value)

    return {
        "sample": str(meta.get("sample") or meta.get("sample_id") or ""),
        "thickness": thickness_text,
        "input_polarization": _format_optional_deg(meta.get("input_polarization")),
        "detected_polarization": _format_optional_deg(meta.get("detected_polarization")),
    }


def _peak_label(meta: dict[str, Any] | None) -> str:
    method = str((meta or {}).get("method") or "").strip().lower()
    if method == "rotation":
        return "Envelope peak"
    if method == "wedge":
        return "Fit amplitude A"
    return "Fit peak"


def _extract_peak(meta: dict[str, Any]) -> float | None:
    for key in ("Pm0", "k_scale"):
        value = _extract_positive_float(meta.get(key))
        if value is not None:
            return value
    return None


def _extract_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _extract_boxcar_input_scale(meta: dict[str, Any]) -> tuple[float | None, str]:
    sensitivity = meta.get("boxcar_sensitivity")
    if sensitivity is None:
        return None, ""

    parsed: dict[str, Any] | None = None
    if isinstance(sensitivity, dict):
        if "magnification" in sensitivity:
            magnification = _extract_positive_float(sensitivity.get("magnification"))
            if magnification is not None:
                label = str(sensitivity.get("label") or "")
                return 1.0 / magnification, label
        label_text = str(sensitivity.get("label") or "").strip()
        if label_text:
            try:
                parsed = parse_boxcar_sensitivity(label_text)
            except ValueError:
                parsed = None
    elif isinstance(sensitivity, str):
        try:
            parsed = parse_boxcar_sensitivity(sensitivity)
        except ValueError:
            parsed = None

    if parsed is None:
        return None, ""
    return 1.0 / float(parsed["magnification"]), str(parsed["label"])


def _extract_filters(meta: dict[str, Any]) -> tuple[dict[str, float], str | None]:
    if "filters" not in meta:
        return {}, "Filter metadata is missing."

    raw_filters = meta.get("filters")
    if raw_filters is None:
        return {}, "Filter metadata is missing."
    if not isinstance(raw_filters, dict):
        return {}, "Filter metadata has an invalid format."

    cleaned: dict[str, float] = {}
    for filter_id, transmission in raw_filters.items():
        parsed = _extract_positive_float(transmission)
        if parsed is None:
            return {}, f"Invalid transmission for filter '{filter_id}'."
        cleaned[str(filter_id)] = parsed
    return cleaned, None


def _transmission_product(filters: dict[str, float]) -> float:
    product = 1.0
    for transmission in filters.values():
        product *= transmission
    return product


def _compute_filter_ratio(reference_filters: dict[str, float], target_filters: dict[str, float]) -> tuple[float, str]:
    common_same = {
        filter_id
        for filter_id in set(reference_filters) & set(target_filters)
        if math.isclose(reference_filters[filter_id], target_filters[filter_id], rel_tol=1e-9, abs_tol=1e-12)
    }
    reference_only = {
        filter_id: transmission
        for filter_id, transmission in reference_filters.items()
        if filter_id not in common_same
    }
    target_only = {
        filter_id: transmission
        for filter_id, transmission in target_filters.items()
        if filter_id not in common_same
    }

    filter_ratio = _transmission_product(reference_only) / _transmission_product(target_only)
    details: list[str] = []
    if reference_only:
        details.append(
            "ref: " + ", ".join(f"{filter_id}={transmission:.6g}" for filter_id, transmission in sorted(reference_only.items()))
        )
    if target_only:
        details.append(
            "target: " + ", ".join(f"{filter_id}={transmission:.6g}" for filter_id, transmission in sorted(target_only.items()))
        )
    return filter_ratio, " | ".join(details) if details else "(same filters or no filters)"


def _format_optional_deg(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):g} deg"
    except (TypeError, ValueError):
        return str(value)
