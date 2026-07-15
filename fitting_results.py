from __future__ import annotations

from typing import Any


FITTING_RESULT_KEYS: tuple[str, ...] = (
    "L_mm",
    "L_mm_std",
    "centering_pos",
    "k_scale",
    "k_scale_std",
    "Pm0",
    "Pm0_stderr",
    "d_rel_abs",
    "d_component",
    "d_factor",
    "Lc_exp_mm",
    "Lc_exp_std_mm",
    "Lc_angle_dependent_n_mm",
    "Lc_angle_dependent_n_std_mm",
    "Lc_constant_n_mm",
    "Lc_constant_n_std_mm",
    "Lc_angle_dependence_delta_mm",
    "Lc_mean_mm",
    "Lc_std_mm",
    "Lc_pair_mean_mm",
    "Lc_pair_std_mm",
    "Lc_theory_mm",
    "delta_k_theory_inv_mm",
    "lc_wedge_minima_mm",
    "lc_wedge_minima_std_mm",
    "lc_extrapolation_order",
    "lc_order_residual_rms",
    "residual_rms",
    "delta_n",
    "delta_n_std",
    "delta_n_fit_cost",
    "delta_n_fit_success",
    "minima_count",
    "n_count",
    "n_peaks",
    "group_size",
    "thickness_group_count",
    "phase_pair_count",
    "n_fit_cost",
    "n_fit_success",
    "n_fit_stage1_cost",
    "n_fit_stage1_success",
    "n_fit_stage1_mean_delta_n_seed",
    "n_fit_stage1_mean_common_offset",
    "n_fit_stage2_start",
    "n_fit_stage2_start_count",
    "dn_w_a",
    "dn_w_b",
    "dn_w_c",
    "dn_2w_a",
    "dn_2w_b",
    "dn_2w_c",
    "n_w_a",
    "n_w_b",
    "n_w_c",
    "n_2w_a",
    "n_2w_b",
    "n_2w_c",
)

FITTING_CONTAINER_KEY = "fitting"
FITTING_ACTIVE_STRATEGY_KEY = "fitting_active_strategy"
FITTING_ACTIVE_RESULT_ID_KEY = "fitting_active_result_id"


def normalize_lc_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "Lc_exp_mm" not in normalized and "Lc_mean_mm" in normalized:
        normalized["Lc_exp_mm"] = normalized["Lc_mean_mm"]
    if "Lc_exp_std_mm" not in normalized and "Lc_std_mm" in normalized:
        normalized["Lc_exp_std_mm"] = normalized["Lc_std_mm"]
    normalized.pop("Lc_mean_mm", None)
    normalized.pop("Lc_std_mm", None)
    return normalized


def migrate_lc_aliases(meta: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    payload = dict(meta) if isinstance(meta, dict) else {}
    changed = False

    def apply_aliases(entry: dict[str, Any]) -> dict[str, Any]:
        nonlocal changed
        updated = normalize_lc_aliases(entry)
        if updated != entry:
            changed = True
        return updated

    payload = apply_aliases(payload)

    raw = payload.get(FITTING_CONTAINER_KEY)
    if isinstance(raw, list):
        entries = []
        for entry in raw:
            entries.append(apply_aliases(entry) if isinstance(entry, dict) else entry)
        payload[FITTING_CONTAINER_KEY] = entries
    elif isinstance(raw, dict):
        if "strategy" in raw:
            payload[FITTING_CONTAINER_KEY] = apply_aliases(raw)
        else:
            payload[FITTING_CONTAINER_KEY] = {
                key: apply_aliases(value) if isinstance(value, dict) else value
                for key, value in raw.items()
            }

    return payload, changed


def normalize_fitting_entries(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(meta, dict):
        return []

    raw = meta.get(FITTING_CONTAINER_KEY)
    if isinstance(raw, list):
        return [normalize_lc_aliases(entry) for entry in raw if isinstance(entry, dict)]

    if isinstance(raw, dict):
        if "strategy" in raw:
            return [normalize_lc_aliases(raw)]
        entries: list[dict[str, Any]] = []
        for strategy_name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            normalized = normalize_lc_aliases(entry)
            normalized.setdefault("strategy", str(strategy_name))
            entries.append(normalized)
        return entries

    return []


def get_fitting_entry(
    meta: dict[str, Any] | None,
    strategy_name: str | None = None,
    result_id: str | None = None,
) -> dict[str, Any] | None:
    entries = normalize_fitting_entries(meta)
    if not entries:
        return None

    if result_id:
        for entry in entries:
            if str(entry.get("result_id") or "").strip() == result_id:
                if not strategy_name or str(entry.get("strategy") or "").strip() == strategy_name:
                    return dict(entry)
        return None

    if strategy_name:
        matching_entries = [
            entry
            for entry in entries
            if str(entry.get("strategy") or "").strip() == strategy_name
        ]
        active_result_id = str((meta or {}).get(FITTING_ACTIVE_RESULT_ID_KEY) or "").strip()
        if active_result_id:
            for entry in matching_entries:
                if str(entry.get("result_id") or "").strip() == active_result_id:
                    return dict(entry)
        for entry in reversed(matching_entries):
            if str(entry.get("strategy") or "").strip() == strategy_name:
                return dict(entry)
        return None

    active_strategy = str((meta or {}).get(FITTING_ACTIVE_STRATEGY_KEY) or "").strip()
    if active_strategy:
        active_result_id = str((meta or {}).get(FITTING_ACTIVE_RESULT_ID_KEY) or "").strip()
        active_entry = get_fitting_entry(meta, active_strategy, active_result_id or None)
        if active_entry is not None:
            return active_entry

    return dict(entries[-1])


def extract_fit_payload(
    meta: dict[str, Any] | None,
    strategy_name: str | None = None,
    result_id: str | None = None,
) -> dict[str, Any]:
    entry = get_fitting_entry(meta, strategy_name, result_id)
    if entry is not None:
        return dict(entry)

    if not isinstance(meta, dict):
        return {}

    fitting_entries = normalize_fitting_entries(meta)
    if strategy_name and fitting_entries:
        return {}

    legacy_payload = {
        key: meta[key]
        for key in FITTING_RESULT_KEYS
        if key in meta
    }
    if legacy_payload:
        return normalize_lc_aliases(legacy_payload)

    return {}


def merge_fit_payload(
    meta: dict[str, Any] | None,
    strategy_name: str | None = None,
    result_id: str | None = None,
) -> dict[str, Any]:
    merged = dict(meta) if isinstance(meta, dict) else {}
    merged.update(extract_fit_payload(meta, strategy_name, result_id))
    return merged


def upsert_fitting_result(
    meta: dict[str, Any],
    strategy_name: str,
    result: dict[str, Any],
    *,
    strategy_module: str | None = None,
    strategy_display_name: str | None = None,
    result_id: str | None = None,
    result_label: str | None = None,
) -> dict[str, Any]:
    payload = dict(meta)
    entries = normalize_fitting_entries(payload)

    result = normalize_lc_aliases(result)
    entry = {
        key: value
        for key, value in result.items()
        if key in FITTING_RESULT_KEYS
    }
    entry["strategy"] = strategy_name
    if strategy_module:
        entry["strategy_module"] = strategy_module
    if strategy_display_name:
        entry["strategy_display_name"] = strategy_display_name
    if result_id:
        entry["result_id"] = result_id
    if result_label:
        entry["result_label"] = result_label

    replaced = False
    for index, existing in enumerate(entries):
        same_strategy = str(existing.get("strategy") or "").strip() == strategy_name
        same_result = str(existing.get("result_id") or "").strip() == str(result_id or "").strip()
        if same_strategy and same_result:
            if "centering_pos" not in entry and "centering_pos" in existing:
                entry["centering_pos"] = existing["centering_pos"]
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    payload[FITTING_CONTAINER_KEY] = entries
    payload[FITTING_ACTIVE_STRATEGY_KEY] = strategy_name
    if result_id:
        payload[FITTING_ACTIVE_RESULT_ID_KEY] = result_id
    else:
        payload.pop(FITTING_ACTIVE_RESULT_ID_KEY, None)

    for key in FITTING_RESULT_KEYS:
        payload.pop(key, None)

    return payload


def remove_fitting_results(
    meta: dict[str, Any] | None,
    strategy_names: list[str] | tuple[str, ...] | set[str],
    result_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    payload = dict(meta) if isinstance(meta, dict) else {}
    remove_targets = {
        str(name).strip()
        for name in strategy_names
        if str(name).strip()
    }
    if not remove_targets:
        return payload

    remove_result_ids = {
        str(result_id).strip()
        for result_id in (result_ids or [])
        if str(result_id).strip()
    }
    entries = []
    for entry in normalize_fitting_entries(payload):
        strategy = str(entry.get("strategy") or "").strip()
        result_id = str(entry.get("result_id") or "").strip()
        remove_entry = (
            result_id in remove_result_ids
            if remove_result_ids
            else strategy in remove_targets
        )
        if not remove_entry:
            entries.append(dict(entry))
    if entries:
        payload[FITTING_CONTAINER_KEY] = entries
    else:
        payload.pop(FITTING_CONTAINER_KEY, None)

    if remove_result_ids:
        global_history = [
            dict(entry)
            for entry in payload.get("n_fit_global_results", [])
            if (
                isinstance(entry, dict)
                and str(entry.get("result_id") or "").strip() not in remove_result_ids
            )
        ]
        if global_history:
            payload["n_fit_global_results"] = global_history
        else:
            payload.pop("n_fit_global_results", None)
        active_global_id = str(payload.get("n_fit_active_result_id") or "").strip()
        if active_global_id in remove_result_ids:
            if global_history:
                latest_global = dict(global_history[-1])
                payload["n_fit_active_result_id"] = str(latest_global.get("result_id") or "")
                payload["n_fit_global_result"] = latest_global
                payload["n_fit_group_results"] = [
                    dict(entry)
                    for entry in latest_global.get("group_results", [])
                    if isinstance(entry, dict)
                ]
                payload["n_fit_thickness_group_results"] = [
                    dict(entry)
                    for entry in latest_global.get("thickness_groups", [])
                    if isinstance(entry, dict)
                ]
            else:
                for key in (
                    "n_fit_active_result_id",
                    "n_fit_global_result",
                    "n_fit_local_result",
                    "n_fit_group_results",
                    "n_fit_thickness_group_results",
                ):
                    payload.pop(key, None)

    active_strategy = str(payload.get(FITTING_ACTIVE_STRATEGY_KEY) or "").strip()
    active_result_id = str(payload.get(FITTING_ACTIVE_RESULT_ID_KEY) or "").strip()
    if active_strategy in remove_targets or active_result_id in remove_result_ids:
        payload[FITTING_ACTIVE_STRATEGY_KEY] = str(entries[-1].get("strategy") or "").strip() if entries else ""
        next_result_id = str(entries[-1].get("result_id") or "").strip() if entries else ""
        if next_result_id:
            payload[FITTING_ACTIVE_RESULT_ID_KEY] = next_result_id
        else:
            payload.pop(FITTING_ACTIVE_RESULT_ID_KEY, None)
        if not payload[FITTING_ACTIVE_STRATEGY_KEY]:
            payload.pop(FITTING_ACTIVE_STRATEGY_KEY, None)

    return payload
