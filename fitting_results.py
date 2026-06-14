from __future__ import annotations

from typing import Any


FITTING_RESULT_KEYS: tuple[str, ...] = (
    "L_mm",
    "L_mm_std",
    "k_scale",
    "k_scale_std",
    "Pm0",
    "Pm0_stderr",
    "d_rel_abs",
    "d_component",
    "d_factor",
    "Lc_mean_mm",
    "Lc_std_mm",
    "Lc_pair_mean_mm",
    "Lc_pair_std_mm",
    "lc_extrapolation_order",
    "lc_order_residual_rms",
    "residual_rms",
    "minima_count",
    "n_count",
    "n_peaks",
    "group_size",
    "thickness_group_count",
    "phase_pair_count",
    "n_fit_cost",
    "n_fit_success",
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


def normalize_fitting_entries(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(meta, dict):
        return []

    raw = meta.get(FITTING_CONTAINER_KEY)
    if isinstance(raw, list):
        return [dict(entry) for entry in raw if isinstance(entry, dict)]

    if isinstance(raw, dict):
        if "strategy" in raw:
            return [dict(raw)]
        entries: list[dict[str, Any]] = []
        for strategy_name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            normalized.setdefault("strategy", str(strategy_name))
            entries.append(normalized)
        return entries

    return []


def get_fitting_entry(meta: dict[str, Any] | None, strategy_name: str | None = None) -> dict[str, Any] | None:
    entries = normalize_fitting_entries(meta)
    if not entries:
        return None

    if strategy_name:
        for entry in entries:
            if str(entry.get("strategy") or "").strip() == strategy_name:
                return dict(entry)
        return None

    active_strategy = str((meta or {}).get(FITTING_ACTIVE_STRATEGY_KEY) or "").strip()
    if active_strategy:
        active_entry = get_fitting_entry(meta, active_strategy)
        if active_entry is not None:
            return active_entry

    return dict(entries[-1])


def extract_fit_payload(meta: dict[str, Any] | None, strategy_name: str | None = None) -> dict[str, Any]:
    entry = get_fitting_entry(meta, strategy_name)
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
        return legacy_payload

    return {}


def merge_fit_payload(meta: dict[str, Any] | None, strategy_name: str | None = None) -> dict[str, Any]:
    merged = dict(meta) if isinstance(meta, dict) else {}
    merged.update(extract_fit_payload(meta, strategy_name))
    return merged


def upsert_fitting_result(
    meta: dict[str, Any],
    strategy_name: str,
    result: dict[str, Any],
    *,
    strategy_module: str | None = None,
    strategy_display_name: str | None = None,
) -> dict[str, Any]:
    payload = dict(meta)
    entries = normalize_fitting_entries(payload)

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

    replaced = False
    for index, existing in enumerate(entries):
        if str(existing.get("strategy") or "").strip() == strategy_name:
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    payload[FITTING_CONTAINER_KEY] = entries
    payload[FITTING_ACTIVE_STRATEGY_KEY] = strategy_name

    for key in FITTING_RESULT_KEYS:
        payload.pop(key, None)

    return payload


def remove_fitting_results(
    meta: dict[str, Any] | None,
    strategy_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    payload = dict(meta) if isinstance(meta, dict) else {}
    remove_targets = {
        str(name).strip()
        for name in strategy_names
        if str(name).strip()
    }
    if not remove_targets:
        return payload

    entries = [
        dict(entry)
        for entry in normalize_fitting_entries(payload)
        if str(entry.get("strategy") or "").strip() not in remove_targets
    ]
    if entries:
        payload[FITTING_CONTAINER_KEY] = entries
    else:
        payload.pop(FITTING_CONTAINER_KEY, None)

    active_strategy = str(payload.get(FITTING_ACTIVE_STRATEGY_KEY) or "").strip()
    if active_strategy in remove_targets:
        payload[FITTING_ACTIVE_STRATEGY_KEY] = str(entries[-1].get("strategy") or "").strip() if entries else ""
        if not payload[FITTING_ACTIVE_STRATEGY_KEY]:
            payload.pop(FITTING_ACTIVE_STRATEGY_KEY, None)

    return payload
