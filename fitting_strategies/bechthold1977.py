import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import argrelextrema

from fitting_strategies.base import BaseWedgeStrategy
from fitting_strategies.base import FittingConfigurationError
from fitting_strategies.jerphagnon1970 import Jerphagnon1970Strategy
from fitting_results import extract_fit_payload, upsert_fitting_result

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *


def _fit_lc_zero_from_minima_pairs(theta_pairs_deg, lc_pairs_mm):
    """
    Empirically extrapolate pair-derived Lc estimates to zero angle.

    Each adjacent-minimum pair is represented at
        s_pair = (sin(theta_1)^2 + sin(theta_2)^2) / 2,
    and the calculated pair value is fitted with
        Lc_pair(s) = c0 + c1*s + c2*s^2.
    The intercept c0 is reported as the experimental estimate of Lc(0).
    """
    theta_pairs_deg = np.asarray(theta_pairs_deg, dtype=float)
    lc_pairs_mm = np.asarray(lc_pairs_mm, dtype=float).reshape(-1)
    if theta_pairs_deg.size == 0 or lc_pairs_mm.size == 0:
        raise ValueError("No adjacent-minimum pairs available for Lc(0) extrapolation.")
    theta_pairs_deg = theta_pairs_deg.reshape((-1, 2))
    if theta_pairs_deg.shape[0] != lc_pairs_mm.size:
        raise ValueError("Angle-pair and Lc-pair counts do not match.")

    s_pair = np.sin(np.deg2rad(theta_pairs_deg)) ** 2
    pair_center_s = np.mean(s_pair, axis=1)
    valid = (
        np.isfinite(pair_center_s)
        & np.isfinite(lc_pairs_mm)
        & (lc_pairs_mm > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid adjacent-minimum pairs are required.")

    theta_pairs_deg = theta_pairs_deg[valid]
    lc_pairs_mm = lc_pairs_mm[valid]
    pair_center_s = pair_center_s[valid]

    order = 2 if lc_pairs_mm.size >= 3 else 1
    design = np.column_stack(
        [pair_center_s**power for power in range(order + 1)]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        lc_pairs_mm,
        rcond=None,
    )
    if (
        rank < order + 1
        or not np.isfinite(coefficients[0])
        or coefficients[0] <= 0.0
    ):
        raise ValueError("Lc(0) extrapolation is ill-conditioned.")

    fitted_pairs_mm = design @ coefficients
    residual_mm = fitted_pairs_mm - lc_pairs_mm
    dof = lc_pairs_mm.size - (order + 1)
    covariance = np.full((order + 1, order + 1), np.nan, dtype=float)
    if dof > 0:
        residual_variance = float(np.dot(residual_mm, residual_mm) / dof)
        covariance = residual_variance * np.linalg.pinv(design.T @ design)

    lc_zero_mm = float(coefficients[0])
    if np.isfinite(covariance[0, 0]) and covariance[0, 0] >= 0.0:
        lc_zero_std_mm = float(np.sqrt(covariance[0, 0]))
    else:
        lc_zero_std_mm = float("nan")

    pair_sign = np.sign(np.mean(theta_pairs_deg, axis=1))
    pair_center_deg = pair_sign * np.rad2deg(
        np.arcsin(np.sqrt(np.clip(pair_center_s, 0.0, 1.0)))
    )

    fit_s = np.linspace(0.0, float(np.max(pair_center_s)), 300)
    fit_lc_mm = np.zeros_like(fit_s)
    for power, coefficient in enumerate(coefficients):
        fit_lc_mm += coefficient * fit_s**power
    fit_theta_deg = np.rad2deg(np.arcsin(np.sqrt(np.clip(fit_s, 0.0, 1.0))))

    return {
        "Lc_zero_mm": lc_zero_mm,
        "Lc_zero_std_mm": lc_zero_std_mm,
        "Lc_pair_mean_mm": float(np.mean(lc_pairs_mm)),
        "Lc_pair_std_mm": (
            float(np.std(lc_pairs_mm, ddof=1))
            if lc_pairs_mm.size >= 2
            else float("nan")
        ),
        "lc_extrapolation_order": int(order),
        "lc_extrapolation_coefficients": np.asarray(coefficients, dtype=float),
        "lc_order_residual_rms": float(np.sqrt(np.mean(residual_mm**2))),
        "pair_theta_bounds_deg": theta_pairs_deg,
        "pair_center_s": pair_center_s,
        "pair_center_deg": pair_center_deg,
        "pair_lc_mm": lc_pairs_mm,
        "fit_theta_deg": fit_theta_deg,
        "fit_lc_mm": fit_lc_mm,
    }


class Bechthold1977Strategy(Jerphagnon1970Strategy):
    """
    Maker fringe theory for biaxial crystals 
    """
    def __init__(self, analysis):
        super().__init__(analysis)

    GEOMETRY_FUNCTIONS = {
        # returns a number in string that indicate one set of experimental configuration shown in Bechthold's paper.
        # key configuration is:("material", (cut), "rot axis", pol_in, pol_out)

        # BMF d31
        ("BaMgF4", "010", "100", 0, 90): "7",
        # BMF d32
        ("BaMgF4", "100", "010", 0, 90): "9",

        # BMF d33
        ("BaMgF4", "100", "001", 0, 0): "11",
        ("BaMgF4", "010", "001", 0, 0): "12",

        # BMF d15
        ("BaMgF4", "010", "100", 45, 0): "13",

        # BMF d24
        ("BaMgF4", "100", "010", 45, 0): "15"
    }

    def _geometry_key(self, meta):
        orientation = meta["crystal_orientation"]
        if isinstance(orientation, list):
            orientation = "".join(map(str, orientation))
        return (
            meta["material"],
            orientation,
            meta["rot/trans_axis"],
            int(round(meta["input_polarization"])),
            int(round(meta["detected_polarization"])),
        )

    def _delta_n_axis_roles(self, meta):
        cut_axis = self.normalize_axis(meta["crystal_orientation"])
        rot_axis = self.normalize_axis(meta["rot/trans_axis"])
        third_axis = self._third_axis(cut_axis, rot_axis)
        axis_label = self.BIAXIAL_N

        key = self._geometry_key(meta)
        exp_config = self.GEOMETRY_FUNCTIONS.get(key)
        if exp_config in ("7", "9"):
            return {
                "w_axes": (axis_label[rot_axis],),
                "two_w_axes": (axis_label[third_axis],),
                "weight": 2.0,
            }
        if exp_config in ("11", "12"):
            return {
                "w_axes": (axis_label[rot_axis],),
                "two_w_axes": (axis_label[rot_axis],),
                "weight": 4.0,
            }
        if exp_config in ("13", "15"):
            return {
                "w_axes": (axis_label[third_axis], axis_label[rot_axis]),
                "two_w_axes": (axis_label[rot_axis],),
                "weight": 1.0,
            }
        return super()._delta_n_axis_roles(meta)

    def _maker_fringes(self, override: dict = {}, envelope=False, return_aux=False):
        
        # loading data
        override = {} if override is None else override
        meta = override.get("meta", "auto")
        data = override.get("data", "auto")
        meta, data = self._resolve_input_info(meta=meta, data=data)

        # basic parameters
        wl1_nm = meta["wavelength_nm"]
        wl1_mm = wl1_nm * 1e-6
        pol_in = meta["input_polarization"] # 0-90 deg
        pol_out = meta["detected_polarization"] # 0-90 deg
        crystal = CRYSTALS[meta["material"]]()

        beam_r_x = meta["beam_r_x"]/2.0
        beam_r_y = meta["beam_r_y"]/2.0

        L = override.get("L", meta["thickness_info"]["t_center_mm"])
        dn_override = override.get("dn_override")
        if "theta_deg" in override.keys():
            theta_deg = override["theta_deg"]
        else:
            theta_deg = np.asarray(data.get("position_centered", data["position"]))
        
        theta = np.deg2rad(theta_deg)

        principle_n_w = self.n_eff(pol_in, wl1_nm, meta=meta, aux=True, dn_override=dn_override)
        n_w_third = principle_n_w["n_third"]
        n_w_rot = principle_n_w["n_rot"]
        n_w_cut = principle_n_w["n_cut"]

        principle_n_2w = self.n_eff(pol_out, wl1_nm / 2.0, meta=meta, aux=True, dn_override=dn_override)
        n_2w_third = principle_n_2w["n_third"]
        n_2w_rot = principle_n_2w["n_rot"]
        n_2w_cut = principle_n_2w["n_cut"]

        v_w = lambda th : (n_w_third / n_w_cut) * np.sqrt(n_w_cut**2 - np.sin(th)**2)
        v_2w = lambda th : (n_2w_third / n_2w_cut) * np.sqrt(n_2w_cut**2 - np.sin(th)**2)

        w_w = lambda th: np.sqrt(n_w_rot**2 - np.sin(th)**2)
        w_2w = lambda th : np.sqrt(n_2w_rot**2 - np.sin(th)**2)
        key = self._geometry_key(meta)
        geometry_functions = override.get("geometry_functions", self.GEOMETRY_FUNCTIONS)
        try:
            exp_config = geometry_functions[key]
        except KeyError:
            raise FittingConfigurationError(
                f"This geometry is not supported: {key}"
            )
        
        if exp_config in ["7", "9"]:
            Psi = 2*np.pi*L *(v_2w(theta) - w_w(theta)) / wl1_mm
            P_nl = lambda theta=theta: 4 * np.cos(theta)**2 /(w_w(theta) + np.cos(theta))**2

            I_2w_env = 2.0 * (P_nl(theta)**2) * (w_w(theta) * (n_2w_third**2) * np.cos(theta) + v_2w(theta)**2) / \
                (((v_2w(theta) - w_w(theta))**2) * (v_2w(theta) + w_w(theta)) * ((v_2w(theta) + (n_2w_third**2) *np.cos(theta))**3))
            
            I_2w_0 = 2.0 *(P_nl(0)**2) * (w_w(0) * (n_2w_third**2) * np.cos(0) + v_2w(0)**2) / \
                (((v_2w(0) - w_w(0))**2) * (v_2w(0) + w_w(0)) * ((v_2w(0) + (n_2w_third**2) *np.cos(0))**3))
        
        elif exp_config in ["11", "12"]:
            Psi = 2*np.pi*L *(w_2w(theta) - w_w(theta)) / wl1_mm
            P_nl = lambda theta=theta: 4 * np.cos(theta)**2 /(w_w(theta) + np.cos(theta))**2

            I_2w_env = 2.0 *(P_nl(theta)**2) * w_2w(theta) * (w_w(theta) + np.cos(theta)) / \
                (((w_2w(theta) - w_w(theta))**2) * (w_2w(theta) + w_w(theta)) * ((w_2w(theta) + np.cos(theta))**3))
            
            I_2w_0 = 2.0 *(P_nl(0)**2) * w_2w(0) * (w_w(0) + np.cos(0)) / \
                (((w_2w(0) - w_w(0))**2) * (w_2w(0) + w_w(0)) * ((w_2w(0) + np.cos(0))**3))
        

        elif exp_config in ["13", "15"]:
            Psi = 2*np.pi*L *(w_2w(theta) - ((v_w(theta) + w_w(theta))/2.0)) / wl1_mm
            P_nl = lambda theta=theta: 4 *v_w(theta)* np.cos(theta)**2 /((v_w(theta) + np.cos(theta)*n_w_third**2) * (w_w(theta) + np.cos(theta)))

            I_2w_env = 2.0 *(P_nl(theta)**2) * w_2w(theta) * (((v_w(theta) + w_w(theta))/2.0) + np.cos(theta)) / \
                (((w_2w(theta) - ((v_w(theta) + w_w(theta))/2.0))**2) * (w_2w(theta) + ((v_w(theta) + w_w(theta))/2.0)) * ((w_2w(theta) + np.cos(theta))**3))
            
            I_2w_0 = 2.0 *(P_nl(0)**2) * w_2w(0) * (((v_w(0) + w_w(0))/2.0) + np.cos(0)) / \
                (((w_2w(0) - ((v_w(0) + w_w(0))/2.0))**2) * (w_2w(0) + ((v_w(0) + w_w(0))/2.0)) * ((w_2w(0) + np.cos(0))**3))
        else:
            raise FittingConfigurationError(
                "This geometry is not supported"
            )
        
        I_2w_N = I_2w_env/I_2w_0

        if envelope:
                model = I_2w_N
        else:
            model = I_2w_N * np.sin(Psi)**2

        if not return_aux:
            return model
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_k = 2.0 * Psi / L
        aux = {
            "d_factor": I_2w_0,
            "Psi": Psi,
            "delta_k": delta_k,
        }
        return model, aux
    

    def _calc_Lc_large_angle(self, meta, data, mask, fitted_L_mm, minima_threshold=None, minima_idx_override=None):
        """
        III D-1 (a): Calculate Lc from large angles (e.g., θ > 30 deg).
        Parameters
        -------------
        mask: list of mask range [min, max]. min < abs(\theta) < max is applied.
        """
        theta_deg = np.asarray(data.get("position_centered", data["position"]))
        I = np.asarray(data["intensity_corrected"])
        
        # --- mask large-angle window and finite values ---
        theta_dropna_inf = np.isfinite(theta_deg) & np.isfinite(I)
        angle_win = (np.abs(theta_deg) >= mask[0]) & (np.abs(theta_deg) <= mask[1])
        m = theta_dropna_inf & angle_win
        if not np.any(m):
            raise ValueError("No data points in the specified theta window.")

        # find minima
        if minima_idx_override is not None:
            minima_idx = np.asarray(minima_idx_override, dtype=int)
            minima_idx = minima_idx[(minima_idx >= 0) & (minima_idx < theta_deg.size)]
        elif minima_threshold is not None:
            minima_idx = self.detect_minima(theta_deg, I, threshold_ratio=minima_threshold)
        else:
            minima_idx = self.detect_minima(theta_deg, I)
        valid_minima_idx = minima_idx[m[minima_idx]]
        
        th_min = theta_deg[valid_minima_idx]
        th_pos = np.sort(th_min[th_min > 0.0])
        th_neg = np.sort(th_min[th_min < 0.0])
        wl1_nm = meta["wavelength_nm"]
        crystal = CRYSTALS[meta["material"]]()
        pol_in = meta["input_polarization"] # 0-90 deg
        pol_out = meta["detected_polarization"] # 0-90 deg


        def differential_L(th_list, pol_in, pol_out):
            if th_list.size < 2:
                return np.array([], dtype=float)
            
            th_rad = np.radians(th_list)
            lc_list = []
            if np.isclose(pol_in, 90):
                raise ValueError(
                    "Input polarization of 90 deg is not yet supported for biaxial crystals."
                )
            elif np.isclose(pol_in, 0):
                if np.isclose(pol_out, 0):
                    n_w = self.n_eff(pol_in, wl1_nm)
                    n_2w = self.n_eff(pol_out, wl1_nm / 2.0)
                    for i in range(th_rad.size - 1):
                        lc = fitted_L_mm * (np.sin(th_rad[i + 1])**2 - np.sin(th_rad[i])**2) / (4.0 * n_2w * n_w)
                        lc_list.append(abs(lc))

                elif np.isclose(pol_out, 90):
                    n_w = self.n_eff(pol_in, wl1_nm)

                    n_2w_dic = self.n_eff(pol_out, wl1_nm / 2.0, aux=True)
                    n_2w_third = n_2w_dic["n_third"]
                    n_2w_cut = n_2w_dic["n_cut"]
                    for i in range(th_rad.size - 1):
                        # Use midpoint refractive indices between adjacent angles
                        A = fitted_L_mm * (np.sin(th_rad[i + 1])**2 - np.sin(th_rad[i])**2) / (4.0 * n_2w_third * n_w)
                        B = fitted_L_mm * n_2w_third * (np.sin(th_rad[i + 1])**2 - np.sin(th_rad[i])**2) * \
                            ((1/n_2w_third**2) - (1/n_2w_cut**2)) / (4 * (n_w - n_2w_third))
                        
                        lc = A - B
                        lc_list.append(abs(lc))
            
            elif np.isclose(pol_in, 45):
                if np.isclose(pol_out, 0):
                    n_w_dic = self.n_eff(pol_in, wl1_nm, aux=True)
                    n_w_rot = n_w_dic["n_rot"]
                    n_w_cut = n_w_dic["n_cut"]
                    n_w_third = n_w_dic["n_third"]
                    n_w = (n_w_rot + n_w_third) / 2.0

                    n_2w = self.n_eff(pol_out, wl1_nm / 2.0, aux=True)
                    n_2w_rot = n_2w["n_rot"]
                    n_2w_cut = n_2w["n_cut"]
                    n_2w_third = n_2w["n_third"]
                    for i in range(th_rad.size - 1):
                        A = fitted_L_mm * (np.sin(th_rad[i + 1])**2 - np.sin(th_rad[i])**2) / (4.0 * n_w * n_2w)
                        B = A* (n_w * (n_w_third**2 - n_w_cut**2) * n_2w_rot / (2*(n_w - n_2w_rot) * n_2w_cut**2 * n_w_rot))
                        
                        lc = A + B
                        lc_list.append(abs(lc))

            else:
                raise ValueError(
                    "Input polarization is supported only for 0, 90, or 45 degrees."
                )

            return np.asarray(lc_list, dtype=float)

        dL_pos = differential_L(th_pos, pol_in, pol_out)
        dL_neg = differential_L(th_neg, pol_in, pol_out)

        parts = [arr for arr in (dL_pos, dL_neg) if len(arr) > 0]
        diffs = np.concatenate(parts) if parts else np.array([], dtype=float)
        diffs = np.asarray(diffs, dtype=float)

        if diffs.size == 0:
            Lc_mean = float("nan")
            Lc_std = float("nan")
            extrapolation = {}
        elif diffs.size == 1:
            Lc_mean = float(diffs[0])
            Lc_std = float("nan")
            extrapolation = {
                "Lc_pair_mean_mm": Lc_mean,
                "Lc_pair_std_mm": float("nan"),
                "lc_extrapolation_order": 0,
                "lc_order_residual_rms": float("nan"),
            }
        else:
            theta_pair_parts = []
            if dL_pos.size:
                theta_pair_parts.append(np.column_stack((th_pos[:-1], th_pos[1:])))
            if dL_neg.size:
                theta_pair_parts.append(np.column_stack((th_neg[:-1], th_neg[1:])))
            theta_pairs = np.vstack(theta_pair_parts)
            extrapolation = _fit_lc_zero_from_minima_pairs(theta_pairs, diffs)
            Lc_mean = float(extrapolation["Lc_zero_mm"])
            Lc_std = float(extrapolation["Lc_zero_std_mm"])

        fit_data = {
            "minima_idx" : valid_minima_idx,
            "x_in_range" : m,
            "dL_pos" : dL_pos,
            "dL_neg" : dL_neg,
            "parts": parts,
            **extrapolation,
        }
        result =  {
            "Lc_mean_mm": Lc_mean,
            "Lc_std_mm": Lc_std,
            "Lc_pair_mean_mm": extrapolation.get("Lc_pair_mean_mm", float("nan")),
            "Lc_pair_std_mm": extrapolation.get("Lc_pair_std_mm", float("nan")),
            "lc_extrapolation_order": extrapolation.get("lc_extrapolation_order", 0),
            "lc_order_residual_rms": extrapolation.get("lc_order_residual_rms", float("nan")),
            "minima_count": int(th_min.size),
            "n_count": int(diffs.size)
            # "theta_used_deg": mask
        }

        return result, fit_data


class GlobalNFitMixin:
    """
    Shared global refractive-index fitting for rotation-scan models.

    The fit uses adjacent fringe minima so that each pair satisfies:
        |Psi(theta[i+1]) - Psi(theta[i])| ~= pi
    while weak priors keep dn and L close to the database / micrometer values.
    """

    DN_PARAMETER_KEYS = (
        "dn_w_a",
        "dn_w_b",
        "dn_w_c",
        "dn_2w_a",
        "dn_2w_b",
        "dn_2w_c",
    )
    N_RESULT_KEYS = (
        ("n_w_a", "w_a"),
        ("n_w_b", "w_b"),
        ("n_w_c", "w_c"),
        ("n_2w_a", "2w_a"),
        ("n_2w_b", "2w_b"),
        ("n_2w_c", "2w_c"),
    )

    DN_PRIOR_SIGMA = 0.01
    L_PRIOR_SIGMA_MM = 0.01
    ANGLE_WEIGHT_SCALE_DEG = 20.0

    def _make_measurement_strategy(self, analysis):
        raise NotImplementedError

    def _validate_measurement_strategy(self, strategy, meta):
        return None

    def _resolve_group_measurement_dirs(self, meta):
        current_dir = None
        if getattr(self.analysis, "base_path", None):
            current_dir = Path(self.analysis.base_path).resolve()

        raw_paths = meta.get("n_fit_group_paths")
        if not isinstance(raw_paths, list) or len(raw_paths) == 0:
            return [current_dir] if current_dir is not None else [None]

        base_dirs = []
        if current_dir is not None:
            base_dirs.append(current_dir)
            base_dirs.append(current_dir.parent)
        base_dirs.append(Path.cwd())

        resolved = []
        seen = set()

        if current_dir is not None:
            resolved.append(current_dir)
            seen.add(str(current_dir))

        for item in raw_paths:
            if not isinstance(item, str) or not item.strip():
                continue
            path_text = item.strip()
            path = Path(path_text).expanduser()
            candidates = []
            if path.is_absolute():
                candidates.append(path)
            else:
                for base_dir in base_dirs:
                    candidates.append((base_dir / path).resolve())

            chosen = None
            for candidate in candidates:
                if candidate.is_file() and candidate.suffix.lower() == ".json":
                    chosen = candidate.parent.resolve()
                    break
                if candidate.is_dir():
                    chosen = candidate.resolve()
                    break
            if chosen is None:
                raise FileNotFoundError(f"Could not resolve n_fit_group_paths entry: {path_text}")

            key = str(chosen)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(chosen)

        return resolved

    def _excluded_group_dirs(self, meta):
        excluded = set()
        for item in meta.get("n_fit_excluded_paths") or []:
            if not isinstance(item, str) or not item.strip():
                continue
            excluded.add(str(Path(item).expanduser().resolve()))
        return excluded

    def _load_saved_extrema_indices(self, meta, data, kind="minima"):
        raw = meta.get(kind)
        if not isinstance(raw, list):
            return None

        if "position" in data.columns:
            raw_x = np.asarray(data["position"], dtype=float)
        else:
            raw_x = np.array([], dtype=float)
        centered_x = np.asarray(data.get("position_centered", data["position"]), dtype=float)

        indices = []
        for item in raw:
            if isinstance(item, dict):
                index = None
                try:
                    index = int(item.get("index"))
                except Exception:
                    index = None
                if index is None:
                    for key, axis in (("position_centered", centered_x), ("position", raw_x)):
                        try:
                            target = float(item.get(key))
                        except Exception:
                            continue
                        if axis.size == 0 or not np.isfinite(target):
                            continue
                        index = int(np.argmin(np.abs(axis - target)))
                        break
                if index is not None:
                    indices.append(index)
                continue
            try:
                indices.append(int(item))
            except Exception:
                continue

        if len(indices) == 0:
            return None

        indices = np.asarray(sorted(set(indices)), dtype=int)
        length = int(len(centered_x))
        valid = (indices >= 0) & (indices < length)
        indices = indices[valid]
        if indices.size == 0:
            return None
        return indices

    def _adjacent_phase_pairs_deg(self, theta_deg, minima_idx):
        theta_deg = np.asarray(theta_deg, dtype=float)
        minima_idx = np.asarray(minima_idx, dtype=int)
        if minima_idx.size == 0:
            return np.empty((0, 2), dtype=float)

        theta_min = theta_deg[minima_idx]
        pairs = []
        for side_mask in (theta_min > 0.0, theta_min < 0.0):
            theta_abs = np.sort(np.abs(theta_min[side_mask]))
            if theta_abs.size < 2:
                continue
            side_pairs = np.column_stack((theta_abs[:-1], theta_abs[1:]))
            pairs.append(side_pairs)

        if not pairs:
            return np.empty((0, 2), dtype=float)
        return np.vstack(pairs).astype(float)

    def _prepare_measurement_bundle(self, analysis, source_dir, *, require_fit_data=True):
        strategy = self._make_measurement_strategy(analysis)
        prepared, centering_info = strategy._position_centering(analysis.data)
        prepared, offset_info = strategy._subtract_offset(prepared)

        minima_idx = self._load_saved_extrema_indices(analysis.meta, prepared, kind="minima")
        if minima_idx is None:
            x = np.asarray(prepared.get("position_centered", prepared["position"]), dtype=float)
            y = np.asarray(prepared.get("offset_corrected", prepared["intensity_corrected"]), dtype=float)
            minima_idx = np.asarray(strategy.detect_minima(x, y), dtype=int)

        theta_deg = np.asarray(prepared.get("position_centered", prepared["position"]), dtype=float)
        theta_min_deg = theta_deg[np.asarray(minima_idx, dtype=int)]
        phase_pairs_deg = self._adjacent_phase_pairs_deg(theta_deg, minima_idx)
        if require_fit_data and phase_pairs_deg.shape[0] == 0:
            raise ValueError(
                f"No adjacent minima pairs available for n fitting in {source_dir or 'current measurement'}."
            )

        self._validate_measurement_strategy(strategy, analysis.meta)
        geometry_key = strategy._geometry_key(analysis.meta)
        thickness_info = analysis.meta.get("thickness_info")
        if not isinstance(thickness_info, dict):
            folder_name = Path(source_dir).name if source_dir is not None else "current measurement"
            raise ValueError(
                f"Missing thickness_info in '{folder_name}'. "
                "Open that folder once and save its metadata, or remove it from the application group."
            )
        thickness_value = thickness_info.get("t_center_mm")
        if thickness_value is None:
            thickness_value = thickness_info.get("t_at_thin_end_mm")
        try:
            L0_mm = float(thickness_value)
        except (TypeError, ValueError):
            folder_name = Path(source_dir).name if source_dir is not None else "current measurement"
            raise ValueError(
                f"Missing t_center_mm in thickness_info for '{folder_name}'."
            ) from None

        return {
            "analysis": analysis,
            "strategy": strategy,
            "meta": analysis.meta,
            "data": prepared,
            "centering_info": centering_info,
            "offset_info": offset_info,
            "minima_idx": np.asarray(minima_idx, dtype=int),
            "theta_min_deg": np.asarray(theta_min_deg, dtype=float),
            "phase_pairs_deg": phase_pairs_deg,
            "L0_mm": L0_mm,
            "source_dir": None if source_dir is None else str(source_dir),
            "geometry_key": geometry_key,
            "included_in_fit": bool(require_fit_data),
        }

    def _load_measurement_group(self):
        from shg_analysis import SHGDataAnalysis

        group_dirs = self._resolve_group_measurement_dirs(self.analysis.meta)
        excluded_dirs = self._excluded_group_dirs(self.analysis.meta)
        measurements = []
        for index, source_dir in enumerate(group_dirs):
            if index == 0:
                analysis = self.analysis
            else:
                analysis = SHGDataAnalysis(source_dir)
            included_in_fit = str(Path(source_dir).resolve()) not in excluded_dirs if source_dir else True
            measurements.append(
                self._prepare_measurement_bundle(
                    analysis,
                    source_dir,
                    require_fit_data=included_in_fit,
                )
            )
        return measurements

    def _dn_dict_from_params(self, params):
        params = np.asarray(params, dtype=float)
        return {
            key: float(params[index])
            for index, key in enumerate(self.DN_PARAMETER_KEYS)
        }

    def _measurement_delta_n_roles(self, measurement):
        strategy = measurement.get("strategy")
        meta = measurement.get("meta") or {}
        if strategy is not None and hasattr(strategy, "_delta_n_axis_roles"):
            roles = strategy._delta_n_axis_roles(meta)
        else:
            roles = {
                "w_axes": ("a", "b", "c"),
                "two_w_axes": ("a", "b", "c"),
                "weight": 1.0,
            }

        def clean_axes(values):
            axes = []
            for axis in values or ():
                axis = str(axis)
                if axis in ("a", "b", "c") and axis not in axes:
                    axes.append(axis)
            return tuple(axes)

        w_axes = clean_axes(roles.get("w_axes"))
        two_w_axes = clean_axes(roles.get("two_w_axes"))
        if not w_axes:
            w_axes = ("a", "b", "c")
        if not two_w_axes:
            two_w_axes = ("a", "b", "c")
        try:
            weight = float(roles.get("weight", 1.0))
        except Exception:
            weight = 1.0
        if not np.isfinite(weight) or weight <= 0.0:
            weight = 1.0
        return {
            "w_axes": w_axes,
            "two_w_axes": two_w_axes,
            "weight": weight,
        }

    def _delta_n_seed_for_measurement(self, measurement):
        strategy = measurement.get("strategy")
        meta = measurement.get("meta") or {}
        strategy_name = strategy.__class__.__name__ if strategy is not None else None
        fit_payload = extract_fit_payload(meta, strategy_name)
        value = self._coerce_scalar(fit_payload.get("delta_n"))
        return float(value) if np.isfinite(value) else 0.0

    def _stage1_dn_override(self, common_offset, delta_n_seed, roles):
        common_offset = float(common_offset)
        delta_n_seed = float(delta_n_seed)
        dn_override = {key: 0.0 for key in self.DN_PARAMETER_KEYS}
        for axis in roles["w_axes"]:
            dn_override[f"dn_w_{axis}"] = common_offset
        for axis in roles["two_w_axes"]:
            dn_override[f"dn_2w_{axis}"] = common_offset + delta_n_seed
        return dn_override

    def _weighted_dn_seed_from_measurements(self, fit_measurements, common_offsets, delta_n_seeds):
        sums = {key: 0.0 for key in self.DN_PARAMETER_KEYS}
        weights = {key: 0.0 for key in self.DN_PARAMETER_KEYS}
        common_offsets = np.asarray(common_offsets, dtype=float)
        delta_n_seeds = np.asarray(delta_n_seeds, dtype=float)

        for index, measurement in enumerate(fit_measurements):
            roles = self._measurement_delta_n_roles(measurement)
            weight = float(roles["weight"])
            common_offset = float(common_offsets[index])
            two_w_offset = common_offset + float(delta_n_seeds[index])
            for axis in roles["w_axes"]:
                key = f"dn_w_{axis}"
                sums[key] += weight * common_offset
                weights[key] += weight
            for axis in roles["two_w_axes"]:
                key = f"dn_2w_{axis}"
                sums[key] += weight * two_w_offset
                weights[key] += weight

        fallback_w = float(np.mean(common_offsets)) if common_offsets.size else 0.0
        fallback_2w = float(np.mean(common_offsets + delta_n_seeds)) if common_offsets.size else 0.0
        dn_seed = {}
        for key in self.DN_PARAMETER_KEYS:
            if weights[key] > 0.0:
                dn_seed[key] = float(sums[key] / weights[key])
            elif key.startswith("dn_w_"):
                dn_seed[key] = fallback_w
            else:
                dn_seed[key] = fallback_2w
        return dn_seed

    def _fit_stage1_difference_offsets(
        self,
        fit_measurements,
        thickness_groups,
        measurement_to_thickness_group,
    ):
        n_measurements = len(fit_measurements)
        n_groups = len(thickness_groups)
        delta_n_seeds = np.asarray(
            [self._delta_n_seed_for_measurement(measurement) for measurement in fit_measurements],
            dtype=float,
        )
        n_minima_residuals = sum(len(measurement["theta_min_deg"]) for measurement in fit_measurements)
        n_data_residuals = sum(len(measurement["phase_pairs_deg"]) for measurement in fit_measurements)
        n_total = n_minima_residuals + n_data_residuals + n_measurements + n_groups

        def residual(params):
            params = np.asarray(params, dtype=float)
            try:
                common_offsets = params[:n_measurements]
                dL_params = params[n_measurements:]
                parts = []
                for index, measurement in enumerate(fit_measurements):
                    group_index = int(measurement_to_thickness_group[index])
                    L_mm = float(thickness_groups[group_index]["L0_mm"]) + float(dL_params[group_index])
                    roles = self._measurement_delta_n_roles(measurement)
                    dn_override = self._stage1_dn_override(common_offsets[index], delta_n_seeds[index], roles)
                    parts.append(self._minima_phase_residuals(measurement, L_mm, dn_override))
                    parts.append(self._phase_pair_residuals(measurement, L_mm, dn_override))
                parts.append(common_offsets / self.DN_PRIOR_SIGMA)
                parts.append(dL_params / self.L_PRIOR_SIGMA_MM)
                out = np.concatenate(parts).astype(float, copy=False)
                if out.size != n_total:
                    raise ValueError("Unexpected stage 1 residual vector size.")
                return np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)
            except Exception:
                return np.full(n_total, 1e6, dtype=float)

        x0 = np.zeros(n_measurements + n_groups, dtype=float)
        for group_index, group in enumerate(thickness_groups):
            saved_offsets = []
            for index, measurement in enumerate(fit_measurements):
                if int(measurement_to_thickness_group[index]) != group_index:
                    continue
                strategy_name = measurement["strategy"].__class__.__name__
                fit_payload = extract_fit_payload(measurement["meta"], strategy_name)
                saved_L = self._coerce_scalar(fit_payload.get("L_mm"))
                if np.isfinite(saved_L):
                    saved_offsets.append(float(saved_L) - float(group["L0_mm"]))
            if saved_offsets:
                x0[n_measurements + group_index] = float(np.clip(np.mean(saved_offsets), -self.L_PRIOR_SIGMA_MM, self.L_PRIOR_SIGMA_MM))

        lower = np.concatenate(
            (
                np.full(n_measurements, -self.DN_PRIOR_SIGMA, dtype=float),
                np.full(n_groups, -self.L_PRIOR_SIGMA_MM, dtype=float),
            )
        )
        upper = np.concatenate(
            (
                np.full(n_measurements, self.DN_PRIOR_SIGMA, dtype=float),
                np.full(n_groups, self.L_PRIOR_SIGMA_MM, dtype=float),
            )
        )
        result = least_squares(residual, x0=x0, bounds=(lower, upper))
        common_offsets = np.asarray(result.x[:n_measurements], dtype=float)
        dL_params = np.asarray(result.x[n_measurements:], dtype=float)
        dn_seed = self._weighted_dn_seed_from_measurements(
            fit_measurements,
            common_offsets,
            delta_n_seeds,
        )
        x0_full = np.concatenate(
            (
                np.asarray([dn_seed[key] for key in self.DN_PARAMETER_KEYS], dtype=float),
                dL_params,
            )
        )
        return {
            "result": result,
            "delta_n_seeds": delta_n_seeds,
            "common_offsets": common_offsets,
            "dL_params": dL_params,
            "dn_seed": dn_seed,
            "x0_full": x0_full,
        }

    def _landscape_candidates_for_measurement(self, measurement, max_candidates=5):
        strategy = measurement["strategy"]
        meta = measurement["meta"]
        data = measurement["data"]
        pos = np.asarray(data.get("position_centered", data["position"]), dtype=float)
        intensity = np.asarray(data.get("offset_corrected", data["intensity_corrected"]), dtype=float)
        finite = np.isfinite(pos) & np.isfinite(intensity)
        if np.count_nonzero(finite) < 3:
            return []

        strategy_name = strategy.__class__.__name__
        fit_payload = extract_fit_payload(meta, strategy_name)
        L_center = self._coerce_scalar(fit_payload.get("L_mm"))
        if not np.isfinite(L_center):
            L_center = float(measurement["L0_mm"])
        delta_center = self._coerce_scalar(fit_payload.get("delta_n"))
        if not np.isfinite(delta_center):
            delta_center = 0.0

        L_grid = np.linspace(float(L_center) - self.L_PRIOR_SIGMA_MM, float(L_center) + self.L_PRIOR_SIGMA_MM, 41)
        delta_grid = np.linspace(float(delta_center) - 0.001, float(delta_center) + 0.001, 41)
        cost = np.full((delta_grid.size, L_grid.size), np.nan, dtype=float)
        x_fit = pos[finite]
        y_fit = intensity[finite]

        for i, delta_n in enumerate(delta_grid):
            dn_override = strategy._delta_n_override(meta, float(delta_n))
            for j, L_mm in enumerate(L_grid):
                try:
                    model = np.asarray(
                        strategy._maker_fringes(
                            override={
                                "meta": meta,
                                "data": data,
                                "theta_deg": x_fit,
                                "L": float(L_mm),
                                "dn_override": dn_override,
                            }
                        ),
                        dtype=float,
                    )
                    valid = np.isfinite(model) & np.isfinite(y_fit)
                    if np.count_nonzero(valid) < 3:
                        continue
                    denom = float(np.dot(model[valid], model[valid]))
                    if denom <= 0.0:
                        continue
                    scale = float(np.dot(model[valid], y_fit[valid]) / denom)
                    residual = scale * model[valid] - y_fit[valid]
                    cost[i, j] = float(np.dot(residual, residual))
                except Exception:
                    continue

        finite_cost = np.isfinite(cost)
        if not np.any(finite_cost):
            return []

        candidates = []
        for i in range(1, cost.shape[0] - 1):
            for j in range(1, cost.shape[1] - 1):
                value = cost[i, j]
                if not np.isfinite(value):
                    continue
                window = cost[i - 1 : i + 2, j - 1 : j + 2]
                finite_window = window[np.isfinite(window)]
                if finite_window.size and value <= float(np.min(finite_window)):
                    candidates.append((float(value), float(L_grid[j]), float(delta_grid[i])))
        if not candidates:
            coords = np.argwhere(finite_cost)
            order = np.argsort(cost[finite_cost])[:max_candidates]
            candidates = [
                (float(cost[tuple(coords[index])]), float(L_grid[coords[index][1]]), float(delta_grid[coords[index][0]]))
                for index in order
            ]
        candidate_records = []
        if candidates:
            coords_norm = np.asarray(
                [
                    (
                        (float(L_mm) - float(L_grid[0])) / max(float(L_grid[-1] - L_grid[0]), 1e-12),
                        (float(delta_n) - float(delta_grid[0])) / max(float(delta_grid[-1] - delta_grid[0]), 1e-12),
                    )
                    for _cost_value, L_mm, delta_n in candidates
                ],
                dtype=float,
            )
            cost_values = np.asarray([item[0] for item in candidates], dtype=float)
            min_cost = float(np.nanmin(cost_values)) if cost_values.size else 0.0
            spread = float(np.nanpercentile(cost_values, 90) - min_cost) if cost_values.size >= 3 else float(np.nanmax(cost_values) - min_cost)
            spread = max(spread, 1e-12)
            for index, (cost_value, L_mm, delta_n) in enumerate(candidates):
                if len(candidates) <= 1:
                    nearest_distance = float("inf")
                    neighbor_count = 0
                else:
                    distances = np.sqrt(np.sum((coords_norm - coords_norm[index]) ** 2, axis=1))
                    distances[index] = np.inf
                    nearest_distance = float(np.min(distances))
                    neighbor_count = int(np.count_nonzero(distances < 0.12))
                candidate_records.append(
                    {
                        "cost": float(cost_value),
                        "L_mm": float(L_mm),
                        "dL_mm": float(L_mm) - float(measurement["L0_mm"]),
                        "delta_n": float(delta_n),
                        "neighbor_count": neighbor_count,
                        "nearest_neighbor_distance": nearest_distance,
                        "relative_cost": float((float(cost_value) - min_cost) / spread),
                    }
                )
        candidate_records = sorted(
            candidate_records,
            key=lambda item: (
                int(item["neighbor_count"]),
                -float(item["nearest_neighbor_distance"]) if np.isfinite(item["nearest_neighbor_distance"]) else -1e9,
                float(item["relative_cost"]),
                float(item["cost"]),
            ),
        )[:max_candidates]
        return candidate_records

    def _stage2_multistart_candidates(
        self,
        x0_zero,
        x0_stage1,
        lower,
        upper,
        fit_measurements,
        thickness_groups,
        measurement_to_thickness_group,
        stage1,
    ):
        x0_zero = np.asarray(x0_zero, dtype=float)
        x0_stage1 = np.asarray(x0_stage1, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        candidates = []
        seen = set()

        def add(name, vector):
            vector = np.clip(np.asarray(vector, dtype=float), lower, upper)
            key = tuple(np.round(vector, 12))
            if key in seen:
                return
            seen.add(key)
            candidates.append((name, vector))

        add("zero", x0_zero)
        add("stage1", x0_stage1)

        landscape_by_measurement = [
            self._landscape_candidates_for_measurement(measurement)
            for measurement in fit_measurements
        ]
        best_candidate_pairs = [
            (measurement_index, items[0])
            for measurement_index, items in enumerate(landscape_by_measurement)
            if items
        ]
        common_offset = float(np.mean(stage1["common_offsets"])) if len(stage1["common_offsets"]) else 0.0

        def apply_measurement_delta_seed(vector, measurement_index, delta_n):
            roles = self._measurement_delta_n_roles(fit_measurements[measurement_index])
            if len(stage1["common_offsets"]) > measurement_index:
                measurement_offset = float(stage1["common_offsets"][measurement_index])
            else:
                measurement_offset = common_offset
            for axis in roles["w_axes"]:
                vector[self.DN_PARAMETER_KEYS.index(f"dn_w_{axis}")] = measurement_offset
            for axis in roles["two_w_axes"]:
                vector[self.DN_PARAMETER_KEYS.index(f"dn_2w_{axis}")] = measurement_offset + float(delta_n)

        if best_candidate_pairs:
            vector = np.array(x0_stage1, dtype=float, copy=True)
            sums = {key: 0.0 for key in self.DN_PARAMETER_KEYS}
            weights = {key: 0.0 for key in self.DN_PARAMETER_KEYS}
            for measurement_index, item in best_candidate_pairs:
                roles = self._measurement_delta_n_roles(fit_measurements[measurement_index])
                weight = float(roles["weight"])
                if len(stage1["common_offsets"]) > measurement_index:
                    measurement_offset = float(stage1["common_offsets"][measurement_index])
                else:
                    measurement_offset = common_offset
                two_w_offset = measurement_offset + float(item["delta_n"])
                for axis in roles["w_axes"]:
                    key = f"dn_w_{axis}"
                    sums[key] += weight * measurement_offset
                    weights[key] += weight
                for axis in roles["two_w_axes"]:
                    key = f"dn_2w_{axis}"
                    sums[key] += weight * two_w_offset
                    weights[key] += weight
            for index, key in enumerate(self.DN_PARAMETER_KEYS):
                if weights[key] > 0.0:
                    vector[index] = float(sums[key] / weights[key])
            for group_index, group in enumerate(thickness_groups):
                group_dL = [
                    item["L_mm"] - float(group["L0_mm"])
                    for measurement_index, item in best_candidate_pairs
                    if int(measurement_to_thickness_group[measurement_index]) == group_index
                ]
                if group_dL:
                    vector[len(self.DN_PARAMETER_KEYS) + group_index] = float(np.mean(group_dL))
            add("landscape:best-mean", vector)

        for measurement_index, items in enumerate(landscape_by_measurement):
            group_index = int(measurement_to_thickness_group[measurement_index])
            for candidate_index, item in enumerate(items[:5]):
                vector = np.array(x0_stage1, dtype=float, copy=True)
                apply_measurement_delta_seed(vector, measurement_index, item["delta_n"])
                vector[len(self.DN_PARAMETER_KEYS) + group_index] = float(item["dL_mm"])
                add(f"landscape:m{measurement_index}:c{candidate_index}", vector)

        return candidates

    def _phase_pair_residuals(self, measurement, L_mm, dn_override):
        theta_pairs_deg = np.asarray(measurement["phase_pairs_deg"], dtype=float)
        theta_eval = theta_pairs_deg.reshape(-1)
        _model, aux = measurement["strategy"]._maker_fringes(
            override={
                "meta": measurement["meta"],
                "data": measurement["data"],
                "theta_deg": theta_eval,
                "L": L_mm,
                "dn_override": dn_override,
            },
            return_aux=True,
        )

        psi = np.asarray(aux["Psi"], dtype=float).reshape(-1)
        if psi.size != theta_eval.size:
            raise ValueError("Phase array size mismatch during global n fit.")

        psi_pairs = psi.reshape((-1, 2))
        residual = np.abs(psi_pairs[:, 1] - psi_pairs[:, 0]) / np.pi - 1.0
        theta_weight_deg = np.mean(np.abs(theta_pairs_deg), axis=1)
        weights = self._angle_priority_weights(theta_weight_deg)
        return np.asarray(weights * residual, dtype=float)

    def _minima_phase_residuals(self, measurement, L_mm, dn_override):
        theta_min_deg = np.asarray(measurement["theta_min_deg"], dtype=float)
        if theta_min_deg.size == 0:
            return np.empty(0, dtype=float)

        _model, aux = measurement["strategy"]._maker_fringes(
            override={
                "meta": measurement["meta"],
                "data": measurement["data"],
                "theta_deg": theta_min_deg,
                "L": L_mm,
                "dn_override": dn_override,
            },
            return_aux=True,
        )
        psi = np.asarray(aux["Psi"], dtype=float).reshape(-1)
        if psi.size != theta_min_deg.size:
            raise ValueError("Phase array size mismatch at minima during global n fit.")
        weights = self._angle_priority_weights(np.abs(theta_min_deg))
        return weights * np.sin(psi)

    def _angle_priority_weights(self, theta_abs_deg):
        theta_abs_deg = np.asarray(theta_abs_deg, dtype=float)
        scale = max(float(self.ANGLE_WEIGHT_SCALE_DEG), 1e-9)
        weights = 1.0 / (1.0 + (theta_abs_deg / scale) ** 2)
        return np.sqrt(np.asarray(weights, dtype=float))

    def _refit_L_small_angle_with_dn(self, measurement, dn_override):
        strategy = measurement["strategy"]
        meta = measurement["meta"]
        data = measurement["data"]

        pos = np.asarray(data.get("position_centered", data["position"]), dtype=float)
        intensity = np.asarray(data["offset_corrected"], dtype=float)
        finite = np.isfinite(pos) & np.isfinite(intensity)
        mask = finite & (np.abs(pos) < 5.0)
        if np.count_nonzero(mask) < 3:
            return {
                "L_mm": float(measurement["L0_mm"]),
                "L_mm_std": float("nan"),
                "k_scale": float("nan"),
                "k_scale_std": float("nan"),
                "success": False,
                "message": "Not enough small-angle points for L refit.",
            }

        theta_small = pos[mask]
        intensity_small = intensity[mask]
        L_guess = float(measurement["L0_mm"])
        peak_guess = float(np.nanmax(intensity_small))
        if not np.isfinite(peak_guess) or peak_guess <= 0.0:
            peak_guess = 1.0

        def residual(params):
            L_mm = float(params[0])
            k_scale = float(params[1])
            model = np.asarray(
                strategy._maker_fringes(
                    override={
                        "meta": meta,
                        "data": data,
                        "theta_deg": theta_small,
                        "L": L_mm,
                        "dn_override": dn_override,
                    }
                ),
                dtype=float,
            )
            if model.shape != intensity_small.shape:
                raise ValueError("Small-angle model length mismatch during L refit.")
            return k_scale * model - intensity_small

        x0 = np.array([L_guess, peak_guess], dtype=float)
        lower = np.array([L_guess - self.L_PRIOR_SIGMA_MM, 0.0], dtype=float)
        upper = np.array([L_guess + self.L_PRIOR_SIGMA_MM, np.inf], dtype=float)
        result = least_squares(residual, x0=x0, bounds=(lower, upper))

        return {
            "L_mm": float(result.x[0]),
            "L_mm_std": float("nan"),
            "k_scale": float(result.x[1]),
            "k_scale_std": float("nan"),
            "success": bool(result.success),
            "message": str(result.message),
        }

    def _format_cut_key(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return ",".join(str(item) for item in value)
        return str(value or "").strip()

    def _thickness_group_key(self, measurement):
        meta = measurement.get("meta") or {}
        sample = str(meta.get("sample") or meta.get("sample_id") or "").strip()
        if not sample:
            sample = self._measurement_source_dir(measurement) or str(id(measurement))
        cut = self._format_cut_key(meta.get("crystal_orientation"))
        return f"{sample}|{cut}"

    def _build_thickness_groups(self, measurements):
        groups = []
        index_by_key = {}
        measurement_to_group = []
        for measurement in measurements:
            key = self._thickness_group_key(measurement)
            index = index_by_key.get(key)
            if index is None:
                index = len(groups)
                index_by_key[key] = index
                groups.append(
                    {
                        "key": key,
                        "sample": str(measurement["meta"].get("sample") or measurement["meta"].get("sample_id") or ""),
                        "crystal_orientation": measurement["meta"].get("crystal_orientation"),
                        "measurement_indices": [],
                    }
                )
            groups[index]["measurement_indices"].append(len(measurement_to_group))
            measurement_to_group.append(index)

        for group in groups:
            l0_values = [float(measurements[index]["L0_mm"]) for index in group["measurement_indices"]]
            group["L0_mm"] = float(np.mean(l0_values)) if l0_values else float("nan")
            group["source_dirs"] = [
                self._measurement_source_dir(measurements[index])
                for index in group["measurement_indices"]
            ]
        return groups, np.asarray(measurement_to_group, dtype=int)

    def _fit_scale_at_L(self, measurement, L_mm, dn_override):
        strategy = measurement["strategy"]
        data = measurement["data"]
        pos = np.asarray(data.get("position_centered", data["position"]), dtype=float)
        intensity = np.asarray(data.get("offset_corrected", data["intensity_corrected"]), dtype=float)
        finite = np.isfinite(pos) & np.isfinite(intensity) & (np.abs(pos) <= 8.0)
        if int(np.sum(finite)) < 3:
            finite = np.isfinite(pos) & np.isfinite(intensity)
        if not np.any(finite):
            return float("nan")

        model = np.asarray(
            strategy._maker_fringes(
                override={
                    "meta": measurement["meta"],
                    "data": data,
                    "theta_deg": pos[finite],
                    "L": float(L_mm),
                    "dn_override": dn_override,
                }
            ),
            dtype=float,
        )
        y = intensity[finite]
        valid = np.isfinite(model) & np.isfinite(y)
        if not np.any(valid):
            return float("nan")
        denom = float(np.dot(model[valid], model[valid]))
        if denom <= 0.0:
            return float("nan")
        return float(np.dot(model[valid], y[valid]) / denom)

    def _measurement_source_dir(self, measurement):
        source_dir = measurement.get("source_dir")
        if source_dir:
            return str(Path(source_dir).resolve())
        analysis = measurement.get("analysis")
        base_path = getattr(analysis, "base_path", None)
        return str(Path(base_path).resolve()) if base_path else ""

    def _local_global_fit_payload(self, global_fit_result, local_result, base_fit_result=None):
        payload = dict(base_fit_result) if isinstance(base_fit_result, dict) else {}
        for key, value in global_fit_result.items():
            if key.startswith("dn_") or key.startswith("n_"):
                payload[key] = value
        for key in (
            "group_size",
            "thickness_group_count",
            "n_fit_cost",
            "n_fit_success",
            "n_fit_stage1_cost",
            "n_fit_stage1_success",
            "n_fit_stage1_mean_delta_n_seed",
            "n_fit_stage1_mean_common_offset",
            "n_fit_stage2_start",
            "n_fit_stage2_start_count",
        ):
            if key in global_fit_result:
                payload[key] = global_fit_result[key]

        payload.update(
            {
                "L_mm": float(local_result["L_mm"]),
                "L_mm_std": float("nan"),
                "k_scale": float(local_result["k_scale_small_angle"]),
                "k_scale_std": float("nan"),
                "Pm0": float(local_result["k_scale_small_angle"]),
                "Pm0_stderr": float("nan"),
                "minima_count": int(local_result["minima_count"]),
                "phase_pair_count": int(local_result["phase_pair_count"]),
            }
        )
        if str(getattr(self, "INTENSITY_SCALE_PARAMETER", "")).strip() == "d_rel_abs":
            payload["d_rel_abs"] = float(
                np.sqrt(max(float(local_result["k_scale_small_angle"]), 0.0))
            )
        return payload

    def _global_result_identity(self, fit_source_dirs):
        normalized = sorted(str(Path(path).resolve()) for path in fit_source_dirs)
        digest = hashlib.sha256(
            (self.__class__.__name__ + "\n" + "\n".join(normalized)).encode("utf-8")
        ).hexdigest()[:16]
        names = [Path(path).name or str(path) for path in normalized]
        label = f"{len(names)} folder(s): " + ", ".join(names)
        return f"{self.__class__.__name__}:{digest}", label

    def _upsert_global_result_history(self, meta, global_fit_result):
        history = [
            dict(entry)
            for entry in meta.get("n_fit_global_results", [])
            if isinstance(entry, dict)
        ]
        result_id = str(global_fit_result["result_id"])
        for index, entry in enumerate(history):
            if str(entry.get("result_id") or "") == result_id:
                history[index] = dict(global_fit_result)
                break
        else:
            history.append(dict(global_fit_result))
        meta["n_fit_global_results"] = history
        meta["n_fit_active_result_id"] = result_id
        return meta

    def _write_global_fit_metadata_to_group(
        self,
        measurements,
        fit_result,
        group_fit_results,
        global_fit_result,
    ):
        group_source_dirs = list(global_fit_result["group_source_dirs"])
        local_by_source = {
            str(entry.get("source_dir") or ""): dict(entry)
            for entry in group_fit_results
            if isinstance(entry, dict)
        }
        current_source = self._measurement_source_dir(measurements[0]) if measurements else ""

        for measurement in measurements:
            analysis = measurement.get("analysis")
            json_path = getattr(analysis, "json_path", None)
            if not json_path:
                continue

            source_dir = self._measurement_source_dir(measurement)
            local_result = local_by_source.get(source_dir)
            if local_result is None:
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = dict(getattr(analysis, "meta", {}) or {})

            local_fit_payload = self._local_global_fit_payload(
                global_fit_result,
                local_result,
                base_fit_result=fit_result if source_dir == current_source else None,
            )
            meta = upsert_fitting_result(
                meta,
                self.__class__.__name__,
                local_fit_payload,
                strategy_module=self.__class__.__module__,
                strategy_display_name=self.__class__.__name__,
                result_id=str(global_fit_result["result_id"]),
                result_label=str(global_fit_result["result_label"]),
            )
            meta = self._upsert_global_result_history(meta, global_fit_result)
            meta["n_fit_global_result"] = dict(global_fit_result)
            meta["n_fit_local_result"] = dict(local_result)
            meta["n_fit_group_results"] = [dict(entry) for entry in group_fit_results]
            meta["n_fit_thickness_group_results"] = [
                dict(entry) for entry in global_fit_result.get("thickness_groups", [])
            ]
            if group_source_dirs:
                meta["n_fit_group_paths"] = group_source_dirs
            excluded_dirs = list(global_fit_result.get("excluded_source_dirs", []))
            if excluded_dirs:
                meta["n_fit_excluded_paths"] = excluded_dirs
            else:
                meta.pop("n_fit_excluded_paths", None)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            analysis.meta = meta

    def fit_all(self):
        measurements = self._load_measurement_group()
        current = measurements[0]
        fit_measurements = [
            measurement for measurement in measurements
            if measurement.get("included_in_fit", True)
        ]
        if not fit_measurements:
            raise ValueError("Select at least one folder to include in the global n fit.")
        thickness_groups, measurement_to_thickness_group = self._build_thickness_groups(fit_measurements)

        n_minima_residuals = sum(len(measurement["theta_min_deg"]) for measurement in fit_measurements)
        n_data_residuals = sum(len(measurement["phase_pairs_deg"]) for measurement in fit_measurements)
        n_total_residuals = n_minima_residuals + n_data_residuals + len(self.DN_PARAMETER_KEYS) + len(thickness_groups)
        stage1 = self._fit_stage1_difference_offsets(
            fit_measurements,
            thickness_groups,
            measurement_to_thickness_group,
        )

        def residual(params):
            params = np.asarray(params, dtype=float)
            try:
                dn_override = self._dn_dict_from_params(params[: len(self.DN_PARAMETER_KEYS)])
                parts = []
                for index, measurement in enumerate(fit_measurements):
                    group_index = int(measurement_to_thickness_group[index])
                    dL_mm = float(params[len(self.DN_PARAMETER_KEYS) + group_index])
                    L_mm = float(thickness_groups[group_index]["L0_mm"]) + dL_mm
                    parts.append(self._minima_phase_residuals(measurement, L_mm, dn_override))
                    parts.append(self._phase_pair_residuals(measurement, L_mm, dn_override))

                parts.append(params[: len(self.DN_PARAMETER_KEYS)] / self.DN_PRIOR_SIGMA)
                parts.append(params[len(self.DN_PARAMETER_KEYS):] / self.L_PRIOR_SIGMA_MM)

                out = np.concatenate(parts).astype(float, copy=False)
                if out.size != n_total_residuals:
                    raise ValueError("Unexpected residual vector size.")
                return np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)
            except Exception:
                return np.full(n_total_residuals, 1e6, dtype=float)

        bounds_dn = np.full(len(self.DN_PARAMETER_KEYS), self.DN_PRIOR_SIGMA, dtype=float)
        bounds_L = np.full(len(thickness_groups), self.L_PRIOR_SIGMA_MM, dtype=float)
        lower = np.concatenate((-bounds_dn, -bounds_L))
        upper = np.concatenate((bounds_dn, bounds_L))

        x0_zero = np.zeros(len(self.DN_PARAMETER_KEYS) + len(thickness_groups), dtype=float)
        x0_stage1 = np.asarray(stage1["x0_full"], dtype=float)
        x0_stage1 = np.clip(x0_stage1, lower, upper)
        starts = self._stage2_multistart_candidates(
            x0_zero,
            x0_stage1,
            lower,
            upper,
            fit_measurements,
            thickness_groups,
            measurement_to_thickness_group,
            stage1,
        )

        fit_attempts = []
        for start_name, x0 in starts:
            fit_attempts.append(
                (
                    start_name,
                    least_squares(
                        residual,
                        x0=x0,
                        bounds=(lower, upper),
                    ),
                )
            )
        start_name, result = min(fit_attempts, key=lambda item: float(item[1].cost))

        dn_override = self._dn_dict_from_params(result.x[: len(self.DN_PARAMETER_KEYS)])
        thickness_group_results = []
        for index, group in enumerate(thickness_groups):
            dL_mm = float(result.x[len(self.DN_PARAMETER_KEYS) + index])
            L_mm = float(group["L0_mm"]) + dL_mm
            thickness_group_results.append(
                {
                    "key": group["key"],
                    "sample": group["sample"],
                    "crystal_orientation": group["crystal_orientation"],
                    "source_dirs": list(group["source_dirs"]),
                    "L0_mm": float(group["L0_mm"]),
                    "L_mm": float(L_mm),
                    "dL_mm": float(dL_mm),
                    "measurement_count": int(len(group["measurement_indices"])),
                }
            )

        group_fit_results = []
        for index, measurement in enumerate(fit_measurements):
            thickness_group_index = int(measurement_to_thickness_group[index])
            thickness_group_result = thickness_group_results[thickness_group_index]
            initial_L_mm = float(thickness_group_result["L_mm"])
            L_mm = initial_L_mm
            dL_mm = L_mm - float(measurement["L0_mm"])
            k_scale_small_angle = self._fit_scale_at_L(measurement, L_mm, dn_override)
            group_fit_results.append(
                {
                    "source_dir": measurement["source_dir"],
                    "thickness_group_key": thickness_group_result["key"],
                    "thickness_group_index": int(thickness_group_index),
                    "sample": str(measurement["meta"].get("sample") or measurement["meta"].get("sample_id") or ""),
                    "material": str(measurement["meta"].get("material") or ""),
                    "crystal_orientation": measurement["meta"].get("crystal_orientation"),
                    "rot_trans_axis": measurement["meta"].get("rot/trans_axis"),
                    "input_polarization": measurement["meta"].get("input_polarization"),
                    "detected_polarization": measurement["meta"].get("detected_polarization"),
                    "L0_mm": float(measurement["L0_mm"]),
                    "L_initial_mm": float(initial_L_mm),
                    "L_mm": float(L_mm),
                    "dL_mm": float(dL_mm),
                    "k_scale_small_angle": float(k_scale_small_angle),
                    "L_refit_success": bool(np.isfinite(k_scale_small_angle)),
                    "minima_count": int(np.asarray(measurement["minima_idx"], dtype=int).size),
                    "phase_pair_count": int(len(measurement["phase_pairs_deg"])),
                    "included_in_fit": True,
                }
            )

        included_sources = {
            self._measurement_source_dir(measurement)
            for measurement in fit_measurements
        }
        for measurement in measurements:
            source_dir = self._measurement_source_dir(measurement)
            if source_dir in included_sources:
                continue
            L_mm = float(measurement["L0_mm"])
            k_scale_small_angle = self._fit_scale_at_L(measurement, L_mm, dn_override)
            group_fit_results.append(
                {
                    "source_dir": measurement["source_dir"],
                    "thickness_group_key": "",
                    "thickness_group_index": -1,
                    "sample": str(measurement["meta"].get("sample") or measurement["meta"].get("sample_id") or ""),
                    "material": str(measurement["meta"].get("material") or ""),
                    "crystal_orientation": measurement["meta"].get("crystal_orientation"),
                    "rot_trans_axis": measurement["meta"].get("rot/trans_axis"),
                    "input_polarization": measurement["meta"].get("input_polarization"),
                    "detected_polarization": measurement["meta"].get("detected_polarization"),
                    "L0_mm": L_mm,
                    "L_initial_mm": L_mm,
                    "L_mm": L_mm,
                    "dL_mm": 0.0,
                    "k_scale_small_angle": float(k_scale_small_angle),
                    "L_refit_success": False,
                    "minima_count": int(np.asarray(measurement["minima_idx"], dtype=int).size),
                    "phase_pair_count": int(len(measurement["phase_pairs_deg"])),
                    "included_in_fit": False,
                }
            )

        local_by_source = {
            str(entry.get("source_dir") or ""): entry
            for entry in group_fit_results
        }
        current_local_result = local_by_source[self._measurement_source_dir(current)]
        current_L_mm = float(current_local_result["L_mm"])

        current_theta = np.asarray(current["data"].get("position_centered", current["data"]["position"]), dtype=float)
        model, fit_aux = current["strategy"]._maker_fringes(
            override={
                "meta": current["meta"],
                "data": current["data"],
                "theta_deg": current_theta,
                "L": current_L_mm,
                "dn_override": dn_override,
            },
            return_aux=True,
        )
        model = np.asarray(model, dtype=float)

        y_offset_corrected = np.asarray(current["data"].get("offset_corrected", current["data"]["intensity_corrected"]), dtype=float)
        finite = np.isfinite(model) & np.isfinite(y_offset_corrected)
        if not np.any(finite):
            raise ValueError("No finite points available to scale the fitted curve.")

        denom = float(np.dot(model[finite], model[finite]))
        if denom <= 0.0:
            raise ValueError("The fitted model has zero norm, so scale fitting failed.")
        k_scale = float(np.dot(model[finite], y_offset_corrected[finite]) / denom)

        offset = float(current["offset_info"]["offset"])
        fit_curve = k_scale * model + offset
        intensity = np.asarray(current["data"]["intensity_corrected"], dtype=float)
        residual_rms = float(np.sqrt(np.mean((intensity[finite] - fit_curve[finite]) ** 2)))

        crystal = CRYSTALS[current["meta"]["material"]]()
        wl1_nm = float(current["meta"]["wavelength_nm"])
        n_values = {
            "w_a": float(crystal.get_n(wl1_nm, polarization="a")) + dn_override["dn_w_a"],
            "w_b": float(crystal.get_n(wl1_nm, polarization="b")) + dn_override["dn_w_b"],
            "w_c": float(crystal.get_n(wl1_nm, polarization="c")) + dn_override["dn_w_c"],
            "2w_a": float(crystal.get_n(wl1_nm / 2.0, polarization="a")) + dn_override["dn_2w_a"],
            "2w_b": float(crystal.get_n(wl1_nm / 2.0, polarization="b")) + dn_override["dn_2w_b"],
            "2w_c": float(crystal.get_n(wl1_nm / 2.0, polarization="c")) + dn_override["dn_2w_c"],
        }

        out = current["data"].copy()
        out["fit"] = fit_curve

        fit_result = {
            "L_mm": current_L_mm,
            "L_mm_std": float("nan"),
            "k_scale": k_scale,
            "k_scale_std": float("nan"),
            "Pm0": k_scale,
            "Pm0_stderr": float("nan"),
            "residual_rms": residual_rms,
            "minima_count": int(current["minima_idx"].size),
            "n_count": int(n_data_residuals),
            "group_size": int(len(fit_measurements)),
            "thickness_group_count": int(len(thickness_group_results)),
            "phase_pair_count": int(n_data_residuals),
            "n_fit_cost": float(result.cost),
            "n_fit_success": bool(result.success),
            "n_fit_stage1_cost": float(stage1["result"].cost),
            "n_fit_stage1_success": bool(stage1["result"].success),
            "n_fit_stage1_mean_delta_n_seed": float(np.mean(stage1["delta_n_seeds"])) if len(stage1["delta_n_seeds"]) else 0.0,
            "n_fit_stage1_mean_common_offset": float(np.mean(stage1["common_offsets"])) if len(stage1["common_offsets"]) else 0.0,
            "n_fit_stage2_start": str(start_name),
            "n_fit_stage2_start_count": int(len(starts)),
        }
        if str(getattr(self, "INTENSITY_SCALE_PARAMETER", "")).strip() == "d_rel_abs":
            fit_result["d_rel_abs"] = float(np.sqrt(max(k_scale, 0.0)))
        fit_result.update(dn_override)
        for result_key, n_key in self.N_RESULT_KEYS:
            fit_result[result_key] = float(n_values[n_key])
        if isinstance(fit_aux, dict) and fit_aux.get("d_factor") is not None:
            fit_result["d_factor"] = self._coerce_scalar(fit_aux["d_factor"])

        group_source_dirs = [self._measurement_source_dir(measurement) for measurement in measurements]
        fit_source_dirs = [self._measurement_source_dir(measurement) for measurement in fit_measurements]
        excluded_source_dirs = [
            self._measurement_source_dir(measurement)
            for measurement in measurements
            if not measurement.get("included_in_fit", True)
        ]
        result_id, result_label = self._global_result_identity(fit_source_dirs)
        global_fit_result = {
            "strategy": self.__class__.__name__,
            "strategy_module": self.__class__.__module__,
            "result_id": result_id,
            "result_label": result_label,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "group_source_dirs": group_source_dirs,
            "fit_source_dirs": fit_source_dirs,
            "excluded_source_dirs": excluded_source_dirs,
            "group_size": int(len(fit_measurements)),
            "application_group_size": int(len(measurements)),
            "thickness_group_count": int(len(thickness_group_results)),
            "thickness_groups": [dict(entry) for entry in thickness_group_results],
            "group_results": [dict(entry) for entry in group_fit_results],
            "n_count": int(n_data_residuals),
            "phase_pair_count": int(n_data_residuals),
            "n_fit_cost": float(result.cost),
            "n_fit_success": bool(result.success),
            "n_fit_stage1_cost": float(stage1["result"].cost),
            "n_fit_stage1_success": bool(stage1["result"].success),
            "n_fit_stage1_mean_delta_n_seed": float(np.mean(stage1["delta_n_seeds"])) if len(stage1["delta_n_seeds"]) else 0.0,
            "n_fit_stage1_mean_common_offset": float(np.mean(stage1["common_offsets"])) if len(stage1["common_offsets"]) else 0.0,
            "n_fit_stage2_start": str(start_name),
            "n_fit_stage2_start_count": int(len(starts)),
        }
        for key, value in fit_result.items():
            if key.startswith("dn_") or key.startswith("n_"):
                global_fit_result[key] = value

        self.analysis.meta = upsert_fitting_result(
            self.analysis.meta,
            self.__class__.__name__,
            fit_result,
            strategy_module=self.__class__.__module__,
            strategy_display_name=self.__class__.__name__,
            result_id=result_id,
            result_label=result_label,
        )
        self.analysis.meta = self._upsert_global_result_history(
            self.analysis.meta,
            global_fit_result,
        )
        self.analysis.meta["n_fit_global_result"] = dict(global_fit_result)
        self.analysis.meta["n_fit_local_result"] = dict(current_local_result)
        self.analysis.meta["n_fit_group_results"] = group_fit_results
        self.analysis.meta["n_fit_thickness_group_results"] = thickness_group_results

        csv_path = self.analysis.csv_path
        json_path = self.analysis.json_path
        self.analysis.data = out
        out.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis.meta, f, ensure_ascii=False, indent=2)
        self._write_global_fit_metadata_to_group(
            measurements,
            fit_result,
            group_fit_results,
            global_fit_result,
        )

        return fit_result


class Bechthold1977GlobalNFitStrategy(GlobalNFitMixin, Bechthold1977Strategy):
    """Global refractive-index fit using the Bechthold 1977 model."""

    def _make_measurement_strategy(self, analysis):
        return Bechthold1977Strategy(analysis)

    def _validate_measurement_strategy(self, strategy, meta):
        geometry_key = strategy._geometry_key(meta)
        if geometry_key not in strategy.GEOMETRY_FUNCTIONS:
            raise FittingConfigurationError(
                f"This geometry is not supported for Bechthold global n fit: {geometry_key}"
            )


class Bechthold1977WedgeStrategy(Bechthold1977Strategy, BaseWedgeStrategy):
    def __init__(self, analysis):
        super().__init__(analysis)

    GEOMETRY_FUNCTIONS = {
        # returns a number in string that indicate one set of experimental configuration shown in Bechthold's paper.
        # this configurations are for Bechthold's equations to be adapted to wedge measurements.
        # key configuration is:("material", (cut), "trans_axis", pol_in, pol_out)

        # BMF d31
        ("BaMgF4", "010", "100", 90, 0): "7",

        # BMF d32
        ("BaMgF4", "100", "001", 0, 90): "9",

        # BMF d33
        ("BaMgF4", "010", "100", 0, 0): "12",

        # BMF d15
        ("BaMgF4", "010", "100", 45, 90): "13",

        # BMF d24
        ("BaMgF4", "100", "001", 45, 0): "15"
    }

    ROTATION_THEORY_KEYS = {
        # key configuration is:
        # (material, cut, trans_axis, pol_in, pol_out) ->
        # (material, cut, rot_axis, pol_in, pol_out)

        # BMF d31
        ("BaMgF4", "010", "100", 90, 0): ("BaMgF4", "010", "100", 0, 90),

        # BMF d32
        ("BaMgF4", "100", "001", 0, 90): ("BaMgF4", "100", "010", 0, 90),

        # BMF d33
        ("BaMgF4", "010", "100", 0, 0): ("BaMgF4", "010", "001", 0, 0),

        # BMF d15
        ("BaMgF4", "010", "100", 45, 90): ("BaMgF4", "010", "100", 45, 0),

        # BMF d24
        ("BaMgF4", "100", "001", 45, 0): ("BaMgF4", "100", "010", 45, 0),
    }

    def _delta_n_axis_roles(self, meta):
        theory_meta = self._rotation_theory_meta(meta)
        cut_axis = self.normalize_axis(theory_meta["crystal_orientation"])
        rot_axis = self.normalize_axis(theory_meta["rot/trans_axis"])
        third_axis = self._third_axis(cut_axis, rot_axis)
        axis_label = self.BIAXIAL_N

        key = self._geometry_key(theory_meta)
        exp_config = Bechthold1977Strategy.GEOMETRY_FUNCTIONS.get(key)
        if exp_config in ("7", "9"):
            return {
                "w_axes": (axis_label[rot_axis],),
                "two_w_axes": (axis_label[third_axis],),
                "weight": 2.0,
            }
        if exp_config in ("11", "12"):
            return {
                "w_axes": (axis_label[rot_axis],),
                "two_w_axes": (axis_label[rot_axis],),
                "weight": 4.0,
            }
        if exp_config in ("13", "15"):
            return {
                "w_axes": (axis_label[third_axis], axis_label[rot_axis]),
                "two_w_axes": (axis_label[rot_axis],),
                "weight": 1.0,
            }
        return super()._delta_n_axis_roles(theory_meta)

    def _rotation_theory_meta(self, meta: dict):
        key = (
            meta["material"],
            meta["crystal_orientation"],
            meta["rot/trans_axis"],
            int(round(meta["input_polarization"])),
            int(round(meta["detected_polarization"])),
        )
        try:
            theory_key = self.ROTATION_THEORY_KEYS[key]
        except KeyError:
            raise FittingConfigurationError(
                f"This wedge geometry has no mapped rotation theory frame: {key}"
            )

        theory_meta = dict(meta)
        theory_meta["material"] = theory_key[0]
        theory_meta["crystal_orientation"] = theory_key[1]
        theory_meta["rot/trans_axis"] = theory_key[2]
        theory_meta["input_polarization"] = theory_key[3]
        theory_meta["detected_polarization"] = theory_key[4]
        return theory_meta

    def _maker_fringes(self, override: dict | None = None, return_aux=False):
        """
        Adapt Bechthold's rotation formula to wedge scans by fixing theta=0
        and evaluating the model on the wedge thickness profile.
        """
        override = {} if override is None else override
        meta = override.get("meta", "auto")
        data = override.get("data", "auto")
        meta, data = self._resolve_input_info(meta=meta, data=data)
        theory_meta = self._rotation_theory_meta(meta)

        L_array = self.calc_thickness_array(override=override, meta=meta, data=data)

        rotation_override = dict(override)
        rotation_override.update(
            {
                "meta": theory_meta,
                "data": data,
                "L": L_array,
                "theta_deg": 0.0,
                "geometry_functions": Bechthold1977Strategy.GEOMETRY_FUNCTIONS,
            }
        )

        model, rot_aux = super()._maker_fringes(
            override=rotation_override,
            envelope=False,
            return_aux=True,
        )

        model = np.asarray(model, dtype=float)
        if model.ndim == 0:
            model = np.full(L_array.shape, float(model), dtype=float)

        if not return_aux:
            return model

        delta_k_values = np.asarray(rot_aux["delta_k"], dtype=float).reshape(-1)
        finite_delta_k = delta_k_values[np.isfinite(delta_k_values)]
        delta_k = float(finite_delta_k[0]) if finite_delta_k.size else float("nan")
        if np.isfinite(delta_k) and not np.isclose(delta_k, 0.0):
            lc = float(np.pi / delta_k)
        else:
            lc = float("nan")

        aux = {
            "d_factor": rot_aux["d_factor"],
            "L_array": L_array,
            "Lc": lc,
        }
        return model, aux
    
    def fit_all(self):
        return BaseWedgeStrategy.fit_all(self)
