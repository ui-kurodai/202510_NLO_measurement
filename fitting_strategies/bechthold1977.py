import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import argrelextrema

from fitting_strategies.base import BaseWedgeStrategy
from fitting_strategies.base import FittingConfigurationError
from fitting_strategies.jerphagnon1970 import Jerphagnon1970Strategy
from fitting_results import upsert_fitting_result

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *

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

            return np.asarray(lc_list, dtype=float)

        dL_pos = differential_L(th_pos, pol_in, pol_out)
        dL_neg = differential_L(th_neg, pol_in, pol_out)

        parts = [arr for arr in (dL_pos, dL_neg) if len(arr) > 0]
        if not parts:
            raise ValueError("No valid adjacent-minima pairs to compute Lc.")

        diffs = np.concatenate(parts)
        diffs = np.asarray(diffs, dtype=float)
        if diffs.size == 0:
            raise ValueError("No valid adjacent-minima pairs after filtering.")

        Lc_mean = float(np.mean(diffs))
        Lc_std = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else float("nan")

        fit_data = {
            "minima_idx" : valid_minima_idx,
            "x_in_range" : m,
            "dL_pos" : dL_pos,
            "dL_neg" : dL_neg,
            "parts": parts
        }
        result =  {
            "Lc_mean_mm": Lc_mean,
            "Lc_std_mm": Lc_std,
            "minima_count": int(th_min.size),
            "n_count": int(diffs.size)
            # "theta_used_deg": mask
        }

        return result, fit_data


class Bechthold1977GlobalNFitStrategy(Bechthold1977Strategy):
    """
    Global refractive-index fitting for Bechthold rotation scans.

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

    def _prepare_measurement_bundle(self, analysis, source_dir):
        strategy = Bechthold1977Strategy(analysis)
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
        if phase_pairs_deg.shape[0] == 0:
            raise ValueError(
                f"No adjacent minima pairs available for n fitting in {source_dir or 'current measurement'}."
            )

        geometry_key = strategy._geometry_key(analysis.meta)
        if geometry_key not in strategy.GEOMETRY_FUNCTIONS:
            raise FittingConfigurationError(f"This geometry is not supported for Bechthold global n fit: {geometry_key}")

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
            "L0_mm": float(analysis.meta["thickness_info"]["t_center_mm"]),
            "source_dir": None if source_dir is None else str(source_dir),
            "geometry_key": geometry_key,
        }

    def _load_measurement_group(self):
        from shg_analysis import SHGDataAnalysis

        group_dirs = self._resolve_group_measurement_dirs(self.analysis.meta)
        measurements = []
        for index, source_dir in enumerate(group_dirs):
            if index == 0:
                analysis = self.analysis
            else:
                analysis = SHGDataAnalysis(source_dir)
            measurements.append(self._prepare_measurement_bundle(analysis, source_dir))
        return measurements

    def _dn_dict_from_params(self, params):
        params = np.asarray(params, dtype=float)
        return {
            key: float(params[index])
            for index, key in enumerate(self.DN_PARAMETER_KEYS)
        }

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

    def fit_all(self):
        measurements = self._load_measurement_group()
        current = measurements[0]

        n_minima_residuals = sum(len(measurement["theta_min_deg"]) for measurement in measurements)
        n_data_residuals = sum(len(measurement["phase_pairs_deg"]) for measurement in measurements)
        n_total_residuals = n_minima_residuals + n_data_residuals + len(self.DN_PARAMETER_KEYS) + len(measurements)

        def residual(params):
            params = np.asarray(params, dtype=float)
            try:
                dn_override = self._dn_dict_from_params(params[: len(self.DN_PARAMETER_KEYS)])
                parts = []
                for index, measurement in enumerate(measurements):
                    dL_mm = float(params[len(self.DN_PARAMETER_KEYS) + index])
                    L_mm = measurement["L0_mm"] + dL_mm
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

        x0 = np.zeros(len(self.DN_PARAMETER_KEYS) + len(measurements), dtype=float)
        bounds_dn = np.full(len(self.DN_PARAMETER_KEYS), self.DN_PRIOR_SIGMA, dtype=float)
        bounds_L = np.full(len(measurements), self.L_PRIOR_SIGMA_MM, dtype=float)
        lower = np.concatenate((-bounds_dn, -bounds_L))
        upper = np.concatenate((bounds_dn, bounds_L))

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
        )

        dn_override = self._dn_dict_from_params(result.x[: len(self.DN_PARAMETER_KEYS)])
        group_fit_results = []
        current_refit = None
        for index, measurement in enumerate(measurements):
            initial_dL_mm = float(result.x[len(self.DN_PARAMETER_KEYS) + index])
            initial_L_mm = measurement["L0_mm"] + initial_dL_mm
            refit = self._refit_L_small_angle_with_dn(measurement, dn_override)
            L_mm = float(refit["L_mm"]) if np.isfinite(float(refit["L_mm"])) else float(initial_L_mm)
            dL_mm = L_mm - float(measurement["L0_mm"])
            if index == 0:
                current_refit = refit
            group_fit_results.append(
                {
                    "source_dir": measurement["source_dir"],
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
                    "k_scale_small_angle": float(refit["k_scale"]) if np.isfinite(float(refit["k_scale"])) else float("nan"),
                    "L_refit_success": bool(refit["success"]),
                    "minima_count": int(np.asarray(measurement["minima_idx"], dtype=int).size),
                    "phase_pair_count": int(len(measurement["phase_pairs_deg"])),
                }
            )

        if current_refit is None:
            raise ValueError("Failed to compute the current measurement L refit.")
        current_L_mm = float(current_refit["L_mm"])

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
            "group_size": int(len(measurements)),
            "phase_pair_count": int(n_data_residuals),
            "n_fit_cost": float(result.cost),
            "n_fit_success": bool(result.success),
        }
        fit_result.update(dn_override)
        for result_key, n_key in self.N_RESULT_KEYS:
            fit_result[result_key] = float(n_values[n_key])
        if isinstance(fit_aux, dict) and fit_aux.get("d_factor") is not None:
            fit_result["d_factor"] = self._coerce_scalar(fit_aux["d_factor"])

        self.analysis.meta = upsert_fitting_result(
            self.analysis.meta,
            self.__class__.__name__,
            fit_result,
            strategy_module=self.__class__.__module__,
            strategy_display_name=self.__class__.__name__,
        )
        self.analysis.meta["n_fit_group_results"] = group_fit_results

        csv_path = self.analysis.csv_path
        json_path = self.analysis.json_path
        self.analysis.data = out
        out.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis.meta, f, ensure_ascii=False, indent=2)

        return fit_result


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
