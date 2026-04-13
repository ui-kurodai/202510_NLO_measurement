import numpy as np
from scipy.signal import argrelextrema

from fitting_strategies.base import BaseWedgeStrategy
from fitting_strategies.base import FittingConfigurationError
from fitting_strategies.jerphagnon1970 import Jerphagnon1970Strategy

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
        if "theta_deg" in override.keys():
            theta_deg = override["theta_deg"]
        else:
            theta_deg = np.asarray(data.get("position_centered", data["position"]))
        
        theta = np.deg2rad(theta_deg)

        principle_n_w = self.n_eff(pol_in, wl1_nm, meta=meta, aux=True)
        n_w_third = principle_n_w["n_third"]
        n_w_rot = principle_n_w["n_rot"]
        n_w_cut = principle_n_w["n_cut"]

        principle_n_2w = self.n_eff(pol_out, wl1_nm / 2.0, meta=meta, aux=True)
        n_2w_third = principle_n_2w["n_third"]
        n_2w_rot = principle_n_2w["n_rot"]
        n_2w_cut = principle_n_2w["n_cut"]

        v_w = lambda th : (n_w_third / n_w_cut) * np.sqrt(n_w_cut**2 - np.sin(th)**2)
        v_2w = lambda th : (n_2w_third / n_2w_cut) * np.sqrt(n_2w_cut**2 - np.sin(th)**2)

        w_w = lambda th: np.sqrt(n_w_rot**2 - np.sin(th)**2)
        w_2w = lambda th : np.sqrt(n_2w_rot**2 - np.sin(th)**2)
        if isinstance(meta["crystal_orientation"], list):
            meta["crystal_orientation"] = "".join(map(str, meta["crystal_orientation"]))

        key = (
                meta["material"],
                meta["crystal_orientation"],
                meta["rot/trans_axis"],
                int(round(meta["input_polarization"])),
                int(round(meta["detected_polarization"])),
            )
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
