import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from fitting_strategies.base import SHGFittingStrategy

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *


class Jerphagnon1970Strategy(SHGFittingStrategy):
    """Fitting strategy based on Jerphagnon et al., 1970."""
    def __init__(self, analysis):
        self.analysis = analysis

    def _maker_fringes(self, override: dict = {}, envelope=False):
        """Full Maker fringes SHG model with Fresnel coefficients and projection factor.
            
        Parameters
        ----------
        analysis : SHGDataAnalysis
            Analysis instance containing meta, data, and utilities.
        override : dict, optional
            If given, overrides the default angle array or thickness.
        envelope : bool, optional
            If True, return envelope values.
        """
         
        meta = self.analysis.meta
        wl1_nm = meta["wavelength_nm"]
        wl1_mm = wl1_nm * 1e-6
        pol_in = meta["input_polarization"]
        pol_out = meta["detected_polarization"]
        crystal = CRYSTALS[meta["material"]]()
        data = self.analysis.data
        theta_deg = np.asarray(data.get("position_centered", data["position"]))

        beam_r_x = meta["beam_r_x"]
        beam_r_y = meta["beam_r_y"]
        beam_r = np.sqrt(beam_r_x * beam_r_y)

        # Refractive indices (extend to o/e as needed)
        n_w = crystal.get_n(wl1_nm, polarization="o")
        n_2w = crystal.get_n(wl1_nm / 2, polarization="o")

        def refraction_angle(theta):
            theta_p_w = np.arcsin(np.sin(theta) / n_w)
            theta_p_2w = np.arcsin(np.sin(theta) / n_2w)
            angles = {"w": theta_p_w, "2w": theta_p_2w}
            return angles

        def fresnel_t(theta, pol):
            """Amplitude transmission coefficient at a planar interface. Eq.(4-5)"""
            n_in = 1
            n_out = n_w
            theta_p_w = refraction_angle(theta)["w"]
            if pol == 0: # s-pol
                t = 2 * n_in * np.cos(theta) / (n_in * np.cos(theta) + n_out * np.cos(theta_p_w))
            elif pol == 90: #p-pol
                t = 2 * n_in * np.cos(theta) / (n_out * np.cos(theta) + n_in * np.cos(theta_p_w))
            else:
                raise ValueError("pol must be 's' or 'p'")
            return t
        
        def T_at_back(theta, pol):
            """
            Calculate back-surface transmission coefficient for SH wave
            Eq.(11-12). Check Eq.(A.36), (A.43, 45) for derivation

            Parameters
            ----------
            pol : str
                'perp' for nonlinear polarization perpendicular to the plane of incidence
                'para' for nonlinear polarization parallel to the plane of incidence
            """
            theta_p_w = refraction_angle(theta)["w"]
            theta_p_2w = refraction_angle(theta)["2w"]

            if pol == "perp":
                T = 2* n_2w * np.cos(theta_p_2w) * \
                    (np.cos(theta) + n_w * np.cos(theta_p_w))* (n_w*np.cos(theta_p_w) + n_2w*np.cos(theta_p_2w)) / \
                    ((n_2w*np.cos(theta_p_2w) + np.cos(theta))**3)
            elif pol == "para":
                T = 2* n_2w * np.cos(theta_p_2w) * \
                    (np.cos(theta_p_w) + n_w * np.cos(theta))* (n_w*np.cos(theta_p_2w) + n_2w*np.cos(theta_p_w)) / \
                    ((n_2w*np.cos(theta) + np.cos(theta_p_2w))**3)
            else:
                raise ValueError("pol must be 'para' or 'perp'")
            
            return T
        
        def multiple_refrection():
            """
            Described in Eq.(B.6)
            """
            R = 1
            return R
        
        def projection_factor(theta, pol_in):
            # Projection factor (placeholder; replace with exact form per geometry)
            theta_p_w = refraction_angle(theta)["w"]
            theta_p_2w = refraction_angle(theta)["2w"]

            if pol_in == 90 and pol_out == "p": # p-pol -> 90 deg
                p_factor = np.cos(theta_p_w) * np.cos(theta_p_2w)
            elif pol_in == 0 and pol_out == "s":    # s-pol -> 0 deg
                p_factor = 1.0
            else:
                p_factor = 1.0
            return p_factor
        
        def beam_size_correction(L_mm, w_um, theta):
            """
            Described in Eq.(16). Check Eq.(C.10) for derivation.
            parameters
            L_mm : float
                Crystal thickness in mm.
            w_um : float
                Spot radius of the Gaussian beam in micron.
            """
            w_mm = w_um * 1e-3
            theta_p_w = refraction_angle(theta)["w"]
            theta_p_2w = refraction_angle(theta)["2w"]

            B = np.exp(-((L_mm/w_mm) * np.cos(theta) * (np.tan(theta_p_w) - np.tan(theta_p_2w)))**2)
            return B
               
        if "L" in override.keys():
            L = override["L"]
        else:
            L = meta["thickness_info"]["t_at_thin_end_mm"] # or analysis.calc_thickness_array

        if "theta_deg" in override.keys():
            theta = np.radians(override["theta_deg"])
        else:
            theta_deg = np.asarray(data.get("position_centered", data["position"]))
            theta = np.radians(theta_deg)
        theta_p_w = refraction_angle(theta)["w"]
        theta_p_2w = refraction_angle(theta)["2w"]

        Psi = (np.pi * L / 2) * (4 / wl1_mm) * \
                (n_w * np.cos(theta_p_w) - n_2w * np.cos(theta_p_2w))    

        # normalized envelope (P"_cw / Pm(0))
        pol = "perp"
        P_env = (fresnel_t(theta, pol_in) / fresnel_t(0, pol_in))**4 * \
                (T_at_back(theta, pol) / T_at_back(0, pol)) * \
                (projection_factor(theta, pol_in) / projection_factor(0, pol_in))**2 * \
                (beam_size_correction(L, beam_r, theta) / beam_size_correction(L, beam_r, 0))
        P_N = P_env * np.sin(Psi)**2

        if envelope:
            return P_env
        else:
            return P_N

    # ---------- stage C/D pipeline ----------
    def _position_centering(self, data):
        x = np.asarray(data["position"])
        y = np.asarray(data["intensity_corrected"])
        n = len(x)

        if n < 10:
            # fallback: not enough points to be fancy
            c_best = np.median(x)
            out = data.copy()
            out["position_centered"] = data["position"] - c_best
            return out

        # candidate centers: coarse → fine search
        c_candidates = np.linspace(x.min(), x.max(), 301)

        def antisym_cost(c):
            # reflect xg around c; keep only points whose mirror is in-range
            xr = 2*c - x
            valid = (xr >= x[0]) & (xr <= x[-1])
            if not np.any(valid) or np.sum(valid) < 10:
                return np.inf
            yr = np.interp(xr[valid], x, y)
            return np.mean((y[valid] - yr)**2)

        # rough search
        costs = np.array([antisym_cost(c) for c in c_candidates])
        idx = np.argmin(costs)
        c0 = c_candidates[idx]

        # refined search around c0
        span = (x.max() - x.min()) * 0.02 or 1e-9
        c_local = np.linspace(max(x.min(), c0 - span), min(x.max(), c0 + span), 101)
        costs_local = np.array([antisym_cost(c) for c in c_local])
        c_best = c_local[np.argmin(costs_local)]

        fit_data = {
            "c_candidates" : c_candidates,
            "costs" : costs,
            "c0" : c0,
            "c_local" : c_local,
            "costs_local" : costs_local,
            "c_best" : c_best
        }

        out = data.copy()
        out["position_centered"] = data["position"] - c_best
        return out, fit_data

    def _subtract_offset(self, data):
        """
        III C-1: Fit and subtract minima offset due to angular averaging.
        Parameters
        ----------
        data : dict
            Must contain the columns:
            - "intensity_corrected" : y-axis values after intensity correction
        Returns
        -------
        dict
            Copy of input dict with a new column:
            - "offset_corrected"
        """
        x = np.asarray(data.get("position_centered", data["position"])) # if exist, use centered position
        y = np.asarray(data["intensity_corrected"])

        # Find local minima indices
        minima_idx = argrelextrema(y, np.less, order=5)[0]

        # Exclude points near the center (e.g. ±5°)
        exclude_range = 5.0
        if minima_idx.size > 0:
            mask = np.abs(x[minima_idx]) > exclude_range
            minima_idx = minima_idx[mask]

        if len(minima_idx) == 0:
            # fallback: use global minimum if no local minima
            offset = y.min()
        else:
            # average of local minima for robustness
            offset = y[minima_idx].mean()

        corrected_y = y - offset

        fit_data = {
            "minima_idx" : minima_idx,
            "offset" : offset
        }

        out = data.copy()
        out["offset_corrected"] = corrected_y
        return out, fit_data

    def _fit_L_small_angle(self, meta, data):
        """III D-1 (a): Fit L at small angles to adjust nominal thickness."""
        pos = np.asarray(data.get("position_centered", data["position"]))
        I = np.asarray(data["offset_corrected"])
        # Small-angle mask (|θ| < 5 deg)
        mask = np.abs(pos) < 5
        theta_small = pos[mask]
        I_small = I[mask]

        # Initial guess: thickness derived from metadata
        L_guess = meta["thickness_info"]["t_at_thin_end_mm"]
        k_guess = float(np.nanmax(I_small) or 1.0)

        def model_small(theta_deg, L, k):
            override = {
                "theta_deg": theta_deg,
                "L": L
            }
            return k* self._maker_fringes(override=override)
        bounds = ([0.0, 0.0],[20, np.inf])   # range of estimated L, k
        popt, pcov = curve_fit(model_small,
                            theta_small,
                            I_small, 
                            p0=[L_guess, k_guess], 
                            bounds=bounds, 
                            maxfev=20000
                            )
        L_fit, k_fit = map(float, popt)
        # rate uncertainty of 1-sigma (pcov: covariance matrix)
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan, np.nan]

        return {
            "L_mm": L_fit,
            "L_mm_std": float(perr[0]),
            "k_scale": k_fit,
            "k_scale_std": float(perr[1]),
            # "theta_window_deg": [mask[0], mask[-1]]
        }

    def _calc_Lc_large_angle(self, meta, data, mask, fitted_L_mm):
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
        minima_idx = argrelextrema(I, np.less, order=5)[0]
        valid_minima_idx = minima_idx[m[minima_idx]]
        
        th_min = theta_deg[valid_minima_idx]
        th_pos = np.sort(th_min[th_min > 0.0])
        th_neg = np.sort(th_min[th_min < 0.0])
        wl1_nm = meta["wavelength_nm"]
        crystal = CRYSTALS[meta["material"]]()
        n_w = crystal.get_n(wl1_nm, polarization="o")
        n_2w = crystal.get_n(wl1_nm / 2, polarization="o")

        # def refraction_angle(theta):
        #     theta_p_w = np.arcsin(np.sin(theta) / n_w)
        #     theta_p_2w = np.arcsin(np.sin(theta) / n_2w)
        #     angles = {"w": theta_p_w, "2w": theta_p_2w}
        #     return angles        

        def differential_L(th_list):
            if th_list.size < 2:
                return np.array([], dtype=float)
            
            th_rad = np.radians(th_list)
            lc_list = []
            for i in range(len(th_rad) - 1):
                lc = fitted_L_mm * (np.sin(th_rad[i+1])**2 - np.sin(th_rad[i])**2) / (4 * n_2w * n_w)
                lc = abs(lc)
                lc_list.append(lc)

            return lc_list
        
            # apparent L list
            # L_app = fitted_L_mm / np.cos(np.deg2rad(th_list))
            # differential（＞0）
            # dL = np.diff(L_app)
            # remove irregular value (oulier, negative, NaN, inf value)
            # if dL.size:
            #     q1, q3 = np.percentile(dL, [25, 75])
            #     iqr = q3 - q1
            #     lo = q1 - 1.5 * iqr
            #     hi = q3 + 1.5 * iqr
            #     keep = (dL > 0) & (dL >= lo) & (dL <= hi) & np.isfinite(dL)

            # return dL   # dL[keep] 

        dL_pos = differential_L(th_pos)
        dL_neg = differential_L(th_neg)

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

    def _fit_Pm0(self, data):
        """
        Fit the absolute envelope amplitude Pm(0) by least squares
        using experimental maxima and the normalized theoretical envelope.

        Parameters
        ----------
        data : dict
            Must contain:
            - "position_centered": np.ndarray [deg]
            - "offset_corrected": np.ndarray [intensity]

        Returns
        -------
        result : dict
            {
            "Pm0": float,
            "Pm0_stderr": float,
            "n_peaks": int,
            "theta_peaks_deg": np.ndarray,
            "I_peaks": np.ndarray,
            "Enorm_peaks": np.ndarray,
            "residual_rms": float,
            }
        """
        theta = np.asarray(data.get("position_centered", data["position"]))
        I_meas = np.asarray(data.get("offset_corrected", data["intensity_corrected"]))

        # Find local maxima (peaks)
        maxima_idx = argrelextrema(I_meas, np.greater, order=5)[0]
        if maxima_idx.size == 0:
            raise RuntimeError("No maxima found in data. Check preprocessing or order parameter.")

        theta_pk = theta[maxima_idx]
        I_pk = I_meas[maxima_idx]

        # Theoretical normalized envelope at maxima positions
        Enorm_pk = self._maker_fringes(override={"theta_deg": theta_pk}, envelope=True)
        Enorm_pk = np.asarray(Enorm_pk)

        if Enorm_pk.shape != I_pk.shape:
            raise RuntimeError("Envelope length mismatch between theory and experimental peaks.")

        # Least-squares closed form: I_pk ≈ Pm0 * Enorm_pk
        denom = np.dot(Enorm_pk, Enorm_pk)
        if denom <= 0:
            raise RuntimeError("Invalid denominator in regression. Check envelope values.")
        Pm0 = float(np.dot(Enorm_pk, I_pk) / denom)

        # Residuals
        resid = I_pk - Pm0 * Enorm_pk
        dof = max(1, I_pk.size - 1)
        sigma2 = float(np.sum(resid**2) / dof)

        # Standard error of parameter estimate
        Pm0_stderr = float(np.sqrt(sigma2 / denom))

        # RMS residual
        rms = float(np.sqrt(np.mean(resid**2)))

        result = {
            "Pm0": Pm0,
            "Pm0_stderr": Pm0_stderr,
            "n_peaks": int(I_pk.size),
            # "theta_peaks_deg": theta_pk,
            # "I_peaks": I_pk,
            # "Enorm_peaks": Enorm_pk,
            "residual_rms": rms
        }
        fit_data = {
            "maxima_idx": maxima_idx
        }

        return result, fit_data

    def fit_all(self):
        """Run full Jerphagnon1970 fitting pipeline and return results."""
        data, _centering = self._position_centering(self.analysis.data)
        data, _offset = self._subtract_offset(data)
        L_fit = self._fit_L_small_angle(self.analysis.meta, data)
        Lc, _Lc = self._calc_Lc_large_angle(self.analysis.meta, data, [15, 180], L_fit["L_mm"])
        Pm0_fit, _Pm0 = self._fit_Pm0(data)

        # add fitted theoretical values to csv
        out = data.copy()
        out["fit"] = Pm0_fit["Pm0"] * self._maker_fringes(override={"L":L_fit["L_mm"], "theta_deg": data["position_centered"]}) \
      + _offset["offset"]

        results = {}
        results.update(L_fit)
        results.update(Lc)
        results.update(Pm0_fit)
        self.analysis.meta.update(results)

        csv_path = self.analysis.csv_path
        json_path = self.analysis.json_path
        self.analysis.data = out
        out.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis.meta, f, ensure_ascii=False, indent=2)

        return results