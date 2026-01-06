import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import find_peaks, savgol_filter  # <-- add
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from fitting_strategies.base import SHGFittingStrategy
from fitting_strategies.base import FittingConfigurationError

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *


class Jerphagnon1970Strategy(SHGFittingStrategy):
    """Fitting strategy based on Jerphagnon et al., 1970."""
    def __init__(self, analysis):
        super().__init__(analysis)

    GEOMETRY_P_FUNCTIONS = {
        # returns p(\theta) for possible combination
        # key configuration is:("material", (cut), "rot axis", pol_in, pol_out)

        # quartz d11
        ("SiO2", (0,1,0), "001", 90, 90): lambda theta_p_w: np.cos(3 * theta_p_w),

        # KDP d36
        ("KH2PO4", (1,1,0), "001", 90, 0): lambda theta_p_w: np.sin(2*theta_p_w - (np.pi/2)),

        # LiNbO3 d33
        ("LiNbO3", (0,1,0), "001", 0, 0): lambda _: 1.0,

        # LiNbO3 d31
        ("LiNbO3", (0,1,0), "001", 90, 0): lambda _: 1.0,

        # BMF d31
        ("BaMgF4", (0,1,0), "100", 0, 90): lambda theta_p_w: np.cos(theta_p_w),

        # BMF d32
        ("BaMgF4", (1,0,0), "010", 0, 90): lambda theta_p_w: np.cos(theta_p_w),

        # BMF d33
        ("BaMgF4", (0,1,0), "001", 0, 0): lambda _:1.0,

        # BMF d33
        ("BaMgF4", (1,0,0), "001", 0, 0): lambda _:1.0
    }

    def normalize_axis(self, axis):
        """
        Normalize axis representation to string '100', '010', or '001'.

        Accepted inputs:
            '100', '010', '001'
            [1,0,0], [0,1,0], [0,0,1]
            (1,0,0), (0,1,0), (0,0,1)
        """
        # String case
        if isinstance(axis, str):
            axis = axis.strip()
            if axis in ("100", "010", "001"):
                return axis
            raise ValueError(f"Invalid axis string: {axis}")

        # Sequence case
        try:
            a = list(axis)
        except TypeError:
            raise ValueError(f"Invalid axis type: {axis}")

        if len(a) != 3:
            raise ValueError(f"Axis must have length 3: {axis}")

        # Allow int / bool / float close to 0 or 1
        tol = 1e-6
        vec = [1 if abs(x - 1) < tol else 0 if abs(x) < tol else None for x in a]

        if vec == [1, 0, 0]:
            return "100"
        if vec == [0, 1, 0]:
            return "010"
        if vec == [0, 0, 1]:
            return "001"

        raise ValueError(f"Invalid axis vector: {axis}")

    def _third_axis(self, cut_axis: str, rot_axis: str) -> str:
        """Return the remaining principal axis label among {'100','010','001'}."""
        axes = {"100", "010", "001"}
        if cut_axis not in axes or rot_axis not in axes or cut_axis == rot_axis:
            raise ValueError(f"Invalid axes: cut_axis={cut_axis}, rot_axis={rot_axis}")
        third = list(axes - {cut_axis, rot_axis})
        if len(third) != 1:
            raise ValueError("Failed to determine third axis.")
        
        return third[0]

    # n_w and n_2w for specific setup (extend angle dependent n_e if needed)
    def n_eff(self, pol_deg, wav_nm, theta_deg=None):
        """Return n for a given polarization angle and crystal setting.

        Parameters
        ----------
        pol_deg : float
            Polarization angle in lab frame [deg], 0 or 90 only (for now).
        wav_nm : float
            Vacuum wavelength [nm].
        theta_deg : float
            Incidence angle (for future angle-dependent n_e).

        Returns
        -------
        float
            Effective refractive index n(wav_nm, pol_deg, theta_deg).
        """
        meta = self.analysis.meta
        crystal = CRYSTALS[meta["material"]]()

        if meta["rot/trans_axis"]=="001":
            if np.isclose(pol_deg, 0, atol=1e-3):
                n = crystal.get_n(wav_nm, polarization="e")
            elif np.isclose(pol_deg, 90, atol=1e-3):
                n = crystal.get_n(wav_nm, polarization="o")
            else:
                raise FittingConfigurationError(
                    f"Input polarization of {pol_deg} deg is not supported. Only 0 or 90 deg is available."
                )
            
        elif meta["rot/trans_axis"] in ("100", "010"):
            if np.isclose(pol_deg, 0, atol=1e-3):
                n = crystal.get_n(wav_nm, polarization="o")
            elif np.isclose(pol_deg, 90, atol=1e-3):
                # angle dependent
                raise FittingConfigurationError(
                    "Angle dependent refractive index is not supported."
                )
            else:
                raise FittingConfigurationError(
                    f"Input polarization of {pol_deg} deg is not supported. Only 0 or 90 deg is available."
                )
        else:
            raise FittingConfigurationError(
            f"Unexpected rotation axis: {meta['rot/trans_axis']}. "
            "Supported values are '001', '100', '010'."
        )
        return n

    def _maker_fringes(self, override: dict = {}, envelope=False, return_aux=False):
        """Full Maker fringes SHG model with Fresnel coefficients and projection factor.
            
        Parameters
        ----------
        analysis : SHGDataAnalysis
            Analysis instance containing meta, data, and utilities.
        override : dict, optional
            If given, overrides the default:
                angle array (deg)
                thickness (mm)
                n_w, n_2w
        envelope : bool, optional
            If True, return envelope values.
        """
         
        meta = self.analysis.meta
        wl1_nm = meta["wavelength_nm"]
        wl1_mm = wl1_nm * 1e-6
        pol_in = meta["input_polarization"] # 0-90 deg
        pol_out = meta["detected_polarization"] # 0-90 deg
        crystal = CRYSTALS[meta["material"]]()
        data = self.analysis.data

        beam_r_x = meta["beam_r_x"]
        beam_r_y = meta["beam_r_y"]
        beam_r = np.sqrt(beam_r_x * beam_r_y)

        if "L" in override.keys():
            L = override["L"]
        else:
            L = meta["thickness_info"]["t_at_thin_end_mm"] # or analysis.calc_thickness_array

        if "theta_deg" in override.keys():
            theta_deg = override["theta_deg"]
        else:
            theta_deg = np.asarray(data.get("position_centered", data["position"]))
        
        if "n" in override.keys():
            # n = override["n"]
            # if np.isclose(pol_in, 0):
            #     n_w = n["n_w"]
            n_w = self.n_eff(pol_in, wl1_nm, theta_deg)
            n_2w = self.n_eff(pol_out, wl1_nm / 2.0, theta_deg)
        else:
            n_w = self.n_eff(pol_in, wl1_nm, theta_deg)
            n_2w = self.n_eff(pol_out, wl1_nm / 2.0, theta_deg)
        

        def refraction_angle(theta):
            n_w = self.n_eff(pol_in, wl1_nm, np.rad2deg(theta))
            n_2w = self.n_eff(pol_out, wl1_nm / 2.0, np.rad2deg(theta))

            theta_p_w = np.arcsin(np.sin(theta) / n_w)
            theta_p_2w = np.arcsin(np.sin(theta) / n_2w)
            angles = {"w": theta_p_w, "2w": theta_p_2w}
            return angles

        def fresnel_t(theta, pol):
            """Amplitude transmission coefficient at a planar interface. Eq.(4-5)"""
            n_in = 1
            n_out = self.n_eff(pol_in, wl1_nm, np.rad2deg(theta))
            theta_p_w = refraction_angle(theta)["w"]
            if pol == 0: # s-pol
                t = 2 * n_in * np.cos(theta) / (n_in * np.cos(theta) + n_out * np.cos(theta_p_w))
            elif pol == 90: #p-pol
                t = 2 * n_in * np.cos(theta) / (n_out * np.cos(theta) + n_in * np.cos(theta_p_w))
            else:
                raise ValueError("pol must be 0 or 90 deg")
            return t
        
        def T_at_back(theta, pol):
            """
            Calculate back-surface transmission coefficient for SH wave
            Eq.(11-12). Check Eq.(A.36), (A.43, 45) for derivation

            Parameters
            ----------
            pol : float
                0 for nonlinear polarization perpendicular to the plane of incidence
                90 for nonlinear polarization parallel to the plane of incidence
            """
            theta_p_w = refraction_angle(theta)["w"]
            theta_p_2w = refraction_angle(theta)["2w"]

            n_w = self.n_eff(pol_in, wl1_nm, np.rad2deg(theta))
            n_2w = self.n_eff(pol_out, wl1_nm / 2.0, np.rad2deg(theta))

            if np.isclose(pol, 0.0, atol=1e-3):
                T = 2* n_2w * np.cos(theta_p_2w) * \
                    (np.cos(theta) + n_w * np.cos(theta_p_w))* (n_w*np.cos(theta_p_w) + n_2w*np.cos(theta_p_2w)) / \
                    ((n_2w*np.cos(theta_p_2w) + np.cos(theta))**3)
            elif np.isclose(pol, 90.0, atol=1e-3):
                T = 2* n_2w * np.cos(theta_p_2w) * \
                    (np.cos(theta_p_w) + n_w * np.cos(theta))* (n_w*np.cos(theta_p_2w) + n_2w*np.cos(theta_p_w)) / \
                    ((n_2w*np.cos(theta) + np.cos(theta_p_2w))**3)
            else:
                raise ValueError("pol must be 0 or 90 degree")
            
            return T
        
        def multiple_refrection():
            """
            Described in Eq.(B.6)
            """
            R = 1
            return R
        
        def projection_factor(theta):
            meta = self.analysis.meta
            theta_p_w = refraction_angle(theta)["w"]

            key = (
                meta["material"],
                tuple(meta["crystal_orientation"]),
                meta["rot/trans_axis"],
                int(round(meta["input_polarization"])),
                int(round(meta["detected_polarization"])),
            )

            try:
                p_func = self.GEOMETRY_P_FUNCTIONS[key]
            except KeyError:
                raise FittingConfigurationError(
                    f"This geometry is not supported: {key}"
                )

            return p_func(theta_p_w)
        
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
        
        theta = np.deg2rad(theta_deg)
        theta_p_w = refraction_angle(theta)["w"]
        theta_p_2w = refraction_angle(theta)["2w"]

        n_w_0 = self.n_eff(pol_in, wl1_nm, 0)
        n_2w_0 = self.n_eff(pol_out, wl1_nm/2.0, 0)

        Psi = (np.pi * L / 2) * (4 / wl1_mm) * \
                (n_w * np.cos(theta_p_w) - n_2w * np.cos(theta_p_2w))    

        # normalized envelope (P"_cw / Pm(0))
        P_env = (fresnel_t(theta, pol_in) / fresnel_t(0, pol_in))**4 * \
                (T_at_back(theta, pol_out) / T_at_back(0, pol_out)) * \
                (projection_factor(theta) / projection_factor(0))**2 * \
                (beam_size_correction(L, beam_r, theta) / beam_size_correction(L, beam_r, 0)) * \
                (n_w_0**2 - n_2w_0**2)**2 / (n_w**2 - n_2w**2)**2
        P_N = P_env * np.sin(Psi)**2

        # env_peak is propotional to d**2 * d_factor.
        d_factor = fresnel_t(0, pol_in)**4 *\
                T_at_back(0, pol_out) * \
                projection_factor(0)**2 * \
                beam_size_correction(L, beam_r, 0) / (n_w_0**2 - n_2w_0**2)**2

        if envelope:
            model = P_env
        else:
            model =  P_N

        if not return_aux:
            return model
        aux = {
            "d_factor": d_factor,
            "t": fresnel_t(0, pol_in),
            "T": T_at_back(0, pol_out)
        }
        return model, aux


    def estimate_fringe_period(self, x, y):
        # Remove mean to suppress DC component
        y_detrended = y - np.mean(y)
        dx = x[1] - x[0]

        # FFT
        fft = np.fft.rfft(y_detrended)
        freq = np.fft.rfftfreq(len(y_detrended), d=dx)

        # Ignore zero frequency
        fft[0] = 0.0
        fft[1] = 0.0

        # Dominant frequency (lowest non-zero peak)
        idx = np.argmax(np.abs(fft))
        dominant_freq = freq[idx]

        if dominant_freq == 0:
            return None

        # Period in x-unit
        period = 1.0 / dominant_freq

        return period, idx, freq, fft
    
    def detect_minima(self, x, y, threshold_ratio=0.02, order=None):
        '''
        Docstring for detect_minima
        
        :param x: Position
        :param y: SHG intensity
        :param threshold_ratio: error ratio to max(y) allowed around y=0
        :param order: Number of datapoints to be compared for detecting minima.
        '''
        # Find local minima indices
        step = np.abs(x[0] - x[1])
        if not order:
            # order is about 1 mm or 1 deg
            order = max(int(1.0/step), 1.0)
        idx_min = argrelextrema(y, np.less, order=order)[0]

        # Global maximum - min
        A = np.max(y) - np.min(y)

        # Filter by intensity threshold
        valid = (y[idx_min] - np.min(y)) < threshold_ratio * A

        return idx_min[valid], idx_min
    
    def detect_extrema(self, x, y, mode, threshold_ratio=0.01, order=3):
        '''
        parameters
        mode : str
            "maxima" or "minima"
        '''
        # Estimate fringe period
        period, idx, freq, fft = self.estimate_fringe_period(x,y)

        if period is None:
            raise RuntimeError("Failed to estimate fringe period")

        # Convert period to number of samples
        dx = x[1] - x[0]
        window = int(period / dx / 3)

        # Window length must be odd and >= 5
        if window < 5:
            window = 5
        if window % 2 == 0:
            window += 1

        # Smooth data
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)

        void = []
        if mode == "maxima":
            # Find local maxima
            idx_max = argrelextrema(y_smooth, np.greater, order=order)[0]
            
            return idx_max, y_smooth, void

        elif mode == "minima":
            # Find local minima indices
            idx_min = argrelextrema(y_smooth, np.less, order=order)[0]
            # Global maximum - min
            A = np.max(y) - np.min(y)

            # Filter by intensity threshold
            valid = (y[idx_min] - np.min(y)) < threshold_ratio * A
            return idx_min[valid], y_smooth, idx_min
        
        else:
            raise ValueError("mode must be 'maxima' or 'minima'")


    # ---------- stage C/D pipeline ----------
    def _position_centering(self, data):
        x = np.asarray(data["position"])
        y = np.asarray(data["intensity_corrected"])
        n = len(x)

        if n < 10:
            # fallback: not enough points to be fancy
            c_best = 0
            out = data.copy()
            out["position_centered"] = data["position"] - c_best
            return out
        
        def antisym_cost(c):
            # reflect xg around c; keep only points whose mirror is in-range
            xr = 2.0 *c - x
            valid = (xr >= x[0]) & (xr <= x[-1])
            if not np.any(valid) or np.sum(valid) < 10: # excluding data points on the side 
                return np.inf
            yr = np.interp(xr[valid], x, y)
            return np.mean((y[valid] - yr)**2)
        
        # Check if measurement range crosses 0 deg
        crosses_zero = (x.min() <= 0.0) and (x.max() >= 0.0)
        
        # Case 1: range does NOT cross 0 → cannot determine center reliably
        if not crosses_zero:
            c_best = 0.0
            fit_data = {
                "c_candidates": None,
                "costs": None,
                "c0": c_best,
                "c_local": None,
                "costs_local": None,
                "c_best": c_best,
            }

        # Case 2: range crosses 0 → search only around 0 deg
        else:
            L = x.max() - x.min()

            # Coarse search window around 0 (e.g. ±20% of total span, clipped to data)
            coarse_span = 10  # "around 0 deg" range; adjust if needed
            c_lo = max(x.min(), -coarse_span)
            c_hi = min(x.max(), +coarse_span)

            # If data range is very narrow, avoid zero-width window
            if c_hi <= c_lo:
                c_lo, c_hi = x.min(), x.max()

            # Coarse candidates around 0
            c_candidates = np.linspace(c_lo, c_hi, 201)
            costs = np.array([antisym_cost(c) for c in c_candidates])

            # If everything failed (all inf), just fall back to 0
            if not np.isfinite(costs).any():
                c0 = 0.0
            else:
                c0 = c_candidates[np.argmin(costs)]

            # Refined search around c0 (still close to 0)
            local_span = 0.02 * L  # 2% of total span
            local_span = max(local_span, 1e-6)  # avoid zero-span

            c_local_lo = max(x.min(), c0 - local_span)
            c_local_hi = min(x.max(), c0 + local_span)
            c_local = np.linspace(c_local_lo, c_local_hi, 101)
            costs_local = np.array([antisym_cost(c) for c in c_local])

            if not np.isfinite(costs_local).any():
                c_best = c0
            else:
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
        according to III D-1:
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
        minima_idx, _ = self.detect_minima(x, y)

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
        """III D-1 (a): Fit L at small angles (not specified) to adjust nominal thickness."""
        pos = np.asarray(data.get("position_centered", data["position"]))
        I = np.asarray(data["offset_corrected"])
        # Small-angle mask (|θ| < 5 deg)
        mask = np.abs(pos) < 5
        theta_small = pos[mask]
        I_small = I[mask]

        # Initial guess: thickness derived from metadata
        L_guess = meta["thickness_info"]["t_at_thin_end_mm"]
        k_guess = float(np.nanmax(I_small) or 1.0)
        # Fallback / clip
        if not np.isfinite(k_guess) or k_guess <= 0:
            k_guess = 1.0

        def model_small(theta_deg, L, k):
            override = {
                "theta_deg": theta_deg,
                "L": L
            }
            return k* self._maker_fringes(override=override)
        # range of estimated L(L +/- 10 um), k(0~inf)
        bounds = ([L_guess - 0.01, 0.0],[L_guess + 0.01, np.inf])
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
        minima_idx, _ = self.detect_minima(theta_deg, I)
        valid_minima_idx = minima_idx[m[minima_idx]]
        
        th_min = theta_deg[valid_minima_idx]
        th_pos = np.sort(th_min[th_min > 0.0])
        th_neg = np.sort(th_min[th_min < 0.0])
        wl1_nm = meta["wavelength_nm"]
        crystal = CRYSTALS[meta["material"]]()
        pol_in = meta["input_polarization"] # 0-90 deg
        pol_out = meta["detected_polarization"] # 0-90 deg


        # def refraction_angle(theta):
        #     theta_p_w = np.arcsin(np.sin(theta) / n_w)
        #     theta_p_2w = np.arcsin(np.sin(theta) / n_2w)
        #     angles = {"w": theta_p_w, "2w": theta_p_2w}
        #     return angles        

        def differential_L(th_list):
            th_list = np.asarray(th_list)
            if th_list.size < 2:
                return np.array([], dtype=float)
            
            th_rad = np.radians(th_list)
            lc_list = []
            for i in range(th_rad.size - 1):
                # Use midpoint refractive indices between adjacent angles
                n_w = self.n_eff(pol_in, wl1_nm, th_list[i])
                n_2w = self.n_eff(pol_out, wl1_nm / 2.0, th_list[i])

                lc = fitted_L_mm * (np.sin(th_rad[i + 1])**2 - np.sin(th_rad[i])**2) / (4.0 * n_2w * n_w)
                lc_list.append(abs(lc))

            return np.asarray(lc_list, dtype=float)
        
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
        theta_deg = np.asarray(data.get("position_centered", data["position"]))
        I_meas = np.asarray(data.get("offset_corrected", data["intensity_corrected"]))

        # Find local maxima (peaks)
        th_step = self.analysis.meta["step"]
        order = max(int(1.0 / th_step), 1)
        maxima_idx = argrelextrema(I_meas, np.greater, order=order)[0]
        if maxima_idx.size == 0:
            raise RuntimeError("No maxima found in data. Check preprocessing or order parameter.")
        # remove the peak aroung 0 deg (usually detected mistakenly)
        valid = np.abs(theta_deg[maxima_idx]) > 3.0
        maxima_idx = maxima_idx[valid]
        
        theta_pk = theta_deg[maxima_idx]
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
        try:
            Lc, _Lc = self._calc_Lc_large_angle(self.analysis.meta, data, [15, 180], L_fit["L_mm"])
        except Exception as e:
            print(f"Error: {e}")
            Lc = []
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