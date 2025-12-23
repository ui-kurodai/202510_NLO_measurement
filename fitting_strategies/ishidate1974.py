import numpy as np
from scipy.signal import argrelextrema

from fitting_strategies.base import FittingConfigurationError
from fitting_strategies.jerphagnon1970 import Jerphagnon1970Strategy

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *

class Ishidate1974Strategy(Jerphagnon1970Strategy):

    INDEX_TO_AXIS = {"100": "a", "010": "b", "001": "c"}

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

    # # override n_eff function for biaxial crystals
    # def n_eff(self, pol_deg, wav_nm, theta_deg=None):
    #     """Return n for a given polarization angle and crystal setting.

    #     Parameters
    #     ----------
    #     pol_deg : float
    #         Polarization angle in lab frame [deg], 0 or 90 only (for now).
    #     wav_nm : float
    #         Vacuum wavelength [nm].
    #     theta_deg : float
    #         Incidence angle (for future angle-dependent n_e).

    #     Returns
    #     -------
    #     float
    #         Effective refractive index n(wav_nm, pol_deg, theta_deg).
    #     """

    #     meta = self.analysis.meta
    #     crystal = CRYSTALS[meta["material"]]()
    #     axiality = crystal.axiality
    #     if axiality != "biaxial":
    #         raise FittingConfigurationError(
    #             f"This strategy is for biaxial crystals. Please use another strategy for {axiality} crystals."
    #         )
        
    #     geometry = {"rot_axis": meta["rot/trans_axis"],
    #                 "cut_axis": meta["crystal_orientation"]
    #     }

    #     cut_axis = self.normalize_axis(geometry["cut_axis"])
    #     rot_axis = self.normalize_axis(geometry["rot_axis"])


    #     third_axis = self._third_axis(cut_axis, rot_axis)

    #     # Principal indices at the wavelength
    #     n_rot = crystal.get_n(wav_nm, polarization=self.INDEX_TO_AXIS[rot_axis])
    #     n_cut = crystal.get_n(wav_nm, polarization=self.INDEX_TO_AXIS[cut_axis])
    #     n_third = crystal.get_n(wav_nm, polarization=self.INDEX_TO_AXIS[third_axis])
        
    #     tol = 1e-3  #tolerance for np.isclose()

    #     if np.isclose(pol_deg, 0, atol=tol):
    #         n = n_rot

    #     elif np.isclose(pol_deg, 90, atol=tol):
    #         if theta_deg is None:
    #             raise FittingConfigurationError(
    #                 "n is angle dependent. Add theta_deg in the argument of def n_eff"
    #             )
    #         theta = np.deg2rad(theta_deg)
    #         n_to_2 = n_third **2 + (1 - (n_third/n_cut)**2)* (np.sin(theta))**2
    #         n = np.sqrt(n_to_2)

    #     else:
    #         raise FittingConfigurationError(
    #             f"Unexpected polarization degree: {pol_deg}. "
    #             "Supported values are 0 or 90 degree."
    #         )
        
    #     return n

    def n_eff(self, pol_deg, wav_nm, theta_deg=None):
        """Return n for a given polarization angle and crystal setting.

        Parameters
        ----------
        pol_deg : float
            Polarization angle in lab frame [deg], 0 or 90 only (for now).
        wav_nm : float
            Vacuum wavelength [nm].
        theta_deg : float or array-like, optional
            Incidence angle(s) [deg]. Required for pol_deg=90.

        Returns
        -------
        float or np.ndarray
            Effective refractive index n(wav_nm, pol_deg, theta_deg).
            If theta_deg is scalar -> float, if array-like -> np.ndarray.
            For pol_deg=0, n is angle-independent but is broadcast to match theta_deg shape if provided.
        """
        import numpy as np

        meta = self.analysis.meta
        crystal = CRYSTALS[meta["material"]]()
        axiality = crystal.axiality
        if axiality != "biaxial":
            raise FittingConfigurationError(
                f"This strategy is for biaxial crystals. Please use another strategy for {axiality} crystals."
            )

        geometry = {
            "rot_axis": meta["rot/trans_axis"],
            "cut_axis": meta["crystal_orientation"],
        }

        cut_axis = self.normalize_axis(geometry["cut_axis"])
        rot_axis = self.normalize_axis(geometry["rot_axis"])
        third_axis = self._third_axis(cut_axis, rot_axis)

        # Principal indices at the wavelength
        n_rot = crystal.get_n(wav_nm, polarization=self.INDEX_TO_AXIS[rot_axis])
        n_cut = crystal.get_n(wav_nm, polarization=self.INDEX_TO_AXIS[cut_axis])
        n_third = crystal.get_n(wav_nm, polarization=self.INDEX_TO_AXIS[third_axis])

        tol = 1e-3  # tolerance for np.isclose()

        # Decide whether we should return scalar or array
        theta_is_none = (theta_deg is None)
        theta_is_scalar = False
        theta_arr = None
        if not theta_is_none:
            theta_arr = np.asarray(theta_deg, dtype=float)
            theta_is_scalar = (theta_arr.ndim == 0)

        if np.isclose(pol_deg, 0, atol=tol):
            # Angle-independent, but broadcast to match theta shape if provided
            if theta_is_none:
                return float(n_rot)
            if theta_is_scalar:
                return float(n_rot)
            return np.full(theta_arr.shape, float(n_rot), dtype=float)

        elif np.isclose(pol_deg, 90, atol=tol):
            if theta_is_none:
                raise FittingConfigurationError(
                    "n is angle dependent. Add theta_deg in the argument of def n_eff"
                )

            theta_rad = np.deg2rad(theta_arr)
            # Elementwise computation; works for both scalar and array
            n_to_2 = (n_third ** 2) + (1.0 - (n_third / n_cut) ** 2) * (np.sin(theta_rad) ** 2)
            n = np.sqrt(n_to_2)

            if theta_is_scalar:
                return float(n)
            return np.asarray(n, dtype=float)

        else:
            raise FittingConfigurationError(
                f"Unexpected polarization degree: {pol_deg}. Supported values are 0 or 90 degree."
            )



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
        th_step = meta["step"]
        order = max(int(1.0 / th_step), 1)
        minima_idx = argrelextrema(I, np.less, order=order)[0]
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
                    n_2w_third = self.n_eff(pol_out, wl1_nm / 2.0, theta_deg=0)
                    n_2w_cut = self.n_eff(pol_out, wl1_nm / 2.0, theta_deg=90)
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