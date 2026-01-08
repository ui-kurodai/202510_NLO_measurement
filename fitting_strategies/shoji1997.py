import numpy as np
from scipy.optimize import least_squares

from fitting_strategies.base import BaseWedgeStrategy
from fitting_strategies.base import FittingConfigurationError

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *

class Shoji1997Strategy(BaseWedgeStrategy):

    UNIAXIAL_N = {"100": "o", "010": "o", "001": "e"}
    BIAXIAL_N = {"100": "a", "010": "b", "001": "c"}

    """Fitting strategy based on Shoji et al., 1997."""
    def __init__(self, analysis):
        super().__init__(analysis)


    # n_w and n_2w for specific setup
    def n_eff(self, pol_deg, wav_nm, meta="auto", aux=False):
        """Return n for a given polarization angle and crystal setting.

        Parameters
        ----------
        pol_deg : float
            Polarization angle in lab frame [deg], 0 or 90 only (for now).
        wav_nm : float
            Vacuum wavelength [nm].

        Returns
        -------
        float
            Refractive index n for the designated setup (polarization, cut plane, AOI...)
        """
        meta = self._resolve_input_info(meta=meta)
        crystal = CRYSTALS[meta["material"]]()


        geometry = {
            "trans_axis": meta["rot/trans_axis"],
            "cut_axis": meta["crystal_orientation"],
        }

        cut_axis = self.normalize_axis(geometry["cut_axis"])
        trans_axis = self.normalize_axis(geometry["trans_axis"])
        third_axis = self._third_axis(cut_axis, trans_axis)

        # Principal indices at the wavelength
        axiality = crystal.axiality
        if axiality == "uniaxial":
            INDEX_TO_N = self.UNIAXIAL_N
        elif axiality =="biaxial":
            INDEX_TO_N = self.BIAXIAL_N
        else:
            INDEX_TO_N = None

        n_trans = crystal.get_n(wav_nm, polarization=INDEX_TO_N[trans_axis])
        n_cut = crystal.get_n(wav_nm, polarization=INDEX_TO_N[cut_axis])
        n_third = crystal.get_n(wav_nm, polarization=INDEX_TO_N[third_axis])

        tol = 1e-3  # tolerance for np.isclose()

        if aux == True:
            return {
                "n_trans" : n_trans,
                "n_cut" : n_cut,
                "n_third" : n_third
            }

        if np.isclose(pol_deg, 0, atol=tol):
            return float(n_third)

        elif np.isclose(pol_deg, 90, atol=tol):
            return float(n_trans)

        else:
            raise FittingConfigurationError(
                f"Unexpected polarization degree: {pol_deg}."
                "Supported values are 0 or 90 degree."
            )

    def _maker_fringe(self, mode="single_path", override: dict | None = None, return_aux=False):
        
        override = {} if override is None else override
        meta = override.get("meta", "auto")
        data = override.get("data", "auto")
        meta, data = self._resolve_input_info(meta=meta, data=data)

        wl1_nm = meta["wavelength_nm"]
        wl1_mm = wl1_nm * 1e-6
        pol_in = meta["input_polarization"] # 0-90 deg
        pol_out = meta["detected_polarization"] # 0-90 deg
        crystal = CRYSTALS[meta["material"]]()

        beam_r_x = meta["beam_r_x"]
        beam_r_y = meta["beam_r_y"]

        L = override.get("L", meta["thickness_info"]["t_center_mm"])
        
        if "n" in override.keys():
            n_w = self.n_eff(pol_in, wl1_nm)
            n_2w = self.n_eff(pol_out, wl1_nm / 2.0)
        else:
            n_w = self.n_eff(pol_in, wl1_nm)
            n_2w = self.n_eff(pol_out, wl1_nm / 2.0)

        L_array = self.calc_thickness_array(override=override, meta=meta, data=data)

        # eq. (3) in Shoji (1997) divided by d**2 (Unnecessary constants are removed)
        K = 1.0 / (n_w**2 * n_2w * wl1_nm**2)
        delta_k = 4.0 * np.pi * np.abs(n_w - n_2w)/ wl1_mm
        Psi = delta_k * L_array / 2.0

        # P_2w = K * L_array**2 * np.sin(Psi)**2 / (beam_r_x*beam_r_y * (Psi)**2)
        P_N = np.sin(Psi)**2

        aux = {"L_array": L_array}

        if return_aux == False:
            return P_N
        
        else:
            return P_N, aux
        

    def _fit_L(self, meta="auto", data="auto"):
        """
        Fit sample thickness L [mm] near the micrometer value using least squares.

        Requirements:
        - meta must contain a nominal thickness L0 in mm (see below).
        - the class must provide a model function that returns y_model given (x, L_mm, meta, data).

        Returns
        -------
        fit : dict
            {
                "L0_mm": float,
                "L_fit_mm": float,
                "dL_um": float,
                "cost": float,
                "success": bool,
                "message": str,
            }
        aux : dict
            {
                "x": np.ndarray,
                "y": np.ndarray,
                "y_fit": np.ndarray,
                "result": scipy.optimize.OptimizeResult,
            }
        """
        meta, data = self._resolve_input_info(meta=meta, data=data)

        x = np.asarray(data["position"], dtype=float)
        y = np.asarray(data["intensity_corrected"], dtype=float)
        L0_mm = meta["thickness_info"]["t_center_mm"]


        search_um = 15   # +/- range around L0
        loss = "linear"         # 'linear', 'soft_l1', 'huber', ...
        f_scale = 1.0

        dL_mm = search_um / 1000.0
        lb = L0_mm - dL_mm
        ub = L0_mm + dL_mm

        # -----------------------------
        # Residual for least squares
        # -----------------------------
        def residual(params):
            L_mm = float(params[0])
            y_model = self._maker_fringe(override={"L": L_mm})
            y_model = np.asarray(y_model, dtype=float)
            if y_model.shape != y.shape:
                raise ValueError(f"Model returned shape {y_model.shape}, expected {y.shape}.")
            return (y_model - y)

        # -----------------------------
        # Fit
        # -----------------------------
        x0 = np.array([L0_mm], dtype=float)
        result = least_squares(
            residual,
            x0=x0,
            bounds=(np.array([lb]), np.array([ub])),
            loss=loss,
            f_scale=f_scale,
        )

        L_fit_mm = float(result.x[0])

        fit = {
            "L0_mm": L0_mm,
            "L_fit_mm": L_fit_mm,
            "dL_um": (L_fit_mm - L0_mm) * 1000.0,
            "cost": float(result.cost),
            "success": bool(result.success),
            "message": str(result.message),
        }
        aux = {
            "result": result,
        }
        return fit, aux



            