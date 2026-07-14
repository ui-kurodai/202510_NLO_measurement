import json

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import argrelextrema

from crystaldatabase import CRYSTALS
from fitting_results import extract_fit_payload, upsert_fitting_result

class BaseFittingStrategy:
    """Abstract base class for SHG fitting algorithms."""

    def __init__(self, analysis):
        self.analysis = analysis

    def fit_all(self):
        """Run the full fitting procedure.

        Parameters
        ----------
        analysis : SHGDataAnalysis
            The analysis object containing metadata, measurement data,
            and utility methods.

        Returns
        -------
        dict
            A dictionary of fitting results.
        """
        raise NotImplementedError("Subclasses must implement fit_all()")

    def _coerce_scalar(self, value, default=float("nan")):
        """
        Convert scalar-like values or arrays to a representative float.
        """
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            try:
                return float(value)
            except Exception:
                return default

        if arr.ndim == 0:
            try:
                return float(arr)
            except Exception:
                return default

        flat = arr.reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return default
        return float(finite[0])

    def _fit_range_bounds(self, meta=None):
        if meta is None:
            meta = getattr(getattr(self, "analysis", None), "meta", None)
        raw = meta.get("fit_range") if isinstance(meta, dict) else None
        if not isinstance(raw, dict):
            return None
        try:
            lo = float(raw.get("min"))
            hi = float(raw.get("max"))
        except Exception:
            return None
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _fit_range_axis(self, data):
        if "position_centered" in data:
            return np.asarray(data["position_centered"], dtype=float)
        return np.asarray(data["position"], dtype=float)

    def _fit_range_mask(self, data, meta=None, base_mask=None, min_points=3):
        axis = self._fit_range_axis(data)
        mask = np.isfinite(axis)
        if base_mask is not None:
            mask &= np.asarray(base_mask, dtype=bool)
        bounds = self._fit_range_bounds(meta)
        if bounds is not None:
            lo, hi = bounds
            mask &= (axis >= lo) & (axis <= hi)
        if np.count_nonzero(mask) < int(min_points):
            raise ValueError("Not enough finite data points inside fit_range.")
        return mask

    def _data_for_fit_range(self, data, meta=None, min_points=3):
        mask = self._fit_range_mask(data, meta=meta, min_points=min_points)
        return data.loc[mask].copy()
    

    _UNSET = object()
    _AUTO = object()

    def _resolve_input_info(self, meta=_UNSET, data=_UNSET):
        """Resolve meta/data inputs.

        - If meta/data is not provided: keep UNSET (This func. doesn't return it too.)
        - If meta/data is "auto": fetch from self.analysis
        - Otherwise: use the provided dict as-is
        """
        if isinstance(meta, str) and meta.strip().lower() == "auto":
            meta = self._AUTO
        if isinstance(data, str) and data.strip().lower() == "auto":
            data = self._AUTO

        resolved_meta = meta
        resolved_data = data

        if meta is self._AUTO:
            resolved_meta = self.analysis.meta
        if data is self._AUTO:
            resolved_data = self.analysis.data

        if resolved_meta is self._UNSET and resolved_data is self._UNSET:
            return None
        if resolved_data is self._UNSET:
            return resolved_meta
        if resolved_meta is self._UNSET:
            return resolved_data
        return resolved_meta, resolved_data

    def _delta_n_axis_roles(self, meta):
        """
        Return axis roles for the simplified delta_n fit.

        Subclasses can narrow this to the geometry component that dominates the
        phase mismatch. The default keeps the previous all-axis behavior.
        """
        return {
            "w_axes": ("a", "b", "c"),
            "two_w_axes": ("a", "b", "c"),
            "weight": 1.0,
        }

    def _delta_n_axes(self, meta):
        roles = self._delta_n_axis_roles(meta)
        return tuple(roles.get("two_w_axes") or ("a", "b", "c"))

    def _delta_n_override(self, meta, delta_n):
        """
        Keep n(w) fixed and offset selected Sellmeier n(2w) axes by delta_n.
        """
        delta_n = float(delta_n)
        axes = self._delta_n_axes(meta)
        dn_override = {}
        for axis in ("a", "b", "c"):
            dn_override[f"dn_w_{axis}"] = 0.0
            dn_override[f"dn_2w_{axis}"] = delta_n if axis in axes else 0.0
        return dn_override

    def _resolve_frequency_tag(self, meta, wav_nm):
        """
        Map the model wavelength onto the fundamental / SH tags used by dn_override.
        """
        try:
            fundamental_nm = float(meta["wavelength_nm"])
            wav_nm = float(wav_nm)
        except Exception:
            return None

        atol = max(abs(fundamental_nm) * 1e-9, 1e-9)
        if np.isclose(wav_nm, fundamental_nm, rtol=0.0, atol=atol):
            return "w"
        if np.isclose(wav_nm, fundamental_nm / 2.0, rtol=0.0, atol=atol):
            return "2w"
        return None

    def _principal_n_with_dn_override(self, meta, wav_nm, index_to_n, dn_override=None):
        """
        Return principal refractive indices keyed by axis label after applying dn overrides.

        Supported dn_override keys are either ``dn_w_a`` style or ``w_a`` style.
        """
        crystal = CRYSTALS[meta["material"]]()
        dn_override = dn_override if isinstance(dn_override, dict) else {}
        freq_tag = self._resolve_frequency_tag(meta, wav_nm)

        principal_n = {}
        for axis_label, pol_label in index_to_n.items():
            n_value = float(crystal.get_n(wav_nm, polarization=pol_label))
            if freq_tag is not None:
                for key in (f"dn_{freq_tag}_{pol_label}", f"{freq_tag}_{pol_label}"):
                    if key in dn_override:
                        n_value += float(dn_override[key])
                        break
            principal_n[axis_label] = n_value

        return principal_n
        
    
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
    
    def _third_axis(self, cut_axis: str, rot_trans_axis: str) -> str:
        """Return the remaining principal axis label among {'100','010','001'}."""
        axes = {"100", "010", "001"}
        if cut_axis not in axes or rot_trans_axis not in axes or cut_axis == rot_trans_axis:
            raise ValueError(f"Invalid axes: cut_axis={cut_axis}, rot_trans_axis={rot_trans_axis}")
        third = list(axes - {cut_axis, rot_trans_axis})
        if len(third) != 1:
            raise ValueError("Failed to determine third axis.")
        
        return third[0]
    
    def detect_minima(self, x, y, threshold_ratio=0.02, order=None, aux=False):
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
            order = max(int(1.0 / step), 1)
        idx_min = argrelextrema(y, np.less, order=order)[0]

        # Global maximum - min
        A = np.max(y) - np.min(y)

        # Filter by intensity threshold
        valid = (y[idx_min] - np.min(y)) < threshold_ratio * A

        if aux==True:
            return idx_min[valid], idx_min
        else:
            return idx_min[valid]
    
    
    def get_rotation_matrix(cut_axis, moving_axis, method, rotation_angle_deg=0.0):
        """
        Compute rotation matrix from crystal coordinates to lab coordinates.

        Parameters
        ----------
        cut_axis : array-like of int
            Miller index of crystal surface normal, e.g., [0, 0, 1].
        moving_axis : array-like of int
            Miller index of rotation/translation axis, e.g., [0, 1, 0].
        method : str
            'rotation' or 'wedge'
        rotation_angle_deg : float
            Rotation about moving_axis in lab coords (only for 'rotation').

        Returns
        -------
        R : ndarray shape (3,3)
            Rotation matrix (crystal coords → lab coords).
        """
        # --- Step 1: normalize input vectors in crystal coords ---
        cut_c = np.array(cut_axis, dtype=float)
        cut_c /= np.linalg.norm(cut_c)
        move_c = np.array(moving_axis, dtype=float)
        move_c /= np.linalg.norm(move_c)

        # --- Step 2: build crystal basis (in crystal coords) ---
        # We'll find the 3 crystal unit vectors that correspond to lab X, Y, Z
        if method == "rotation":
            # Lab: Y_lab = cut, Z_lab = moving
            Y_c = cut_c
            Z_c = move_c
            X_c = np.cross(Y_c, Z_c)
            X_c /= np.linalg.norm(X_c)
        elif method == "wedge":
            # Lab: Y_lab = cut, X_lab = moving
            Y_c = cut_c
            X_c = move_c
            Z_c = np.cross(X_c, Y_c)
            Z_c /= np.linalg.norm(Z_c)
        else:
            raise ValueError("Unknown method")

        # --- Step 3: crystal basis in lab coords ---
        # Lab basis vectors in lab frame are just identity axes
        X_lab = np.array([1, 0, 0], dtype=float)
        Y_lab = np.array([0, 1, 0], dtype=float)
        Z_lab = np.array([0, 0, 1], dtype=float)

        # The columns of this matrix are the lab coords of the crystal's unit vectors
        if method == "rotation":
            crystal_in_lab = np.column_stack((X_lab, Y_lab, Z_lab))
            lab_in_crystal = np.column_stack((X_c, Y_c, Z_c))
        else:  # wedge
            crystal_in_lab = np.column_stack((X_lab, Y_lab, Z_lab))
            lab_in_crystal = np.column_stack((X_c, Y_c, Z_c))

        # Rotation matrix: crystal → lab
        R = crystal_in_lab @ lab_in_crystal.T

        # --- Step 4: apply sample rotation (rotation method only) ---
        if method == "rotation" and rotation_angle_deg != 0.0:
            theta = np.radians(rotation_angle_deg)
            k = Z_lab  # rotation axis in lab coords
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R_rot = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            R = R_rot @ R

        return R
    



# Common functions for Rotation Maker fringes
class BaseRotationStrategy(BaseFittingStrategy):
    UNIAXIAL_N = {"100": "o", "010": "o", "001": "e"}
    BIAXIAL_N = {"100": "a", "010": "b", "001": "c"}

    def __init__(self, analysis=None):
        super().__init__(analysis)

    def n_eff(self, pol_deg, wav_nm, theta_deg=None, meta="auto", aux=False, dn_override=None):
        """Return effective refractive index for rotation Maker fringe geometry."""
        meta = self._resolve_input_info(meta=meta)
        crystal = CRYSTALS[meta["material"]]()

        geometry = {
            "rot_axis": meta["rot/trans_axis"],
            "cut_axis": meta["crystal_orientation"],
        }

        cut_axis = self.normalize_axis(geometry["cut_axis"])
        rot_axis = self.normalize_axis(geometry["rot_axis"])
        third_axis = self._third_axis(cut_axis, rot_axis)

        axiality = crystal.axiality
        if axiality == "uniaxial":
            index_to_n = self.UNIAXIAL_N
        elif axiality == "biaxial":
            index_to_n = self.BIAXIAL_N
        else:
            raise FittingConfigurationError(
                f"Unsupported crystal axiality: {axiality}."
            )

        principal_n = self._principal_n_with_dn_override(
            meta,
            wav_nm,
            index_to_n=index_to_n,
            dn_override=dn_override,
        )
        n_rot = principal_n[rot_axis]
        n_cut = principal_n[cut_axis]
        n_third = principal_n[third_axis]

        tol = 1e-3
        theta_is_none = theta_deg is None
        theta_is_scalar = False
        theta_arr = None
        if not theta_is_none:
            theta_arr = np.asarray(theta_deg, dtype=float)
            theta_is_scalar = theta_arr.ndim == 0

        if aux:
            return {
                "n_rot": n_rot,
                "n_cut": n_cut,
                "n_third": n_third,
            }

        if np.isclose(pol_deg, 0, atol=tol):
            if theta_is_none or theta_is_scalar:
                return float(n_rot)
            return np.full(theta_arr.shape, float(n_rot), dtype=float)

        if np.isclose(pol_deg, 90, atol=tol):
            if theta_is_none:
                raise FittingConfigurationError(
                    "n is angle dependent. Add theta_deg in the argument of def n_eff"
                )

            theta_rad = np.deg2rad(theta_arr)
            n_sq = (n_third ** 2) + (1.0 - (n_third / n_cut) ** 2) * (np.sin(theta_rad) ** 2)
            n = np.sqrt(n_sq)

            if theta_is_scalar:
                return float(n)
            return np.asarray(n, dtype=float)

        raise FittingConfigurationError(
            f"Unexpected polarization degree: {pol_deg}. Supported values are 0 or 90 degree."
        )

    def _position_centering(self, data):
        """
        Find the accurate position of 0 degree assuming a symmetry Maker fringe.
        """
        saved_center = self._saved_centering_pos()
        if np.isfinite(saved_center):
            out = data.copy()
            out["position_centered"] = data["position"] - saved_center
            fit_data = {
                "c_candidates": None,
                "costs": None,
                "c0": saved_center,
                "c_local": None,
                "costs_local": None,
                "c_best": saved_center,
                "source": "saved",
            }
            return out, fit_data

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

    def _saved_centering_pos(self):
        meta = getattr(getattr(self, "analysis", None), "meta", None)
        saved_fit = extract_fit_payload(meta, self.__class__.__name__)
        saved_center = self._coerce_scalar(saved_fit.get("centering_pos"))
        if np.isfinite(saved_center):
            return saved_center
        return float("nan")






# Common functions for Wedge Maker fringes
class BaseWedgeStrategy(BaseFittingStrategy):
    UNIAXIAL_N = {"100": "o", "010": "o", "001": "e"}
    BIAXIAL_N = {"100": "a", "010": "b", "001": "c"}

    def __init__(self, analysis=None):
        super().__init__(analysis)

        # stage position where the laser hit the center of the sample holder 
        self.center_pos = 18.05

    def n_eff(self, pol_deg, wav_nm, meta="auto", aux=False, dn_override=None):
        """Return effective refractive index for wedge Maker fringe geometry."""
        meta = self._resolve_input_info(meta=meta)
        crystal = CRYSTALS[meta["material"]]()

        geometry = {
            "trans_axis": meta["rot/trans_axis"],
            "cut_axis": meta["crystal_orientation"],
        }

        cut_axis = self.normalize_axis(geometry["cut_axis"])
        trans_axis = self.normalize_axis(geometry["trans_axis"])
        third_axis = self._third_axis(cut_axis, trans_axis)

        axiality = crystal.axiality
        if axiality == "uniaxial":
            index_to_n = self.UNIAXIAL_N
        elif axiality == "biaxial":
            index_to_n = self.BIAXIAL_N
        else:
            raise FittingConfigurationError(
                f"Unsupported crystal axiality: {axiality}."
            )

        principal_n = self._principal_n_with_dn_override(
            meta,
            wav_nm,
            index_to_n=index_to_n,
            dn_override=dn_override,
        )
        n_trans = principal_n[trans_axis]
        n_cut = principal_n[cut_axis]
        n_third = principal_n[third_axis]

        tol = 1e-3

        if aux:
            return {
                "n_trans": n_trans,
                "n_cut": n_cut,
                "n_third": n_third,
            }

        if np.isclose(pol_deg, 0, atol=tol):
            return float(n_third)

        if np.isclose(pol_deg, 90, atol=tol):
            return float(n_trans)

        raise FittingConfigurationError(
            f"Unexpected polarization degree: {pol_deg}."
            "Supported values are 0 or 90 degree."
        )

    def _theory_n_at_zero(self, pol_deg, wav_nm, meta=None, dn_override=None):
        meta = self._resolve_input_info(meta=meta)
        if np.isclose(pol_deg, 45.0, atol=1e-3):
            principal = self.n_eff(pol_deg, wav_nm, meta=meta, aux=True, dn_override=dn_override)
            return 0.5 * (float(principal["n_trans"]) + float(principal["n_third"]))
        return float(self.n_eff(pol_deg, wav_nm, meta=meta, dn_override=dn_override))

    def _calc_theoretical_lc(self, meta=None, dn_override=None):
        meta = self._resolve_input_info(meta=meta)
        wl1_nm = float(meta["wavelength_nm"])
        wl1_mm = wl1_nm * 1e-6
        n_w = self._theory_n_at_zero(
            meta["input_polarization"],
            wl1_nm,
            meta=meta,
            dn_override=dn_override,
        )
        n_2w = self._theory_n_at_zero(
            meta["detected_polarization"],
            wl1_nm / 2.0,
            meta=meta,
            dn_override=dn_override,
        )
        dn = abs(float(n_w) - float(n_2w))
        lc = wl1_mm / (4.0 * dn) if np.isfinite(dn) and not np.isclose(dn, 0.0) else float("nan")
        return {"Lc_theory_mm": float(lc)}

    def calc_thickness_array(self, override: dict = {}, meta="auto", data="auto"):
        """
        Calculate crystal thickness array from metadata
        meta/data is fetched from self.analysis by default.
        Otherwise you can assign other meta/data to the argument
        """
        meta, data = self._resolve_input_info(meta=meta, data=data)
        
        if "L" in override.keys():
            t_center = override["L"]
        else:
            t_center = meta["thickness_info"]["t_center_mm"]
        if "wedge_deg" in override.keys():
            wedge_angle_deg = override["wedge_deg"]
        else:
            wedge_angle_deg = meta["thickness_info"]["wedge_angle_deg"]

        L_array = t_center + (data["position"] - self.center_pos) * np.tan(np.radians(wedge_angle_deg))
        return np.asarray(L_array, dtype=float)

    def _calc_wedge_minima_lc(self, meta, data, intensity_column="intensity_corrected"):
        x = np.asarray(data["position"], dtype=float)
        y = np.asarray(data[intensity_column], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite) < 3:
            return {}

        x_fit = x[finite]
        y_fit = y[finite]
        minima_idx = self.detect_minima(x_fit, y_fit)
        if minima_idx.size < 2:
            return {}

        minima_x = np.sort(x_fit[minima_idx])
        wedge_deg = float(meta["thickness_info"].get("wedge_angle_deg", float("nan")))
        if not np.isfinite(wedge_deg):
            return {}

        lc_values = 0.5 * np.abs(np.diff(minima_x)) * abs(np.tan(np.deg2rad(wedge_deg)))
        lc_values = lc_values[np.isfinite(lc_values) & (lc_values > 0.0)]
        if lc_values.size == 0:
            return {}

        return {
            "lc_wedge_minima_mm": float(np.mean(lc_values)),
            "lc_wedge_minima_std_mm": (
                float(np.std(lc_values, ddof=1))
                if lc_values.size >= 2
                else float("nan")
            ),
        }

    def _fit_L(self, meta="auto", data="auto"):
        """
        Fit sample thickness L [mm] near the nominal thickness for wedge scans.
        """
        meta, data = self._resolve_input_info(meta=meta, data=data)

        fit_data = self._data_for_fit_range(data, meta=meta)
        y = np.asarray(fit_data["intensity_corrected"], dtype=float)
        L0_mm = meta["thickness_info"]["t_center_mm"]

        search_um = 15
        loss = "linear"
        f_scale = 1.0

        dL_mm = search_um / 1000.0
        lb = L0_mm - dL_mm
        ub = L0_mm + dL_mm

        def residual(params):
            L_mm = float(params[0])
            y_model = self._maker_fringes(override={"L": L_mm, "meta": meta, "data": fit_data})
            y_model = np.asarray(y_model, dtype=float)
            if y_model.shape != y.shape:
                raise ValueError(f"Model returned shape {y_model.shape}, expected {y.shape}.")
            return y_model - y

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

    def fit_all(self):
        """
        Wedge fitting pipeline:
        1. fit thickness with _fit_L()
        2. use measured max intensity as amplitude
        3. write fit curve and fitting summary
        """
        meta = self.analysis.meta
        data = self.analysis.data

        fit_L, _fit_aux = self._fit_L(meta=meta, data=data)
        fitted_L_mm = float(fit_L["L_fit_mm"])

        model_result = self._maker_fringes(
            override={"L": fitted_L_mm, "meta": meta, "data": data},
            return_aux=True,
        )
        if isinstance(model_result, tuple) and len(model_result) == 2:
            model, fit_aux = model_result
        else:
            model, fit_aux = model_result, {}

        model = np.asarray(model, dtype=float)
        intensity = np.asarray(data["intensity_corrected"], dtype=float)
        if intensity.size == 0 or not np.isfinite(intensity).any():
            raise ValueError("No finite intensity data available for wedge fitting.")

        peak = float(np.nanmax(intensity))
        fit_curve = peak * model
        finite = np.isfinite(intensity) & np.isfinite(fit_curve)
        if not np.any(finite):
            raise ValueError("No finite residual points available for wedge fitting.")
        fit_eval_mask = self._fit_range_mask(data, meta=meta, base_mask=finite, min_points=1)
        residual_rms = float(np.sqrt(np.mean((intensity[fit_eval_mask] - fit_curve[fit_eval_mask]) ** 2)))

        out = data.copy()
        out["fit"] = fit_curve

        results = {
            "L_mm": fitted_L_mm,
            "L_mm_std": float("nan"),
            "k_scale": peak,
            "k_scale_std": 0.0,
            "Pm0": peak,
            "Pm0_stderr": 0.0,
            "residual_rms": residual_rms,
        }
        if isinstance(fit_aux, dict) and fit_aux.get("d_factor") is not None:
            d_factor = self._coerce_scalar(fit_aux["d_factor"])
            if np.isfinite(d_factor):
                results["d_factor"] = d_factor
        results.update(self._calc_wedge_minima_lc(meta, data))
        if hasattr(self, "_calc_theoretical_lc"):
            results.update(self._calc_theoretical_lc(meta))

        self.analysis.meta = upsert_fitting_result(
            self.analysis.meta,
            self.__class__.__name__,
            results,
            strategy_module=self.__class__.__module__,
            strategy_display_name=self.__class__.__name__,
        )

        csv_path = self.analysis.csv_path
        json_path = self.analysis.json_path
        self.analysis.data = out
        out.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis.meta, f, ensure_ascii=False, indent=2)

        return results


# Error handling
class FittingConfigurationError(Exception):
    """Raised when the fitting configuration (meta, pol, axis, etc.) is invalid."""
    pass
