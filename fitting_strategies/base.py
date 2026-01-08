import numpy as np
from scipy.signal import argrelextrema

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
            order = max(int(1.0/step), 1.0)
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
    def __init__(self, analysis=None):
        super().__init__(analysis)

    def _position_centering(self, data):
        """
        Find the accurate position of 0 degree assuming a symmetry Maker fringe.
        """
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






# Common functions for Wedge Maker fringes
class BaseWedgeStrategy(BaseFittingStrategy):
    def __init__(self, analysis=None):
        super().__init__(analysis)

        # stage position where the laser hit the center of the sample holder 
        self.center_pos = 18.05

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


# Error handling
class FittingConfigurationError(Exception):
    """Raised when the fitting configuration (meta, pol, axis, etc.) is invalid."""
    pass
