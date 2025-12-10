import numpy as np

class SHGFittingStrategy:
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


# Error handling
class FittingConfigurationError(Exception):
    """Raised when the fitting configuration (meta, pol, axis, etc.) is invalid."""
    pass
