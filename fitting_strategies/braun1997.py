import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from fitting_strategies.base import FittingConfigurationError
from fitting_strategies.bechthold1977 import GlobalNFitMixin
from fitting_strategies.jerphagnon1970 import Jerphagnon1970Strategy
from fitting_results import upsert_fitting_result

from crystaldatabase import CRYSTALS


@dataclass(frozen=True)
class _ModeSet:
    """Four Braun/Yeh plane-wave modes in one layer."""

    a: float
    g: np.ndarray
    p: np.ndarray
    h: np.ndarray
    labels: tuple[str, str, str, str] = ("s+", "s-", "p+", "p-")


class Braun1997Strategy(Jerphagnon1970Strategy):
    """
    Braun et al. 1997 Maker-fringe model for biaxial crystals.

    This first implementation covers the geometry used by the paper and by the
    existing rotation strategies: incidence in a principal plane with
    ``z = cut_axis``, ``y = rot/trans_axis`` (s polarization), and ``x`` the
    remaining principal axis. The calculation uses a 4x4 dynamic-matrix solve
    for the fundamental boundary conditions, then evaluates a Braun-style
    free/bound-wave phase term. A finite-beam ray-trace correction can be
    enabled with ``override={"use_raytrace": True}``.
    """

    BIAXIAL_N = {"100": "a", "010": "b", "001": "c"}
    UNIAXIAL_N = {"100": "o", "010": "o", "001": "e"}
    DEFAULT_MAX_REFLECTIONS = 20
    DEFAULT_PARALLEL = False
    DEFAULT_PARALLEL_THRESHOLD = 64
    LIVE_UPDATE_ON_SLIDER = False
    INTENSITY_SCALE_PARAMETER = "d_rel_abs"
    GEOMETRY_D_COMPONENTS = {
        ("BaMgF4", "010", "100", 0, 90): "d_31",
        ("BaMgF4", "100", "010", 0, 90): "d_32",
        ("BaMgF4", "100", "001", 0, 0): "d_33",
        ("BaMgF4", "010", "001", 0, 0): "d_33",
        ("BaMgF4", "010", "100", 45, 0): "d_15",
        ("BaMgF4", "100", "010", 45, 0): "d_24",
    }

    def _braun_axes(self, meta):
        cut_axis = self.normalize_axis(meta["crystal_orientation"])
        rot_axis = self.normalize_axis(meta["rot/trans_axis"])
        x_axis = self._third_axis(cut_axis, rot_axis)
        return {
            "x": x_axis,
            "y": rot_axis,
            "z": cut_axis,
        }

    def _principal_indices(self, meta, wav_nm, dn_override=None):
        crystal = CRYSTALS[meta["material"]]()
        if crystal.axiality == "biaxial":
            index_to_n = self.BIAXIAL_N
        elif crystal.axiality == "uniaxial":
            index_to_n = self.UNIAXIAL_N
        else:
            raise FittingConfigurationError(
                f"Unsupported crystal axiality: {crystal.axiality}."
            )
        return self._principal_n_with_dn_override(
            meta,
            wav_nm,
            index_to_n=index_to_n,
            dn_override=dn_override,
        )

    def _braun_n_xyz(self, meta, wav_nm, dn_override=None):
        axes = self._braun_axes(meta)
        n_by_axis = self._principal_indices(meta, wav_nm, dn_override=dn_override)
        return np.array(
            [n_by_axis[axes["x"]], n_by_axis[axes["y"]], n_by_axis[axes["z"]]],
            dtype=float,
        )

    def _incident_amplitudes(self, pol_deg):
        pol_rad = np.deg2rad(float(pol_deg))
        return float(np.cos(pol_rad)), float(np.sin(pol_rad))

    def _geometry_key(self, meta):
        orientation = meta["crystal_orientation"]
        if isinstance(orientation, list):
            orientation = "".join(map(str, orientation))
        return (
            meta["material"],
            self.normalize_axis(orientation),
            self.normalize_axis(meta["rot/trans_axis"]),
            int(round(float(meta["input_polarization"]))),
            int(round(float(meta["detected_polarization"]))),
        )

    def _default_d_component(self, meta):
        return self.GEOMETRY_D_COMPONENTS.get(self._geometry_key(meta))

    def _delta_n_axis_roles(self, meta):
        cut_axis = self.normalize_axis(meta["crystal_orientation"])
        rot_axis = self.normalize_axis(meta["rot/trans_axis"])
        third_axis = self._third_axis(cut_axis, rot_axis)
        axis_label = self.BIAXIAL_N
        d_component = str(self._default_d_component(meta) or meta.get("d_component") or "")
        if d_component.startswith("d") and not d_component.startswith("d_") and len(d_component) >= 3:
            d_component = "d_" + d_component[1:]

        if d_component in ("d_31", "d_32"):
            return {
                "w_axes": (axis_label[rot_axis],),
                "two_w_axes": (axis_label[third_axis],),
                "weight": 2.0,
            }
        if d_component == "d_33":
            return {
                "w_axes": (axis_label[rot_axis],),
                "two_w_axes": (axis_label[rot_axis],),
                "weight": 4.0,
            }
        if d_component in ("d_15", "d_24"):
            return {
                "w_axes": (axis_label[third_axis], axis_label[rot_axis]),
                "two_w_axes": (axis_label[rot_axis],),
                "weight": 1.0,
            }
        return super()._delta_n_axis_roles(meta)

    def _model_meta(self, meta, override):
        model_meta = dict(meta)
        if (
            "d_component" not in model_meta
            and "d_component" not in override
            and "d_matrix" not in model_meta
            and "d_matrix" not in override
            and "d_values" not in model_meta
            and "d_values" not in override
        ):
            d_component = self._default_d_component(model_meta)
            if d_component is not None:
                model_meta["d_component"] = d_component
        return model_meta

    def _axis_to_index(self, axis):
        return {"100": 0, "010": 1, "001": 2}[self.normalize_axis(axis)]

    def _lab_field_to_crystal(self, field_lab, meta):
        axes = self._braun_axes(meta)
        field_lab = np.asarray(field_lab, dtype=complex)
        field_crystal = np.zeros(3, dtype=complex)
        for lab_index, lab_axis in enumerate(("x", "y", "z")):
            field_crystal[self._axis_to_index(axes[lab_axis])] = field_lab[lab_index]
        return field_crystal

    def _crystal_field_to_lab(self, field_crystal, meta):
        axes = self._braun_axes(meta)
        field_crystal = np.asarray(field_crystal, dtype=complex)
        field_lab = np.zeros(3, dtype=complex)
        for lab_index, lab_axis in enumerate(("x", "y", "z")):
            field_lab[lab_index] = field_crystal[self._axis_to_index(axes[lab_axis])]
        return field_lab

    def _numeric_d_matrix(self, meta, override):
        if "_numeric_d_matrix" in override:
            return override["_numeric_d_matrix"]

        d_matrix = override.get("d_matrix", meta.get("d_matrix"))
        d_component = override.get("d_component", meta.get("d_component", self._default_d_component(meta)))
        d_values = override.get("d_values", meta.get("d_values", {}))

        if d_matrix is None and (d_component is not None or d_values):
            crystal = CRYSTALS[meta["material"]]()
            if not hasattr(crystal, "d_matrix"):
                return None
            d_matrix = crystal.d_matrix(kleinmann=False)

        if d_matrix is None:
            return None

        arr = np.asarray(d_matrix, dtype=object)
        if arr.shape != (3, 6):
            raise ValueError("d_matrix must have shape (3, 6).")

        substitutions = {}
        for value in arr.reshape(-1):
            for symbol in getattr(value, "free_symbols", set()):
                substitutions[symbol] = 0.0

        if d_component is not None:
            component_name = str(d_component).strip()
            for symbol in list(substitutions.keys()):
                if str(symbol) == component_name:
                    substitutions[symbol] = 1.0

        if isinstance(d_values, dict):
            for symbol in list(substitutions.keys()):
                for key, value in d_values.items():
                    if str(symbol) == str(key):
                        substitutions[symbol] = float(value)

        numeric = np.zeros(arr.shape, dtype=complex)
        for index, value in np.ndenumerate(arr):
            if hasattr(value, "subs"):
                value = value.subs(substitutions)
            try:
                numeric[index] = complex(value)
            except TypeError as exc:
                raise ValueError(
                    "d_matrix contains symbolic entries. Pass d_component='d_31' "
                    "or d_values={'d_31': value, ...}."
                ) from exc
        return numeric

    def _safe_sqrt(self, value):
        value = complex(value)
        root = np.sqrt(value)
        if abs(root.imag) < 1e-12:
            return float(root.real)
        return root

    def _normalize_vector(self, vector):
        vector = np.asarray(vector, dtype=complex)
        norm = np.sqrt(np.sum(np.abs(vector) ** 2))
        if norm == 0:
            return vector
        out = vector / norm
        if np.all(np.abs(out.imag) < 1e-12):
            return out.real.astype(float)
        return out

    def _layer_modes(self, theta_rad, n_xyz):
        """
        Return the four modes ordered as transmitted/reflected s and p waves.

        ``a`` and ``g`` are dimensionless wave-vector components normalized by
        omega/c, matching Braun's Eqs. (6)-(10).
        """
        nx, ny, nz = [float(x) for x in n_xyz]
        a = float(np.sin(theta_rad))

        g_s = self._safe_sqrt(ny**2 - a**2)
        g_p = self._safe_sqrt((nx**2 / nz**2) * (nz**2 - a**2))
        g = np.array([g_s, -g_s, g_p, -g_p], dtype=complex)

        p = np.zeros((3, 4), dtype=complex)
        p[:, 0] = (0.0, 1.0, 0.0)
        p[:, 1] = (0.0, 1.0, 0.0)

        for index in (2, 3):
            gi = g[index]
            # Braun 1997 Eq. (9b), up to an irrelevant overall sign.
            p[:, index] = self._normalize_vector([nz**2 - a**2, 0.0, -a * gi])

        h = np.zeros((3, 4), dtype=complex)
        for index, gi in enumerate(g):
            k = np.array([a, 0.0, gi], dtype=complex)
            h[:, index] = np.cross(k, p[:, index])

        return _ModeSet(a=a, g=g, p=p, h=h)

    def _dynamic_matrix(self, modes):
        d = np.zeros((4, 4), dtype=complex)
        d[0, :] = modes.p[0, :]
        d[1, :] = modes.h[1, :]
        d[2, :] = modes.p[1, :]
        d[3, :] = modes.h[0, :]
        return d

    def _propagation_matrix(self, modes, thickness_mm, wav_nm):
        wav_mm = float(wav_nm) * 1e-6
        phase = 2.0 * np.pi * np.asarray(modes.g, dtype=complex) * float(thickness_mm) / wav_mm
        return np.diag(np.exp(1j * phase))

    def _solve_linear_slab(self, theta_rad, wav_nm, thickness_mm, n_xyz, pol_deg):
        """
        Solve the fundamental 4x4 boundary problem for air | crystal | air.
        """
        air = np.ones(3, dtype=float)
        modes_0 = self._layer_modes(theta_rad, air)
        modes_1 = self._layer_modes(theta_rad, n_xyz)
        modes_2 = self._layer_modes(theta_rad, air)
        d0 = self._dynamic_matrix(modes_0)
        d1 = self._dynamic_matrix(modes_1)
        d2 = self._dynamic_matrix(modes_2)
        p1 = self._propagation_matrix(modes_1, thickness_mm, wav_nm)

        inc_s, inc_p = self._incident_amplitudes(pol_deg)
        q0_fixed = np.array([inc_s, 0.0, inc_p, 0.0], dtype=complex)

        # Unknowns are r_s, r_p, four crystal amplitudes, t_s, t_p.
        mat = np.zeros((8, 8), dtype=complex)
        rhs = np.zeros(8, dtype=complex)

        r_s_col = d0[:, 1]
        r_p_col = d0[:, 3]
        t_s_col = d2[:, 0]
        t_p_col = d2[:, 2]

        mat[:4, 0] = r_s_col
        mat[:4, 1] = r_p_col
        mat[:4, 2:6] = -d1
        rhs[:4] = -d0 @ q0_fixed

        mat[4:, 2:6] = d1 @ p1
        mat[4:, 6] = -t_s_col
        mat[4:, 7] = -t_p_col

        try:
            solution = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(mat, rhs, rcond=None)[0]

        q0 = q0_fixed.copy()
        q0[1] = solution[0]
        q0[3] = solution[1]
        q1 = solution[2:6]
        q2 = np.array([solution[6], 0.0, solution[7], 0.0], dtype=complex)
        return {
            "modes_air_in": modes_0,
            "modes_crystal": modes_1,
            "modes_air_out": modes_2,
            "Q_air_in": q0,
            "Q_crystal": q1,
            "Q_air_out": q2,
            "D_crystal": d1,
        }

    def _mode_weights(self, q_crystal):
        q_crystal = np.asarray(q_crystal, dtype=complex)
        weights = np.abs(q_crystal[[0, 2]]) ** 2
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            return np.array([1.0, 0.0], dtype=float)
        return np.asarray(weights / total, dtype=float)

    def _selected_g(self, modes, pol_deg):
        if np.isclose(float(pol_deg), 0.0, atol=1e-3):
            return modes.g[0]
        if np.isclose(float(pol_deg), 90.0, atol=1e-3):
            return modes.g[2]
        weights = self._incident_amplitudes(pol_deg)
        w_s = abs(weights[0]) ** 2
        w_p = abs(weights[1]) ** 2
        denom = w_s + w_p
        if denom <= 0:
            return modes.g[0]
        return (w_s * modes.g[0] + w_p * modes.g[2]) / denom

    def _mode_poynting_vector(self, modes, index):
        e_vec = np.asarray(modes.p[:, index], dtype=complex)
        h_vec = np.asarray(modes.h[:, index], dtype=complex)
        return np.real(np.cross(e_vec, np.conj(h_vec)))

    def _selected_poynting_slope(self, modes, pol_deg):
        if np.isclose(float(pol_deg), 0.0, atol=1e-3):
            s_vec = self._mode_poynting_vector(modes, 0)
        elif np.isclose(float(pol_deg), 90.0, atol=1e-3):
            s_vec = self._mode_poynting_vector(modes, 2)
        else:
            amp_s, amp_p = self._incident_amplitudes(pol_deg)
            w_s = abs(amp_s) ** 2
            w_p = abs(amp_p) ** 2
            denom = w_s + w_p
            if denom <= 0:
                s_vec = self._mode_poynting_vector(modes, 0)
            else:
                s_vec = (
                    w_s * self._mode_poynting_vector(modes, 0)
                    + w_p * self._mode_poynting_vector(modes, 2)
                ) / denom

        if np.isclose(s_vec[2], 0.0):
            return np.nan
        return float(s_vec[0] / s_vec[2])

    def _raytrace_lateral_slope(self, modes, theta_rad, pol_deg, beam_direction):
        beam_direction = str(beam_direction or "k").strip().lower()
        if beam_direction in {"poynting", "s", "energy"}:
            return self._selected_poynting_slope(modes, pol_deg)

        g_eff = self._selected_g(modes, pol_deg)
        if np.isclose(g_eff, 0.0):
            return np.nan
        return float(np.real(np.sin(theta_rad) / g_eff))

    def _detected_transmission_weight(self, linear_solution, pol_out):
        q_out = np.asarray(linear_solution["Q_air_out"], dtype=complex)
        if np.isclose(float(pol_out), 0.0, atol=1e-3):
            return float(np.abs(q_out[0]) ** 2)
        if np.isclose(float(pol_out), 90.0, atol=1e-3):
            return float(np.abs(q_out[2]) ** 2)
        amp_s, amp_p = self._incident_amplitudes(pol_out)
        return float(np.abs(amp_s * q_out[0] + amp_p * q_out[2]) ** 2)

    def _solve_front_interface(self, theta_rad, n_xyz, q_air_forward, q_crystal_backward):
        """Front interface step used by the Braun partial-beam ray tracer."""
        air_modes = self._layer_modes(theta_rad, np.ones(3, dtype=float))
        crystal_modes = self._layer_modes(theta_rad, n_xyz)
        d_air = self._dynamic_matrix(air_modes)
        d_crystal = self._dynamic_matrix(crystal_modes)

        q_air_forward = np.asarray(q_air_forward, dtype=complex)
        q_crystal_backward = np.asarray(q_crystal_backward, dtype=complex)

        mat = np.column_stack(
            (
                d_air[:, 1],
                d_air[:, 3],
                -d_crystal[:, 0],
                -d_crystal[:, 2],
            )
        )
        rhs = -(d_air @ q_air_forward) + (d_crystal @ q_crystal_backward)
        try:
            r_s, r_p, t_s, t_p = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            r_s, r_p, t_s, t_p = np.linalg.lstsq(mat, rhs, rcond=None)[0]

        q_air = q_air_forward.copy()
        q_air[1] = r_s
        q_air[3] = r_p

        q_crystal = q_crystal_backward.copy()
        q_crystal[0] = t_s
        q_crystal[2] = t_p
        return q_air, q_crystal, crystal_modes

    def _solve_back_interface(self, theta_rad, n_xyz, q_crystal_forward):
        """Back interface step used by the Braun partial-beam ray tracer."""
        air_modes = self._layer_modes(theta_rad, np.ones(3, dtype=float))
        crystal_modes = self._layer_modes(theta_rad, n_xyz)
        d_air = self._dynamic_matrix(air_modes)
        d_crystal = self._dynamic_matrix(crystal_modes)

        q_crystal_forward = np.asarray(q_crystal_forward, dtype=complex)
        q_air_backward = np.zeros(4, dtype=complex)

        mat = np.column_stack(
            (
                d_crystal[:, 1],
                d_crystal[:, 3],
                -d_air[:, 0],
                -d_air[:, 2],
            )
        )
        rhs = -(d_crystal @ q_crystal_forward) + (d_air @ q_air_backward)
        try:
            rb_s, rb_p, tt_s, tt_p = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            rb_s, rb_p, tt_s, tt_p = np.linalg.lstsq(mat, rhs, rcond=None)[0]

        q_crystal = q_crystal_forward.copy()
        q_crystal[1] = rb_s
        q_crystal[3] = rb_p

        q_air = q_air_backward.copy()
        q_air[0] = tt_s
        q_air[2] = tt_p
        return q_air, q_crystal, crystal_modes

    def _detected_air_intensity(self, q_air, pol_deg):
        if np.isclose(float(pol_deg), 0.0, atol=1e-3):
            return float(np.abs(q_air[0]) ** 2)
        if np.isclose(float(pol_deg), 90.0, atol=1e-3):
            return float(np.abs(q_air[2]) ** 2)
        amp_s, amp_p = self._incident_amplitudes(pol_deg)
        return float(np.abs(amp_s * q_air[0] + amp_p * q_air[2]) ** 2)

    def _forward_crystal_intensity(self, q_crystal):
        q_crystal = np.asarray(q_crystal, dtype=complex)
        return float(np.abs(q_crystal[0]) ** 2 + np.abs(q_crystal[2]) ** 2)

    def _scaled_linear_solution(self, linear_solution, field_scale):
        out = dict(linear_solution)
        out["Q_crystal"] = np.asarray(linear_solution["Q_crystal"], dtype=complex) * complex(field_scale)
        return out

    def _source_strength(self, linear_w, meta, override):
        """
        Nonlinear source proxy from the fundamental field in the crystal.

        If a 3x6 contracted tensor is supplied as ``override["d_matrix"]`` or
        ``meta["d_matrix"]``, it is used. Otherwise the field component selected
        by the analyzer is used; this preserves the fringe phase/envelope shape
        while leaving absolute d extraction to a later refinement.
        """
        modes = linear_w["modes_crystal"]
        q = np.asarray(linear_w["Q_crystal"], dtype=complex)
        e_field = modes.p @ q

        d_matrix = self._numeric_d_matrix(meta, override)
        if d_matrix is not None:
            e_crystal = self._lab_field_to_crystal(e_field, meta)
            ex, ey, ez = e_crystal
            quadratic = np.array(
                [ex * ex, ey * ey, ez * ez, 2 * ey * ez, 2 * ex * ez, 2 * ex * ey],
                dtype=complex,
            )
            p2_crystal = d_matrix @ quadratic
            p2_lab = self._crystal_field_to_lab(p2_crystal, meta)
            pol_out = float(meta["detected_polarization"])
            if np.isclose(pol_out, 0.0, atol=1e-3):
                return float(np.abs(p2_lab[1]) ** 2)
            if np.isclose(pol_out, 90.0, atol=1e-3):
                return float(np.abs(p2_lab[0]) ** 2 + np.abs(p2_lab[2]) ** 2)
            amp_s, amp_p = self._incident_amplitudes(pol_out)
            projected = amp_p * p2_lab[0] + amp_s * p2_lab[1] + amp_p * p2_lab[2]
            return float(np.abs(projected) ** 2)

        # Without tensor values we keep only the angle-dependent field build-up.
        # This makes the strategy useful for phase/envelope fitting while the
        # absolute d-tensor extraction remains opt-in via d_matrix.
        field_intensity = float(np.sum(np.abs(e_field) ** 2))
        return field_intensity**2

    def _wave_operator(self, a, g, n_xyz):
        """Dimensionless wave-equation operator used for Braun Eq. (17)."""
        q = np.array([a, 0.0, g], dtype=complex)
        eps = np.diag(np.asarray(n_xyz, dtype=complex) ** 2)
        return eps - np.dot(q, q) * np.eye(3, dtype=complex) + np.outer(q, q)

    def _nonlinear_p_lab(self, e_lab, meta, override, e_lab_2=None):
        d_matrix = self._numeric_d_matrix(meta, override)
        e_crystal = self._lab_field_to_crystal(e_lab, meta)
        if e_lab_2 is None:
            ex, ey, ez = e_crystal
            quadratic = np.array(
                [ex * ex, ey * ey, ez * ez, 2 * ey * ez, 2 * ex * ez, 2 * ex * ey],
                dtype=complex,
            )
        else:
            e_crystal_2 = self._lab_field_to_crystal(e_lab_2, meta)
            ex1, ey1, ez1 = e_crystal
            ex2, ey2, ez2 = e_crystal_2
            quadratic = np.array(
                [
                    2 * ex1 * ex2,
                    2 * ey1 * ey2,
                    2 * ez1 * ez2,
                    2 * (ey1 * ez2 + ey2 * ez1),
                    2 * (ex1 * ez2 + ex2 * ez1),
                    2 * (ex1 * ey2 + ex2 * ey1),
                ],
                dtype=complex,
            )

        if d_matrix is None:
            # Fallback for geometry/phase checks when no tensor is supplied.
            p2_crystal = np.array(quadratic[:3], dtype=complex)
        else:
            p2_crystal = d_matrix @ quadratic
        return self._crystal_field_to_lab(p2_crystal, meta)

    def _tangential_vector(self, e_vec, a, g):
        e_vec = np.asarray(e_vec, dtype=complex)
        h_vec = np.cross(np.array([a, 0.0, g], dtype=complex), e_vec)
        return np.array([e_vec[0], h_vec[1], e_vec[1], h_vec[0]], dtype=complex)

    def _bound_wave_terms(self, linear_w, meta, override, n_2w):
        """
        Build the four inhomogeneous bound waves from Braun Eqs. (16)-(19).

        Each fundamental mode inside the crystal produces one bound wave with
        k_b = 2 k_w. In dimensionless 2w units this has the same (a, g) as the
        fundamental mode.
        """
        modes_w = linear_w["modes_crystal"]
        q_w = np.asarray(linear_w["Q_crystal"], dtype=complex)
        a_b = modes_w.a

        mode = str(override.get("bound_wave_mode", meta.get("bound_wave_mode", "auto"))).lower()
        if mode == "auto":
            mode = "ten" if not (
                np.isclose(float(meta["input_polarization"]), 0.0, atol=1e-3)
                or np.isclose(float(meta["input_polarization"]), 90.0, atol=1e-3)
            ) else "four"
        if mode not in {"four", "ten"}:
            raise ValueError("bound_wave_mode must be 'auto', 'four', or 'ten'.")

        pairs = [(index, index) for index in range(4)]
        if mode == "ten":
            pairs.extend((i, j) for i in range(4) for j in range(i + 1, 4))

        e_bound = []
        g_bound = []
        for i, j in pairs:
            e_i = modes_w.p[:, i] * q_w[i]
            if i == j:
                p_nl = self._nonlinear_p_lab(e_i, meta, override)
                g_b = modes_w.g[i]
            else:
                e_j = modes_w.p[:, j] * q_w[j]
                p_nl = self._nonlinear_p_lab(e_i, meta, override, e_lab_2=e_j)
                g_b = 0.5 * (modes_w.g[i] + modes_w.g[j])
            operator = self._wave_operator(a_b, g_b, n_2w)
            try:
                e_b = np.linalg.solve(operator, p_nl)
            except np.linalg.LinAlgError:
                e_b = np.linalg.lstsq(operator, p_nl, rcond=None)[0]
            e_bound.append(e_b)
            g_bound.append(g_b)

        return {
            "a": a_b,
            "g": np.asarray(g_bound, dtype=complex),
            "e": np.asarray(e_bound, dtype=complex).T,
        }

    def _bound_boundary_vector(self, bound, thickness_mm, wav_nm, at_back):
        wav_mm = float(wav_nm) * 1e-6
        total = np.zeros(4, dtype=complex)
        for index, g_b in enumerate(bound["g"]):
            phase = 1.0
            if at_back:
                phase = np.exp(1j * 2.0 * np.pi * g_b * float(thickness_mm) / wav_mm)
            total += self._tangential_vector(bound["e"][:, index], bound["a"], g_b) * phase
        return total

    def _solve_nonlinear_shg(self, linear_w, theta_rad, wl2_nm, thickness_mm, n_2w, meta, override):
        """
        Solve Braun Eq. (21) for free SHG waves with known bound waves.

        The unknowns are the reflected SHG amplitudes in layer 0, the four free
        waves inside the crystal, and the transmitted SHG amplitudes in layer 2.
        No incoming SHG is imposed in the outer air layers.
        """
        air_modes = self._layer_modes(theta_rad, np.ones(3, dtype=float))
        free_modes = self._layer_modes(theta_rad, n_2w)
        d_air = self._dynamic_matrix(air_modes)
        d_free = self._dynamic_matrix(free_modes)
        p_free = self._propagation_matrix(free_modes, thickness_mm, wl2_nm)

        bound = self._bound_wave_terms(linear_w, meta, override, n_2w)
        b_front = self._bound_boundary_vector(bound, thickness_mm, wl2_nm, at_back=False)
        b_back = self._bound_boundary_vector(bound, thickness_mm, wl2_nm, at_back=True)

        mat = np.zeros((8, 8), dtype=complex)
        rhs = np.zeros(8, dtype=complex)

        # Front boundary: D_air Q(0) = D_free Q_f + D_bound Q_b.
        mat[:4, 0:2] = d_air[:, [1, 3]]
        mat[:4, 2:6] = -d_free
        rhs[:4] = b_front

        # Back boundary: D_air Q(2) = D_free P_free Q_f + D_bound P_bound Q_b.
        mat[4:, 2:6] = -(d_free @ p_free)
        mat[4:, 6:8] = d_air[:, [0, 2]]
        rhs[4:] = b_back

        try:
            solution = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(mat, rhs, rcond=None)[0]

        q_left = np.array([0.0, solution[0], 0.0, solution[1]], dtype=complex)
        q_free = solution[2:6]
        q_right = np.array([solution[6], 0.0, solution[7], 0.0], dtype=complex)

        pol_out = float(meta["detected_polarization"])
        intensity = self._detected_air_intensity(q_right, pol_out)
        return {
            "intensity": intensity,
            "Q_left": q_left,
            "Q_free": q_free,
            "Q_right": q_right,
            "bound": bound,
            "bound_front_norm": float(np.linalg.norm(b_front)),
            "bound_back_norm": float(np.linalg.norm(b_back)),
        }

    def _raytrace_correction(self, theta_rad, wav_nm, thickness_mm, n_xyz, pol_deg, beam_radius_mm, reference_internal_intensity, max_reflections, beam_direction="k"):
        """
        Partial-beam ray tracer after Braun 1997 Section 4.

        The incident Gaussian beam is discretized into partial beams. Each front
        interface step combines the local incident beam with the internally
        reflected beam generated by a preceding partial beam; each back
        interface step returns a new internal reflection to a later partial beam.
        The partial-beam width is exactly Braun's Eq. (24), so the internally
        reflected beam from one partial beam overlaps the neighboring partial
        beam at the front surface.
        """
        if beam_radius_mm <= 0.0:
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": np.nan,
                "bin_width": np.nan,
                "lateral_slope": np.nan,
            }

        modes = self._layer_modes(theta_rad, n_xyz)
        lateral_slope = self._raytrace_lateral_slope(modes, theta_rad, pol_deg, beam_direction)
        if not np.isfinite(lateral_slope):
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": np.nan,
                "bin_width": np.nan,
                "lateral_slope": np.nan,
            }

        wav_mm = float(wav_nm) * 1e-6
        b = abs(2.0 * lateral_slope * float(thickness_mm) * np.cos(theta_rad))
        if not np.isfinite(float(np.real(b))):
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": np.nan,
                "bin_width": np.nan,
                "lateral_slope": lateral_slope,
            }
        b = float(np.real(b))

        span = 8.0 * float(beam_radius_mm)
        if b <= 0.0 or b < span * 1e-9:
            # Complete-overlap limit. A self-consistent field treatment is the
            # correct limiting case; the already solved 4x4 slab model supplies
            # that field, so no finite-beam correction is needed here.
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": np.nan,
                "bin_width": b,
                "lateral_slope": lateral_slope,
            }

        bin_width = b
        n_bins = max(int(np.ceil(span / bin_width)), 1)
        shift_bins = 1

        centers = (np.arange(n_bins, dtype=float) - 0.5 * (n_bins - 1)) * bin_width
        profile = np.exp(-0.5 * (centers / max(float(beam_radius_mm), 1e-12)) ** 2)
        profile_sum = float(np.sum(profile))
        if profile_sum <= 0.0 or not np.isfinite(profile_sum):
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": np.nan,
                "bin_width": bin_width,
                "lateral_slope": lateral_slope,
            }
        profile = profile / profile_sum

        inc_s, inc_p = self._incident_amplitudes(pol_deg)
        k0d = 2.0 * np.pi * float(thickness_mm) / wav_mm
        forward_phase = np.exp(1j * np.asarray(modes.g, dtype=complex) * k0d)
        backward_return_phase = np.array(
            [0.0, np.exp(1j * modes.g[1] * k0d), 0.0, np.exp(1j * modes.g[3] * k0d)],
            dtype=complex,
        )

        air_modes = self._layer_modes(theta_rad, np.ones(3, dtype=float))
        d_air = self._dynamic_matrix(air_modes)
        d_crystal = self._dynamic_matrix(modes)
        front_mat = np.column_stack(
            (
                d_air[:, 1],
                d_air[:, 3],
                -d_crystal[:, 0],
                -d_crystal[:, 2],
            )
        )
        back_mat = np.column_stack(
            (
                d_crystal[:, 1],
                d_crystal[:, 3],
                -d_air[:, 0],
                -d_air[:, 2],
            )
        )
        q_air_backward = np.zeros(4, dtype=complex)

        def solve_front(q_air_forward, q_crystal_backward):
            rhs = -(d_air @ q_air_forward) + (d_crystal @ q_crystal_backward)
            try:
                r_s, r_p, t_s, t_p = np.linalg.solve(front_mat, rhs)
            except np.linalg.LinAlgError:
                r_s, r_p, t_s, t_p = np.linalg.lstsq(front_mat, rhs, rcond=None)[0]

            q_air = q_air_forward.copy()
            q_air[1] = r_s
            q_air[3] = r_p

            q_crystal = q_crystal_backward.copy()
            q_crystal[0] = t_s
            q_crystal[2] = t_p
            return q_air, q_crystal

        def solve_back(q_crystal_forward):
            rhs = -(d_crystal @ q_crystal_forward) + (d_air @ q_air_backward)
            try:
                rb_s, rb_p, tt_s, tt_p = np.linalg.solve(back_mat, rhs)
            except np.linalg.LinAlgError:
                rb_s, rb_p, tt_s, tt_p = np.linalg.lstsq(back_mat, rhs, rcond=None)[0]

            q_crystal = q_crystal_forward.copy()
            q_crystal[1] = rb_s
            q_crystal[3] = rb_p

            q_air = q_air_backward.copy()
            q_air[0] = tt_s
            q_air[2] = tt_p
            return q_air, q_crystal

        returned_front = np.zeros((n_bins + shift_bins + 1, 4), dtype=complex)
        ray_intensity = 0.0
        single_pass_intensity = 0.0
        single_pass_internal_intensity = 0.0

        for index, weight in enumerate(profile):
            amp = np.sqrt(weight)
            q_air_forward = np.array([amp * inc_s, 0.0, amp * inc_p, 0.0], dtype=complex)

            q_front_back = np.zeros(4, dtype=complex)
            q_front_back[[1, 3]] = returned_front[index, [1, 3]]
            _q_air_front, q_crystal_front = solve_front(q_air_forward, q_front_back)

            q_crystal_back_forward = np.zeros(4, dtype=complex)
            q_crystal_back_forward[[0, 2]] = q_crystal_front[[0, 2]] * forward_phase[[0, 2]]
            q_air_out, q_crystal_back = solve_back(q_crystal_back_forward)
            ray_intensity += self._detected_air_intensity(q_air_out, pol_deg)

            returned_index = index + shift_bins
            if returned_index < returned_front.shape[0]:
                returned_front[returned_index, [1, 3]] += q_crystal_back[[1, 3]] * backward_return_phase[[1, 3]]

            q_front_back_zero = np.zeros(4, dtype=complex)
            _q_air_single, q_crystal_single = solve_front(q_air_forward, q_front_back_zero)
            single_pass_internal_intensity += self._forward_crystal_intensity(q_crystal_single)
            q_crystal_single_back = np.zeros(4, dtype=complex)
            q_crystal_single_back[[0, 2]] = q_crystal_single[[0, 2]] * forward_phase[[0, 2]]
            q_air_single_out, _q_crystal_single_back = solve_back(q_crystal_single_back)
            single_pass_intensity += self._detected_air_intensity(q_air_single_out, pol_deg)

        if (
            single_pass_intensity <= 0.0
            or single_pass_internal_intensity <= 0.0
            or reference_internal_intensity <= 0.0
            or not np.isfinite(single_pass_intensity)
            or not np.isfinite(single_pass_internal_intensity)
            or not np.isfinite(reference_internal_intensity)
        ):
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": ray_intensity,
                "bin_width": bin_width,
                "lateral_slope": lateral_slope,
            }

        back_transmission = single_pass_intensity / single_pass_internal_intensity
        if back_transmission <= 0.0 or not np.isfinite(back_transmission):
            return {
                "field_scale": 1.0,
                "intensity_factor": 1.0,
                "internal_intensity": reference_internal_intensity,
                "ray_transmitted_intensity": ray_intensity,
                "bin_width": bin_width,
                "lateral_slope": lateral_slope,
            }

        # Braun Eq. (27): infer the fundamental intensity inside the crystal
        # from the ray-traced transmitted intensity and the back-surface
        # transmission. The nonlinear source is then recomputed from this field.
        ray_internal_intensity = ray_intensity / back_transmission
        field_scale = np.sqrt(max(ray_internal_intensity, 0.0) / reference_internal_intensity)
        intensity_factor = ray_internal_intensity / reference_internal_intensity
        if not np.isfinite(field_scale) or not np.isfinite(intensity_factor):
            field_scale = 1.0
            intensity_factor = 1.0
            ray_internal_intensity = reference_internal_intensity

        return {
            "field_scale": float(field_scale),
            "intensity_factor": float(max(intensity_factor, 0.0)),
            "internal_intensity": float(max(ray_internal_intensity, 0.0)),
            "ray_transmitted_intensity": float(max(ray_intensity, 0.0)),
            "bin_width": bin_width,
            "lateral_slope": lateral_slope,
        }

    def _single_angle_model(self, theta_rad, meta, override):
        wl1_nm = float(meta["wavelength_nm"])
        wl2_nm = wl1_nm / 2.0
        wl1_mm = wl1_nm * 1e-6
        thickness_mm = float(np.asarray(override.get("L", meta["thickness_info"]["t_center_mm"])).reshape(-1)[0])
        dn_override = override.get("dn_override")

        n_w = override.get("_n_w")
        if n_w is None:
            n_w = self._braun_n_xyz(meta, wl1_nm, dn_override=dn_override)
        n_2w = override.get("_n_2w")
        if n_2w is None:
            n_2w = self._braun_n_xyz(meta, wl2_nm, dn_override=dn_override)

        linear_w = self._solve_linear_slab(
            theta_rad,
            wl1_nm,
            thickness_mm,
            n_w,
            meta["input_polarization"],
        )
        reference_internal_intensity = self._forward_crystal_intensity(linear_w["Q_crystal"])
        g_w = self._selected_g(linear_w["modes_crystal"], meta["input_polarization"])
        free_2w_modes = self._layer_modes(theta_rad, n_2w)
        g_2w = self._selected_g(free_2w_modes, meta["detected_polarization"])
        delta_g = g_2w - g_w
        psi = 2.0 * np.pi * thickness_mm * delta_g / wl1_mm

        use_raytrace = bool(override.get("use_raytrace", meta.get("use_raytrace", False)))
        ray_factor = 1.0
        ray_field_scale = 1.0
        ray_internal_intensity = reference_internal_intensity
        ray_transmitted_intensity = float("nan")
        ray_bin_width = float("nan")
        ray_lateral_slope = float("nan")
        if use_raytrace:
            beam_r_x = float(meta.get("beam_r_x", 0.0)) / 2.0
            beam_r_y = float(meta.get("beam_r_y", 0.0)) / 2.0
            beam_radius_mm = np.sqrt(max(beam_r_x, 0.0) * max(beam_r_y, 0.0)) * 1e-3
            raytrace_beam_direction = override.get(
                "raytrace_beam_direction",
                meta.get("raytrace_beam_direction", "k"),
            )
            ray = self._raytrace_correction(
                theta_rad,
                wl1_nm,
                thickness_mm,
                n_w,
                meta["input_polarization"],
                beam_radius_mm,
                reference_internal_intensity,
                override.get("max_reflections", self.DEFAULT_MAX_REFLECTIONS),
                beam_direction=raytrace_beam_direction,
            )
            ray_factor = float(ray["intensity_factor"])
            ray_field_scale = float(ray["field_scale"])
            ray_internal_intensity = float(ray["internal_intensity"])
            ray_transmitted_intensity = float(ray["ray_transmitted_intensity"])
            ray_bin_width = float(ray["bin_width"])
            ray_lateral_slope = float(ray["lateral_slope"])
            linear_w = self._scaled_linear_solution(linear_w, ray_field_scale)

        shg = self._solve_nonlinear_shg(
            linear_w,
            theta_rad,
            wl2_nm,
            thickness_mm,
            n_2w,
            meta,
            override,
        )
        intensity = float(np.real(shg["intensity"]))
        source = float(shg["bound_front_norm"] ** 2 + shg["bound_back_norm"] ** 2)

        return {
            "envelope": max(intensity, 0.0),
            "psi": psi,
            "delta_k": 2.0 * psi / thickness_mm if thickness_mm != 0 else np.nan,
            "raytrace_factor": ray_factor,
            "raytrace_field_scale": ray_field_scale,
            "raytrace_bin_width": ray_bin_width,
            "raytrace_lateral_slope": ray_lateral_slope,
            "raytrace_internal_intensity": ray_internal_intensity,
            "raytrace_transmitted_intensity": ray_transmitted_intensity,
            "reference_internal_intensity": reference_internal_intensity,
            "source_strength": source,
            "sh_transmission": float(shg["intensity"]),
        }

    def _parallel_workers(self, meta, override, n_points):
        if n_points < int(override.get("parallel_threshold", meta.get("parallel_threshold", self.DEFAULT_PARALLEL_THRESHOLD))):
            return 1
        if override.get("parallel", meta.get("parallel", self.DEFAULT_PARALLEL)) is False:
            return 1

        requested = override.get("parallel_workers", meta.get("parallel_workers"))
        if requested is None:
            requested = min(32, (os.cpu_count() or 1))
        workers = max(int(requested), 1)
        return min(workers, int(n_points))

    def _evaluate_angles(self, theta_flat, meta, override):
        theta_rad = np.deg2rad(np.asarray(theta_flat, dtype=float))
        workers = self._parallel_workers(meta, override, theta_rad.size)
        if workers <= 1:
            return [self._single_angle_model(theta, meta, override) for theta in theta_rad]

        def evaluate_chunk(chunk):
            return [self._single_angle_model(theta, meta, override) for theta in chunk]

        chunks = [chunk for chunk in np.array_split(theta_rad, workers) if chunk.size]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(evaluate_chunk, chunks))
        return [item for chunk_values in nested for item in chunk_values]

    def _maker_fringes(self, override: dict | None = None, envelope=False, return_aux=False):
        override = {} if override is None else dict(override)
        meta = override.get("meta", "auto")
        data = override.get("data", "auto")
        meta, data = self._resolve_input_info(meta=meta, data=data)
        meta = self._model_meta(meta, override)
        wl1_nm = float(meta["wavelength_nm"])
        wl2_nm = wl1_nm / 2.0
        dn_override = override.get("dn_override")
        override.setdefault("_n_w", self._braun_n_xyz(meta, wl1_nm, dn_override=dn_override))
        override.setdefault("_n_2w", self._braun_n_xyz(meta, wl2_nm, dn_override=dn_override))
        override.setdefault("_numeric_d_matrix", self._numeric_d_matrix(meta, override))

        if "theta_deg" in override:
            theta_deg = override["theta_deg"]
        else:
            theta_deg = np.asarray(data.get("position_centered", data["position"]), dtype=float)

        theta_arr = np.asarray(theta_deg, dtype=float)
        scalar_input = theta_arr.ndim == 0
        theta_flat = np.ravel(theta_arr)

        values = self._evaluate_angles(theta_flat, meta, override)
        raw_intensity = np.array([item["envelope"] for item in values], dtype=float).reshape(theta_arr.shape)
        psi = np.array([np.real(item["psi"]) for item in values], dtype=float).reshape(theta_arr.shape)

        intensity0 = self._single_angle_model(0.0, meta, override)["envelope"]
        if not np.isfinite(intensity0) or intensity0 <= 0.0:
            intensity0 = (
                float(np.nanmax(raw_intensity))
                if np.isfinite(raw_intensity).any() and np.nanmax(raw_intensity) > 0.0
                else 1.0
            )

        normalize = bool(override.get("normalize", False))
        model = raw_intensity / intensity0 if normalize else raw_intensity

        if scalar_input:
            model = float(np.asarray(model))

        if not return_aux:
            return model

        aux = {
            "d_factor": float(intensity0),
            "Psi": psi,
            "delta_k": np.array([np.real(item["delta_k"]) for item in values], dtype=float).reshape(theta_arr.shape),
            "raytrace_factor": np.array([item["raytrace_factor"] for item in values], dtype=float).reshape(theta_arr.shape),
            "raytrace_field_scale": np.array([item["raytrace_field_scale"] for item in values], dtype=float).reshape(theta_arr.shape),
            "raytrace_bin_width": np.array([item["raytrace_bin_width"] for item in values], dtype=float).reshape(theta_arr.shape),
            "raytrace_lateral_slope": np.array([item["raytrace_lateral_slope"] for item in values], dtype=float).reshape(theta_arr.shape),
            "raytrace_internal_intensity": np.array([item["raytrace_internal_intensity"] for item in values], dtype=float).reshape(theta_arr.shape),
            "raytrace_transmitted_intensity": np.array([item["raytrace_transmitted_intensity"] for item in values], dtype=float).reshape(theta_arr.shape),
            "reference_internal_intensity": np.array([item["reference_internal_intensity"] for item in values], dtype=float).reshape(theta_arr.shape),
            "source_strength": np.array([item["source_strength"] for item in values], dtype=float).reshape(theta_arr.shape),
            "sh_transmission": np.array([item["sh_transmission"] for item in values], dtype=float).reshape(theta_arr.shape),
        }
        return model, aux

    def fit_all(self):
        """
        App-friendly Braun 1997 fit.

        Fit the relative absolute d coefficient using the raw Braun intensity.
        No external intensity scale or additive offset is fitted here:
        measured_voltage(theta) ~= d_rel_abs**2 * I_Braun(theta; d_ij=1).
        """
        data, _centering_info = self._position_centering(self.analysis.data)
        meta = self._model_meta(self.analysis.meta, {})

        theta_deg = np.asarray(data.get("position_centered", data["position"]), dtype=float)
        y = np.asarray(data.get("intensity_corrected", data.get("ch2")), dtype=float)
        finite_y = np.isfinite(theta_deg) & np.isfinite(y)
        fit_mask = self._fit_range_mask(data, meta=meta, base_mask=finite_y, min_points=3)
        if not np.any(fit_mask):
            raise ValueError("No finite points available for Braun1997 fitting.")

        theta_fit = theta_deg[fit_mask]
        y_fit = y[fit_mask]
        L0_mm = float(meta["thickness_info"]["t_center_mm"])

        def residual(params):
            L_mm = float(params[0])
            delta_n = float(params[1])
            d_squared = float(params[2])
            dn_override = self._delta_n_override(meta, delta_n)
            model = np.asarray(
                self._maker_fringes(
                    override={
                        "meta": meta,
                        "data": data,
                        "theta_deg": theta_fit,
                        "L": L_mm,
                        "dn_override": dn_override,
                        "normalize": False,
                    }
                ),
                dtype=float,
            )
            return d_squared * model - y_fit

        unit_model0 = np.asarray(
            self._maker_fringes(
                override={
                    "meta": meta,
                    "data": data,
                    "theta_deg": theta_fit,
                    "L": L0_mm,
                    "normalize": False,
                }
            ),
            dtype=float,
        )
        denom = float(np.dot(unit_model0, unit_model0))
        d_squared0 = float(np.dot(unit_model0, y_fit) / denom) if denom > 0.0 else 1.0
        if not np.isfinite(d_squared0) or d_squared0 < 0.0:
            d_squared0 = 1.0
        result = least_squares(
            residual,
            x0=np.array([L0_mm, 0.0, d_squared0], dtype=float),
            bounds=(
                np.array([L0_mm - 0.01, -0.001, 0.0], dtype=float),
                np.array([L0_mm + 0.01, 0.001, np.inf], dtype=float),
            ),
            max_nfev=20000,
        )

        fitted_L_mm = float(result.x[0])
        delta_n = float(result.x[1])
        d_squared = float(result.x[2])
        dn_override = self._delta_n_override(meta, delta_n)
        unit_model, _fit_aux = self._maker_fringes(
            override={
                "meta": meta,
                "data": data,
                "theta_deg": theta_deg,
                "L": fitted_L_mm,
                "dn_override": dn_override,
                "normalize": False,
            },
            return_aux=True,
        )
        unit_model = np.asarray(unit_model, dtype=float)

        d_rel_abs = float(np.sqrt(max(d_squared, 0.0)))
        fit_curve = d_rel_abs**2 * unit_model

        out = data.copy()
        out["fit"] = fit_curve

        results = {
            "d_component": str(meta.get("d_component", "")),
            "d_rel_abs": d_rel_abs,
            "L_mm": fitted_L_mm,
            "L_mm_std": float("nan"),
            "delta_n": delta_n,
            "delta_n_std": float("nan"),
            "delta_n_fit_cost": float(result.cost),
            "delta_n_fit_success": bool(result.success),
        }
        results.update(dn_override)
        if isinstance(_fit_aux, dict) and _fit_aux.get("delta_k") is not None:
            delta_k = self._coerce_scalar(_fit_aux["delta_k"])
            if np.isfinite(delta_k):
                results["delta_k_theory_inv_mm"] = float(delta_k)
                results["Lc_theory_mm"] = float(np.pi / abs(delta_k)) if not np.isclose(delta_k, 0.0) else float("nan")

        self.analysis.meta = upsert_fitting_result(
            self.analysis.meta,
            self.__class__.__name__,
            results,
            strategy_module=self.__class__.__module__,
            strategy_display_name=self.__class__.__name__,
        )

        self.analysis.data = out
        out.to_csv(self.analysis.csv_path, index=False)
        with open(self.analysis.json_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis.meta, f, ensure_ascii=False, indent=2)

        return results


class Braun1997GlobalNFitStrategy(GlobalNFitMixin, Braun1997Strategy):
    """Global refractive-index fit using the Braun 1997 model."""

    def _make_measurement_strategy(self, analysis):
        return Braun1997Strategy(analysis)

    def _validate_measurement_strategy(self, strategy, meta):
        geometry_key = strategy._geometry_key(meta)
        if strategy._default_d_component(meta) is None:
            raise FittingConfigurationError(
                f"This geometry is not supported for Braun global n fit: {geometry_key}"
            )
