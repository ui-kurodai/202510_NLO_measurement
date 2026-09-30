"""Finite-beam, multiple-reflection Maker fringes after Abe et al. (2008).

This module implements the one-transverse-coordinate form of Eqs. (18)-(46)
from M. Abe et al., JOSA B 25, 1616-1624 (2008).  The implementation is meant
as a transparent numerical reference model: the fundamental and generated SH
fields retain their complex phase and Gaussian transverse profiles until the
final output-plane power integral.

The first implementation is restricted to a principal-plane geometry with a
diagonal dielectric tensor.  One s or p mode is selected at each frequency.
That covers the s-in/p-out 6H-SiC calculation in Fig. 4 of the paper and the
principal-plane d31/d32 measurements used by this project.

Lengths are in micrometres and vacuum wavelengths are in micrometres.
Absolute SI prefactors common to all angles are omitted; returned powers are
therefore in consistent relative units.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
from scipy.integrate import simpson


Polarization = Literal["s", "p"]


@dataclass(frozen=True)
class Abe2008Parameters:
    """Physical and numerical inputs for the finite-beam calculation.

    ``beam_radius_um`` is Abe's field radius ``a``: at the input surface the
    field amplitude is proportional to ``exp(-x**2 cos(theta)**2 / a**2)``.
    ``n_w_xyz`` and ``n_2w_xyz`` are principal refractive indices ordered as
    the laboratory x, y, and z axes, with z normal to the plate and y the
    rotation axis.
    """

    wavelength_um: float = 1.064
    thickness_um: float = 400.0
    beam_radius_um: float = 400.0
    n_w_xyz: tuple[float, float, float] = (2.5464, 2.5464, 2.5464)
    n_2w_xyz: tuple[float, float, float] = (2.6289, 2.6289, 2.6716)
    fundamental_polarization: Polarization | float = "s"
    sh_polarization: Polarization = "p"
    nonlinear_source_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    nonlinear_d_matrix: tuple[tuple[complex, ...], ...] | None = None
    lab_to_crystal_axes: tuple[int, int, int] = (0, 1, 2)
    max_fundamental_round_trips: int = 4
    max_sh_round_trips: int = 4
    include_fundamental_reflections: bool = True
    include_sh_reflections: bool = True
    x_points: int = 401
    z_points: int = 801
    transverse_margin_radii: float = 4.5

    def validated(self) -> "Abe2008Parameters":
        if self.wavelength_um <= 0 or self.thickness_um <= 0:
            raise ValueError("wavelength_um and thickness_um must be positive.")
        if self.beam_radius_um <= 0:
            raise ValueError("beam_radius_um must be positive.")
        if isinstance(self.fundamental_polarization, str):
            if self.fundamental_polarization not in {"s", "p"}:
                raise ValueError("fundamental_polarization must be 's', 'p', or an angle in degrees.")
        elif not np.isfinite(float(self.fundamental_polarization)):
            raise ValueError("fundamental_polarization angle must be finite.")
        if self.sh_polarization not in {"s", "p"}:
            raise ValueError("sh_polarization must be 's' or 'p'.")
        if self.x_points < 101 or self.z_points < 101:
            raise ValueError("x_points and z_points must both be at least 101.")
        if self.max_fundamental_round_trips < 0 or self.max_sh_round_trips < 0:
            raise ValueError("Reflection orders must be non-negative.")
        for name, values in (("n_w_xyz", self.n_w_xyz), ("n_2w_xyz", self.n_2w_xyz)):
            if len(values) != 3 or np.any(np.asarray(values, dtype=float) <= 0):
                raise ValueError(f"{name} must contain three positive indices.")
        source = np.asarray(self.nonlinear_source_xyz, dtype=float)
        if source.shape != (3,) or not np.all(np.isfinite(source)) or np.linalg.norm(source) == 0:
            raise ValueError("nonlinear_source_xyz must be a finite nonzero 3-vector.")
        if self.nonlinear_d_matrix is not None:
            d_matrix = np.asarray(self.nonlinear_d_matrix, dtype=complex)
            if d_matrix.shape != (3, 6) or not np.all(np.isfinite(d_matrix)):
                raise ValueError("nonlinear_d_matrix must be a finite 3x6 matrix.")
        if tuple(sorted(self.lab_to_crystal_axes)) != (0, 1, 2):
            raise ValueError("lab_to_crystal_axes must be a permutation of (0, 1, 2).")
        return self


@dataclass(frozen=True)
class _Mode:
    a: float
    g: complex
    e: np.ndarray
    h: np.ndarray
    slope: float


@dataclass(frozen=True)
class Abe2008AngleResult:
    """Diagnostics returned for one external incidence angle."""

    theta_deg: float
    power: float
    direct_sh_power: float
    x_um: np.ndarray
    e_out: np.ndarray
    e_plus_single_path: np.ndarray
    e_minus_single_path: np.ndarray
    r_w: complex
    r_2w: complex
    t01_w: complex
    t12_2w: complex
    fundamental_slope: float
    sh_slope: float


def sic_fig4_parameters(**updates) -> Abe2008Parameters:
    """Return the material and beam parameters quoted for Abe et al. Fig. 4."""

    base = Abe2008Parameters(
        wavelength_um=1.064,
        thickness_um=400.0,
        beam_radius_um=400.0,
        n_w_xyz=(2.5464, 2.5464, 2.5464),
        n_2w_xyz=(2.6289, 2.6289, 2.6716),
        fundamental_polarization="s",
        sh_polarization="p",
        # For s-p 6H-SiC only d31=d32 contributes: P_z = d31 E_y^2.
        nonlinear_source_xyz=(0.0, 0.0, 1.0),
    )
    return replace(base, **updates).validated()


def bmf_parameters(
    *,
    d_component: str,
    crystal_orientation: str,
    rotation_axis: str,
    input_polarization_deg: float,
    detected_polarization_deg: float,
    wavelength_nm: float = 1064.0,
    thickness_mm: float = 2.0,
    beam_diameter_um: float = 400.0,
    **updates,
) -> Abe2008Parameters:
    """Build an Abe-model parameter set for a principal-plane BaMgF4 scan.

    Refractive indices, the symbolic 3x6 tensor, and the mapping between lab
    and crystal axes are taken from the same helpers as ``Braun1997Strategy``.
    The selected ``d_component`` is assigned unit amplitude, so the resulting
    fringe is on a relative scale suitable for shape comparisons.
    """

    from types import SimpleNamespace

    from fitting_strategies.braun1997 import Braun1997Strategy

    if np.isclose(float(detected_polarization_deg), 0.0, atol=1e-9):
        sh_polarization: Polarization = "s"
    elif np.isclose(float(detected_polarization_deg), 90.0, atol=1e-9):
        sh_polarization = "p"
    else:
        raise ValueError("BMF helper currently supports detected polarization 0 or 90 degrees.")

    meta = {
        "material": "BaMgF4",
        "wavelength_nm": float(wavelength_nm),
        "input_polarization": float(input_polarization_deg),
        "detected_polarization": float(detected_polarization_deg),
        "crystal_orientation": str(crystal_orientation),
        "rot/trans_axis": str(rotation_axis),
        "thickness_info": {"t_center_mm": float(thickness_mm)},
        "beam_r_x": float(beam_diameter_um),
        "beam_r_y": float(beam_diameter_um),
        "d_component": str(d_component),
    }
    strategy = Braun1997Strategy(SimpleNamespace(meta=meta, data=None))
    n_w = strategy._braun_n_xyz(meta, float(wavelength_nm))
    n_2w = strategy._braun_n_xyz(meta, float(wavelength_nm) / 2.0)
    d_matrix = strategy._numeric_d_matrix(meta, {})
    axes = strategy._braun_axes(meta)
    lab_to_crystal = tuple(
        strategy._axis_to_index(axes[label]) for label in ("x", "y", "z")
    )

    base = Abe2008Parameters(
        wavelength_um=float(wavelength_nm) * 1e-3,
        thickness_um=float(thickness_mm) * 1e3,
        # Metadata uses the full beam diameter; Abe's a is a field radius.
        beam_radius_um=float(beam_diameter_um) / 2.0,
        n_w_xyz=tuple(float(value) for value in n_w),
        n_2w_xyz=tuple(float(value) for value in n_2w),
        fundamental_polarization=float(input_polarization_deg),
        sh_polarization=sh_polarization,
        nonlinear_d_matrix=tuple(
            tuple(complex(value) for value in row) for row in np.asarray(d_matrix)
        ),
        lab_to_crystal_axes=lab_to_crystal,
    )
    return replace(base, **updates).validated()


class Abe2008FiniteBeamModel:
    """Numerical implementation of Abe's partially-overlapping beam sums."""

    def __init__(self, parameters: Abe2008Parameters):
        self.parameters = parameters.validated()

    def _fundamental_components(self) -> dict[Polarization, float]:
        polarization = self.parameters.fundamental_polarization
        if isinstance(polarization, str):
            return {"s": 1.0, "p": 0.0} if polarization == "s" else {"s": 0.0, "p": 1.0}
        angle_rad = np.deg2rad(float(polarization))
        return {"s": float(np.cos(angle_rad)), "p": float(np.sin(angle_rad))}

    @staticmethod
    def _normalize(vector: Sequence[complex]) -> np.ndarray:
        vector = np.asarray(vector, dtype=complex)
        norm = float(np.sqrt(np.sum(np.abs(vector) ** 2)))
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    @classmethod
    def _mode(
        cls,
        n_xyz: Sequence[float],
        tangential_k: float,
        polarization: Polarization,
        direction: int,
    ) -> _Mode:
        """Return an s or p eigenmode for diagonal epsilon, Eqs. (2)-(13)."""

        nx, ny, nz = np.asarray(n_xyz, dtype=float)
        a = float(tangential_k)
        sign = 1 if direction >= 0 else -1
        if polarization == "s":
            g_abs = np.sqrt(complex(ny**2 - a**2))
            e = np.array([0.0, 1.0, 0.0], dtype=complex)
        else:
            g_abs = np.sqrt(complex(nx**2 - (nx**2 / nz**2) * a**2))
            # Equivalent to Abe Eqs. (11)-(12), up to an overall mode sign.
            g_tmp = sign * g_abs
            e = cls._normalize([g_tmp / nx**2, 0.0, -a / nz**2])
        g = sign * g_abs
        k_vec = np.array([a, 0.0, g], dtype=complex)
        h = np.cross(k_vec, e)
        poynting = np.real(np.cross(e, np.conj(h)))
        slope = float(poynting[0] / poynting[2])
        return _Mode(a=a, g=g, e=e, h=h, slope=slope)

    @staticmethod
    def _boundary_vector(mode: _Mode, polarization: Polarization) -> np.ndarray:
        if polarization == "s":
            return np.array([mode.e[1], mode.h[0]], dtype=complex)
        return np.array([mode.e[0], mode.h[1]], dtype=complex)

    @classmethod
    def _forward_interface(
        cls,
        n_left: Sequence[float],
        n_right: Sequence[float],
        tangential_k: float,
        polarization: Polarization,
    ) -> tuple[complex, complex]:
        """Electric-field reflection/transmission amplitudes for left-to-right incidence."""

        incident = cls._mode(n_left, tangential_k, polarization, +1)
        reflected = cls._mode(n_left, tangential_k, polarization, -1)
        transmitted = cls._mode(n_right, tangential_k, polarization, +1)
        matrix = np.column_stack(
            (
                cls._boundary_vector(reflected, polarization),
                -cls._boundary_vector(transmitted, polarization),
            )
        )
        rhs = -cls._boundary_vector(incident, polarization)
        reflection, transmission = np.linalg.solve(matrix, rhs)
        return complex(reflection), complex(transmission)

    @staticmethod
    def _interp_complex(x_new: np.ndarray, x: np.ndarray, values: np.ndarray) -> np.ndarray:
        real = np.interp(x_new, x, values.real, left=0.0, right=0.0)
        imag = np.interp(x_new, x, values.imag, left=0.0, right=0.0)
        return real + 1j * imag

    @staticmethod
    def _green_projection(mode: _Mode, n_xyz: Sequence[float], source_xyz: np.ndarray) -> complex:
        """Polarization projection in Abe Eqs. (40)-(43), sans common constants."""

        kz_magnitude = max(abs(mode.g), 1e-15)
        if np.allclose(mode.e[[0, 2]], 0.0):
            return complex(np.dot(mode.e, source_xyz) / kz_magnitude)
        nx = float(n_xyz[0])
        # n_p = |k|/k0 in Abe's notation.
        n_p_sq = mode.g**2 + mode.a**2
        return complex((nx**4 / (n_p_sq * kz_magnitude)) * np.dot(mode.e, source_xyz))

    def _field_radius_on_surface(self, theta_rad: float) -> float:
        return self.parameters.beam_radius_um / max(abs(np.cos(theta_rad)), 1e-12)

    def _x_grid(
        self,
        theta_rad: float,
        fundamental_slopes: Sequence[float],
        slope_2w: float,
    ) -> np.ndarray:
        p = self.parameters
        radius_surface = self._field_radius_on_surface(theta_rad)
        m_w = p.max_fundamental_round_trips if p.include_fundamental_reflections else 0
        m_2w = p.max_sh_round_trips if p.include_sh_reflections else 0
        max_slope_w = max((abs(float(value)) for value in fundamental_slopes), default=0.0)
        shift_w = 2.0 * (m_w + 1) * p.thickness_um * max_slope_w
        shift_2w = (2.0 * m_2w + 1.0) * p.thickness_um * abs(slope_2w)
        half_width = p.transverse_margin_radii * radius_surface + shift_w + shift_2w
        return np.linspace(-half_width, half_width, int(p.x_points), dtype=float)

    def _nonlinear_polarization(self, e_lab: np.ndarray) -> np.ndarray:
        """Evaluate the contracted 3x6 d tensor in crystal coordinates."""

        p = self.parameters
        d_matrix = np.asarray(p.nonlinear_d_matrix, dtype=complex)
        e_crystal = np.zeros_like(e_lab, dtype=complex)
        for lab_index, crystal_index in enumerate(p.lab_to_crystal_axes):
            e_crystal[..., crystal_index] = e_lab[..., lab_index]

        ex, ey, ez = (e_crystal[..., index] for index in range(3))
        quadratic = np.stack(
            (ex * ex, ey * ey, ez * ez, 2 * ey * ez, 2 * ex * ez, 2 * ex * ey),
            axis=-1,
        )
        p_crystal = np.einsum("...j,ij->...i", quadratic, d_matrix)
        p_lab = np.zeros_like(p_crystal, dtype=complex)
        for lab_index, crystal_index in enumerate(p.lab_to_crystal_axes):
            p_lab[..., lab_index] = p_crystal[..., crystal_index]
        return p_lab

    @staticmethod
    def _green_project_field(
        mode: _Mode,
        n_xyz: Sequence[float],
        p_nl: np.ndarray,
    ) -> np.ndarray:
        """Project a spatially varying nonlinear-polarization vector."""

        kz_magnitude = max(abs(mode.g), 1e-15)
        if np.allclose(mode.e[[0, 2]], 0.0):
            factor = 1.0 / kz_magnitude
        else:
            nx = float(n_xyz[0])
            n_p_sq = mode.g**2 + mode.a**2
            factor = nx**4 / (n_p_sq * kz_magnitude)
        return factor * np.einsum("...i,i->...", p_nl, mode.e)

    def _fundamental_envelopes(
        self,
        x: np.ndarray,
        z: np.ndarray,
        theta_rad: float,
        slope_w: float,
        kz_w: complex,
        r_w: complex,
        t01_w: complex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return E+ and E- with the common exp(i kx x) factor removed."""

        p = self.parameters
        gaussian_factor = np.cos(theta_rad) ** 2 / p.beam_radius_um**2
        e_plus = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)), dtype=complex)
        max_m = p.max_fundamental_round_trips if p.include_fundamental_reflections else 0
        for m in range(max_m + 1):
            center = (z + 2.0 * m * p.thickness_um) * slope_w
            profile = np.exp(-gaussian_factor * (x - center) ** 2)
            e_plus += profile * r_w ** (2 * m) * np.exp(2j * m * kz_w * p.thickness_um)
        e_plus *= t01_w * np.exp(1j * kz_w * z)

        e_minus = np.zeros_like(e_plus)
        if p.include_fundamental_reflections:
            for m in range(p.max_fundamental_round_trips + 1):
                center = ((2.0 * m + 2.0) * p.thickness_um - z) * slope_w
                profile = np.exp(-gaussian_factor * (x - center) ** 2)
                e_minus += (
                    profile
                    * r_w ** (2 * m + 1)
                    * np.exp(1j * (2 * m + 1) * kz_w * p.thickness_um)
                )
            e_minus *= t01_w * np.exp(-1j * kz_w * z)
        return e_plus, e_minus

    def calculate_angle(self, theta_deg: float) -> Abe2008AngleResult:
        """Calculate the output-plane field and relative power for one angle."""

        p = self.parameters
        theta_rad = float(np.deg2rad(theta_deg))
        tangential_k = float(np.sin(theta_rad))
        air = (1.0, 1.0, 1.0)

        sh_plus = self._mode(p.n_2w_xyz, tangential_k, p.sh_polarization, +1)
        sh_minus = self._mode(p.n_2w_xyz, tangential_k, p.sh_polarization, -1)
        r_2w, t12_2w = self._forward_interface(p.n_2w_xyz, air, tangential_k, p.sh_polarization)

        k0_w = 2.0 * np.pi / p.wavelength_um
        k0_2w = 2.0 * k0_w
        kz_2w = k0_2w * sh_plus.g

        fundamental_modes = {}
        for polarization, amplitude in self._fundamental_components().items():
            if np.isclose(amplitude, 0.0, atol=1e-15):
                continue
            mode = self._mode(p.n_w_xyz, tangential_k, polarization, +1)
            _, transmission = self._forward_interface(
                air, p.n_w_xyz, tangential_k, polarization
            )
            reflection, _ = self._forward_interface(
                p.n_w_xyz, air, tangential_k, polarization
            )
            fundamental_modes[polarization] = {
                "polarization": polarization,
                "amplitude": float(amplitude),
                "mode": mode,
                "mode_minus": self._mode(p.n_w_xyz, tangential_k, polarization, -1),
                "t01": transmission,
                "r": reflection,
                "kz": k0_w * mode.g,
            }
        if not fundamental_modes:
            raise ValueError("The incident fundamental field has zero amplitude.")
        if len(fundamental_modes) > 1 and p.nonlinear_d_matrix is None:
            raise ValueError(
                "Mixed fundamental polarization requires nonlinear_d_matrix."
            )

        primary = max(fundamental_modes.values(), key=lambda item: abs(item["amplitude"]))

        x_grid = self._x_grid(
            theta_rad,
            [item["mode"].slope for item in fundamental_modes.values()],
            sh_plus.slope,
        )
        z_grid = np.linspace(0.0, p.thickness_um, int(p.z_points), dtype=float)
        x_2d = x_grid[None, :]
        z_2d = z_grid[:, None]
        source = self._normalize(p.nonlinear_source_xyz)

        def fundamental_fields(x_coordinates):
            shape = np.broadcast_shapes(np.shape(x_coordinates), np.shape(z_2d))
            plus_vector = np.zeros(shape + (3,), dtype=complex)
            minus_vector = np.zeros_like(plus_vector)
            plus_scalar = np.zeros(shape, dtype=complex)
            minus_scalar = np.zeros(shape, dtype=complex)
            for item in fundamental_modes.values():
                mode = item["mode"]
                e_plus_mode, e_minus_mode = self._fundamental_envelopes(
                    x_coordinates,
                    z_2d,
                    theta_rad,
                    mode.slope,
                    item["kz"],
                    item["r"],
                    item["t01"],
                )
                e_plus_mode *= item["amplitude"]
                e_minus_mode *= item["amplitude"]
                plus_vector += e_plus_mode[..., None] * mode.e
                minus_vector += e_minus_mode[..., None] * item["mode_minus"].e
                plus_scalar += e_plus_mode
                minus_scalar += e_minus_mode
            return plus_vector, minus_vector, plus_scalar, minus_scalar

        # Eqs. (40)-(41): trace each output coordinate back along the SH ray.
        x_source_plus = x_2d - (p.thickness_um - z_2d) * sh_plus.slope
        e_w_plus, _, e_w_plus_scalar, _ = fundamental_fields(x_source_plus)
        if p.nonlinear_d_matrix is None:
            projected_plus = (
                self._green_projection(sh_plus, p.n_2w_xyz, source)
                * e_w_plus_scalar**2
            )
        else:
            p_plus = self._nonlinear_polarization(e_w_plus)
            projected_plus = self._green_project_field(sh_plus, p.n_2w_xyz, p_plus)
        integrand_plus = projected_plus * np.exp(
            1j * kz_2w * (p.thickness_um - z_2d)
        )
        e_plus = simpson(integrand_plus, x=z_grid, axis=0)

        # Eqs. (42)-(43): the source is the backward fundamental field.
        e_minus = np.zeros_like(e_plus)
        if p.include_fundamental_reflections:
            x_source_minus = x_2d - z_2d * sh_plus.slope
            _, e_w_minus, _, e_w_minus_scalar = fundamental_fields(x_source_minus)
            if p.nonlinear_d_matrix is None:
                projected_minus = (
                    self._green_projection(sh_minus, p.n_2w_xyz, source)
                    * e_w_minus_scalar**2
                )
            else:
                p_minus = self._nonlinear_polarization(e_w_minus)
                projected_minus = self._green_project_field(sh_minus, p.n_2w_xyz, p_minus)
            integrand_minus = projected_minus * np.exp(1j * kz_2w * z_2d)
            e_minus = simpson(integrand_minus, x=z_grid, axis=0)

        # Eqs. (44)-(45): coherently add all later SH reflection paths.
        e_out = np.zeros_like(e_plus)
        if p.include_sh_reflections:
            for m in range(p.max_sh_round_trips + 1):
                plus_shift = 2.0 * m * p.thickness_um * sh_plus.slope
                minus_shift = (2.0 * m + 1.0) * p.thickness_um * sh_plus.slope
                plus_path = self._interp_complex(x_grid - plus_shift, x_grid, e_plus)
                minus_path = self._interp_complex(x_grid - minus_shift, x_grid, e_minus)
                e_out += plus_path * r_2w ** (2 * m) * np.exp(2j * m * kz_2w * p.thickness_um)
                e_out += minus_path * r_2w ** (2 * m + 1) * np.exp(
                    1j * (2 * m + 1) * kz_2w * p.thickness_um
                )
        else:
            e_out = e_plus.copy()
        e_out *= t12_2w

        # Eq. (46), omitting the angle-independent epsilon0*c*sqrt(pi)*a/4.
        power = float(max(np.cos(theta_rad), 0.0) * np.trapezoid(np.abs(e_out) ** 2, x_grid))
        single_field = t12_2w * e_plus
        direct_sh_power = float(
            max(np.cos(theta_rad), 0.0) * np.trapezoid(np.abs(single_field) ** 2, x_grid)
        )
        return Abe2008AngleResult(
            theta_deg=float(theta_deg),
            power=power,
            direct_sh_power=direct_sh_power,
            x_um=x_grid,
            e_out=e_out,
            e_plus_single_path=e_plus,
            e_minus_single_path=e_minus,
            r_w=primary["r"],
            r_2w=r_2w,
            t01_w=primary["t01"],
            t12_2w=t12_2w,
            fundamental_slope=primary["mode"].slope,
            sh_slope=sh_plus.slope,
        )

    def power(
        self,
        theta_deg: Sequence[float] | float,
        return_results: bool = False,
        workers: int = 1,
    ):
        """Evaluate relative SH power at one or more external angles."""

        theta = np.asarray(theta_deg, dtype=float)
        flat_theta = [float(value) for value in theta.reshape(-1)]
        workers = max(int(workers), 1)
        if workers == 1 or len(flat_theta) < 2:
            flat_results = [self.calculate_angle(value) for value in flat_theta]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(flat_theta))) as executor:
                flat_results = list(executor.map(self.calculate_angle, flat_theta))
        powers = np.asarray([result.power for result in flat_results], dtype=float).reshape(theta.shape)
        if theta.ndim == 0:
            powers = float(powers)
        if return_results:
            return powers, flat_results
        return powers

    def single_path_power(self, theta_deg: Sequence[float] | float, workers: int = 1):
        """Evaluate the no-reflection baseline using the same numerical grid."""

        baseline = replace(
            self.parameters,
            include_fundamental_reflections=False,
            include_sh_reflections=False,
            max_fundamental_round_trips=0,
            max_sh_round_trips=0,
        )
        return Abe2008FiniteBeamModel(baseline).power(theta_deg, workers=workers)


__all__ = [
    "Abe2008AngleResult",
    "Abe2008FiniteBeamModel",
    "Abe2008Parameters",
    "bmf_parameters",
    "sic_fig4_parameters",
]
