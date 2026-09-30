from dataclasses import replace

import numpy as np

from fitting_strategies.abe2008 import (
    Abe2008FiniteBeamModel,
    bmf_parameters,
    sic_fig4_parameters,
)


BMF_CONFIGS = [
    ("d_31", "010", "100", 0, 90),
    ("d_32", "100", "010", 0, 90),
    ("d_33", "100", "001", 0, 0),
    ("d_33", "010", "001", 0, 0),
    ("d_15", "010", "100", 45, 0),
    ("d_24", "100", "010", 45, 0),
]


def _fast_parameters(**updates):
    base = sic_fig4_parameters(x_points=201, z_points=401)
    return replace(base, **updates)


def test_no_reflection_configuration_matches_single_path_helper():
    angles = np.array([12.0, 27.0, 43.0])
    model = Abe2008FiniteBeamModel(_fast_parameters())
    expected = model.single_path_power(angles)

    no_reflections = Abe2008FiniteBeamModel(
        _fast_parameters(
            include_fundamental_reflections=False,
            include_sh_reflections=False,
            max_fundamental_round_trips=0,
            max_sh_round_trips=0,
        )
    ).power(angles)

    assert np.allclose(no_reflections, expected, rtol=1e-12, atol=1e-12)


def test_paper_parameters_produce_finite_nonnegative_power():
    model = Abe2008FiniteBeamModel(_fast_parameters())
    power = model.power([0.0, 15.0, 30.0, 45.0, 60.0])

    assert power.shape == (5,)
    assert np.all(np.isfinite(power))
    assert np.all(power >= 0.0)
    assert np.any(power > 0.0)


def test_multiple_reflections_change_the_fig4_curve_by_order_unity():
    angles = np.linspace(15.0, 60.0, 19)
    model = Abe2008FiniteBeamModel(_fast_parameters())
    multiple = model.power(angles)
    single = model.single_path_power(angles)
    scale = max(float(np.max(single)), 1e-30)

    # This is deliberately a broad regression guard: Fig. 4 predicts a large
    # effect, while exact peak sampling depends on the angular grid.
    assert float(np.max(np.abs(multiple - single))) / scale > 0.25


def test_z_quadrature_is_nearly_converged_at_default_resolution():
    angles = np.array([30.0, 54.0])
    medium = Abe2008FiniteBeamModel(_fast_parameters(x_points=301, z_points=801)).power(angles)
    fine = Abe2008FiniteBeamModel(_fast_parameters(x_points=301, z_points=1201)).power(angles)

    assert np.allclose(medium, fine, rtol=0.01, atol=1e-10)


def test_bmf_two_mm_geometries_are_finite_and_symmetric():
    angles = np.array([-10.0, 0.0, 10.0])
    for d_component, cut, axis, pol_in, pol_out in BMF_CONFIGS:
        parameters = bmf_parameters(
            d_component=d_component,
            crystal_orientation=cut,
            rotation_axis=axis,
            input_polarization_deg=pol_in,
            detected_polarization_deg=pol_out,
            thickness_mm=2.0,
            x_points=101,
            z_points=401,
            max_fundamental_round_trips=1,
            max_sh_round_trips=1,
        )
        power = Abe2008FiniteBeamModel(parameters).power(angles)

        assert parameters.thickness_um == 2000.0
        assert np.all(np.isfinite(power))
        assert np.all(power >= 0.0)
        assert np.any(power > 0.0)
        assert np.isclose(power[0], power[-1], rtol=1e-9, atol=1e-9)
