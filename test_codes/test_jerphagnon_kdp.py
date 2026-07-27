from types import SimpleNamespace

import numpy as np
import pandas as pd

from fitting_strategies.jerphagnon1970 import Jerphagnon1970Strategy


def _kdp_strategy():
    theta = np.linspace(-30.0, 30.0, 61)
    meta = {
        "material": "KH2PO4",
        "wavelength_nm": 1064.0,
        "input_polarization": 90,
        "detected_polarization": 0,
        "crystal_orientation": "110",
        "rot/trans_axis": "001",
        "thickness_info": {"t_center_mm": 0.5},
        "beam_r_x": 50.0,
        "beam_r_y": 50.0,
    }
    data = pd.DataFrame(
        {
            "position": theta,
            "position_centered": theta,
            "intensity_corrected": np.ones_like(theta),
        }
    )
    return Jerphagnon1970Strategy(SimpleNamespace(meta=meta, data=data))


def test_kdp_110_input_index_is_constant_ordinary_index():
    strategy = _kdp_strategy()
    theta = strategy.analysis.data["position"].to_numpy()

    n_input = strategy.n_eff(90, 1064.0, theta)
    roles = strategy.n_eff(90, 1064.0, 0.0, aux=True)

    assert roles["n_cut"] == roles["n_third"]
    assert np.allclose(n_input, roles["n_cut"])


def test_kdp_110_maker_fringes_are_finite():
    strategy = _kdp_strategy()

    fringes = strategy._maker_fringes()

    assert fringes.shape == strategy.analysis.data["position"].shape
    assert np.all(np.isfinite(fringes))
