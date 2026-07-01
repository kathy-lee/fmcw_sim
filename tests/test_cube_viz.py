import matplotlib
matplotlib.use("Agg")  # must precede pyplot import

import numpy as np
from matplotlib.figure import Figure

from processing.cube_viz import show_adc, show_rd_map, show_rad_slice


def _fake_cubes(nv=4, nc=8, nr=16, naz=181):
    rng = np.random.default_rng(0)
    return {
        "adc": (rng.standard_normal((nv, nc, nr))
                + 1j * rng.standard_normal((nv, nc, nr))).astype(np.complex64),
        "power": rng.standard_normal((nc, nr // 2)) ** 2,
        "rad": rng.standard_normal((naz, nc, nr // 2)) ** 2,
        "axes": {"range_m": np.linspace(0, 10, nr // 2),
                 "velocity_mps": np.linspace(-5, 5, nc),
                 "azimuth_deg": np.linspace(-90, 90, naz)},
        "meta": {"n_tx": 1, "n_rx": 4},
    }


def test_show_adc_returns_figure():
    assert isinstance(show_adc(_fake_cubes(), channel=1), Figure)


def test_show_rd_map_returns_figure():
    assert isinstance(show_rd_map(_fake_cubes()), Figure)


def test_show_rad_slice_returns_figure():
    assert isinstance(show_rad_slice(_fake_cubes(), doppler_bin=3), Figure)
