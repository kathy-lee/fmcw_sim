import numpy as np

from processing.range_doppler import range_doppler


def _frame(n_chirps=16, n_range=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_chirps, n_range))
            + 1j * rng.standard_normal((n_chirps, n_range))).astype(np.complex64)


def test_default_return_unchanged():
    frame = _frame()
    out = range_doppler(frame)
    assert out.shape == (16, 32)
    assert np.iscomplexobj(out)


def test_return_range_cube_shapes():
    frame = _frame()
    out, rfft = range_doppler(frame, return_range_cube=True)
    assert out.shape == (16, 32)
    assert rfft.shape == (16, 32)
    assert np.iscomplexobj(rfft)


def test_out_identical_with_and_without_flag():
    frame = _frame(seed=3)
    out_only = range_doppler(frame)
    out_pair, _ = range_doppler(frame, return_range_cube=True)
    np.testing.assert_array_equal(out_only, out_pair)
