"""CFAR threshold under MIMO non-coherent integration.

After NCI of V channels the per-cell noise is Gamma(V), not exponential, so the
threshold multiplier alpha must be re-calibrated (not scaled). These tests verify
that (a) the recalibrated alpha differs from the single-channel one, and (b) the
realized false-alarm rate on the integrated map stays near the design Pfa, while
the single-channel alpha applied to integrated noise is too conservative.
"""
import numpy as np

from processing.cfar import _alpha_empirical, _windowed_noise_rd_maps, cfar_2d

_SHAPE = (32, 64)
_GUARD, _TRAIN = (2, 4), (4, 8)
_PAD_D, _PAD_R = 6, 12          # G+T per axis: trim the edge-replicated border


def _fa_rate(maps, method, pfa, n_integrate):
    rates = []
    for m in maps:
        mask = cfar_2d(10.0 * np.log10(m + 1e-12), method=method, guard=_GUARD,
                       training=_TRAIN, pfa=pfa, n_integrate=n_integrate)
        rates.append(mask[_PAD_D:-_PAD_D, _PAD_R:-_PAD_R].mean())
    return float(np.mean(rates))


def test_nci_alpha_differs_from_single_channel():
    kw = dict(window="hann", dc_removal=True, clutter_removal=False)
    a1 = _alpha_empirical("ca", _SHAPE, _GUARD, _TRAIN, 1e-3, *kw.values(),
                          n_integrate=1)
    a4 = _alpha_empirical("ca", _SHAPE, _GUARD, _TRAIN, 1e-3, *kw.values(),
                          n_integrate=4)
    assert a4 != a1
    assert a4 < a1          # Gamma(V) has lighter tails -> lower multiplier


def test_nci_cfar_recalibration_keeps_pfa_nominal():
    V, pfa = 4, 1e-3
    # out-of-sample integrated noise (calibration uses seed=0)
    maps = _windowed_noise_rd_maps(_SHAPE, "hann", True, False, 150, 12345, V)
    rate_corr = _fa_rate(maps, "ca", pfa, n_integrate=V)     # recalibrated
    rate_uncorr = _fa_rate(maps, "ca", pfa, n_integrate=1)   # single-channel alpha
    assert 0.3 * pfa < rate_corr < 3.0 * pfa                 # near nominal
    assert rate_uncorr < rate_corr                           # uncorrected too low
