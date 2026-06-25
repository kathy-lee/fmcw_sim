"""Tests of direction-of-arrival estimators.

Convention: synthesize a snapshot from known azimuths via the steering vector,
then assert the estimator recovers them — isolates doa.py from the full pipeline.
Element positions: lambda/2 ULA, 12 elements (pos = 0.5 * arange(12)).
"""
import numpy as np

from processing.doa import estimate_doa

POS = 0.5 * np.arange(12)


def _snapshot(azimuths_deg, amps=None):
    az = np.deg2rad(np.atleast_1d(azimuths_deg).astype(float))
    amps = np.ones(len(az)) if amps is None else np.asarray(amps, dtype=complex)
    A = np.exp(1j * 2 * np.pi * np.outer(POS, np.sin(az)))   # (12, K)
    return A @ amps


def test_bartlett_recovers_single_azimuth():
    est = estimate_doa(_snapshot(20.0), POS, method="bartlett", n_sources=1)
    assert abs(est[0] - 20.0) < 2.0


def test_bartlett_boresight():
    est = estimate_doa(_snapshot(0.0), POS, method="bartlett", n_sources=1)
    assert abs(est[0]) < 2.0


def test_capon_recovers_single_azimuth():
    est = estimate_doa(_snapshot(-12.0), POS, method="capon", n_sources=1)
    assert abs(est[0] - (-12.0)) < 2.0


def test_music_recovers_single_azimuth():
    est = estimate_doa(_snapshot(8.0), POS, method="music", n_sources=1)
    assert abs(est[0] - 8.0) < 1.5


def test_music_resolves_two_close_targets():
    # Two coherent sources 7 deg apart — below the smoothed 8-element subarray's
    # Rayleigh limit (~12.7 deg). MUSIC + FB smoothing resolves both. This is the
    # unit-level check of the primitive; the Bartlett-vs-super-resolution contrast
    # is quantified statistically in test_doa_resolution_superres_beats_bartlett
    est = sorted(estimate_doa(_snapshot([-3.0, 4.0]), POS,
                              method="music", n_sources=2))
    assert len(est) == 2
    assert abs(est[0] - (-3.0)) < 2.0 and abs(est[1] - 4.0) < 2.0


def test_esprit_recovers_single_azimuth_gridless():
    est = estimate_doa(_snapshot(15.0), POS, method="esprit", n_sources=1)
    assert abs(est[0] - 15.0) < 1.0          # gridless -> tighter than 1 deg


def test_esprit_resolves_two_close_targets():
    est = sorted(estimate_doa(_snapshot([-3.0, 4.0]), POS,
                              method="esprit", n_sources=2))
    assert len(est) == 2
    assert abs(est[0] - (-3.0)) < 2.0 and abs(est[1] - 4.0) < 2.0
