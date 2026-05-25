import os

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import

import numpy as np
import pytest

from processing.clustering import Detection
from processing.tracking import Track
from processing.viz import LayeredRDView
from collections import deque


def _make_view():
    return LayeredRDView(
        n_doppler=32,
        n_range=640,
        range_res_m=0.075,
        chirp_period_s=50e-6,
        lambda_m=3.9e-3,
    )


def test_view_construction_does_not_crash():
    v = _make_view()
    assert v is not None


def test_view_update_with_synthetic_data():
    v = _make_view()
    rd = np.full((32, 640), -60.0, dtype=np.float32)
    rd[16, 100] = 10.0
    mask = np.zeros((32, 640), dtype=bool)
    mask[16, 100] = True
    dets = [Detection(range_bin=100.0, doppler_bin=16.0, peak_db=10.0, cell_count=1)]
    t = Track(
        id=0,
        state=np.array([100.0, 1.0]),
        cov=np.diag([0.5, 4.0]),
        last_doppler_bin=16.0,
        hits=3, misses=0, age=3,
        status="confirmed",
        hit_history=deque([True, True, True], maxlen=5),
        history=deque([(98.0, 16.0), (99.0, 16.0), (100.0, 16.0)], maxlen=10),
    )
    v.update(rd, mask, dets, [t], frame_id=0)  # should not raise


def test_view_toggles_via_keypress_callbacks_dont_crash():
    v = _make_view()
    rd = np.full((32, 640), -60.0, dtype=np.float32)
    mask = np.zeros((32, 640), dtype=bool)
    v.update(rd, mask, [], [], frame_id=0)
    # Simulate the toggle keys via the public methods directly.
    v.toggle_layer("cfar")
    v.toggle_layer("clusters")
    v.toggle_layer("tracks")
    v.update(rd, mask, [], [], frame_id=1)  # should still not raise after toggles


def test_view_on_reset_callback_invoked():
    v = _make_view()
    called = [0]
    def on_reset():
        called[0] += 1
    v.on_reset(on_reset)
    v._invoke_reset()  # internal helper exposed for testing
    assert called[0] == 1


