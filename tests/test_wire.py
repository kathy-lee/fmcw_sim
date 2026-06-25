"""Single-sourced wire protocol (header schema + (de)serialize)."""
import numpy as np

from simulator.wire import build_header, deserialize_frame, frame_payload


def test_build_header_single_channel_schema():
    frame = np.zeros((64, 256), dtype=np.complex64)
    h = build_header(frame=frame, frame_id=3, topic="fmcw",
                     metadata={"config": {}}, ground_truth=[],
                     timestamp_unix_s=123.0)
    assert h["frame_id"] == 3
    assert h["dtype"] == "complex64"
    assert h["shape"] == [64, 256]
    assert h["order"] == "C"
    assert h["topic"] == "fmcw"


def test_build_header_mimo_shape_is_3d():
    frame = np.zeros((12, 64, 256), dtype=np.complex64)
    h = build_header(frame=frame, frame_id=0, topic="fmcw",
                     metadata={}, ground_truth=[], timestamp_unix_s=0.0)
    assert h["shape"] == [12, 64, 256]      # consumers see V via len(shape) == 3


def test_round_trip_single_channel():
    rng = np.random.default_rng(0)
    frame = (rng.standard_normal((64, 256))
             + 1j * rng.standard_normal((64, 256))).astype(np.complex64)
    h = build_header(frame=frame, frame_id=0, topic="t",
                     metadata={}, ground_truth=[], timestamp_unix_s=0.0)
    out = deserialize_frame(h, frame_payload(frame))
    np.testing.assert_array_equal(out, frame)


def test_round_trip_mimo():
    rng = np.random.default_rng(1)
    frame = (rng.standard_normal((12, 64, 256))
             + 1j * rng.standard_normal((12, 64, 256))).astype(np.complex64)
    h = build_header(frame=frame, frame_id=7, topic="t",
                     metadata={}, ground_truth=[], timestamp_unix_s=0.0)
    out = deserialize_frame(h, frame_payload(frame))
    assert out.shape == (12, 64, 256)
    np.testing.assert_array_equal(out, frame)


# --- C2: full publish round-trip for a MIMO frame ---

from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, RadarConfig, Target)


def test_publish_round_trip_reconstructs_mimo_frame():
    gen = FMCWBeatSignalGenerator(RadarConfig(n_tx=3, n_rx=4))
    targets = [Target(x=8.0, y=2.0, vx=-1.0, vy=0.5, motion_model="cv_xy",
                      amplitude=0.01)]
    frame = gen.generate_frame_mimo(targets, frame_index=5)
    header = build_header(frame=frame, frame_id=5, topic="fmcw",
                          metadata=gen.metadata(),
                          ground_truth=gen.ground_truth(targets, frame_index=5),
                          timestamp_unix_s=0.0)
    out = deserialize_frame(header, frame_payload(frame))
    np.testing.assert_array_equal(out, frame)
    assert header["metadata"]["config"]["n_tx"] == 3
    assert "azimuth_deg" in header["ground_truth"][0]
