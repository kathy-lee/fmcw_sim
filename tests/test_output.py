"""Tests for DetectionWriter / TrackWriter CSV output."""

from __future__ import annotations

import csv
from collections import deque

import numpy as np
import pytest

from processing.clustering import Detection
from processing.output import DetectionWriter, TrackWriter
from processing.tracking import Track


def _make_track(track_id: int, range_bin: float, doppler_bin: float,
                status: str = "tentative", assoc_det_idx=None) -> Track:
    return Track(
        id=track_id,
        state=np.array([range_bin, 0.0]),
        cov=np.eye(2),
        last_doppler_bin=doppler_bin,
        hits=1,
        misses=0,
        age=1,
        status=status,
        hit_history=deque([True], maxlen=5),
        history=deque([(range_bin, doppler_bin)], maxlen=10),
        assoc_det_idx=assoc_det_idx,
    )


def test_detection_writer_emits_header_and_row(tmp_path):
    path = tmp_path / "detections.csv"
    w = DetectionWriter(path, range_res_m=0.25, chirp_period_s=1e-4,
                        lambda_m=0.004, n_doppler=64)
    w.write(frame_id=7, detections=[
        Detection(range_bin=10.0, doppler_bin=36.0, peak_db=15.0, cell_count=3),
    ])
    w.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert r["frame_id"] == "7"
    assert r["det_idx"] == "0"
    assert float(r["range_bin"]) == pytest.approx(10.0)
    assert float(r["doppler_bin"]) == pytest.approx(36.0)
    # range_m = range_bin * range_res_m
    assert float(r["range_m"]) == pytest.approx(2.5)
    # velocity_mps = (doppler_bin - n_doppler/2) / (n_doppler * chirp_period_s) * lambda_m / 2
    # = (36 - 32) / (64 * 1e-4) * 0.004 / 2 = 4 / 6.4e-3 * 0.002 = 1.25
    assert float(r["velocity_mps"]) == pytest.approx(1.25)
    assert float(r["peak_db"]) == pytest.approx(15.0)
    assert int(r["cell_count"]) == 3


def test_detection_writer_assigns_sequential_det_idx_per_frame(tmp_path):
    path = tmp_path / "detections.csv"
    w = DetectionWriter(path, range_res_m=1.0, chirp_period_s=1.0,
                        lambda_m=1.0, n_doppler=4)
    w.write(frame_id=0, detections=[
        Detection(range_bin=1.0, doppler_bin=2.0, peak_db=0.0, cell_count=1),
        Detection(range_bin=5.0, doppler_bin=3.0, peak_db=0.0, cell_count=1),
    ])
    w.write(frame_id=1, detections=[
        Detection(range_bin=2.0, doppler_bin=2.0, peak_db=0.0, cell_count=1),
    ])
    w.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["frame_id"], r["det_idx"]) for r in rows] == [
        ("0", "0"), ("0", "1"), ("1", "0"),
    ]


def test_detection_writer_empty_frame_writes_no_rows(tmp_path):
    path = tmp_path / "detections.csv"
    w = DetectionWriter(path, range_res_m=1.0, chirp_period_s=1.0,
                        lambda_m=1.0, n_doppler=4)
    w.write(frame_id=0, detections=[])
    w.close()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_track_writer_emits_status_and_assoc_det_idx(tmp_path):
    path = tmp_path / "tracks.csv"
    w = TrackWriter(path, range_res_m=0.25, chirp_period_s=1e-4,
                    lambda_m=0.004, n_doppler=64)
    tracks = [
        _make_track(0, range_bin=10.0, doppler_bin=36.0,
                    status="confirmed", assoc_det_idx=1),
        _make_track(1, range_bin=20.0, doppler_bin=32.0,
                    status="coasting", assoc_det_idx=None),
    ]
    w.write(frame_id=7, tracks=tracks)
    w.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["frame_id"] == "7"
    assert rows[0]["track_id"] == "0"
    assert rows[0]["status"] == "confirmed"
    assert rows[0]["assoc_det_idx"] == "1"
    assert float(rows[0]["range_m"]) == pytest.approx(2.5)
    assert float(rows[0]["velocity_mps"]) == pytest.approx(1.25)
    # Coasting: assoc_det_idx blank.
    assert rows[1]["track_id"] == "1"
    assert rows[1]["status"] == "coasting"
    assert rows[1]["assoc_det_idx"] == ""


def test_track_writer_empty_frame_writes_no_rows(tmp_path):
    path = tmp_path / "tracks.csv"
    w = TrackWriter(path, range_res_m=1.0, chirp_period_s=1.0,
                    lambda_m=1.0, n_doppler=4)
    w.write(frame_id=0, tracks=[])
    w.close()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_open_csv_writers_returns_none_when_save_dir_is_none():
    from processing.output import open_csv_writers
    det_w, trk_w = open_csv_writers(None, range_res_m=1.0, chirp_period_s=1.0,
                                    lambda_m=1.0, n_doppler=4)
    assert det_w is None and trk_w is None


def test_open_csv_writers_creates_files_under_save_dir(tmp_path):
    from processing.output import open_csv_writers
    det_w, trk_w = open_csv_writers(tmp_path, range_res_m=1.0,
                                    chirp_period_s=1.0, lambda_m=1.0,
                                    n_doppler=4)
    try:
        assert (tmp_path / "detections.csv").exists()
        assert (tmp_path / "tracks.csv").exists()
        assert det_w is not None and trk_w is not None
    finally:
        det_w.close()
        trk_w.close()
