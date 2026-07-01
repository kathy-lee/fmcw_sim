"""CSV writers for per-frame detections and tracks.

Each writer opens a file on construction, writes a header row, and appends
one row per detection / track per call to `write(frame_id, ...)`.

Detection and track rows can be joined on (frame_id, det_idx ↔ assoc_det_idx).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple

from processing.clustering import Detection
from processing.tracking import Track
from processing.layered_rd_view import doppler_bin_to_mps


class DetectionWriter:
    _FIELDS = ["frame_id", "det_idx", "range_bin", "doppler_bin",
               "range_m", "velocity_mps", "peak_db", "cell_count"]

    def __init__(self, path: Path, range_res_m: float, chirp_period_s: float,
                 lambda_m: float, n_doppler: int):
        self.range_res_m = range_res_m
        self.chirp_period_s = chirp_period_s
        self.lambda_m = lambda_m
        self.n_doppler = n_doppler
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self._FIELDS)

    def write(self, frame_id: int, detections: Iterable[Detection]) -> None:
        for di, d in enumerate(detections):
            v = doppler_bin_to_mps(d.doppler_bin, self.n_doppler,
                                   self.chirp_period_s, self.lambda_m)
            self._w.writerow([
                frame_id, di,
                f"{d.range_bin:.3f}", f"{d.doppler_bin:.3f}",
                f"{d.range_bin * self.range_res_m:.3f}", f"{v:.3f}",
                f"{d.peak_db:.3f}", d.cell_count,
            ])

    def close(self) -> None:
        if self._f is not None and not self._f.closed:
            self._f.close()


def open_csv_writers(
    save_dir: Optional[Path],
    range_res_m: float,
    chirp_period_s: float,
    lambda_m: float,
    n_doppler: int,
) -> Tuple[Optional["DetectionWriter"], Optional["TrackWriter"]]:
    """Construct (DetectionWriter, TrackWriter) under save_dir, or (None, None)."""
    if save_dir is None:
        return None, None
    det = DetectionWriter(save_dir / "detections.csv", range_res_m,
                          chirp_period_s, lambda_m, n_doppler)
    trk = TrackWriter(save_dir / "tracks.csv", range_res_m,
                      chirp_period_s, lambda_m, n_doppler)
    return det, trk


class TrackWriter:
    _FIELDS = ["frame_id", "track_id", "range_bin", "doppler_bin",
               "range_m", "velocity_mps", "status", "hits", "misses", "age",
               "assoc_det_idx"]

    def __init__(self, path: Path, range_res_m: float, chirp_period_s: float,
                 lambda_m: float, n_doppler: int):
        self.range_res_m = range_res_m
        self.chirp_period_s = chirp_period_s
        self.lambda_m = lambda_m
        self.n_doppler = n_doppler
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self._FIELDS)

    def write(self, frame_id: int, tracks: Iterable[Track]) -> None:
        for t in tracks:
            range_bin = float(t.state[0])
            v = doppler_bin_to_mps(t.last_doppler_bin, self.n_doppler,
                                   self.chirp_period_s, self.lambda_m)
            self._w.writerow([
                frame_id, t.id,
                f"{range_bin:.3f}", f"{t.last_doppler_bin:.3f}",
                f"{range_bin * self.range_res_m:.3f}", f"{v:.3f}",
                t.status, t.hits, t.misses, t.age,
                "" if t.assoc_det_idx is None else t.assoc_det_idx,
            ])

    def close(self) -> None:
        if self._f is not None and not self._f.closed:
            self._f.close()
