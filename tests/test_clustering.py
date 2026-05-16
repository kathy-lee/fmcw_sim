import numpy as np
import pytest

from processing.clustering import Detection, cluster
from _synth import make_noise_frame


def _empty_mask(shape=(32, 640)):
    return np.zeros(shape, dtype=bool)


def _frame_db_with_blocks(blocks, shape=(32, 640), bg_db=-60.0, peak_db=10.0):
    """Build a dB frame with rectangular bright blocks for centroid tests."""
    frame = np.full(shape, bg_db, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    for d0, d1, r0, r1 in blocks:
        frame[d0:d1, r0:r1] = peak_db
        mask[d0:d1, r0:r1] = True
    return frame, mask


def test_cc_returns_empty_list_on_empty_mask():
    frame = make_noise_frame()
    out = cluster(_empty_mask(), frame, method="cc")
    assert out == []


def test_cc_single_3x3_block_one_detection_centroid():
    frame, mask = _frame_db_with_blocks([(15, 18, 99, 102)])
    out = cluster(mask, frame, method="cc")
    assert len(out) == 1
    det = out[0]
    assert isinstance(det, Detection)
    # Centroid = mean of (15,16,17) and (99,100,101) = (16, 100)
    assert det.doppler_bin == pytest.approx(16.0, abs=1e-6)
    assert det.range_bin == pytest.approx(100.0, abs=1e-6)
    assert det.cell_count == 9
    assert det.peak_db == pytest.approx(10.0, abs=1e-6)


def test_cc_two_separated_blocks_two_detections():
    frame, mask = _frame_db_with_blocks(
        [(15, 18, 99, 102), (10, 13, 399, 402)]
    )
    out = cluster(mask, frame, method="cc")
    assert len(out) == 2
    centroids = sorted([(d.doppler_bin, d.range_bin) for d in out])
    assert centroids[0] == pytest.approx((11.0, 400.0), abs=1e-6)
    assert centroids[1] == pytest.approx((16.0, 100.0), abs=1e-6)


def test_cc_corner_touching_blocks_8conn_merges():
    frame, mask = _frame_db_with_blocks([(15, 18, 99, 102), (18, 21, 102, 105)])
    out_8 = cluster(mask, frame, method="cc", connectivity=8)
    out_4 = cluster(mask, frame, method="cc", connectivity=4)
    assert len(out_8) == 1, "8-connectivity merges corner-touching blocks"
    assert len(out_4) == 2, "4-connectivity keeps corner-touching blocks separate"


def test_cc_power_weighted_centroid_biased_toward_brighter_corner():
    """3x3 block, top-left corner 20 dB brighter → centroid pulled toward it."""
    frame = np.full((32, 640), -60.0, dtype=np.float32)
    mask = np.zeros((32, 640), dtype=bool)
    # 3x3 block centered at (16, 100): rows 15..17, cols 99..101
    for d in range(15, 18):
        for r in range(99, 102):
            frame[d, r] = 10.0
            mask[d, r] = True
    frame[15, 99] = 30.0   # top-left corner is 20 dB hotter

    out = cluster(mask, frame, method="cc")
    assert len(out) == 1
    det = out[0]
    # Geometric centroid would be (16, 100). Power-weighted is pulled toward (15, 99).
    assert det.doppler_bin < 16.0
    assert det.range_bin < 100.0


def test_cluster_rejects_unknown_method():
    frame = make_noise_frame()
    with pytest.raises(ValueError, match="method"):
        cluster(_empty_mask(), frame, method="bogus")


def test_cluster_min_cells_filter():
    frame = np.full((32, 640), -60.0, dtype=np.float32)
    mask = np.zeros((32, 640), dtype=bool)
    mask[15, 100] = True   # singleton
    mask[10, 200] = True
    mask[10, 201] = True
    mask[10, 202] = True   # 1x3 block of 3 cells
    out = cluster(mask, frame, method="cc", min_cells=2)
    assert len(out) == 1
    assert out[0].cell_count == 3


def test_dbscan_returns_one_cluster_for_3x3_block():
    frame, mask = _frame_db_with_blocks([(15, 18, 99, 102)])
    out = cluster(mask, frame, method="dbscan", eps=1.5, min_samples=2)
    assert len(out) == 1
    det = out[0]
    assert det.doppler_bin == pytest.approx(16.0, abs=1e-6)
    assert det.range_bin == pytest.approx(100.0, abs=1e-6)


def test_dbscan_two_separated_blocks_two_clusters():
    frame, mask = _frame_db_with_blocks(
        [(15, 18, 99, 102), (10, 13, 399, 402)]
    )
    out = cluster(mask, frame, method="dbscan", eps=1.5, min_samples=2)
    assert len(out) == 2


def test_dbscan_with_axis_weights_separates_close_blocks():
    """Two blocks 6 range bins apart, same doppler. With heavy range weight, separate."""
    frame, mask = _frame_db_with_blocks(
        [(15, 18, 100, 103), (15, 18, 109, 112)]
    )
    # Default weights: blocks may merge if eps>=3. Make range axis "longer" so the
    # gap looks larger to DBSCAN.
    out_merged = cluster(mask, frame, method="dbscan", eps=1.5,
                         min_samples=2, axis_weights=(1.0, 1.0))
    out_split = cluster(mask, frame, method="dbscan", eps=1.5,
                        min_samples=2, axis_weights=(1.0, 0.3))
    assert len(out_split) >= len(out_merged)


def test_dbscan_raises_helpful_error_when_sklearn_missing(monkeypatch):
    """Simulate missing sklearn; cluster(method='dbscan') should raise ImportError."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.cluster" or name.startswith("sklearn"):
            raise ImportError("No module named sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    frame, mask = _frame_db_with_blocks([(15, 18, 99, 102)])
    with pytest.raises(ImportError, match="scikit-learn"):
        cluster(mask, frame, method="dbscan")
