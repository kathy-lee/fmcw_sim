import pytest

from processing.config import CfarConfig, ClusteringConfig, TrackerConfig


# --- CfarConfig ---

def test_cfar_default_constructs_without_error():
    cfg = CfarConfig()
    assert cfg.method == "ca"


def test_cfar_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        CfarConfig(method="nonsense")


@pytest.mark.parametrize("bad_pfa", [0.0, -0.1, 1.0, 2.0])
def test_cfar_rejects_bad_pfa(bad_pfa):
    with pytest.raises(ValueError, match="pfa"):
        CfarConfig(pfa=bad_pfa)


@pytest.mark.parametrize("bad_rank_frac", [0.0, 1.0, -0.5, 1.5])
def test_cfar_rejects_bad_rank_frac(bad_rank_frac):
    with pytest.raises(ValueError, match="rank_frac"):
        CfarConfig(rank_frac=bad_rank_frac)


@pytest.mark.parametrize("bad_guard", [
    (2,),               # wrong arity
    (2, 3, 4),          # wrong arity
    (2, -1),            # negative
    [2, 4],             # not a tuple
    (2.0, 4),           # not int
])
def test_cfar_rejects_bad_guard(bad_guard):
    with pytest.raises(ValueError, match="guard"):
        CfarConfig(guard=bad_guard)


# --- ClusteringConfig ---

def test_clustering_default_constructs_without_error():
    cfg = ClusteringConfig()
    assert cfg.method == "cc"


def test_clustering_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        ClusteringConfig(method="kmeans")


def test_clustering_rejects_bad_connectivity():
    with pytest.raises(ValueError, match="connectivity"):
        ClusteringConfig(connectivity=6)


def test_clustering_rejects_zero_min_cells():
    with pytest.raises(ValueError, match="min_cells"):
        ClusteringConfig(min_cells=0)


def test_clustering_rejects_non_positive_eps():
    with pytest.raises(ValueError, match="eps"):
        ClusteringConfig(eps=0.0)


# --- TrackerConfig ---

def test_tracker_default_constructs_without_error():
    cfg = TrackerConfig()
    assert cfg.method == "gnn"


def test_tracker_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        TrackerConfig(method="particle")


def test_tracker_rejects_M_greater_than_N():
    with pytest.raises(ValueError, match="confirm_M"):
        TrackerConfig(confirm_M=6, confirm_N=5)


@pytest.mark.parametrize("field, value", [
    ("gate_threshold", 0.0),
    ("process_noise_q", -1.0),
    ("range_meas_var", 0.0),
    ("doppler_meas_var", -0.1),
    ("init_velocity_var", 0.0),
])
def test_tracker_rejects_non_positive_variance(field, value):
    with pytest.raises(ValueError, match=field):
        TrackerConfig(**{field: value})


@pytest.mark.parametrize("bad_beta", [0.0, -0.1, 1.5])
def test_tracker_rejects_bad_doppler_smoothing_beta(bad_beta):
    with pytest.raises(ValueError, match="doppler_smoothing_beta"):
        TrackerConfig(doppler_smoothing_beta=bad_beta)


def test_tracker_rejects_negative_max_coast():
    with pytest.raises(ValueError, match="max_coast"):
        TrackerConfig(max_coast=-1)


def test_tracker_rejects_zero_history_len():
    with pytest.raises(ValueError, match="history_len"):
        TrackerConfig(history_len=0)
