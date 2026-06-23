"""M3 (M3a): Cartesian point-cloud clustering (DBSCAN on x, y, v_radial)."""
import numpy as np

from processing.clustering import ClusterObject, cluster_point_cloud


def test_collapses_single_target():
    pts = np.array([[8.0, 2.0, -0.8], [8.05, 2.02, -0.85],
                    [7.96, 1.98, -0.82], [8.02, 2.01, -0.79]])
    objs = cluster_point_cloud(pts, eps_m=0.5, min_samples=2)
    assert len(objs) == 1
    assert isinstance(objs[0], ClusterObject)
    assert abs(objs[0].x - 8.0) < 0.1 and abs(objs[0].y - 2.0) < 0.1
    assert objs[0].n_points == 4


def test_separates_two_targets():
    a = np.array([[8, 2, -0.8], [8.05, 2.0, -0.8], [7.97, 2.02, -0.8]])
    b = np.array([[3, -1, 2.0], [3.04, -1.02, 2.0], [2.98, -0.97, 2.0]])
    objs = cluster_point_cloud(np.vstack([a, b]), eps_m=0.5, min_samples=2)
    assert len(objs) == 2


def test_velocity_feature_separates_same_position():
    slow = np.array([[5, 0, 0.0], [5.05, 0.02, 0.05], [4.97, -0.01, -0.03]])
    fast = np.array([[5, 0, 6.0], [5.04, 0.01, 6.05], [4.98, 0.02, 5.97]])
    pts = np.vstack([slow, fast])
    assert len(cluster_point_cloud(pts, eps_m=0.5, min_samples=2,
                                   vel_scale=1.0)) == 2     # velocity splits
    assert len(cluster_point_cloud(pts, eps_m=0.5, min_samples=2,
                                   vel_scale=0.0)) == 1     # spatial-only merges


def test_isolated_noise_dropped():
    blob = np.array([[8, 2, -0.8], [8.05, 2.0, -0.8], [7.97, 2.02, -0.8]])
    noise = np.array([[20.0, 15.0, 3.0]])
    objs = cluster_point_cloud(np.vstack([blob, noise]), eps_m=0.5, min_samples=2)
    assert len(objs) == 1


def test_empty_cloud_returns_empty():
    assert cluster_point_cloud(np.empty((0, 3)), eps_m=0.5) == []
