"""Cluster CFAR detection cells into discrete target detections.

Public entry point: `cluster(mask, rd_db, method=...)`. Returns a list of
`Detection` objects. Two backends: connected-components (default) and DBSCAN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

_VALID_METHODS = {"cc", "dbscan", "peak"}


@dataclass
class Detection:
    range_bin: float
    doppler_bin: float
    peak_db: float
    cell_count: int


@dataclass
class ClusterObject:
    """A Cartesian object from point-cloud clustering (the tracker's input)."""
    x: float            # centroid, metres
    y: float
    v_radial: float     # mean radial velocity, m/s
    n_points: int       # points in the cluster
    extent_m: float     # spatial spread (2*std of |point - centroid|)


def cluster(
    mask: np.ndarray,
    rd_db: np.ndarray,
    method: str = "cc",
    min_cells: int = 1,
    connectivity: int = 8,
    eps: float = 1.5,
    min_samples: int = 2,
    axis_weights: tuple = (1.0, 1.0),
    peak_neighborhood: int = 3,
) -> list:
    """Cluster CFAR detection cells into Detection objects.

    Parameters used by each method:
      method="cc":     min_cells, connectivity
      method="dbscan": min_cells, eps, min_samples, axis_weights
      method="peak":   min_cells, peak_neighborhood
    Parameters not used by the chosen method are accepted but silently
    ignored, so a caller can switch methods without changing the call site.

    `axis_weights` semantics for DBSCAN: each index is *divided* by its weight
    before fitting, so larger weight de-emphasises that axis (smaller scaled
    distances). E.g., axis_weights=(1.0, 0.3) makes range gaps look 1/0.3=3.3x
    larger, splitting blocks that the default would merge.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"cluster.method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )
    if not mask.any():
        return []
    if method == "cc":
        return cluster_connected_components(
            mask, rd_db, min_cells=min_cells, connectivity=connectivity
        )
    if method == "dbscan":
        return cluster_dbscan(
            mask, rd_db, eps=eps, min_samples=min_samples,
            axis_weights=axis_weights, min_cells=min_cells,
        )
    if method == "peak":
        return cluster_peak_picking(
            mask, rd_db, min_cells=min_cells,
            peak_neighborhood=peak_neighborhood,
        )
    raise NotImplementedError(f"cluster method {method!r} not yet implemented")


def cluster_connected_components(
    mask: np.ndarray,
    rd_db: np.ndarray,
    min_cells: int = 1,
    connectivity: int = 8,
) -> list:
    if connectivity == 8:
        struct = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    labels, n_labels = scipy.ndimage.label(mask, structure=struct)
    if n_labels == 0:
        return []

    rd_lin = np.power(10.0, rd_db / 10.0)

    out = []
    for label_id in range(1, n_labels + 1):
        cells = labels == label_id
        n = int(cells.sum())
        if n < min_cells:
            continue
        # Power-weighted centroid in (doppler_bin, range_bin).
        d_idx, r_idx = np.where(cells)
        w = rd_lin[d_idx, r_idx]
        w_sum = w.sum()
        if w_sum <= 0:
            # Reachable only if every cell is exactly -inf dB (caller used -inf
            # as a no-data sentinel). Fall back to geometric centroid.
            d_c = float(d_idx.mean())
            r_c = float(r_idx.mean())
        else:
            d_c = float((d_idx * w).sum() / w_sum)
            r_c = float((r_idx * w).sum() / w_sum)
        peak_db = float(rd_db[d_idx, r_idx].max())
        out.append(Detection(
            range_bin=r_c,
            doppler_bin=d_c,
            peak_db=peak_db,
            cell_count=n,
        ))
    return out


def cluster_dbscan(
    mask: np.ndarray,
    rd_db: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 2,
    axis_weights: tuple = (1.0, 1.0),
    min_cells: int = 1,
) -> list:
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        raise ImportError(
            "DBSCAN clustering requires scikit-learn; install with "
            "`pip install scikit-learn`"
        ) from e

    d_idx, r_idx = np.where(mask)
    if len(d_idx) == 0:
        return []

    w_d, w_r = axis_weights
    points = np.column_stack([d_idx / w_d, r_idx / w_r])

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    rd_lin = np.power(10.0, rd_db / 10.0)

    out = []
    for label_id in sorted(set(labels)):
        if label_id == -1:
            continue  # DBSCAN noise
        sel = labels == label_id
        ds = d_idx[sel]
        rs = r_idx[sel]
        n = int(sel.sum())
        if n < min_cells:
            continue
        w = rd_lin[ds, rs]
        w_sum = w.sum()
        if w_sum <= 0:
            d_c = float(ds.mean())
            r_c = float(rs.mean())
        else:
            d_c = float((ds * w).sum() / w_sum)
            r_c = float((rs * w).sum() / w_sum)
        peak_db = float(rd_db[ds, rs].max())
        out.append(Detection(
            range_bin=r_c, doppler_bin=d_c,
            peak_db=peak_db, cell_count=n,
        ))
    return out


def cluster_peak_picking(
    mask: np.ndarray,
    rd_db: np.ndarray,
    min_cells: int = 1,
    peak_neighborhood: int = 3,
) -> list:
    """Peak-picking clustering: one detection per local maximum of `rd_db`.

    A cell is a peak if (a) it is in the mask and (b) its rd_db value equals
    the max within a `peak_neighborhood`-sized window centred on it. Non-mask
    cells are suppressed so adjacent bright-but-rejected cells can't hide a
    masked peak.

    Each emitted Detection's `cell_count` is the size of the connected-component
    blob the peak belongs to; the `min_cells` filter operates on that blob size,
    so a single isolated CFAR-positive cell can be rejected via min_cells>=2 just
    like with the "cc" method.

    Unlike `cc` and `dbscan`, this method uses the **intensity surface** under
    the mask, not just mask topology. Two CFAR-positive cells whose blob has
    two local maxima will produce two detections — which is the property that
    resolves the close-target merge problem CC and DBSCAN suffer from.
    """
    if peak_neighborhood < 1:
        raise ValueError(
            f"peak_neighborhood must be >= 1, got {peak_neighborhood}"
        )

    rd_masked = np.where(mask, rd_db, -np.inf)
    # Strict local max: the CUT is strictly greater than every other cell in
    # its neighbourhood. Excluding the centre from the footprint sidesteps
    # plateau ambiguity — flat regions never qualify as peaks.
    footprint = np.ones((peak_neighborhood, peak_neighborhood), dtype=bool)
    centre = peak_neighborhood // 2
    footprint[centre, centre] = False
    max_neighbour = scipy.ndimage.maximum_filter(
        rd_masked, footprint=footprint, mode="nearest"
    )
    is_peak = (rd_masked > max_neighbour) & mask

    if not is_peak.any():
        return []

    struct8 = np.ones((3, 3), dtype=bool)
    # Underlying blob labels of the mask, for cell_count and min_cells filter.
    blob_labels, n_blobs = scipy.ndimage.label(mask, structure=struct8)
    if n_blobs == 0:
        return []
    blob_sizes = np.bincount(blob_labels.ravel())  # blob_sizes[0] = background

    # A plateau of equal-valued cells all pass the (rd == max) test. Group
    # them into one detection by labelling the peak mask and emitting the
    # centroid of each peak-region.
    peak_labels, n_peaks = scipy.ndimage.label(is_peak, structure=struct8)

    out = []
    for pid in range(1, n_peaks + 1):
        pd_idx, pr_idx = np.where(peak_labels == pid)
        if pd_idx.size == 0:
            continue
        blob_id = int(blob_labels[pd_idx[0], pr_idx[0]])
        n = int(blob_sizes[blob_id])
        if n < min_cells:
            continue
        d_c = float(pd_idx.mean())
        r_c = float(pr_idx.mean())
        peak_val = float(rd_db[pd_idx, pr_idx].max())
        out.append(Detection(
            range_bin=r_c,
            doppler_bin=d_c,
            peak_db=peak_val,
            cell_count=n,
        ))
    return out


def cluster_point_cloud(
    points: np.ndarray,
    method: str = "dbscan",
    eps_m: float = 0.8,
    min_samples: int = 2,
    vel_scale: float = 1.0,
) -> list:
    """Cluster a Cartesian point cloud into ClusterObject centroids.

    `points`: (M, >=3) array of [x, y, v_radial, ...] (the MIMO point cloud).
    Velocity is folded into the metric as `vel_scale * v_radial`, so objects at
    the same position but different radial velocity (crossing / overtaking)
    separate; `vel_scale = 0` clusters on (x, y) only. `eps_m` is the DBSCAN
    radius in metres over the (x, y, vel_scale*v) space. Mirrors `cluster()` for
    the RD grid; the grid methods (peak/cc/dbscan there) do not apply to a
    scattered point cloud, so DBSCAN is the point-cloud method.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return []
    if method != "dbscan":
        raise ValueError(
            f"cluster_point_cloud.method must be 'dbscan', got {method!r}"
        )
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        raise ImportError(
            "DBSCAN clustering requires scikit-learn; install with "
            "`pip install scikit-learn`"
        ) from e

    xy = points[:, :2]
    v = points[:, 2] if points.shape[1] >= 3 else np.zeros(len(points))
    feat = np.column_stack([xy, vel_scale * v])
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(feat)

    out = []
    for label_id in sorted(set(labels)):
        if label_id == -1:
            continue  # DBSCAN noise
        sel = labels == label_id
        cx, cy = xy[sel].mean(axis=0)
        dist = np.hypot(xy[sel, 0] - cx, xy[sel, 1] - cy)
        out.append(ClusterObject(
            x=float(cx), y=float(cy), v_radial=float(v[sel].mean()),
            n_points=int(sel.sum()), extent_m=2.0 * float(dist.std()),
        ))
    return out
