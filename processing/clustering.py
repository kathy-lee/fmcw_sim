"""Cluster CFAR detection cells into discrete target detections.

Public entry point: `cluster(mask, rd_db, method=...)`. Returns a list of
`Detection` objects. Two backends: connected-components (default) and DBSCAN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

_VALID_METHODS = {"cc", "dbscan"}


@dataclass
class Detection:
    range_bin: float
    doppler_bin: float
    peak_db: float
    cell_count: int


def cluster(
    mask: np.ndarray,
    rd_db: np.ndarray,
    method: str = "cc",
    min_cells: int = 1,
    connectivity: int = 8,
    eps: float = 1.5,
    min_samples: int = 2,
    axis_weights: tuple = (1.0, 1.0),
) -> list:
    """Cluster CFAR detection cells into Detection objects.

    Parameters used by each method:
      method="cc":     min_cells, connectivity
      method="dbscan": min_cells, eps, min_samples, axis_weights
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
