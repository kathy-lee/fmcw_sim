"""Wire protocol for the streaming beat-signal transport (single-sourced).

The publisher and all consumers share this header schema and (de)serialization, so
the on-wire format is defined in exactly one place. It is shape-polymorphic:
single-channel frames are 2-D (n_chirps, n_range); MIMO frames are 3-D
(n_virtual, n_chirps, n_range). Consumers disambiguate via len(header["shape"]) and
metadata.config.n_tx / n_rx. The single-channel header+payload bytes are unchanged
from the pre-MIMO publisher (only the timestamp varies between runs).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def build_header(*, frame: np.ndarray, frame_id: int, topic: str,
                 metadata: dict, ground_truth: list,
                 timestamp_unix_s: float) -> dict[str, Any]:
    """Assemble the JSON-serializable frame header.

    `shape` follows the frame's own shape, so single-channel (2-D) and MIMO (3-D)
    share the same schema with no format break.
    """
    return {
        "frame_id": frame_id,
        "timestamp_unix_s": timestamp_unix_s,
        "topic": topic,
        "dtype": str(frame.dtype),
        "shape": list(frame.shape),
        "order": "C",
        "metadata": metadata,
        "ground_truth": ground_truth,
    }


def frame_payload(frame: np.ndarray) -> bytes:
    """Serialize a frame to bytes (C order), matching header['shape']/['dtype']."""
    return frame.ravel(order="C").tobytes()


def deserialize_frame(header: dict, payload: bytes) -> np.ndarray:
    """Reconstruct a frame from header + payload (inverse of frame_payload)."""
    shape = tuple(header["shape"])
    dtype = np.dtype(header["dtype"])
    return np.frombuffer(payload, dtype=dtype).reshape(shape)
