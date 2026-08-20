"""Extract and export the front-end cube representations.

extract_cubes runs the same primitives the detection pipeline uses (quantize ->
range_doppler -> integrate) plus the dense RAD beamformer, returning the
requested cubes as a dict. save_cubes writes one .npz per frame.

L1 `adc` is the Q_adc quantized frame when `fp` is set (what the downstream cubes
derive from, so the .npz is internally reproducible), the float frame otherwise.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from processing.doa import _GRID_DEG
from processing.fixed_point import quantize
from processing.mimo_frame import frame_params
from processing.point_cloud import integrate
from processing.rad import rad_bartlett
from processing.range_doppler import range_doppler

LEVELS = ("adc", "range_cube", "rd_cube", "power", "rad")


def extract_cubes(frame, header, fp=None, levels=LEVELS):
    """Return the requested cube tensors + axes + meta for one MIMO frame."""
    unknown = set(levels) - set(LEVELS)
    if unknown:
        raise ValueError(f"unknown levels {sorted(unknown)}; valid: {LEVELS}")

    p = frame_params(header)
    if fp is not None:                       # Q_adc (digitizer; fixed scale)
        adc = quantize(frame, fp.adc_bits, fp.adc_full_scale, is_complex=True,
                       rounding=fp.rounding, overflow=fp.overflow)
    else:
        adc = np.asarray(frame)

    out = {}
    if "adc" in levels:
        out["adc"] = adc

    if set(levels) & {"range_cube", "rd_cube", "power", "rad"}:
        rd_list, cube_list = [], []
        for v in range(adc.shape[0]):
            o, rf = range_doppler(adc[v], fp=fp, return_range_cube=True)
            rd_list.append(o)
            cube_list.append(rf)
        rd = np.stack(rd_list)
        if "range_cube" in levels:
            out["range_cube"] = np.stack(cube_list)
        if "rd_cube" in levels:
            out["rd_cube"] = rd
        if "power" in levels:
            out["power"] = integrate(rd, fp=fp)
        if "rad" in levels:
            out["rad"] = rad_bartlett(rd, p["pos"], p["n_tx"], p["n_rx"])

    out["axes"] = {"range_m": np.asarray(p["range_axis"]),
                   "velocity_mps": np.asarray(p["vel_axis"]),
                   "azimuth_deg": np.asarray(_GRID_DEG)}
    out["meta"] = {"n_tx": p["n_tx"], "n_rx": p["n_rx"],
                   "frame_id": header.get("frame_id"),
                   "fp": dataclasses.asdict(fp) if fp is not None else None}
    return out


def save_cubes(cubes, path):
    """Write cubes to one compressed .npz (arrays by name, axes as axis_*,
    meta as a JSON string). Recover with np.load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in cubes.items()
              if k not in ("axes", "meta")}
    axes = {f"axis_{k}": np.asarray(v) for k, v in cubes["axes"].items()}
    np.savez_compressed(path, **arrays, **axes,
                        meta=np.array(json.dumps(cubes["meta"])))
