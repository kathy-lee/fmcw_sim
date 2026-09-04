"""ctypes wrapper around the one-pass composition (c_frontend/libfmcw.so).

Build the library first:  make -C c_frontend lib

CFrontendOnepass feeds a frame row by row, v-major, exactly as the wire does, and
returns the same (N, 7) detection rows as CFrontend.q15_process_frame —
which is the equivalence the streaming bridge leans on. quantize() is the
laptop-side ADC: the same adc_to_q15 the frozen chain applies internally,
so the bytes sent to a target and the reference's input are identical by
construction.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[1] / "c_frontend" / "libfmcw.so"
_lib = None


def read_params(path: str | Path) -> dict:
    """params.cfg as a dict; scalar values parsed to float, lists left as
    comma-joined strings (nothing here needs them)."""
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, val = line.split("=", 1)
        try:
            out[key] = float(val) if "," not in val else val
        except ValueError:
            out[key] = val
    return out


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        raise OSError(f"{_LIB_PATH} not found — build it with: "
                      f"make -C c_frontend lib")
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.fmcw_onepass_create.argtypes = [ctypes.c_char_p]
    lib.fmcw_onepass_create.restype = ctypes.c_void_p
    lib.fmcw_onepass_destroy.argtypes = [ctypes.c_void_p]
    lib.fmcw_onepass_destroy.restype = None
    lib.fmcw_onepass_dims.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_int)]
    lib.fmcw_onepass_dims.restype = ctypes.c_int
    lib.fmcw_onepass_row.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int16)]
    lib.fmcw_onepass_row.restype = ctypes.c_int
    lib.fmcw_onepass_finish.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_int]
    lib.fmcw_onepass_finish.restype = ctypes.c_int
    lib.fmcw_q15_quantize.argtypes = [ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_longlong, ctypes.c_double,
                                      ctypes.POINTER(ctypes.c_int16)]
    lib.fmcw_q15_quantize.restype = ctypes.c_longlong
    _lib = lib
    return lib


def _ptr(arr: np.ndarray, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


class CFrontendOnepass:
    """One one-pass context: params + caller-invisible storage in C."""

    def __init__(self, params_cfg: str | Path):
        lib = _load()
        self._lib = lib
        self._h = lib.fmcw_onepass_create(str(params_cfg).encode())
        if not self._h:
            raise ValueError(f"fmcw_onepass_create failed for {params_cfg}")
        d = (ctypes.c_int * 3)()
        lib.fmcw_onepass_dims(self._h, d)
        self.dims = (d[0], d[1], d[2])
        self.adc_full_scale = float(read_params(params_cfg)["adc_full_scale"])
        self.last_sat = 0

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.fmcw_onepass_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def quantize(self, adc: np.ndarray) -> np.ndarray:
        """(V, C, S) complex64 -> (V, C, S, 2) int16 on the ADC grid."""
        V, C, S = self.dims
        adc = np.ascontiguousarray(adc)
        if adc.dtype != np.complex64 or adc.shape != (V, C, S):
            raise ValueError(f"adc: expected complex64 {(V, C, S)}, "
                             f"got {adc.dtype} {adc.shape}")
        out = np.empty((V, C, S, 2), dtype=np.int16)
        sat = self._lib.fmcw_q15_quantize(
            _ptr(adc, ctypes.c_float), V * C * S, self.adc_full_scale,
            _ptr(out, ctypes.c_int16))
        if sat < 0:
            raise RuntimeError("fmcw_q15_quantize failed")
        self.last_sat = int(sat)
        return out

    def process(self, rows_q: np.ndarray) -> np.ndarray:
        """(V, C, S, 2) int16, fed v-major row by row -> (N, 7) float64."""
        V, C, S = self.dims
        rows_q = np.ascontiguousarray(rows_q)
        if rows_q.dtype != np.int16 or rows_q.shape != (V, C, S, 2):
            raise ValueError(f"rows_q: expected int16 {(V, C, S, 2)}, "
                             f"got {rows_q.dtype} {rows_q.shape}")
        for v in range(V):
            for c in range(C):
                row = np.ascontiguousarray(rows_q[v, c])
                if self._lib.fmcw_onepass_row(
                        self._h, v, c, _ptr(row, ctypes.c_int16)) != 0:
                    raise RuntimeError(f"fmcw_onepass_row({v}, {c}) failed")
        cap = 64
        det = np.empty((cap, 7), dtype=np.float64)
        n = self._lib.fmcw_onepass_finish(
            self._h, _ptr(det, ctypes.c_double), cap)
        if n < 0:
            raise RuntimeError("fmcw_onepass_finish failed")
        return det[:n].copy()
