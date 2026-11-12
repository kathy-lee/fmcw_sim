"""ctypes wrapper around the C front end (c_frontend/libfmcw.so).

Build the library first:  make -C c_frontend lib

CFrontend exposes the C implementation's whole-frame facade and its
individual stages, in both arithmetics: the float chain (rd, nci, cfar,
pointcloud, process_frame) and the fixed-point one (the q15_* methods).
One handle serves both, so the two can be run against the same frame and
compared directly.

Equivalence contract for the float chain: same numbers as range_doppler /
integrate / cfar_2d / mimo_point_cloud within the tolerances enforced by
tests/test_c_frontend.py. The fixed-point chain is held to a statistical
standard instead — per-stage SQNR floors and mask agreement — because
integer arithmetic is not expected to reproduce float results exactly.

All validation (dtype, contiguity, shape) happens here, before any
pointer crosses into C — a ValueError beats a segfault.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[1] / "c_frontend" / "libfmcw.so"
_lib = None
_other_libs: dict[str, ctypes.CDLL] = {}    # loaded from a given lib_path


def _load(lib_path: str | Path | None = None) -> ctypes.CDLL:
    """Load libfmcw.so once and declare every signature (the one place a
    contract mismatch with fmcw_api.h could hide). lib_path names another
    build of the same library, loaded once and kept beside it."""
    global _lib
    if lib_path is None:
        if _lib is not None:
            return _lib
        if not _LIB_PATH.exists():
            raise OSError(f"{_LIB_PATH} not found — build it with: "
                          f"make -C c_frontend lib")
        path = _LIB_PATH
    else:
        path = Path(lib_path)
        if str(path) in _other_libs:
            return _other_libs[str(path)]
        if not path.exists():
            raise OSError(f"{path} not found")
    lib = ctypes.CDLL(str(path))
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
    # void *fmcw_create(const char *params_cfg);
    lib.fmcw_create.argtypes = [ctypes.c_char_p]
    lib.fmcw_create.restype = ctypes.c_void_p
    # void fmcw_destroy(void *h);
    lib.fmcw_destroy.argtypes = [ctypes.c_void_p]
    lib.fmcw_destroy.restype = None
    # int fmcw_dims(void *h, int out[3]);
    lib.fmcw_dims.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.fmcw_dims.restype = ctypes.c_int
    # int fmcw_process_frame(void *h, const float *adc, double *det, int max);
    lib.fmcw_process_frame.argtypes = [ctypes.c_void_p, c_float_p,
                                       c_double_p, ctypes.c_int]
    lib.fmcw_process_frame.restype = ctypes.c_int
    # int fmcw_rd(void *h, const float *adc, float *rd);
    lib.fmcw_rd.argtypes = [ctypes.c_void_p, c_float_p, c_float_p]
    lib.fmcw_rd.restype = ctypes.c_int
    # int fmcw_nci(void *h, const float *rd, double *power);
    lib.fmcw_nci.argtypes = [ctypes.c_void_p, c_float_p, c_double_p]
    lib.fmcw_nci.restype = ctypes.c_int
    # int fmcw_cfar(void *h, const double *power, uint8_t *mask);
    lib.fmcw_cfar.argtypes = [ctypes.c_void_p, c_double_p, c_uint8_p]
    lib.fmcw_cfar.restype = ctypes.c_int
    # int fmcw_pointcloud(void *h, const float *rd, const uint8_t *mask,
    #                     double *det, int max_out);
    lib.fmcw_pointcloud.argtypes = [ctypes.c_void_p, c_float_p, c_uint8_p,
                                    c_double_p, ctypes.c_int]
    lib.fmcw_pointcloud.restype = ctypes.c_int

    c_int16_p = ctypes.POINTER(ctypes.c_int16)
    c_int64_p = ctypes.POINTER(ctypes.c_int64)
    c_ll_p = ctypes.POINTER(ctypes.c_longlong)
    # int fmcw_q15_rd(void *h, const float *adc, int16_t *rd_q);
    lib.fmcw_q15_rd.argtypes = [ctypes.c_void_p, c_float_p, c_int16_p]
    lib.fmcw_q15_rd.restype = ctypes.c_int
    # int fmcw_q15_nci(void *h, const int16_t *rd_q, int64_t *power_q);
    lib.fmcw_q15_nci.argtypes = [ctypes.c_void_p, c_int16_p, c_int64_p]
    lib.fmcw_q15_nci.restype = ctypes.c_int
    # int fmcw_q15_cfar(void *h, const int64_t *power_q, uint8_t *mask);
    lib.fmcw_q15_cfar.argtypes = [ctypes.c_void_p, c_int64_p, c_uint8_p]
    lib.fmcw_q15_cfar.restype = ctypes.c_int
    # int fmcw_q15_pointcloud(void *h, const int16_t *rd_q,
    #                         const uint8_t *mask, double *det, int max_out);
    lib.fmcw_q15_pointcloud.argtypes = [ctypes.c_void_p, c_int16_p, c_uint8_p,
                                        c_double_p, ctypes.c_int]
    lib.fmcw_q15_pointcloud.restype = ctypes.c_int
    # int fmcw_q15_process_frame(void *h, const float *adc,
    #                            double *det, int max_out);
    lib.fmcw_q15_process_frame.argtypes = [ctypes.c_void_p, c_float_p,
                                           c_double_p, ctypes.c_int]
    lib.fmcw_q15_process_frame.restype = ctypes.c_int
    # double fmcw_q15_rd_scale(void *h);
    lib.fmcw_q15_rd_scale.argtypes = [ctypes.c_void_p]
    lib.fmcw_q15_rd_scale.restype = ctypes.c_double
    # int fmcw_q15_saturations(void *h, long long out[2]);
    lib.fmcw_q15_saturations.argtypes = [ctypes.c_void_p, c_ll_p]
    lib.fmcw_q15_saturations.restype = ctypes.c_int
    # void fmcw_q15_reset_saturations(void *h);
    lib.fmcw_q15_reset_saturations.argtypes = [ctypes.c_void_p]
    lib.fmcw_q15_reset_saturations.restype = None
    if lib_path is None:
        _lib = lib
    else:
        _other_libs[str(path)] = lib
    return lib


def _ptr(arr: np.ndarray, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def _checked(arr: np.ndarray, dtype, shape, name: str) -> np.ndarray:
    """Require exact dtype and shape; return a C-contiguous view/copy."""
    if arr.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {arr.dtype}")
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    return np.ascontiguousarray(arr)


class CFrontend:
    """One C front-end context: params + buffers + plans behind a handle."""

    def __init__(self, params_cfg: str | Path,
                 lib_path: str | Path | None = None):
        lib = _load(lib_path)
        self._lib = lib
        self._h = lib.fmcw_create(str(params_cfg).encode())
        if not self._h:
            raise ValueError(f"fmcw_create failed for {params_cfg} "
                             f"(missing file or invalid config)")
        d = (ctypes.c_int * 3)()
        lib.fmcw_dims(self._h, d)
        self.dims = (d[0], d[1], d[2])          # (V, C, S)

    # -- lifecycle ---------------------------------------------------- #
    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.fmcw_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def _handle(self):
        if not self._h:
            raise ValueError("CFrontend is closed")
        return self._h

    # -- float stages ------------------------------------------------- #
    def rd(self, adc: np.ndarray) -> np.ndarray:
        """Per-channel range-Doppler: (V, C, S) complex64 -> (V, C, S//2)."""
        V, C, S = self.dims
        adc = _checked(adc, np.complex64, (V, C, S), "adc")
        out = np.empty((V, C, S // 2), dtype=np.complex64)
        if self._lib.fmcw_rd(self._handle(), _ptr(adc, ctypes.c_float),
                             _ptr(out, ctypes.c_float)) != 0:
            raise RuntimeError("fmcw_rd failed")
        return out

    def nci(self, rd: np.ndarray) -> np.ndarray:
        """Non-coherent integration: (V, C, S//2) complex64 -> (C, S//2) f64."""
        V, C, S = self.dims
        rd = _checked(rd, np.complex64, (V, C, S // 2), "rd")
        out = np.empty((C, S // 2), dtype=np.float64)
        if self._lib.fmcw_nci(self._handle(), _ptr(rd, ctypes.c_float),
                              _ptr(out, ctypes.c_double)) != 0:
            raise RuntimeError("fmcw_nci failed")
        return out

    def cfar(self, power: np.ndarray) -> np.ndarray:
        """CA-CFAR detection mask: (C, S//2) float64 -> (C, S//2) bool."""
        V, C, S = self.dims
        power = _checked(power, np.float64, (C, S // 2), "power")
        out = np.empty((C, S // 2), dtype=np.uint8)
        if self._lib.fmcw_cfar(self._handle(), _ptr(power, ctypes.c_double),
                               _ptr(out, ctypes.c_uint8)) != 0:
            raise RuntimeError("fmcw_cfar failed")
        return out.astype(bool)

    def pointcloud(self, rd: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detections for masked cells: rows of
        [range_bin, doppler_bin, azimuth_deg, x, y, v_radial, snr_db]."""
        V, C, S = self.dims
        rd = _checked(rd, np.complex64, (V, C, S // 2), "rd")
        mask8 = np.ascontiguousarray(mask.astype(np.uint8))
        if mask8.shape != (C, S // 2):
            raise ValueError(f"mask: expected shape {(C, S // 2)}, "
                             f"got {mask8.shape}")
        cap = C * (S // 2)
        det = np.empty((cap, 7), dtype=np.float64)
        n = self._lib.fmcw_pointcloud(
            self._handle(), _ptr(rd, ctypes.c_float),
            _ptr(mask8, ctypes.c_uint8), _ptr(det, ctypes.c_double), cap)
        if n < 0:
            raise RuntimeError("fmcw_pointcloud failed")
        return det[:n].copy()

    def process_frame(self, adc: np.ndarray) -> np.ndarray:
        """Whole chain: (V, C, S) complex64 ADC frame -> (N, 7) detections."""
        V, C, S = self.dims
        adc = _checked(adc, np.complex64, (V, C, S), "adc")
        cap = C * (S // 2)
        det = np.empty((cap, 7), dtype=np.float64)
        n = self._lib.fmcw_process_frame(
            self._handle(), _ptr(adc, ctypes.c_float),
            _ptr(det, ctypes.c_double), cap)
        if n < 0:
            raise RuntimeError("fmcw_process_frame failed")
        return det[:n].copy()

    # -- fixed-point stages ------------------------------------------- #
    #
    # The integer chain takes the same float ADC frame the float stages
    # take, because quantization is its first step. It returns native
    # int16/int64 rather than pre-scaled floats, so the caller can see
    # how many counts a value actually occupies; dequantize_rd and the
    # scale properties do the conversion when it is wanted.
    #
    # NumPy has no complex-int16 dtype, so an integer RD cube is carried
    # as (V, C, S//2, 2) with the last axis holding I and Q — the same
    # bytes the C side reads as cplx_q15.

    @property
    def rd_scale(self) -> float:
        """Multiply an integer RD value by this to reach float-path units."""
        return float(self._lib.fmcw_q15_rd_scale(self._handle()))

    @property
    def power_scale(self) -> float:
        """Multiply an integer NCI value by this to reach float-path power.

        The square of rd_scale, since power is a squared magnitude."""
        return self.rd_scale ** 2

    def saturations(self) -> tuple[int, int]:
        """(quantization, range-Doppler) clip counts since the last reset.

        Nonzero means an input assumption failed — the signal exceeded the
        ADC's full scale, or a stage's magnitude bound — not merely that
        precision was lost."""
        out = (ctypes.c_longlong * 2)()
        if self._lib.fmcw_q15_saturations(self._handle(), out) != 0:
            raise RuntimeError("fmcw_q15_saturations failed")
        return int(out[0]), int(out[1])

    def reset_saturations(self) -> None:
        self._lib.fmcw_q15_reset_saturations(self._handle())

    def dequantize_rd(self, rd_q: np.ndarray) -> np.ndarray:
        """(V, C, S//2, 2) int16 -> complex128 in the float path's units."""
        return (rd_q[..., 0] + 1j * rd_q[..., 1]) * self.rd_scale

    def q15_rd(self, adc: np.ndarray) -> np.ndarray:
        """Integer range-Doppler: (V, C, S) complex64 -> (V, C, S//2, 2)."""
        V, C, S = self.dims
        adc = _checked(adc, np.complex64, (V, C, S), "adc")
        out = np.empty((V, C, S // 2, 2), dtype=np.int16)
        if self._lib.fmcw_q15_rd(self._handle(), _ptr(adc, ctypes.c_float),
                                 _ptr(out, ctypes.c_int16)) != 0:
            raise RuntimeError("fmcw_q15_rd failed")
        return out

    def q15_nci(self, rd_q: np.ndarray) -> np.ndarray:
        """Integer integration: (V, C, S//2, 2) int16 -> (C, S//2) int64."""
        V, C, S = self.dims
        rd_q = _checked(rd_q, np.int16, (V, C, S // 2, 2), "rd_q")
        out = np.empty((C, S // 2), dtype=np.int64)
        if self._lib.fmcw_q15_nci(self._handle(), _ptr(rd_q, ctypes.c_int16),
                                  _ptr(out, ctypes.c_int64)) != 0:
            raise RuntimeError("fmcw_q15_nci failed")
        return out

    def q15_cfar(self, power_q: np.ndarray) -> np.ndarray:
        """Integer CA-CFAR: (C, S//2) int64 -> (C, S//2) bool."""
        V, C, S = self.dims
        power_q = _checked(power_q, np.int64, (C, S // 2), "power_q")
        out = np.empty((C, S // 2), dtype=np.uint8)
        if self._lib.fmcw_q15_cfar(self._handle(), _ptr(power_q, ctypes.c_int64),
                                   _ptr(out, ctypes.c_uint8)) != 0:
            raise RuntimeError("fmcw_q15_cfar failed")
        return out.astype(bool)

    def q15_pointcloud(self, rd_q: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detections from an integer cube; same (N, 7) rows as the float
        stage, since DoA and the Cartesian conversion are float in both."""
        V, C, S = self.dims
        rd_q = _checked(rd_q, np.int16, (V, C, S // 2, 2), "rd_q")
        mask8 = np.ascontiguousarray(mask.astype(np.uint8))
        if mask8.shape != (C, S // 2):
            raise ValueError(f"mask: expected shape {(C, S // 2)}, "
                             f"got {mask8.shape}")
        cap = C * (S // 2)
        det = np.empty((cap, 7), dtype=np.float64)
        n = self._lib.fmcw_q15_pointcloud(
            self._handle(), _ptr(rd_q, ctypes.c_int16),
            _ptr(mask8, ctypes.c_uint8), _ptr(det, ctypes.c_double), cap)
        if n < 0:
            raise RuntimeError("fmcw_q15_pointcloud failed")
        return det[:n].copy()

    def q15_process_frame(self, adc: np.ndarray) -> np.ndarray:
        """Whole integer chain; same signature and rows as process_frame."""
        V, C, S = self.dims
        adc = _checked(adc, np.complex64, (V, C, S), "adc")
        cap = C * (S // 2)
        det = np.empty((cap, 7), dtype=np.float64)
        n = self._lib.fmcw_q15_process_frame(
            self._handle(), _ptr(adc, ctypes.c_float),
            _ptr(det, ctypes.c_double), cap)
        if n < 0:
            raise RuntimeError("fmcw_q15_process_frame failed")
        return det[:n].copy()
