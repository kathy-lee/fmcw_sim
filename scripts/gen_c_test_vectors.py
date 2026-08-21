"""Generate golden binary vectors for the c_frontend unit tests.

Deterministic (seed 2026). Reference values are computed in float64 and cast
to the file dtype, so the vectors are precision-stable. Formats match the C
`cplx` struct (interleaved float32 I/Q) and plain float64 for windows.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.range_doppler import range_doppler

OUT = Path(__file__).resolve().parents[1] / "c_frontend" / "tests" / "vectors"


def cvec(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    for n in (16, 64):
        x = cvec(rng, n)
        y = np.fft.fft(x.astype(np.complex128)).astype(np.complex64)
        x.tofile(OUT / f"fft{n}_in.bin")
        y.tofile(OUT / f"fft{n}_out.bin")

    for n in (64, 256):
        np.hanning(n).astype(np.float64).tofile(OUT / f"hann{n}.bin")

    frame = cvec(rng, 8 * 16).reshape(8, 16)          # (n_chirps, n_samples)
    out = range_doppler(frame)                        # float path, defaults
    frame.tofile(OUT / "rd_in.bin")
    out.astype(np.complex64).tofile(OUT / "rd_out.bin")

    # --- NCI: non-coherent integration ----------------------------------- #
    stack = cvec(rng, 3 * 4 * 6).reshape(3, 4, 6)        # (V, C, R)
    power = np.sum(np.abs(stack.astype(np.complex128)) ** 2, axis=0)
    stack.tofile(OUT / "nci_in.bin")
    power.astype(np.float64).tofile(OUT / "nci_out.bin")

    # --- CA-CFAR (fixed alpha — window mechanics, not calibration) -------- #
    D, R = 16, 24
    Gd, Gr, Td, Tr = 1, 2, 2, 3
    ALPHA, MIN_RANGE = 8.0, 4
    pw = rng.exponential(1.0, (D, R))
    pw[5, 10] = 60.0       # interior peak
    pw[12, 21] = 45.0      # near-edge peak: window overhangs the border
    pw[0, 23] = 45.0       # corner peak: edge replication clones the cell
                           # into its own training ring -> self-suppressed
    pw[7, 2] = 80.0        # inside the min-range gate -> must end up masked off
    lin = pw + 1e-12
    fd, fr = 2 * (Gd + Td) + 1, 2 * (Gr + Tr) + 1
    gd, gr = 2 * Gd + 1, 2 * Gr + 1
    n_full, n_inner = fd * fr, gd * gr
    sum_full = scipy.ndimage.uniform_filter(lin, size=(fd, fr), mode="nearest") * n_full
    sum_inner = scipy.ndimage.uniform_filter(lin, size=(gd, gr), mode="nearest") * n_inner
    noise = (sum_full - sum_inner) / (n_full - n_inner)
    ratio = lin / (ALPHA * noise)
    # Committed-vector determinism: no cell may sit on the threshold, or C's
    # different summation order could legitimately flip it forever.
    assert np.min(np.abs(ratio - 1.0)) > 1e-6, "borderline cell in CFAR vector"
    cmask = ratio > 1.0
    assert cmask[5, 10] and cmask[12, 21] and cmask[7, 2]
    assert not cmask[0, 23]        # corner: self-contaminated noise estimate
    cmask[:, :MIN_RANGE] = False
    pw.tofile(OUT / "cfar_power.bin")
    cmask.astype(np.uint8).tofile(OUT / "cfar_mask.bin")

    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
