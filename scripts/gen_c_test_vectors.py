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

from processing.doa import _peaks, bartlett, estimate_doa, spectrum, steering_matrix
from processing.point_cloud import tdm_doppler_compensation
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

    # --- Bartlett DoA (two sources, 8-element half-wavelength ULA) -------- #
    pos8 = 0.5 * np.arange(8)
    a20 = np.exp(1j * 2 * np.pi * pos8 * np.sin(np.deg2rad(20.0)))
    am40 = np.exp(1j * 2 * np.pi * pos8 * np.sin(np.deg2rad(-40.0)))
    snap = (a20 + 0.5 * am40 + 0.05 * cvec(rng, 8)).astype(np.complex64)
    g, P = spectrum(snap, pos8, "bartlett")
    pk = bartlett(snap, pos8, 2)
    # Determinism guard: C stores the steering matrix in float32. Recompute
    # the spectrum through a float32 matrix and require the same peaks.
    A32 = steering_matrix(pos8, np.deg2rad(g)).astype(np.complex64)
    P32 = np.abs(A32.conj().T.astype(np.complex128) @ snap.astype(np.complex128)) ** 2
    assert np.array_equal(_peaks(P32, g, 2), pk), "float32 steering flips a DoA peak"
    snap.tofile(OUT / "doa_snap.bin")
    pos8.astype(np.float64).tofile(OUT / "doa_pos.bin")
    P.astype(np.float64).tofile(OUT / "doa_spec.bin")
    pk.astype(np.float64).tofile(OUT / "doa_peaks.bin")

    # --- point cloud (V=4 as 2x2 TDM, C=8, R=6, 3 detections) ------------- #
    PV, PC, PR, NTX, NRX = 4, 8, 6, 2, 2
    RPB = 0.3
    pvel = np.linspace(-2.0, 1.5, PC)                # exact halves
    ppos = 0.5 * np.arange(PV)
    pc_stack = cvec(rng, PV * PC * PR).reshape(PV, PC, PR)
    pc_mask = np.zeros((PC, PR), dtype=bool)
    pc_mask[0, 5] = pc_mask[3, 2] = pc_mask[7, 4] = True
    grid_deg = np.arange(-90.0, 90.0 + 1e-9, 1.0)
    A32p = steering_matrix(ppos, np.deg2rad(grid_deg)).astype(np.complex64)
    rows, snap0 = [], None
    for d, r in np.argwhere(pc_mask):
        cs = tdm_doppler_compensation(pc_stack[:, d, r], d, PC, NTX, NRX)
        if snap0 is None:
            snap0 = cs.astype(np.complex64)
        snr = 10.0 * np.log10(np.sum(np.abs(cs) ** 2) + 1e-12)
        for az in estimate_doa(cs, ppos, method="bartlett", n_sources=1):
            # Determinism guard: float32 steering + float32 snapshot (the C
            # arithmetic) must land on the same grid maximum.
            P32p = np.abs(A32p.conj().T.astype(np.complex128)
                          @ cs.astype(np.complex64).astype(np.complex128)) ** 2
            assert _peaks(P32p, grid_deg, 1)[0] == az, "float32 flips a PC angle"
            rows.append([r, d, az, r * RPB * np.cos(np.deg2rad(az)),
                         r * RPB * np.sin(np.deg2rad(az)), pvel[d], snr])
    pc_stack.tofile(OUT / "pc_in.bin")
    pc_mask.astype(np.uint8).tofile(OUT / "pc_mask.bin")
    np.array(rows, dtype=np.float64).tofile(OUT / "pc_out.bin")
    snap0.tofile(OUT / "pc_snap0.bin")

    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
