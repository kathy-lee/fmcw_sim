"""Generate golden binary vectors for the c_frontend unit tests.

Deterministic (seed 2026). Reference values are computed in float64 and cast
to the file dtype, so the vectors are precision-stable. Formats match the C
`cplx` struct (interleaved float32 I/Q) and plain float64 for windows.
"""
import sys
from pathlib import Path

import numpy as np

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

    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
