"""One-shot generator for the byte-for-byte reference data.

NOT a pytest test — the filename has no `test_` prefix, so pytest does not collect
or run it. Run it ONCE by hand on the pre-refactor (radial) code, then commit the
produced tests/golden/*.npy. They pin generate_frame's exact output so the
Cartesian-primary reparametrization can be proven byte-for-byte identical for
on-axis (migrated) scenarios. Consumed by tests/test_generate_frame_byte_identical.py.

Usage (from anywhere):  python3 -m tests.golden.capture_goldens
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root
sys.path.insert(0, str(ROOT))

import numpy as np

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario

SCENARIOS = [
    "scenarios/scenario_missed_detections.json",
    "scenarios/scenario_extended_tracking.json",
    "scenarios/scenario_drag_decel.json",
    "scenarios/scenario_imm_switching.json",
]
FRAMES = [0, 1, 7, 23]


def main() -> None:
    out = Path(__file__).parent          # tests/golden/
    # NOTE: generate_frame draws thermal noise from a running RNG that advances
    # per call (not per frame_index). The test regenerates each frame from a
    # FRESH generator, so we must capture the same way — one generator per frame
    # — or the references would encode a call-sequence the test never reproduces.
    for sc in SCENARIOS:
        for fi in FRAMES:
            cfg, targets = load_scenario(str(ROOT / sc))
            gen = FMCWBeatSignalGenerator(cfg)
            frame = gen.generate_frame(targets, frame_index=fi)
            name = Path(sc).stem + f"_f{fi}.npy"
            np.save(out / name, frame)
            print("saved", name, frame.shape, frame.dtype)


if __name__ == "__main__":
    main()
