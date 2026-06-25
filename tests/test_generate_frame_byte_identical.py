"""Byte-for-byte guard for the Cartesian-primary refactor.

Loads the reference frames captured (on the pre-refactor radial code) by
tests/golden/capture_goldens.py and asserts generate_frame still produces
bit-identical output for the migrated on-axis scenarios. Any single-sample drift
fails — this is what makes the "byte-for-byte" claim enforceable rather than
aspirational.
"""
from pathlib import Path

import numpy as np
import pytest

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario

ROOT = Path(__file__).resolve().parent.parent     # repo root
GOLDEN = Path(__file__).parent / "golden"
CASES = [
    (p.stem.rsplit("_f", 1)[0], int(p.stem.rsplit("_f", 1)[1]), p)
    for p in sorted(GOLDEN.glob("*.npy"))
]


@pytest.mark.parametrize(
    "stem,frame_idx,golden_path", CASES, ids=[f"{s}_f{f}" for s, f, _ in CASES]
)
def test_generate_frame_byte_identical(stem, frame_idx, golden_path):
    cfg, targets = load_scenario(str(ROOT / "scenarios" / f"{stem}.json"))
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame(targets, frame_index=frame_idx)
    golden = np.load(golden_path)
    assert frame.dtype == golden.dtype
    assert np.array_equal(frame, golden), f"{stem} f{frame_idx} drifted"
