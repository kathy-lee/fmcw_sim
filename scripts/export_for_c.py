"""Export MIMO ADC frames + params.cfg for the C front end.

Frames: raw complex64 (interleaved float32 I/Q), shape V x C x S, C-order —
the C `cplx` buffer format. params.cfg: flat key=value, the 16 keys the C
`params_load` requires (spec 2026-08-20-c-frontend-design.md §4). cfar_alpha
is calibrated here (offline, as firmware config) with the exact arguments
`process_mimo_frame` uses: CA, guard/training/pfa from _CFAR_KW,
window="hann", dc_removal=True, clutter_removal=False, n_integrate=V.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.cfar import _alpha_empirical
from processing.mimo_frame import _CFAR_KW, frame_params
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.wire import build_header

MIN_RANGE_BINS = 6   # near-DC gate, matching process_mimo_frame's default


def export(scenario, out_dir, frames):
    cfg, targets = load_scenario(scenario)
    gen = FMCWBeatSignalGenerator(cfg)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    header = None
    frame = None
    for i in range(frames):
        frame = gen.generate_frame_mimo(targets, frame_index=i)
        if header is None:
            header = build_header(frame=frame, frame_id=i, topic="cube",
                                  metadata=gen.metadata(), ground_truth=[],
                                  timestamp_unix_s=0.0)
        frame.astype(np.complex64).tofile(out / "frames" / f"frame_{i:04d}.bin")

    p = frame_params(header)
    n_virtual, n_chirps, n_samples = frame.shape
    guard, training, pfa = _CFAR_KW["guard"], _CFAR_KW["training"], _CFAR_KW["pfa"]
    alpha = _alpha_empirical(
        "ca", (n_chirps, n_samples // 2), tuple(guard), tuple(training), pfa,
        "hann", True, False, n_integrate=n_virtual)

    def fmt_list(values):
        return ",".join(repr(float(v)) for v in values)

    lines = [
        f"n_virtual={n_virtual}",
        f"n_chirps={n_chirps}",
        f"n_samples={n_samples}",
        f"n_tx={p['n_tx']}",
        f"n_rx={p['n_rx']}",
        "doa_n_sources=1",
        f"cfar_guard_d={guard[0]}",
        f"cfar_guard_r={guard[1]}",
        f"cfar_training_d={training[0]}",
        f"cfar_training_r={training[1]}",
        f"min_range_bins={MIN_RANGE_BINS}",
        f"range_per_bin={float(p['range_axis'][1])!r}",
        f"pfa={pfa!r}",
        f"cfar_alpha={alpha!r}",
        f"velocity_axis={fmt_list(p['vel_axis'])}",
        f"pos_wl={fmt_list(p['pos'])}",
    ]
    (out / "params.cfg").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export ADC frames for the C front end.")
    ap.add_argument("--scenario", default="scenarios/scenario_mimo.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args(argv)
    export(args.scenario, args.out, args.frames)
    print(f"wrote {args.frames} frame(s) + params.cfg to {args.out}")


if __name__ == "__main__":
    main()
