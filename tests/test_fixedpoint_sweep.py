import numpy as np

from processing.config import FixedPointConfig
from simulator.beat_signal_generator import load_scenario
from evaluation.mimo_pipeline import run_pipeline_mimo


def test_run_pipeline_mimo_accepts_fp():
    cfg, targets = load_scenario("scenarios/scenario_mimo_fixedpoint_dynrange.json")
    fp = FixedPointConfig(adc_full_scale=2.0)
    hist = run_pipeline_mimo(cfg, targets, n_frames=3, fp=fp)
    assert len(hist) == 3


# --- Task 7: metrics + sweep harness ---

from evaluation.fixedpoint_sweep import (
    frame_metrics, sweep_stage, _weak_target_cell)


def test_weak_target_cell_in_bounds():
    cfg, targets = load_scenario("scenarios/scenario_mimo_fixedpoint_dynrange.json")
    d, r = _weak_target_cell(cfg, targets, frame_index=0)
    assert 0 <= d and 0 <= r


def test_frame_metrics_keys():
    cfg, targets = load_scenario("scenarios/scenario_mimo_fixedpoint_dynrange.json")
    m = frame_metrics(cfg, targets, fp=None, n_frames=2, n_seeds=1)
    for k in ("pd", "pfa", "snr_db", "sfdr_db"):
        assert k in m


def test_sweep_stage_wide_recovers_float():
    # A wide bit width should reproduce the float-golden Pd (within MC tolerance):
    # at high precision the fixed-point path collapses to float. (We do NOT assert
    # monotone noise-floor elevation: under block scaling the floor can *drop* at
    # coarse bits as sub-LSB cells quantize to zero -- see the report.)
    cfg, targets = load_scenario("scenarios/scenario_mimo_fixedpoint_dynrange.json")
    res = sweep_stage(cfg, targets, stage="cube_bits", bits=[16],
                      n_frames=10, n_seeds=3)
    assert abs(res["rows"][-1]["pd"] - res["float_pd"]) <= 0.15
