#!/usr/bin/env bash
# Reproduce the clutter-removal velocity-bias experiment.
#
# Runs the streaming pipeline twice against the scenarios/scenario_two_targets.json
# scenario (targets at v = +/- 0.1 m/s):
#   1. With clutter removal ON  (default; biases sub-bin Doppler targets)
#   2. With clutter removal OFF (--no-clutter-removal; recovers v ~ +/- 0.1 m/s)
#
# Prints the first detections for each case, side-by-side, so the bias is visible.
#
# Usage: ./scripts/verify_clutter_bias.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="docs/notch_out_range_bin_0"
mkdir -p "$OUT_DIR"
SCENARIO=scenarios/scenario_two_targets.json
N_FRAMES_PUB=30
N_FRAMES_SUB=25

run_case() {
    local label=$1
    local extra_flag=$2
    local sub_dir="$OUT_DIR/$label"
    rm -rf "$sub_dir"
    mkdir -p "$sub_dir"
    local sub_log="$sub_dir/sub.log"
    local pub_log="$sub_dir/pub.log"

    echo "=== [$label] starting subscriber (flag: ${extra_flag:-<none>}) ==="
    MPLBACKEND=Agg uv run python -m simulator.streaming_pipeline \
        --max-frames "$N_FRAMES_SUB" \
        --save-dir "$sub_dir" \
        --stats-window 100 \
        $extra_flag \
        > "$sub_log" 2>&1 &
    local sub_pid=$!

    # Give SUB a moment to connect before PUB starts sending.
    sleep 0.4

    echo "=== [$label] publishing $N_FRAMES_PUB frames ==="
    MPLBACKEND=Agg uv run python -m simulator.streaming_publisher \
        --scenario "$SCENARIO" \
        --frames "$N_FRAMES_PUB" \
        > "$pub_log" 2>&1

    wait "$sub_pid"
    echo "=== [$label] subscriber finished, detections.csv head: ==="
    head -8 "$sub_dir/detections.csv"
    echo
}

# Case A: default (clutter_removal enabled).
run_case "with_clutter_removal" ""

# Case B: --no-clutter-removal.
run_case "no_clutter_removal" "--no-clutter-removal"

echo "All output saved under: $OUT_DIR"
