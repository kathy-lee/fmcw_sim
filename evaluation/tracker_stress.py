"""Tracker stress test (pilot): crossing-targets scenario.

Runs the existing pipeline (range_doppler → cfar_2d → cluster → Tracker)
on a two-target range-axis crossing and reports whether the tracker
maintains stable track IDs through the crossing.

Usage:
    python -m evaluation.tracker_stress [--frames N] [--out PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.config import TrackerConfig
from processing.range_doppler import range_doppler
from processing.tracking import Tracker
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


DEFAULT_SCENARIO_PATH = "scenarios/scenario_crossing_targets.json"
DEFAULT_FRAMES = 80


def _scenario_display_name(scenario_path: str) -> str:
    """Derive a human-readable name from the scenario file path.

    scenarios/scenario_crossing_targets.json -> "crossing targets"
    scenarios/scenario_overtaking_targets.json -> "overtaking targets"
    """
    stem = Path(scenario_path).stem
    if stem.startswith("scenario_"):
        stem = stem[len("scenario_"):]
    return stem.replace("_", " ")


def _default_out_path(scenario_path: str) -> Path:
    """Derive the default PNG output path from the scenario filename.

    scenarios/scenario_crossing_targets.json -> docs/tracker_stress_crossing_targets.png
    scenarios/scenario_overtaking_targets.json -> docs/tracker_stress_overtaking_targets.png
    """
    stem = Path(scenario_path).stem
    if stem.startswith("scenario_"):
        stem = stem[len("scenario_"):]
    return Path(f"docs/tracker_stress_{stem}.png")


def build_gt_trajectory(
    initial_targets: list,
    gen: FMCWBeatSignalGenerator,
    n_frames: int,
) -> list:
    """Per-frame ground truth positions.

    Returns: list[(frame_idx, [(gt_id, range_m, velocity_mps,
                                  range_bin, doppler_bin), ...])]

    Uses gen.ground_truth() so the GT trajectory stays in sync with the
    simulator's internal motion model (no drift between what the simulator
    generates and what we compute as GT).
    """
    out = []
    for frame_idx in range(n_frames):
        gts = gen.ground_truth(initial_targets, frame_index=frame_idx)
        gt_tuples = [
            (gt_id, gt["range_m"], gt["velocity_mps"],
             gt["range_bin"], gt["doppler_bin_shifted"])
            for gt_id, gt in enumerate(gts)
        ]
        out.append((frame_idx, gt_tuples))
    return out


def associate_per_frame(
    gt_at_frame: list,
    tracks_at_frame: list,
    max_distance_bins: float = 3.0,
) -> dict:
    """Hungarian assignment of GT positions to confirmed tracks.

    Args:
        gt_at_frame: list of (gt_id, range_m, velocity_mps,
                              range_bin, doppler_bin) tuples.
        tracks_at_frame: list of Track-like objects with attributes
                         track_id, range_bin, doppler_bin, status.
                         Only `status == "confirmed"` tracks participate.
        max_distance_bins: pairs with Euclidean (range_bin, doppler_bin)
                           distance above this become None.

    Returns: {gt_id: track_id_or_None}.
    """
    confirmed = [t for t in tracks_at_frame if t.status == "confirmed"]
    out = {gt[0]: None for gt in gt_at_frame}
    if not confirmed:
        return out

    cost = np.full((len(gt_at_frame), len(confirmed)), np.inf)
    for i, (_, _, _, r_bin, d_bin) in enumerate(gt_at_frame):
        for j, tr in enumerate(confirmed):
            cost[i, j] = np.hypot(r_bin - tr.state[0],
                                  d_bin - tr.last_doppler_bin)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_distance_bins:
            out[gt_at_frame[r][0]] = confirmed[c].id
    return out


def count_id_switches(
    association_history: list,
    min_persistence_frames: int = 2,
) -> list:
    """Detect ID switches with a persistence filter.

    An ID switch is counted when a GT's associated track ID changes from
    one non-None value to a different non-None value, AND the new
    assignment persists for at least `min_persistence_frames`
    consecutive frames immediately after the change. Gaps (None) don't
    count as switches by themselves; a switch is from "track A" to
    "track B" where B != A.

    Args:
        association_history: list of (frame_idx, {gt_id: track_id_or_None}).
        min_persistence_frames: minimum consecutive frames the new track
            must persist for the switch to be counted.

    Returns: list of (frame_idx, gt_id, old_track_id, new_track_id) tuples.
    """
    events = []
    if not association_history:
        return events

    gt_ids = sorted({gt_id for _, m in association_history for gt_id in m})
    for gt_id in gt_ids:
        last_stable = None
        for k, (frame_idx, mapping) in enumerate(association_history):
            current = mapping.get(gt_id)
            if current is None:
                continue
            if last_stable is None:
                last_stable = current
                continue
            if current == last_stable:
                continue
            # Apparent switch: check persistence over the next K frames
            # (including the current frame).
            window = association_history[k:k + min_persistence_frames]
            persisted = sum(
                1 for (_, m) in window if m.get(gt_id) == current
            )
            if persisted >= min_persistence_frames:
                events.append((frame_idx, gt_id, last_stable, current))
                last_stable = current
    return events


def evaluate_run(
    association_history: list,
    switch_events: list,
    tracks_history: list,
    track_break_pd_threshold: float = 0.5,
) -> dict:
    """Compute per-GT metrics and classify the run.

    Args:
        association_history: list of (frame_idx, {gt_id: track_id_or_None}).
        switch_events: list of (frame_idx, gt_id, old_id, new_id) -- output of
            count_id_switches().
        tracks_history: list of (frame_idx, [TrackSnapshot, ...]) -- output of
            run_pipeline(). Pass [] to skip coast-frame counting (resulting
            coast_frames will be 0 for every GT).
        track_break_pd_threshold: a GT's Pd below this triggers TRACK BREAK.

    Returns:
        {
          "per_gt": {gt_id: {"pd": float, "distinct_track_ids": int,
                              "n_frames_associated": int,
                              "coast_frames": int}},
          "n_switches": int,
          "switch_events": [...],
          "classification": "CLEAN" | "ID SWITCH" | "TRACK BREAK"
                            | "TOTAL CONFUSION",
        }

    coast_frames counts frames where some track that was ever-associated
    with this GT was alive in status='coasting'. This is the metric the
    spec calls "how long the tracker coasted through misses" -- the same
    track ID is being kept alive across CFAR misses via Kalman prediction.
    """
    gt_ids = sorted({gt_id for _, m in association_history for gt_id in m})

    # Tracks ever associated with each GT (used to attribute coast frames).
    gt_to_tracks = {gt_id: set() for gt_id in gt_ids}
    for _, mapping in association_history:
        for gt_id, track_id in mapping.items():
            if track_id is not None:
                gt_to_tracks[gt_id].add(track_id)

    coast_count = {gt_id: 0 for gt_id in gt_ids}
    for _, snapshots in tracks_history:
        for snap in snapshots:
            if snap.status != "coasting":
                continue
            for gt_id, track_set in gt_to_tracks.items():
                if snap.id in track_set:
                    coast_count[gt_id] += 1
                    break

    per_gt = {}
    for gt_id in gt_ids:
        n_total = len(association_history)
        n_with = sum(
            1 for _, m in association_history if m.get(gt_id) is not None
        )
        distinct = {
            m[gt_id] for _, m in association_history
            if m.get(gt_id) is not None
        }
        per_gt[gt_id] = {
            "pd": n_with / n_total if n_total else 0.0,
            "distinct_track_ids": len(distinct),
            "n_frames_associated": n_with,
            "coast_frames": coast_count[gt_id],
        }

    has_switches = len(switch_events) > 0
    has_break = any(
        per_gt[g]["pd"] < track_break_pd_threshold for g in gt_ids
    )
    if has_break and has_switches:
        classification = "TOTAL CONFUSION"
    elif has_break:
        classification = "TRACK BREAK"
    elif has_switches:
        classification = "ID SWITCH"
    else:
        classification = "CLEAN"

    return {
        "per_gt": per_gt,
        "n_switches": len(switch_events),
        "switch_events": list(switch_events),
        "classification": classification,
    }


def track_position_rmse(
    gt_trajectory: list,
    tracks_history: list,
    association_history: list,
    range_per_bin: float,
) -> float:
    """RMSE (metres) of each GT-owning track's range estimate vs the GT range.

    Averaged over every frame where a track is associated with a GT. This is
    the *continuous* estimate-quality metric that the process_noise_q sweep
    needs: the discrete metrics in evaluate_run (Pd, distinct IDs,
    classification) are blind to estimate quality, whereas q trades estimate
    smoothness for responsiveness and shows up here.

    Args:
        gt_trajectory: list of (frame_idx, [(gt_id, range_m, ...), ...]) --
            output of build_gt_trajectory().
        tracks_history: list of (frame_idx, [TrackSnapshot, ...]) -- output of
            run_pipeline().
        association_history: list of (frame_idx, {gt_id: track_id_or_None}).
        range_per_bin: metres per range bin, used to convert a track's
            state[0] (range-bin units) into metres.

    Returns: RMSE in metres over all associated GT-frames, or NaN when no
        frame had an association (nothing to average).
    """
    sq_errors = []
    for (_, gts), (_, snaps), (_, mapping) in zip(
            gt_trajectory, tracks_history, association_history):
        snap_by_id = {s.id: s for s in snaps}
        gt_range = {gt_id: range_m for gt_id, range_m, *_ in gts}
        for gt_id, track_id in mapping.items():
            if track_id is None:
                continue
            snap = snap_by_id.get(track_id)
            if snap is None:
                continue
            est_m = float(snap.state[0]) * range_per_bin
            sq_errors.append((est_m - gt_range[gt_id]) ** 2)
    if not sq_errors:
        return float("nan")
    return float(np.sqrt(np.mean(sq_errors)))


def count_false_confirmed_tracks(
    association_history: list,
    tracks_history: list,
) -> dict:
    """Count confirmed tracks that never associate with any GT (clutter-born).

    A track ID is *false* if it reaches status 'confirmed' in some frame yet is
    never the associated track of any GT anywhere in association_history. These
    are the spurious tracks the clutter spawns -- the suppression target of the
    confirm_M/confirm_N sweep (A.3.2). Tentative tracks are excluded: only a
    *confirmed* clutter track is a real false alarm at the track level.

    Args:
        association_history: list of (frame_idx, {gt_id: track_id_or_None}) --
            output of associate_per_frame() per frame.
        tracks_history: list of (frame_idx, [TrackSnapshot, ...]) -- output of
            run_pipeline().

    Returns:
        {"n_false_confirmed": int, "max_simultaneous_false": int}
          n_false_confirmed      — distinct false confirmed track IDs over the
                                    whole run (the clutter false-track count).
          max_simultaneous_false — peak number of distinct false confirmed
                                    tracks alive in any single frame.
    """
    real_ids = {
        track_id
        for _, mapping in association_history
        for track_id in mapping.values()
        if track_id is not None
    }

    false_ids = set()
    max_simultaneous = 0
    for _, snapshots in tracks_history:
        frame_false = {
            s.id for s in snapshots
            if s.status == "confirmed" and s.id not in real_ids
        }
        false_ids |= frame_false
        max_simultaneous = max(max_simultaneous, len(frame_false))

    return {
        "n_false_confirmed": len(false_ids),
        "max_simultaneous_false": max_simultaneous,
    }


from collections import namedtuple

# Immutable per-frame track snapshot. Storing live Track references would
# alias the tracker's mutable internal state — by end-of-run, every entry
# would reflect the latest state, hiding per-frame trajectory information.
TrackSnapshot = namedtuple(
    "TrackSnapshot", ["id", "state", "last_doppler_bin", "status"]
)


def _snapshot_track(tr) -> TrackSnapshot:
    return TrackSnapshot(
        id=tr.id,
        state=np.array(tr.state, copy=True),
        last_doppler_bin=float(tr.last_doppler_bin),
        status=str(tr.status),
    )


def run_pipeline(
    radar_cfg,
    initial_targets: list,
    n_frames: int,
    max_coast: int | None = None,
    process_noise_q: float | None = None,
    confirm_M: int | None = None,
    confirm_N: int | None = None,
    seed: int | None = None,
    fluctuation_seed: int | None = None,
    algorithm: str = "kf",
    drag_coefficient_k: float = 0.5,
) -> list:
    """Run generate_frame -> range_doppler -> cfar_2d -> cluster -> tracker.step
    for n_frames, with the lens chosen for this pilot:
        CFAR=os, clustering=peak(min_cells=2), tracker=gnn (defaults).

    Returns: list[(frame_idx, list of TrackSnapshot objects)].

    TrackSnapshot is an immutable per-frame copy of the relevant Track fields,
    needed because Tracker mutates Track objects in place across frames.

    max_coast / process_noise_q / confirm_M / confirm_N override defaults when
    not None. This is the hook the hyperparameter sweeps use to vary tracker
    behavior across runs without editing config files; confirm_M/confirm_N drive
    the A.3.2 confirmation sweep. Leave them at None to get default tuning.

    seed (when not None) re-seeds the radar noise via dataclasses.replace on the
    radar config, leaving the scenario JSON untouched. This is the hook the A.4
    Monte Carlo matrix uses to draw independent noise realizations per run; the
    GT trajectory is noise-independent, so only the detector input changes.

    fluctuation_seed (when not None) re-seeds the Swerling fade stream the
    same way `seed` re-seeds the noise: via dataclasses.replace on the radar
    config. Sweeps that want independent fade realisations per run vary
    both; leaving it None holds the scenario's fade pattern fixed across
    noise seeds.

    algorithm selects one of three tracker configurations:
        "kf"  → single Tracker with CV motion model (linear Kalman filter).
        "ekf" → single Tracker with drag motion model (Extended Kalman
                filter). drag_coefficient_k tunes the drag rate (bin units).
        "imm" → IMMTracker blending {CV, drag-EKF}. drag_coefficient_k
                tunes the drag-EKF constituent.
    """
    if algorithm == "imm":
        from processing.config import IMMConfig, MotionModelSpec
        from processing.imm import IMMTracker
        imm_kwargs = {
            "models": [
                MotionModelSpec(motion_model="cv"),
                MotionModelSpec(
                    motion_model="drag",
                    motion_params={"drag_coefficient_k": drag_coefficient_k},
                ),
            ],
            "transition_matrix": [[0.95, 0.05], [0.05, 0.95]],
            "initial_mode_probs": [0.5, 0.5],
        }
        if max_coast is not None:
            imm_kwargs["max_coast"] = max_coast
        if process_noise_q is not None:
            imm_kwargs["process_noise_q"] = process_noise_q
        if confirm_M is not None:
            imm_kwargs["confirm_M"] = confirm_M
        if confirm_N is not None:
            imm_kwargs["confirm_N"] = confirm_N
        tracker = IMMTracker(IMMConfig(**imm_kwargs))
    elif algorithm in ("kf", "ekf"):
        motion_model = "cv" if algorithm == "kf" else "drag"
        tracker_kwargs = {"method": "gnn",
                           "motion_model": motion_model,
                           "drag_coefficient_k": drag_coefficient_k}
        if max_coast is not None:
            tracker_kwargs["max_coast"] = max_coast
        if process_noise_q is not None:
            tracker_kwargs["process_noise_q"] = process_noise_q
        if confirm_M is not None:
            tracker_kwargs["confirm_M"] = confirm_M
        if confirm_N is not None:
            tracker_kwargs["confirm_N"] = confirm_N
        tracker = Tracker(TrackerConfig(**tracker_kwargs))
    else:
        raise ValueError(
            f"algorithm must be one of 'kf', 'ekf', 'imm'; got {algorithm!r}"
        )
    if seed is not None:
        radar_cfg = dataclasses.replace(radar_cfg, seed=seed)
    if fluctuation_seed is not None:
        radar_cfg = dataclasses.replace(
            radar_cfg, fluctuation_seed=fluctuation_seed)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    tracks_history = []
    for frame_idx in range(n_frames):
        frame = gen.generate_frame(initial_targets, frame_index=frame_idx)
        rd = range_doppler(
            frame, window="hann", dc_removal=True, clutter_removal=False,
        )
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(
            rd_db, method="os", pfa=1e-4, window="hann",
            dc_removal=True, clutter_removal=False,
        )
        mask[:, 0] = False
        detections = cluster(
            mask, rd_db, method="peak", min_cells=2, peak_neighborhood=3,
        )
        tracks = tracker.step(detections)
        snapshots = [_snapshot_track(t) for t in tracks]
        tracks_history.append((frame_idx, snapshots))
    return tracks_history


def _per_track_gt_lookup(association_history: list) -> dict:
    """Invert {gt_id: track_id} per frame into {track_id: {frame_idx: gt_id}}.

    Useful for coloring each track's trajectory by which GT it's associated
    with at each frame.
    """
    out = {}
    for frame_idx, mapping in association_history:
        for gt_id, track_id in mapping.items():
            if track_id is None:
                continue
            out.setdefault(track_id, {})[frame_idx] = gt_id
    return out


def plot_results(
    gt_trajectory: list,
    tracks_history: list,
    association_history: list,
    switch_events: list,
    out: Path,
    range_per_bin: float = 0.0749,
    scenario_name: str = "",
) -> None:
    """One panel: range vs frame. GT trajectories as solid lines; tracks as
    line+markers colored by their currently-associated GT. ID switch events
    annotated with vertical lines.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    gt_color = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}
    color_for_none = "tab:gray"

    # GT trajectories (solid lines).
    n_gts = len(gt_trajectory[0][1])
    for gt_id in range(n_gts):
        frames = [f for f, _ in gt_trajectory]
        ranges = [gt[gt_id][1] for _, gt in gt_trajectory]
        ax.plot(frames, ranges, color=gt_color.get(gt_id, "black"),
                linewidth=2.2, alpha=0.5, label=f"GT {gt_id} (truth)")

    # Track trajectories, colored per-frame by associated GT.
    per_track_gt = _per_track_gt_lookup(association_history)
    track_frames = {}     # track_id -> list of (frame_idx, range_bin, status)
    for frame_idx, tracks in tracks_history:
        for tr in tracks:
            if tr.status not in ("confirmed", "coasting"):
                continue
            # Track stores range as state[0] in range-bin units. Convert
            # to metres so it sits on the same y-axis as the GT lines.
            range_m = float(tr.state[0]) * range_per_bin
            track_frames.setdefault(tr.id, []).append(
                (frame_idx, range_m, tr.status)
            )

    for track_id, segments in track_frames.items():
        frames = [s[0] for s in segments]
        ranges = [s[1] for s in segments]
        statuses = [s[2] for s in segments]
        gt_at_frame = per_track_gt.get(track_id, {})
        colors = [
            gt_color.get(gt_at_frame.get(f), color_for_none)
            for f in frames
        ]
        # Thin line connecting points for trajectory visibility.
        for i in range(len(frames) - 1):
            ax.plot(frames[i:i+2], ranges[i:i+2], color=colors[i],
                    linewidth=1.0, alpha=0.6)
        # Per-frame scatter; filled for confirmed, hollow for coasted.
        ax.scatter(
            frames, ranges,
            c=colors,
            marker="o",
            s=[35 if st == "confirmed" else 18 for st in statuses],
            edgecolors=colors,
            linewidth=1.0,
        )

    # ID switch markers.
    for frame_idx, gt_id, old, new in switch_events:
        ax.axvline(frame_idx, color="orange", linestyle="--",
                   linewidth=1.0, alpha=0.7)
        ax.text(frame_idx, ax.get_ylim()[1] * 0.98,
                f" switch: gt{gt_id} {old}→{new}",
                rotation=90, va="top", ha="left", fontsize=8,
                color="darkorange")

    ax.set_xlabel("frame")
    ax.set_ylabel("range [m]")
    title_main = (
        f"Tracker stress test — {scenario_name}"
        if scenario_name else "Tracker stress test"
    )
    ax.set_title(
        f"{title_main} (CFAR=os, clustering=peak, tracker=gnn)\n"
        "Lines = ground truth · markers = tracker output (color = which GT "
        "this track is associated with; gray = no GT association)"
    )
    ax.grid(True, alpha=0.3)

    # Build an explicit legend distinguishing GT lines, track markers per
    # GT-association color, and confirmed vs coasting marker style. The
    # GT-specific entries are generated dynamically so single-target scenarios
    # don't see a phantom "GT 1" in the legend.
    legend_handles = []
    for gt_id in range(n_gts):
        legend_handles.append(plt.Line2D(
            [0], [0], color=gt_color.get(gt_id, "black"),
            linewidth=2.2, alpha=0.5, label=f"GT {gt_id} (truth line)",
        ))
    for gt_id in range(n_gts):
        legend_handles.append(plt.Line2D(
            [0], [0], color=gt_color.get(gt_id, "black"), marker="o",
            markersize=8, linestyle="",
            label=f"track assoc. with GT {gt_id}",
        ))
    legend_handles.append(plt.Line2D(
        [0], [0], color=color_for_none, marker="o",
        markersize=8, linestyle="", label="track (no association)",
    ))
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def print_summary(summary: dict, n_frames: int, scenario_path: str) -> None:
    """Compact human-readable summary of the run."""
    print()
    print("─" * 69)
    print(f"Tracker stress test — {_scenario_display_name(scenario_path)} scenario")
    print("─" * 69)
    print(f"Pipeline:    CFAR=os, clustering=peak(min_cells=2), tracker=gnn")
    print(f"Scenario:    {scenario_path}")
    print(f"Run length:  {n_frames} frames "
          f"({n_frames * 0.05:.1f} s at 20 Hz)")
    print()
    print("Per-target metrics:")
    headers = ["gt_id", "Pd", "distinct_tracks", "n_assoc_frames",
               "coast_frames"]
    print(f"  {headers[0]:8s} {headers[1]:>6s} {headers[2]:>16s} "
          f"{headers[3]:>16s} {headers[4]:>13s}")
    for gt_id in sorted(summary["per_gt"]):
        m = summary["per_gt"][gt_id]
        print(f"  gt_{gt_id:<5d} {m['pd']:>6.2f} "
              f"{m['distinct_track_ids']:>16d} "
              f"{m['n_frames_associated']:>16d} "
              f"{m['coast_frames']:>13d}")
    print()
    if summary["switch_events"]:
        print("ID-switch events:")
        for frame_idx, gt_id, old, new in summary["switch_events"]:
            print(f"  frame {frame_idx}: gt_{gt_id}: track {old} → {new}")
    else:
        print("ID-switch events:    (none)")
    print()
    cls = summary["classification"]
    marker = "✓" if cls == "CLEAN" else "⚠"
    print(f"Classification:      {marker} {cls}")
    print("─" * 69)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO_PATH,
                   help="path to scenario JSON")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="number of frames to run")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default: derived from scenario name)")
    p.add_argument("--max-coast", type=int, default=None,
                   help="override TrackerConfig.max_coast (default: 5)")
    p.add_argument("--algorithm", type=str, default="kf",
                   choices=["kf", "ekf", "imm"],
                   help="tracker algorithm: kf (linear Kalman, CV motion), "
                        "ekf (Extended Kalman, drag motion), imm "
                        "(IMM blending CV+drag). Default: kf.")
    p.add_argument("--drag-coefficient-k", type=float, default=0.5,
                   help="drag coefficient k in bin-units (used by ekf and "
                        "imm algorithms; default: 0.5)")
    p.add_argument("--process-noise-q", type=float, default=None,
                   help="override TrackerConfig.process_noise_q (default: 0.5)")
    args = p.parse_args()

    if args.out is None:
        args.out = _default_out_path(args.scenario)

    radar_cfg, initial_targets = load_scenario(args.scenario)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    print(f"loaded {args.scenario}: {len(initial_targets)} targets, "
          f"{args.frames} frames")

    gt_trajectory = build_gt_trajectory(initial_targets, gen, args.frames)
    tracks_history = run_pipeline(
        radar_cfg, initial_targets, args.frames,
        max_coast=args.max_coast, process_noise_q=args.process_noise_q,
        algorithm=args.algorithm,
        drag_coefficient_k=args.drag_coefficient_k,
    )

    association_history = []
    for (f_gt, gt_at_frame), (f_tr, tracks) in zip(gt_trajectory,
                                                    tracks_history):
        assert f_gt == f_tr, "frame index mismatch"
        mapping = associate_per_frame(gt_at_frame, tracks)
        association_history.append((f_gt, mapping))

    switch_events = count_id_switches(association_history,
                                       min_persistence_frames=2)
    summary = evaluate_run(association_history, switch_events,
                            tracks_history)

    # Convert track range_bin (state[0]) to metres for the plot's y-axis.
    range_per_bin = float(gen.range_axis_m()[1] - gen.range_axis_m()[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_results(gt_trajectory, tracks_history, association_history,
                 switch_events, args.out, range_per_bin=range_per_bin,
                 scenario_name=_scenario_display_name(args.scenario))
    print_summary(summary, args.frames, args.scenario)


if __name__ == "__main__":
    main()
