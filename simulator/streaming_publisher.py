#!/usr/bin/env python3
"""Publish simulated FMCW beat-signal frames over ZeroMQ PUB socket."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import zmq
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, default_scenario, load_scenario


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="tcp://127.0.0.1:5555")
    p.add_argument("--topic", default="fmcw")
    p.add_argument("--scenario", default=None, help="JSON scenario file")
    p.add_argument("--frames", type=int, default=0, help="0 means run forever")
    p.add_argument("--startup-delay", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.scenario:
        cfg, targets = load_scenario(args.scenario)
    else:
        cfg, targets = default_scenario()

    gen = FMCWBeatSignalGenerator(cfg)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(args.bind)

    print(f"Publishing on {args.bind}, topic={args.topic!r}")
    print(json.dumps(gen.metadata(), indent=2))
    print("Targets:")
    for t in targets:
        print("  ", asdict(t))

    time.sleep(args.startup_delay)

    period = cfg.frame_period_s
    next_t = time.monotonic()
    frame_id = 0

    try:
        while args.frames == 0 or frame_id < args.frames:
            frame = gen.generate_frame(targets, frame_index=frame_id)

            header = {
                "frame_id": frame_id,
                "timestamp_unix_s": time.time(),
                "topic": args.topic,
                "dtype": "complex64",
                "shape": list(frame.shape),
                "order": "C",
                "metadata": gen.metadata(),
                "ground_truth": gen.ground_truth(targets, frame_index=frame_id),
            }

            sock.send_multipart([
                args.topic.encode("utf-8"),
                json.dumps(header).encode("utf-8"),
                frame.ravel(order="C").tobytes(),
            ])

            if frame_id % max(1, int(round(1.0 / cfg.frame_period_s))) == 0:
                print(f"\nSent frame {frame_id}")

            frame_id += 1
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # If processing falls behind, resynchronize instead of accumulating lag.
                next_t = time.monotonic()

        print(f"Finished publishing {frame_id} frames.")

    except KeyboardInterrupt:
        print("\nPublisher stopped.")
    finally:
        sock.close(linger=0)


if __name__ == "__main__":
    main()
