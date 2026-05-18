"""Layered range-Doppler visualization with toggleable overlays."""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from processing.clustering import Detection
from processing.tracking import Track

_LAYERS = {"cfar", "clusters", "tracks"}


def doppler_bin_to_mps(doppler_bin: float, n_doppler: int,
                       chirp_period_s: float, lambda_m: float) -> float:
    """Convert a (centered) Doppler bin index to radial velocity in m/s."""
    return (doppler_bin - n_doppler / 2.0) / (n_doppler * chirp_period_s) * lambda_m / 2.0


class LayeredRDView:
    """Owns the matplotlib figure and stacked overlays.

    Public API:
        update(rd_db, mask, detections, tracks)
        toggle_layer(name)
        on_reset(callback)
        show()
    """

    def __init__(
        self,
        n_doppler: int,
        n_range: int,
        range_res_m: float,
        chirp_period_s: float,
        lambda_m: float,
        vmin: float = -120.0,
        vmax: float = 20.0,
    ):
        self.n_doppler = n_doppler
        self.n_range = n_range
        self.range_res_m = range_res_m
        self.chirp_period_s = chirp_period_s
        self.lambda_m = lambda_m
        self._reset_cb: Optional[Callable[[], None]] = None
        self._visible = {"cfar": True, "clusters": True, "tracks": True}
        # Track-id-keyed cache of per-track artist groups; rebuilt each frame for simplicity.
        self._track_artists: list = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel("Range bin")
        self.ax.set_ylabel("Doppler bin")
        self.ax.set_title("Frame 0  |  tracks: 0")

        # Heatmap (background).
        self._heatmap = self.ax.imshow(
            np.full((n_doppler, n_range), vmin, dtype=np.float32),
            aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
        )
        self.fig.colorbar(self._heatmap, ax=self.ax, label="dB")

        # CFAR mask layer (transparent → red).
        cfar_cmap = mcolors.LinearSegmentedColormap.from_list(
            "cfar_overlay", [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 0.45))], N=2,
        )
        self._mask_img = self.ax.imshow(
            np.zeros((n_doppler, n_range), dtype=np.uint8),
            aspect="auto", origin="lower", cmap=cfar_cmap, vmin=0, vmax=1,
            interpolation="nearest",
        )

        # Cluster scatter (centroids).
        self._cluster_scatter = self.ax.scatter([], [], marker="+", s=80,
                                                c="yellow", linewidths=1.5)

        # Secondary axes (decorative — bins ↔ physical units).
        rr = range_res_m
        self.ax.secondary_xaxis(
            "top",
            functions=(lambda b, rr=rr: b * rr, lambda m, rr=rr: m / rr),
        ).set_xlabel("Range (m)")
        n_d = n_doppler
        cp = chirp_period_s
        lm = lambda_m
        def b2mps(b, n_d=n_d, cp=cp, lm=lm):
            return doppler_bin_to_mps(b, n_d, cp, lm)
        def mps2b(v, n_d=n_d, cp=cp, lm=lm):
            return v * 2.0 / lm * (n_d * cp) + n_d / 2.0
        self.ax.secondary_yaxis("right", functions=(b2mps, mps2b)).set_ylabel("Velocity (m/s)")

        # Hotkeys.
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ---- Public API ----

    def show(self) -> None:
        plt.show(block=False)

    def update(self, rd_db, mask, detections, tracks, frame_id: int) -> None:
        n_confirmed = sum(1 for t in tracks if t.status == "confirmed")
        self.ax.set_title(
            f"Frame {frame_id}  |  detections: {len(detections)}  |  "
            f"tracks: {len(tracks)} ({n_confirmed} confirmed)"
        )
        self._heatmap.set_data(rd_db)

        if self._visible["cfar"] and mask is not None:
            self._mask_img.set_data(mask.astype(np.uint8))
            self._mask_img.set_visible(True)
        else:
            self._mask_img.set_visible(False)

        if self._visible["clusters"] and detections:
            xs = [d.range_bin for d in detections]
            ys = [d.doppler_bin for d in detections]
            self._cluster_scatter.set_offsets(np.column_stack([xs, ys]))
            self._cluster_scatter.set_visible(True)
        else:
            self._cluster_scatter.set_offsets(np.empty((0, 2)))
            self._cluster_scatter.set_visible(self._visible["clusters"])

        # Tracks: clear and recreate. Cheap at our scale.
        for artist in self._track_artists:
            artist.remove()
        self._track_artists = []
        if self._visible["tracks"]:
            cmap = plt.get_cmap("tab20")
            for t in tracks:
                color = cmap(t.id % 20)
                trail_x = [p[0] for p in (t.history or [])]
                trail_y = [p[1] for p in (t.history or [])]
                if trail_x:
                    line, = self.ax.plot(
                        trail_x, trail_y,
                        color=color, alpha=0.6,
                        linestyle=("--" if t.status == "tentative" else "-"),
                        linewidth=1.2,
                    )
                    self._track_artists.append(line)
                marker_face = color if t.status != "coasting" else "none"
                marker_size = 50 if t.status == "confirmed" else 25
                sc = self.ax.scatter(
                    [t.state[0]], [t.last_doppler_bin],
                    s=marker_size, edgecolors=color, facecolors=marker_face, linewidths=1.5,
                )
                self._track_artists.append(sc)
                # Label.
                rng_m = t.state[0] * self.range_res_m
                vel_mps = doppler_bin_to_mps(
                    t.last_doppler_bin, self.n_doppler,
                    self.chirp_period_s, self.lambda_m,
                )
                txt = self.ax.annotate(
                    f"T{t.id}  r={rng_m:.2f}m  v={vel_mps:+.2f}m/s",
                    xy=(t.state[0], t.last_doppler_bin),
                    xytext=(6, 4), textcoords="offset points",
                    color=color, fontsize=8,
                )
                self._track_artists.append(txt)

        # Force immediate redraw. draw_idle() only schedules at the next idle
        # tick, which never arrives if the caller's loop is tight (as with the
        # ZMQ subscriber). draw() + flush_events() matches receive_real_time.py.
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass  # headless backends (Agg in tests) don't have a real event loop

    def toggle_layer(self, name: str) -> None:
        if name not in _LAYERS:
            raise ValueError(f"unknown layer {name!r}; expected one of {sorted(_LAYERS)}")
        self._visible[name] = not self._visible[name]

    def on_reset(self, callback: Callable[[], None]) -> None:
        self._reset_cb = callback

    # ---- Internal ----

    def _invoke_reset(self) -> None:
        if self._reset_cb is not None:
            self._reset_cb()

    def _on_key(self, event) -> None:
        key = event.key
        if key == "c":
            self.toggle_layer("cfar")
        elif key == "k":
            self.toggle_layer("clusters")
        elif key == "t":
            self.toggle_layer("tracks")
        elif key == "r":
            self._invoke_reset()
        elif key == "q":
            plt.close(self.fig)
