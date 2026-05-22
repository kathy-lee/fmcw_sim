#!/usr/bin/env python3
"""
Beat-signal FMCW simulator.

This generates ADC-like complex samples after analog dechirping:
    frame shape = [n_chirps, n_range]

It does not generate the RF chirp itself. It directly generates the sampled
beat signal used as input to range FFT / Doppler FFT processing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json
import numpy as np

C = 299_792_458.0


@dataclass
class Target:
    range_m: float
    velocity_mps: float
    amplitude: float = 1.0
    phase_rad: float = 0.0


@dataclass
class ClutterRegion:
    """Spatially-bounded Gaussian noise region with elevated variance.

    Cells inside the region receive an additional complex-Gaussian
    contribution whose per-cell variance is `noise_power * 10^(cnr_db/10)`
    above the base noise. velocity bounds = None means unbounded
    (the region covers all Doppler — Level 1 in the clutter-edge eval).
    """
    range_m_min: float
    range_m_max: float
    velocity_mps_min: float | None = None
    velocity_mps_max: float | None = None
    cnr_db: float = 10.0

    def __post_init__(self):
        if self.range_m_min >= self.range_m_max:
            raise ValueError(
                f"ClutterRegion.range_m_min ({self.range_m_min}) must be < "
                f"range_m_max ({self.range_m_max})"
            )
        if (self.velocity_mps_min is not None
                and self.velocity_mps_max is not None
                and self.velocity_mps_min >= self.velocity_mps_max):
            raise ValueError(
                f"ClutterRegion.velocity_mps_min ({self.velocity_mps_min}) "
                f"must be < velocity_mps_max ({self.velocity_mps_max})"
            )
        if self.cnr_db < 0:
            raise ValueError(
                f"ClutterRegion.cnr_db must be >= 0, got {self.cnr_db}"
            )


@dataclass
class RadarConfig:
    fc_hz: float = 77e9
    bandwidth_hz: float = 2e9
    ramp_duration_s: float = 40e-6
    chirp_period_s: float = 50e-6
    n_range: int = 256
    n_chirps: int = 64
    noise_power: float = 1e-4
    leakage_amplitude: float = 0.0
    frame_period_s: float | None = None
    # Default seed != 0 so this generator's RNG sequence is decoupled from
    # processing.cfar's calibration RNG (which uses seed=0 internally). With
    # the same seed, the two paths produce identical noise realizations, which
    # makes any downstream Pfa measurement self-consistent with the calibration
    # data rather than out-of-sample.
    seed: int = 99999

    def __post_init__(self):
        if self.frame_period_s is None:
            self.frame_period_s = self.n_chirps * self.chirp_period_s


class FMCWBeatSignalGenerator:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.fs_hz = cfg.n_range / cfg.ramp_duration_s
        self.slope_hz_per_s = cfg.bandwidth_hz / cfg.ramp_duration_s
        self.lambda_m = C / cfg.fc_hz

        self._validate_config()

    def _validate_config(self) -> None:
        if self.cfg.chirp_period_s < self.cfg.ramp_duration_s:
            raise ValueError("chirp_period_s must be >= ramp_duration_s")
        if self.cfg.n_range <= 0 or self.cfg.n_chirps <= 0:
            raise ValueError("n_range and n_chirps must be positive")
        if self.cfg.bandwidth_hz <= 0 or self.cfg.ramp_duration_s <= 0:
            raise ValueError("bandwidth_hz and ramp_duration_s must be positive")
        if self.cfg.frame_period_s < self.cfg.n_chirps * self.cfg.chirp_period_s:
            raise ValueError(
                "frame_period_s must be >= n_chirps * chirp_period_s"
            )

    @property
    def max_unambiguous_range_m(self) -> float:
        # Positive beat frequencies up to fs/2.
        fb_max = self.fs_hz / 2.0
        return C * fb_max / (2.0 * self.slope_hz_per_s)

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        # Doppler PRF = 1 / chirp_period_s, unambiguous fd is +/- PRF/2.
        fd_max = 1.0 / (2.0 * self.cfg.chirp_period_s)
        return self.lambda_m * fd_max / 2.0

    def range_axis_m(self, n_fft: int | None = None, positive_only: bool = True) -> np.ndarray:
        n_fft = int(n_fft or self.cfg.n_range)
        freqs = np.fft.fftfreq(n_fft, d=1.0 / self.fs_hz)
        ranges = C * freqs / (2.0 * self.slope_hz_per_s)
        if positive_only:
            return ranges[: n_fft // 2]
        return ranges

    def velocity_axis_mps(self, n_fft: int | None = None, shifted: bool = True) -> np.ndarray:
        n_fft = int(n_fft or self.cfg.n_chirps)
        fd = np.fft.fftfreq(n_fft, d=self.cfg.chirp_period_s)
        if shifted:
            fd = np.fft.fftshift(fd)
        return self.lambda_m * fd / 2.0

    def frame_start_time_s(self, frame_index: int) -> float:
        if self.cfg.frame_period_s is not None:
            return frame_index * self.cfg.frame_period_s
        return frame_index * self.cfg.n_chirps * self.cfg.chirp_period_s

    def generate_frame(self, targets: list[Target], frame_index: int = 0) -> np.ndarray:
        """
        Returns a complex64 frame with shape [n_chirps, n_range].

        Signal convention:
            positive target range -> positive range FFT bin
            positive radial velocity -> positive Doppler bin after fftshift

        Motion model:
            stop-and-hop across chirps: range is constant inside one chirp,
            but may change from chirp to chirp and frame to frame.
        """
        cfg = self.cfg
        m = np.arange(cfg.n_chirps, dtype=np.float64)[:, None]
        n = np.arange(cfg.n_range, dtype=np.float64)[None, :]

        t_fast = n / self.fs_hz
        t_slow = self.frame_start_time_s(frame_index) + m * cfg.chirp_period_s

        x = np.zeros((cfg.n_chirps, cfg.n_range), dtype=np.complex128)

        for target in targets:
            # Range at each chirp. Constant inside a chirp.
            range_m = target.range_m + target.velocity_mps * t_slow
            range_m = np.maximum(range_m, 0.0)

            fb_hz = 2.0 * self.slope_hz_per_s * range_m / C
            fd_hz = 2.0 * target.velocity_mps / self.lambda_m

            phase = 2.0 * np.pi * (
                fb_hz * t_fast +
                fd_hz * t_slow
            ) + target.phase_rad

            x += target.amplitude * np.exp(1j * phase)

        if cfg.leakage_amplitude != 0.0:
            x += cfg.leakage_amplitude

        if cfg.noise_power > 0.0:
            sigma = np.sqrt(cfg.noise_power / 2.0)
            noise = sigma * (
                self.rng.standard_normal(x.shape) +
                1j * self.rng.standard_normal(x.shape)
            )
            x += noise

        return x.astype(np.complex64)

    def ground_truth(self, targets: list[Target], frame_index: int = 0) -> list[dict[str, float]]:
        """Approximate target range/Doppler bin locations at frame center."""
        cfg = self.cfg
        frame_center_time = self.frame_start_time_s(frame_index) + 0.5 * cfg.n_chirps * cfg.chirp_period_s

        range_bin_spacing_hz = self.fs_hz / cfg.n_range
        doppler_bin_spacing_hz = 1.0 / (cfg.n_chirps * cfg.chirp_period_s)

        out: list[dict[str, float]] = []
        for target in targets:
            r = max(0.0, target.range_m + target.velocity_mps * frame_center_time)
            fb = 2.0 * self.slope_hz_per_s * r / C
            fd = 2.0 * target.velocity_mps / self.lambda_m

            range_bin = fb / range_bin_spacing_hz
            doppler_bin_unshifted = fd / doppler_bin_spacing_hz
            doppler_bin_shifted = doppler_bin_unshifted + cfg.n_chirps / 2.0

            out.append({
                "range_m": float(r),
                "velocity_mps": float(target.velocity_mps),
                "amplitude": float(target.amplitude),
                "beat_frequency_hz": float(fb),
                "doppler_frequency_hz": float(fd),
                "range_bin": float(range_bin),
                "doppler_bin_unshifted": float(doppler_bin_unshifted),
                "doppler_bin_shifted": float(doppler_bin_shifted),
            })
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "derived": {
                "fs_hz": self.fs_hz,
                "slope_hz_per_s": self.slope_hz_per_s,
                "lambda_m": self.lambda_m,
                "max_unambiguous_range_m": self.max_unambiguous_range_m,
                "max_unambiguous_velocity_mps": self.max_unambiguous_velocity_mps,
            },
        }


def load_scenario(path: str) -> tuple[RadarConfig, list[Target]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    radar = data.get("radar", {})
    targets_raw = data.get("targets", [])

    cfg = RadarConfig(**radar)
    targets = [Target(**item) for item in targets_raw]
    return cfg, targets


def default_scenario() -> tuple[RadarConfig, list[Target]]:
    cfg = RadarConfig(
        fc_hz=77e9,
        bandwidth_hz=2e9,
        ramp_duration_s=40e-6,
        chirp_period_s=50e-6,
        n_range=256,
        n_chirps=64,
        noise_power=1e-4,
        leakage_amplitude=0.02,
        frame_period_s=0.05,
        seed=0,
    )
    targets = [
        Target(range_m=3.0, velocity_mps=5.0, amplitude=1.0),
        Target(range_m=6.0, velocity_mps=-3.0, amplitude=0.6),
    ]
    return cfg, targets
