#!/usr/bin/env python3
"""
Beat-signal FMCW simulator.

This generates ADC-like complex samples after analog dechirping:
    frame shape = [n_chirps, n_range]

It does not generate the RF chirp itself. It directly generates the sampled
beat signal used as input to range FFT / Doppler FFT processing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
import json
import math
import numpy as np

from simulator.motion_models import range_at
from simulator.fluctuation_models import amplitude_factors, validate_fluctuation
from simulator.extended_target import make_scatterers

C = 299_792_458.0


@dataclass
class Target:
    # Cartesian world frame (boresight = +x). Radial range / velocity / azimuth
    # are DERIVED (read-only properties below), so motion_models and
    # generate_frame keep their radial interface unchanged for on-axis targets.
    x: float
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    amplitude: float = 1.0
    phase_rad: float = 0.0
    motion_model: str = "constant_velocity"
    motion_params: dict = field(default_factory=dict)
    # Optional RCS fluctuation (Swerling), e.g.
    # {"model": "swerling1", "frame_corr": 0.9}. None = steady amplitude.
    # Treated as constant for the lifetime of a generator instance.
    fluctuation: dict | None = None
    # Optional extended-target model (K coherent scatterers over a range
    # extent), e.g. {"n_scatterers": 10, "range_extent_m": 0.6}. None = point
    # target (K=1). Spatial; orthogonal to the temporal `fluctuation` knob.
    extent: dict | None = None

    @property
    def range_m(self) -> float:
        # y == 0 fast path is EXACT: avoids sqrt(x*x) != x by 1 ULP, which would
        # shift the beat phase and break byte-for-byte for migrated scenarios.
        if self.y == 0.0:
            return abs(self.x)
        return math.hypot(self.x, self.y)

    @property
    def velocity_mps(self) -> float:
        # Radial velocity d(range)/dt. On-axis: sign(x)*vx (== vx after migration,
        # where x > 0); off-axis: projection of (vx, vy) onto the range direction.
        if self.y == 0.0:
            return math.copysign(self.vx, self.x) if self.x != 0.0 else self.vx
        r = math.hypot(self.x, self.y)
        return (self.x * self.vx + self.y * self.vy) / r

    @property
    def azimuth_rad(self) -> float:
        return math.atan2(self.y, self.x)


@dataclass
class ClutterRegion:
    """Spatially-bounded Gaussian noise region with elevated variance.

    Cells inside the region receive an additional complex-Gaussian
    contribution whose per-cell variance is `noise_power * 10^(cnr_db/10)`
    above the base noise. velocity bounds = None means unbounded
    (the region covers all Doppler — Level 1 in the clutter-edge eval).

    Time window: frame_min / frame_max bracket the frames during which the
    region is active (inclusive lower, exclusive upper, matching range()).
    None means unbounded on that side. With both at None (the default), the
    region is active for every frame — preserves prior behavior.
    """
    range_m_min: float
    range_m_max: float
    velocity_mps_min: float | None = None
    velocity_mps_max: float | None = None
    cnr_db: float = 10.0
    frame_min: int | None = None
    frame_max: int | None = None

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
        if (self.frame_min is not None and self.frame_max is not None
                and self.frame_min >= self.frame_max):
            raise ValueError(
                f"ClutterRegion.frame_min ({self.frame_min}) must be < "
                f"frame_max ({self.frame_max})"
            )

    def is_active(self, frame_index: int) -> bool:
        """Return True when this region contributes clutter on `frame_index`."""
        if self.frame_min is not None and frame_index < self.frame_min:
            return False
        if self.frame_max is not None and frame_index >= self.frame_max:
            return False
        return True


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
    # Seed for the per-target RCS-fluctuation stream (Swerling models): a
    # THIRD RNG stream, independent of both the thermal-noise stream
    # (`seed`) and CFAR's calibration stream (0). Decoupling it lets a
    # Monte Carlo redraw the noise while holding the fade pattern fixed,
    # or vary both. 77777 != 0 and != 99999 by design.
    fluctuation_seed: int = 77777
    # Seed for the per-target extended-target scatterer stream: a FOURTH RNG
    # stream, independent of noise (`seed`), CFAR calibration (0), and the
    # fluctuation stream (`fluctuation_seed`). Scatterers are drawn once per
    # target from default_rng([extent_seed, target_index]). 555 by design.
    extent_seed: int = 555
    # Optional clutter regions; empty by default (existing behavior unchanged).
    clutter: list = field(default_factory=list)   # list[ClutterRegion]

    def __post_init__(self):
        if self.frame_period_s is None:
            self.frame_period_s = self.n_chirps * self.chirp_period_s


class FMCWBeatSignalGenerator:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Lazily-grown per-target fluctuation factor sequences, keyed by the
        # target's index in the `targets` list passed to generate_frame().
        self._fluct_factors: dict[int, np.ndarray] = {}
        # Lazily-drawn per-target scatterer sets (delta_r_m, complex amps),
        # keyed by target index; drawn once from the extent_seed stream.
        self._extent_scatterers: dict[int, tuple] = {}

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

    def fluctuation_factor(
        self, target_index: int, target: Target, frame_index: int
    ) -> float:
        """Per-frame amplitude factor for a fluctuating target (1.0 if steady).

        Factors come from a dedicated stream seeded by
        (cfg.fluctuation_seed, target_index) — independent of the
        thermal-noise stream, so redrawing the noise (RadarConfig.seed)
        leaves the fade pattern unchanged unless fluctuation_seed is also
        changed. Sequences are cached per target index and regrown by
        regenerating from a fresh identically-seeded rng; the
        prefix-stability of amplitude_factors guarantees history is preserved.
        """
        if target.fluctuation is None:
            return 1.0
        model, frame_corr = validate_fluctuation(target.fluctuation)
        seq = self._fluct_factors.get(target_index)
        if seq is None or seq.size <= frame_index:
            n = max(128, 1 << (frame_index + 1).bit_length())
            rng = np.random.default_rng(
                [self.cfg.fluctuation_seed, target_index])
            seq = amplitude_factors(model, n, rng, frame_corr)
            self._fluct_factors[target_index] = seq
        return float(seq[frame_index])

    def _scatterers_for(self, target_index: int, target: Target) -> tuple:
        """(delta_r_m, amplitudes) for an extended target, drawn once.

        Returns None for a point target (extent is None). Scatterers come
        from default_rng([extent_seed, target_index]) — independent of the
        noise and fluctuation streams — and are cached for the generator's
        lifetime (static geometry; glint arises from target motion).
        """
        if target.extent is None:
            return None
        cached = self._extent_scatterers.get(target_index)
        if cached is None:
            rng = np.random.default_rng(
                [self.cfg.extent_seed, target_index])
            cached = make_scatterers(target.extent, target.amplitude, rng)
            self._extent_scatterers[target_index] = cached
        return cached

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

        for target_index, target in enumerate(targets):
            # Per-frame position/velocity from the target's motion model.
            # Stop-and-hop approximation: velocity is constant within one
            # frame (frames are ~3.2ms long, intra-frame drag/maneuver
            # change is negligible), so range advances linearly across
            # chirps within the frame using v_at_frame_start.
            r_at_frame_start, v_at_frame_start = range_at(
                target, frame_index, cfg.frame_period_s,
            )
            range_m = r_at_frame_start + v_at_frame_start * m * cfg.chirp_period_s
            range_m = np.maximum(range_m, 0.0)

            fb_hz = 2.0 * self.slope_hz_per_s * range_m / C
            fd_hz = 2.0 * v_at_frame_start / self.lambda_m

            phase = 2.0 * np.pi * (
                fb_hz * t_fast +
                fd_hz * t_slow
            ) + target.phase_rad

            # Original point-target accumulation (kept for reference; the
            # branch below generalises it to K scatterers, point = K=1):
            #   amp = target.amplitude * self.fluctuation_factor(
            #       target_index, target, frame_index)
            #   x += amp * np.exp(1j * phase)
            fluct = self.fluctuation_factor(target_index, target, frame_index)
            scatterers = self._scatterers_for(target_index, target)
            if scatterers is None:
                amp = target.amplitude * fluct
                x += amp * np.exp(1j * phase)
            else:
                delta_r, amps = scatterers
                for dr_i, a_i in zip(delta_r, amps):
                    # Each scatterer is offset in range; its beat phase shifts
                    # by the offset's contribution. range_m here is the array
                    # of per-chirp ranges for the target centre.
                    fb_i = 2.0 * self.slope_hz_per_s * (range_m + dr_i) / C
                    phase_i = 2.0 * np.pi * (
                        fb_i * t_fast + fd_hz * t_slow
                    ) + target.phase_rad
                    x += (fluct * a_i) * np.exp(1j * phase_i)

        if cfg.leakage_amplitude != 0.0:
            x += cfg.leakage_amplitude

        for region in cfg.clutter:
            if not region.is_active(frame_index):
                continue
            x += self._generate_clutter_beat_signal(region)

        if cfg.noise_power > 0.0:
            sigma = np.sqrt(cfg.noise_power / 2.0)
            noise = sigma * (
                self.rng.standard_normal(x.shape) +
                1j * self.rng.standard_normal(x.shape)
            )
            x += noise

        return x.astype(np.complex64)

    def _generate_clutter_beat_signal(self, region: ClutterRegion) -> np.ndarray:
        """Synthesize the beat-signal contribution for one clutter region.

        Approach: build the desired RD-bin-space description (complex
        Gaussian inside the region's bins, zero outside, scaled by
        sqrt(noise_power) * 10^(cnr_db/20)), then invert the FFT chain
        that range_doppler applies (Doppler-fftshift, Doppler-IFFT, range
        zero-pad, range-IFFT).

        Window losses in range_doppler are not pre-compensated, so the
        realised CNR in the RD map is approximately cnr_db within a few dB
        (off by ~12 dB for Hann × Hann). Adequate for the clutter-edge
        experiment; exact calibration would require dividing by window
        coherent gain.

        Returns:
            Complex (n_chirps, n_range) beat-signal contribution to add to x.
        """
        cfg = self.cfg
        n_d = cfg.n_chirps
        n_r_out = cfg.n_range // 2
        n_r_in = cfg.n_range

        range_axis = self.range_axis_m()                  # (n_r_out,)
        vel_axis = self.velocity_axis_mps()               # (n_d,), fftshifted

        r_mask = (range_axis >= region.range_m_min) & (range_axis <= region.range_m_max)
        d_mask = np.ones(n_d, dtype=bool)
        if region.velocity_mps_min is not None:
            d_mask &= (vel_axis >= region.velocity_mps_min)
        if region.velocity_mps_max is not None:
            d_mask &= (vel_axis <= region.velocity_mps_max)

        cells_d = np.where(d_mask)[0]
        cells_r = np.where(r_mask)[0]
        if cells_d.size == 0 or cells_r.size == 0:
            return np.zeros((n_d, n_r_in), dtype=np.complex64)

        # CNR is defined relative to the RD-output noise floor per cell, which
        # for rect window is N_d × N_r_in × noise_power (the FFT processing
        # gain applied to per-sample noise). Windowed FFTs change this by a
        # multiplicative factor that affects both clutter and noise equally,
        # so the realised CNR is preserved (within ~1-2 dB).
        noise_floor_per_cell = n_d * n_r_in * cfg.noise_power
        sigma = np.sqrt(noise_floor_per_cell) * 10.0 ** (region.cnr_db / 20.0)
        rd_clutter = np.zeros((n_d, n_r_out), dtype=np.complex128)
        noise = (
            self.rng.standard_normal((cells_d.size, cells_r.size)) +
            1j * self.rng.standard_normal((cells_d.size, cells_r.size))
        ) / np.sqrt(2.0)
        rd_clutter[np.ix_(cells_d, cells_r)] = sigma * noise

        # Invert range_doppler's FFT chain. numpy's ifft is the exact inverse
        # of fft (fft(ifft(X)) == X), so no extra N-scaling needed.
        # Windowing is not pre-compensated — produces approximate CNR.
        rfft = np.fft.ifftshift(rd_clutter, axes=0)
        rfft = np.fft.ifft(rfft, axis=0)
        rfft_full = np.zeros((n_d, n_r_in), dtype=np.complex128)
        rfft_full[:, :n_r_out] = rfft
        x_clutter = np.fft.ifft(rfft_full, axis=1)

        return x_clutter.astype(np.complex64)

    def ground_truth(self, targets: list[Target], frame_index: int = 0) -> list[dict[str, float]]:
        """Approximate target range/Doppler bin locations at frame center."""
        cfg = self.cfg
        frame_center_time = self.frame_start_time_s(frame_index) + 0.5 * cfg.n_chirps * cfg.chirp_period_s

        range_bin_spacing_hz = self.fs_hz / cfg.n_range
        doppler_bin_spacing_hz = 1.0 / (cfg.n_chirps * cfg.chirp_period_s)

        out: list[dict[str, float]] = []
        frame_center_offset = 0.5 * cfg.n_chirps * cfg.chirp_period_s
        for target in targets:
            r_start, v_start = range_at(
                target, frame_index, cfg.frame_period_s,
            )
            # Frame-center position via the same stop-and-hop assumption
            # generate_frame uses: velocity constant within one frame.
            r = max(0.0, r_start + v_start * frame_center_offset)
            fb = 2.0 * self.slope_hz_per_s * r / C
            fd = 2.0 * v_start / self.lambda_m

            range_bin = fb / range_bin_spacing_hz
            doppler_bin_unshifted = fd / doppler_bin_spacing_hz
            doppler_bin_shifted = doppler_bin_unshifted + cfg.n_chirps / 2.0

            if target.extent is None:
                range_half_width_bins = 0.0
            else:
                from simulator.extended_target import validate_extent
                _, range_extent_m = validate_extent(target.extent)
                # Half-extent expressed in range bins, via the same fb->bin
                # mapping used above (so it tracks the generator's geometry).
                fb_half = 2.0 * self.slope_hz_per_s * (0.5 * range_extent_m) / C
                range_half_width_bins = fb_half / range_bin_spacing_hz

            out.append({
                "range_m": float(r),
                "velocity_mps": float(v_start),
                "amplitude": float(target.amplitude),
                "beat_frequency_hz": float(fb),
                "doppler_frequency_hz": float(fd),
                "range_bin": float(range_bin),
                "doppler_bin_unshifted": float(doppler_bin_unshifted),
                "doppler_bin_shifted": float(doppler_bin_shifted),
                "range_half_width_bins": float(range_half_width_bins),
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


def _target_from_dict(item: dict) -> Target:
    """Build a Target, mapping legacy radial keys to on-axis Cartesian.

    Native Cartesian scenarios (with "x") pass through. Legacy radial scenarios
    (with "range_m"/"velocity_mps") map to the boresight: x = range_m, vx =
    velocity_mps, y = vy = 0 — which the derived properties reproduce bit-for-bit.
    """
    if "x" in item:
        return Target(**item)
    item = dict(item)
    range_m = item.pop("range_m")
    velocity_mps = item.pop("velocity_mps", 0.0)
    return Target(x=float(range_m), y=0.0, vx=float(velocity_mps), vy=0.0, **item)


def load_scenario(path: str) -> tuple[RadarConfig, list[Target]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    radar = data.get("radar", {})
    targets_raw = data.get("targets", [])
    clutter_raw = data.get("clutter", [])

    cfg = RadarConfig(**radar)
    cfg.clutter = [ClutterRegion(**item) for item in clutter_raw]
    targets = [_target_from_dict(item) for item in targets_raw]
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
        Target(x=3.0, vx=5.0, amplitude=1.0),
        Target(x=6.0, vx=-3.0, amplitude=0.6),
    ]
    return cfg, targets
