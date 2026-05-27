from dataclasses import dataclass


_CFAR_METHODS = {"ca", "os", "go", "so"}
_CLUSTERING_METHODS = {"cc", "dbscan", "peak"}
_TRACKER_METHODS = {"nn", "gnn"}


@dataclass
class CfarConfig:
    method: str = "ca"
    guard: tuple = (2, 4)              # (G_doppler, G_range)
    training: tuple = (4, 8)           # (T_doppler, T_range)
    pfa: float = 1e-4
    rank_frac: float = 0.75            # OS-CFAR only

    def __post_init__(self):
        if self.method not in _CFAR_METHODS:
            raise ValueError(
                f"CfarConfig.method must be one of {sorted(_CFAR_METHODS)}, "
                f"got {self.method!r}"
            )
        if self.pfa <= 0.0 or self.pfa >= 1.0:
            raise ValueError(f"CfarConfig.pfa must be in (0, 1), got {self.pfa}")
        if not (0.0 < self.rank_frac < 1.0):
            raise ValueError(
                f"CfarConfig.rank_frac must be in (0, 1), got {self.rank_frac}"
            )
        for name, val in (("guard", self.guard), ("training", self.training)):
            if (
                not isinstance(val, tuple)
                or len(val) != 2
                or any((not isinstance(v, int)) or v < 0 for v in val)
            ):
                raise ValueError(
                    f"CfarConfig.{name} must be a (int, int) tuple of non-negative ints, "
                    f"got {val!r}"
                )


@dataclass
class ClusteringConfig:
    method: str = "cc"
    min_cells: int = 2
    connectivity: int = 8              # cc only: 4 or 8
    eps: float = 1.5                   # dbscan only
    min_samples: int = 2               # dbscan only
    axis_weights: tuple = (1.0, 1.0)   # (doppler, range)
    peak_neighborhood: int = 3         # peak only: local-max window size

    def __post_init__(self):
        if self.method not in _CLUSTERING_METHODS:
            raise ValueError(
                f"ClusteringConfig.method must be one of "
                f"{sorted(_CLUSTERING_METHODS)}, got {self.method!r}"
            )
        if self.connectivity not in (4, 8):
            raise ValueError(
                f"ClusteringConfig.connectivity must be 4 or 8, "
                f"got {self.connectivity}"
            )
        if self.min_cells < 1:
            raise ValueError(
                f"ClusteringConfig.min_cells must be >= 1, got {self.min_cells}"
            )
        if self.eps <= 0:
            raise ValueError(f"ClusteringConfig.eps must be > 0, got {self.eps}")
        if self.min_samples < 1:
            raise ValueError(
                f"ClusteringConfig.min_samples must be >= 1, got {self.min_samples}"
            )
        if self.peak_neighborhood < 1:
            raise ValueError(
                f"ClusteringConfig.peak_neighborhood must be >= 1, "
                f"got {self.peak_neighborhood}"
            )


@dataclass
class TrackerConfig:
    method: str = "gnn"
    gate_threshold: float = 9.21
    process_noise_q: float = 0.5
    range_meas_var: float = 0.5
    doppler_meas_var: float = 1.0
    init_velocity_var: float = 4.0
    doppler_smoothing_beta: float = 0.5
    confirm_M: int = 3
    confirm_N: int = 5
    max_misses_tentative: int = 2
    max_coast: int = 5
    history_len: int = 10
    motion_model: str = "cv"             # "cv" or "drag"
    drag_coefficient_k: float = 0.5     # only used when motion_model == "drag"

    def __post_init__(self):
        if self.motion_model not in ("cv", "drag"):
            raise ValueError(
                f"TrackerConfig.motion_model must be 'cv' or 'drag', "
                f"got {self.motion_model!r}"
            )
        if self.drag_coefficient_k <= 0:
            raise ValueError(
                f"TrackerConfig.drag_coefficient_k must be > 0, "
                f"got {self.drag_coefficient_k}"
            )
        if self.method not in _TRACKER_METHODS:
            raise ValueError(
                f"TrackerConfig.method must be one of {sorted(_TRACKER_METHODS)}, "
                f"got {self.method!r}"
            )
        if self.confirm_M > self.confirm_N:
            raise ValueError(
                f"TrackerConfig.confirm_M ({self.confirm_M}) must be "
                f"<= confirm_N ({self.confirm_N})"
            )
        for name in ("gate_threshold", "process_noise_q", "range_meas_var",
                     "doppler_meas_var", "init_velocity_var"):
            if getattr(self, name) <= 0:
                raise ValueError(f"TrackerConfig.{name} must be > 0")
        if not (0.0 < self.doppler_smoothing_beta <= 1.0):
            raise ValueError(
                f"TrackerConfig.doppler_smoothing_beta must be in (0, 1], "
                f"got {self.doppler_smoothing_beta}"
            )
        if self.confirm_N < 1 or self.confirm_M < 1:
            raise ValueError("confirm_M and confirm_N must be >= 1")
        if self.max_coast < 0:
            raise ValueError(f"TrackerConfig.max_coast must be >= 0, got {self.max_coast}")
        if self.max_misses_tentative < 0:
            raise ValueError(
                f"TrackerConfig.max_misses_tentative must be >= 0, "
                f"got {self.max_misses_tentative}"
            )
        if self.history_len < 1:
            raise ValueError(
                f"TrackerConfig.history_len must be >= 1, got {self.history_len}"
            )
