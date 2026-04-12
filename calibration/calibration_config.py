from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

# Fixed simulation time scale: one tick equals 0.5 seconds.
TIME_PER_TICK = 0.5

# Baseline calibration scenario controls.
BASELINE_BOARDING_POLICY = "std"
CALIBRATION_LOAD_FACTOR = 0.95
CALIBRATION_LUGGAGE_PROBABILITY = 0.60
CALIBRATION_N_MANIFESTS = 30

# Luggage stow model defaults (seconds).
STOW_DIST = "weibull"
STOW_SHAPE = 1.7
STOW_SCALE_S = 16.0
STOW_UNIFORM_LOW_S = 15.0
STOW_UNIFORM_HIGH_S = 45.0

# Seat-shuffle delay defaults (seconds).
SHUFFLE_MODEL = "triangular"
SHUFFLE_LOW_S = 15.0
SHUFFLE_MODE_S = 20.5
SHUFFLE_HIGH_S = 26.0

# Initial sweep grid (stow x shuffle settings).
CANDIDATE_STOW = [
    ("weibull", 1.7, 14.0),
    ("weibull", 1.7, 16.0),
    ("weibull", 1.7, 18.0),
]
CANDIDATE_SHUFFLE = [
    {"model": "triangular", "low_s": 15.0, "mode_s": 20.0, "high_s": 26.0},
    {"model": "triangular", "low_s": 15.0, "mode_s": 20.5, "high_s": 26.0},
    {"model": "triangular", "low_s": 15.0, "mode_s": 21.0, "high_s": 26.0},
]


@dataclass(frozen=True)
class StowConfig:
    dist: str
    shape_k: float = STOW_SHAPE
    scale_lambda_s: float = STOW_SCALE_S
    low_s: float = STOW_UNIFORM_LOW_S
    high_s: float = STOW_UNIFORM_HIGH_S


@dataclass(frozen=True)
class ShuffleConfig:
    model: str
    low_s: float
    high_s: float
    mode_s: Optional[float] = None


def seconds_to_ticks(seconds: float, min_ticks: int = 0) -> int:
    ticks = int(math.ceil(float(seconds) / TIME_PER_TICK))
    return max(min_ticks, ticks)


def ticks_to_seconds(ticks: float) -> float:
    return float(ticks) * TIME_PER_TICK


def validate_bounds(low: float, high: float, label: str) -> Tuple[float, float]:
    low_f = float(low)
    high_f = float(high)
    if low_f <= 0 or high_f <= 0 or high_f < low_f:
        raise ValueError(f"Invalid {label} bounds: low={low_f}, high={high_f}")
    return low_f, high_f


def validate_triangular_mode(low: float, mode: float, high: float, label: str) -> float:
    mode_f = float(mode)
    if not (low <= mode_f <= high):
        raise ValueError(
            f"Invalid {label} mode: mode={mode_f} not in [{low}, {high}]"
        )
    return mode_f


def candidate_parameter_grid() -> Iterable[tuple[StowConfig, ShuffleConfig]]:
    for stow_dist, stow_shape, stow_scale in CANDIDATE_STOW:
        stow = StowConfig(dist=stow_dist, shape_k=stow_shape, scale_lambda_s=stow_scale)
        for shuffle_entry in CANDIDATE_SHUFFLE:
            model = str(shuffle_entry["model"]).strip().lower()
            low_s, high_s = validate_bounds(
                float(shuffle_entry["low_s"]),
                float(shuffle_entry["high_s"]),
                "shuffle",
            )
            mode_s = (
                validate_triangular_mode(
                    low_s,
                    float(shuffle_entry["mode_s"]),
                    high_s,
                    "shuffle triangular",
                )
                if model == "triangular"
                else None
            )
            shuffle = ShuffleConfig(
                model=model,
                low_s=low_s,
                high_s=high_s,
                mode_s=mode_s,
            )
            yield stow, shuffle
