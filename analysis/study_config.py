from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results" / "paired_strategy"


@dataclass(frozen=True)
class StudyConfig:
    master_seed: int = 20260413
    policy_a: str = "std"
    policy_b: str = "pyramid"
    load_factor: float = 0.85
    luggage_probability: float = 0.75
    cross_zone_violation_rate: float = 0.05
    batch_size: int = 10

    ci_level: float = 0.95
    target_ci_half_width_s: float = 10.0
    min_replications: int = 30
    max_replications: int = 500

    default_replications: int = 100


DEFAULT_STUDY_CONFIG = StudyConfig()
