from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_strategy_core import vargha_delaney_a_from_differences
from analysis.study_config import DEFAULT_OUTPUT_DIR


def compute_paired_ttest_summary(pairs_df: pd.DataFrame) -> pd.DataFrame:
    valid = pairs_df[pairs_df["pair_completed"].astype(bool)].copy()
    if len(valid) < 2:
        return pd.DataFrame(
            [
                {
                    "n_pairs": int(len(valid)),
                    "mean_paired_difference": float("nan"),
                    "mean_relative_improvement": float("nan"),
                    "normality_p_value": float("nan"),
                    "selected_test": "paired_t",
                    "test_statistic": float("nan"),
                    "p_value": float("nan"),
                    "effect_size_vargha_delaney_A": float("nan"),
                    "effect_size_paired_d": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                }
            ]
        )

    zonal = valid["boarding_time_zonal"].to_numpy(dtype=float)
    pyramid = valid["boarding_time_pyramid"].to_numpy(dtype=float)
    differences = valid["difference"].to_numpy(dtype=float)

    test_result = stats.ttest_rel(zonal, pyramid, alternative="two-sided")
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    paired_d = mean_diff / std_diff if std_diff != 0 else float("nan")

    t_crit = float(stats.t.ppf(0.975, df=len(differences) - 1))
    margin = t_crit * (std_diff / np.sqrt(len(differences)))

    return pd.DataFrame(
        [
            {
                "n_pairs": int(len(differences)),
                "mean_paired_difference": mean_diff,
                "mean_relative_improvement": float(np.mean(valid["relative_improvement"])),
                "normality_p_value": float("nan"),
                "selected_test": "paired_t",
                "test_statistic": float(test_result.statistic),
                "p_value": float(test_result.pvalue),
                "effect_size_vargha_delaney_A": float(vargha_delaney_a_from_differences(differences)),
                "effect_size_paired_d": float(paired_d),
                "ci95_low": float(mean_diff - margin),
                "ci95_high": float(mean_diff + margin),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired t-test inference for completed strategy pairs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    pairs_path = output_dir / "paired_runs_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError("Missing paired_runs_pairs.csv. Run run_paired_replications.py first.")

    pairs_df = pd.read_csv(pairs_path)
    summary_df = compute_paired_ttest_summary(pairs_df)
    summary_df.to_csv(output_dir / "paired_inferential_summary.csv", index=False)

    print("Paired t-test complete.")
    print(f"Rows written: {len(summary_df)}")


if __name__ == "__main__":
    main()
