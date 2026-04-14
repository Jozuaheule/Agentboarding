from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_strategy_core import plot_outputs
from analysis.study_config import DEFAULT_OUTPUT_DIR, DEFAULT_STUDY_CONFIG

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_stabilization_df(pairs_df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    valid = pairs_df[pairs_df["pair_completed"].astype(bool)].sort_values("replication_id")
    diffs = valid["difference"].to_numpy(dtype=float)

    rows: List[dict] = []
    running: List[float] = []
    alpha = 1.0 - ci_level

    for idx, diff in enumerate(diffs, start=1):
        running.append(float(diff))
        if len(running) < 2:
            rows.append(
                {
                    "n_completed_pairs": idx,
                    "running_mean_diff": float(np.mean(running)),
                    "running_cv": float("nan"),
                    "running_ci_half_width": float("nan"),
                }
            )
            continue

        arr = np.array(running, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        se = sd / np.sqrt(len(arr))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=len(arr) - 1))
        ci_half = t_crit * se
        cv = sd / abs(mean) if mean != 0 else float("nan")

        rows.append(
            {
                "n_completed_pairs": idx,
                "running_mean_diff": mean,
                "running_cv": cv,
                "running_ci_half_width": ci_half,
            }
        )

    return pd.DataFrame(rows)


def _plot_ci_cv_stabilization(stab_df: pd.DataFrame, output_dir: Path) -> None:
    if stab_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = stab_df["n_completed_pairs"].to_numpy(dtype=float)

    ax1.plot(x, stab_df["running_ci_half_width"], color="#1d4ed8", linewidth=2, label="CI half-width")
    ax1.set_xlabel("Completed pairs")
    ax1.set_ylabel("CI half-width (s)", color="#1d4ed8")
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(x, stab_df["running_cv"], color="#b91c1c", linewidth=2, label="CV")
    ax2.set_ylabel("Coefficient of variation", color="#b91c1c")
    ax2.tick_params(axis="y", labelcolor="#b91c1c")

    fig.suptitle("CI and CV Stabilization Across Completed Pairs")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_ci_cv_stabilization.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate descriptive diagnostics for paired strategy analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ci-level", type=float, default=DEFAULT_STUDY_CONFIG.ci_level)
    args = parser.parse_args()

    output_dir = args.output_dir
    runs_path = output_dir / "paired_runs_long.csv"
    pairs_path = output_dir / "paired_runs_pairs.csv"

    if not runs_path.exists() or not pairs_path.exists():
        raise FileNotFoundError("Missing paired run outputs. Run run_paired_replications.py first.")

    runs_df = pd.read_csv(runs_path)
    pairs_df = pd.read_csv(pairs_path)

    plot_outputs(runs_df, pairs_df, output_dir)

    stab_df = _build_stabilization_df(pairs_df, ci_level=args.ci_level)
    stab_df.to_csv(output_dir / "replication_stability_summary.csv", index=False)
    _plot_ci_cv_stabilization(stab_df, output_dir)

    print("Diagnostics complete.")
    print(f"Stability rows: {len(stab_df)}")


if __name__ == "__main__":
    main()
