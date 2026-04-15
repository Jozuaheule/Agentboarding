from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results" / "paired_strategy"

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize_by_strategy(runs_df: pd.DataFrame) -> pd.DataFrame:
    completed = runs_df[runs_df["completed"].astype(bool)].copy()
    summary_rows: List[Dict[str, object]] = []
    for strategy, group in completed.groupby("strategy"):
        values = group["total_boarding_time"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        summary_rows.append(
            {
                "strategy": strategy,
                "n_completed": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("strategy")


def vargha_delaney_a_from_differences(differences: np.ndarray) -> float:
    wins = float(np.sum(differences > 0))
    ties = float(np.sum(differences == 0))
    total = float(len(differences))
    if total == 0:
        return float("nan")
    return (wins + 0.5 * ties) / total


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
                    "selected_test": "one-sided paired t-test (zonal - pyramid > 0)",
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

    test_result = stats.ttest_rel(zonal, pyramid, alternative="greater")
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
                "selected_test": "one-sided paired t-test (zonal - pyramid > 0)",
                "test_statistic": float(test_result.statistic),
                "p_value": float(test_result.pvalue),
                "effect_size_vargha_delaney_A": float(vargha_delaney_a_from_differences(differences)),
                "effect_size_paired_d": float(paired_d),
                "ci95_low": float(mean_diff - margin),
                "ci95_high": float(mean_diff + margin),
            }
        ]
    )


def plot_outputs(runs_df: pd.DataFrame, pairs_df: pd.DataFrame, output_dir: Path) -> None:
    valid_pairs = pairs_df[pairs_df["pair_completed"].astype(bool)].copy()
    completed_runs = runs_df[runs_df["completed"].astype(bool)].copy()

    if not completed_runs.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        order = sorted(completed_runs["strategy"].unique().tolist())
        data = [
            completed_runs.loc[completed_runs["strategy"] == strategy, "total_boarding_time"].to_numpy(dtype=float)
            for strategy in order
        ]
        ax.boxplot(data, tick_labels=order)
        ax.set_title("Boarding Time by Strategy")
        ax.set_ylabel("Total boarding time (s)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_boxplot_boarding_time.png", dpi=160)
        plt.close(fig)

    if not valid_pairs.empty:
        differences = valid_pairs["difference"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(differences, bins=20, density=True, edgecolor="black", alpha=0.65, label="Paired differences")

        if len(differences) >= 2:
            mean = float(np.mean(differences))
            sd = float(np.std(differences, ddof=1))
            if np.isfinite(sd) and sd > 0:
                x_min = float(np.min(differences))
                x_max = float(np.max(differences))
                x = np.linspace(x_min, x_max, 300)
                y = stats.norm.pdf(x, loc=mean, scale=sd)
                ax.plot(x, y, color="#dc2626", linewidth=2, label="Fitted normal density")

        ax.set_title("Histogram of Paired Differences (zonal - pyramid)")
        ax.set_xlabel("Difference in total boarding time (s)")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_hist_paired_differences.png", dpi=160)
        plt.close(fig)

        if len(differences) >= 3:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            stats.probplot(differences, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot of Paired Differences")
            fig.tight_layout()
            fig.savefig(output_dir / "fig_qq_paired_differences.png", dpi=160)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: build statistical summaries and plots from step-1 outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    runs_path = output_dir / "paired_runs_long.csv"
    pairs_path = output_dir / "paired_runs_pairs.csv"

    if not runs_path.exists() or not pairs_path.exists():
        raise FileNotFoundError(
            "Missing step-1 outputs paired_runs_long.csv or paired_runs_pairs.csv. "
            "Run 1_Estimate_Required_Replications.py first."
        )

    runs_df = pd.read_csv(runs_path)
    pairs_df = pd.read_csv(pairs_path)

    descriptive_df = summarize_by_strategy(runs_df)
    descriptive_df.to_csv(output_dir / "strategy_descriptive_summary.csv", index=False)

    inferential_df = compute_paired_ttest_summary(pairs_df)
    inferential_df.to_csv(output_dir / "paired_inferential_summary.csv", index=False)

    plot_outputs(runs_df, pairs_df, output_dir)

    print("Step 2 complete: analysis outputs and plots generated.")
    print(f"Descriptive rows: {len(descriptive_df)}")
    print(f"Inferential rows: {len(inferential_df)}")


if __name__ == "__main__":
    main()
