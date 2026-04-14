from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_strategy_core import (
    build_shuffle_config,
    build_stow_config,
    replication_seed,
    run_one_strategy,
)
from analysis.study_config import DEFAULT_OUTPUT_DIR, DEFAULT_STUDY_CONFIG, StudyConfig
from manifest_generation.generate_passenger_manifest_run import (
    BUSINESS_LOAD_FACTOR,
    ECONOMY_LOAD_FACTOR,
    EDGES_FILE,
    NODES_FILE,
    PREMIUM_LOAD_FACTOR,
    build_manifest,
    build_seat_map,
    load_input_files,
    resolve_graph_file,
)
from simulation import CabinEnvironment

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ci_half_width(differences: np.ndarray, ci_level: float) -> float:
    if len(differences) < 2:
        return float("nan")
    alpha = 1.0 - ci_level
    sd = float(np.std(differences, ddof=1))
    se = sd / np.sqrt(len(differences))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=len(differences) - 1))
    return t_crit * se


def _running_stats(differences: np.ndarray, ci_level: float) -> Dict[str, float]:
    n = int(len(differences))
    if n == 0:
        return {
            "sample_mean": float("nan"),
            "sample_sd": float("nan"),
            "standard_error": float("nan"),
            "t_critical": float("nan"),
            "ci_half_width": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "cv": float("nan"),
        }

    sample_mean = float(np.mean(differences))
    if n < 2:
        return {
            "sample_mean": sample_mean,
            "sample_sd": float("nan"),
            "standard_error": float("nan"),
            "t_critical": float("nan"),
            "ci_half_width": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "cv": float("nan"),
        }

    sample_sd = float(np.std(differences, ddof=1))
    standard_error = sample_sd / np.sqrt(n)
    t_critical = float(stats.t.ppf(1.0 - (1.0 - ci_level) / 2.0, df=n - 1))
    ci_half_width = t_critical * standard_error
    cv = sample_sd / abs(sample_mean) if sample_mean != 0 else float("nan")
    return {
        "sample_mean": sample_mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "t_critical": t_critical,
        "ci_half_width": ci_half_width,
        "ci_low": sample_mean - ci_half_width,
        "ci_high": sample_mean + ci_half_width,
        "cv": cv,
    }


def _completed_pair_trace_view(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return trace_df
    view = (
        trace_df[trace_df["n_completed_pairs"] > 0]
        .drop_duplicates(subset=["n_completed_pairs"], keep="last")
        .sort_values("n_completed_pairs")
        .reset_index(drop=True)
    )
    return view


def _write_live_stabilization_plots(trace_df: pd.DataFrame, output_dir: Path, ci_level: float, target_half_width: float) -> None:
    view = _completed_pair_trace_view(trace_df)
    if view.empty:
        return

    x = view["n_completed_pairs"].to_numpy(dtype=float)
    running_mean = view["sample_mean_diff"].to_numpy(dtype=float)
    ci_low = view["ci_low"].to_numpy(dtype=float)
    ci_high = view["ci_high"].to_numpy(dtype=float)
    ci_half = view["ci_half_width"].to_numpy(dtype=float)
    cv = view["cv"].to_numpy(dtype=float)

    # 1) Running mean of paired differences.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, running_mean, color="#1d4ed8", linewidth=2)
    ax.set_title("Running Mean of Paired Differences")
    ax.set_xlabel("Completed pairs")
    ax.set_ylabel("Mean difference (s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_estimate_running_mean_paired_diff.png", dpi=160)
    plt.close(fig)

    # 2) Running confidence interval around the paired mean difference.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, running_mean, color="#1d4ed8", linewidth=2, label="Running mean")
    finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
    if np.any(finite_ci):
        ax.fill_between(x[finite_ci], ci_low[finite_ci], ci_high[finite_ci], color="#93c5fd", alpha=0.35, label=f"{int(ci_level * 100)}% CI")
        ax.plot(x[finite_ci], ci_low[finite_ci], color="#60a5fa", linewidth=1)
        ax.plot(x[finite_ci], ci_high[finite_ci], color="#60a5fa", linewidth=1)
    ax.set_title("Running Confidence Interval of Paired Mean Difference")
    ax.set_xlabel("Completed pairs")
    ax.set_ylabel("Difference (s)")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_estimate_running_ci_paired_diff.png", dpi=160)
    plt.close(fig)

    # 3) Running CI half-width (formal Approach 1 stopping criterion).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, ci_half, color="#1d4ed8", linewidth=2, label="Observed CI half-width")
    ax.axhline(target_half_width, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Target = {target_half_width:.2f}s")
    ax.set_title("Running CI Half-Width (Approach 1)")
    ax.set_xlabel("Completed pairs")
    ax.set_ylabel("CI half-width (s)")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_estimate_running_ci_half_width.png", dpi=160)
    plt.close(fig)

    # 4) Running CV (supporting Approach 2 stabilization check).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, cv, color="#059669", linewidth=2)
    ax.set_title("Running Coefficient of Variation (Approach 2)")
    ax.set_xlabel("Completed pairs")
    ax.set_ylabel("CV")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_estimate_running_cv.png", dpi=160)
    plt.close(fig)

    # Live combined panel for quick monitoring during execution.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, running_mean, color="#1d4ed8", linewidth=2)
    axes[0, 0].set_title("Running Mean")
    axes[0, 0].set_xlabel("Completed pairs")
    axes[0, 0].set_ylabel("Diff (s)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)

    axes[0, 1].plot(x, running_mean, color="#1d4ed8", linewidth=2)
    if np.any(finite_ci):
        axes[0, 1].fill_between(x[finite_ci], ci_low[finite_ci], ci_high[finite_ci], color="#93c5fd", alpha=0.35)
    axes[0, 1].set_title(f"Running {int(ci_level * 100)}% CI")
    axes[0, 1].set_xlabel("Completed pairs")
    axes[0, 1].set_ylabel("Diff (s)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)

    axes[1, 0].plot(x, ci_half, color="#1d4ed8", linewidth=2)
    axes[1, 0].axhline(target_half_width, color="#b91c1c", linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("CI Half-Width")
    axes[1, 0].set_xlabel("Completed pairs")
    axes[1, 0].set_ylabel("Half-width (s)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.35)

    axes[1, 1].plot(x, cv, color="#059669", linewidth=2)
    axes[1, 1].set_title("Running CV")
    axes[1, 1].set_xlabel("Completed pairs")
    axes[1, 1].set_ylabel("CV")
    axes[1, 1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Estimator Stabilization Dashboard (Live)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_estimate_stabilization_live.png", dpi=160)
    plt.close(fig)


def _build_config(args: argparse.Namespace) -> StudyConfig:
    base = DEFAULT_STUDY_CONFIG
    return StudyConfig(
        master_seed=args.master_seed,
        policy_a=args.policy_a,
        policy_b=args.policy_b,
        load_factor=args.load_factor,
        luggage_probability=args.luggage_probability,
        cross_zone_violation_rate=args.cross_zone_violation_rate,
        batch_size=base.batch_size,
        ci_level=args.ci_level,
        target_ci_half_width_s=args.target_ci_half_width,
        min_replications=args.min_replications,
        max_replications=args.max_replications,
        default_replications=base.default_replications,
    )


def estimate_required_replications(config: StudyConfig, output_dir: Path) -> Dict[str, object]:
    nodes_path = resolve_graph_file(NODES_FILE)
    edges_path = resolve_graph_file(EDGES_FILE)
    nodes_df, edges_df = load_input_files(nodes_path, edges_path)
    seat_map_df = build_seat_map(nodes_df, edges_df)

    env = CabinEnvironment(nodes_path, edges_path)
    stow_config = build_stow_config()
    shuffle_config = build_shuffle_config()

    differences: List[float] = []
    trace_rows: List[Dict[str, object]] = []
    pair_attempt_rows: List[Dict[str, object]] = []
    stop_replication = config.max_replications

    print(
        "[estimate] starting "
        f"max_replications={config.max_replications} "
        f"min_replications={config.min_replications} "
        f"target_ci_half_width={config.target_ci_half_width_s:.2f}s",
        flush=True,
    )

    for replication_id in range(1, config.max_replications + 1):
        seed_i = replication_seed(config.master_seed, replication_id)
        rng = np.random.default_rng(seed_i)

        manifest_df = build_manifest(
            seat_map_df=seat_map_df,
            global_load_factor=config.load_factor,
            business_load_factor=BUSINESS_LOAD_FACTOR,
            premium_load_factor=PREMIUM_LOAD_FACTOR,
            economy_load_factor=ECONOMY_LOAD_FACTOR,
            rng=rng,
            luggage_probability=config.luggage_probability,
            stow_config=stow_config,
        )

        result_a, completed_a, error_a = run_one_strategy(
            env=env,
            manifest_df=manifest_df,
            strategy=config.policy_a,
            seed=seed_i,
            shuffle_config=shuffle_config,
            cross_zone_violation_rate=config.cross_zone_violation_rate,
        )
        result_b, completed_b, error_b = run_one_strategy(
            env=env,
            manifest_df=manifest_df,
            strategy=config.policy_b,
            seed=seed_i,
            shuffle_config=shuffle_config,
            cross_zone_violation_rate=config.cross_zone_violation_rate,
        )

        total_a = float(result_a.get("total_seconds", np.nan))
        total_b = float(result_b.get("total_seconds", np.nan))
        pair_completed = bool(completed_a and completed_b)

        diff = float("nan")
        if completed_a and completed_b:
            diff = total_a - total_b
            if np.isfinite(diff):
                differences.append(diff)

        pair_attempt_rows.append(
            {
                "replication_id": replication_id,
                "scenario_seed": seed_i,
                "pair_completed": pair_completed,
                "completed_pairs_so_far": int(len(differences)),
                "total_seconds_a": total_a,
                "total_seconds_b": total_b,
                "difference": diff,
                "error_a": error_a,
                "error_b": error_b,
            }
        )

        n = len(differences)
        diffs_np = np.array(differences, dtype=float)
        running = _running_stats(diffs_np, config.ci_level)
        sample_mean = running["sample_mean"]
        sample_sd = running["sample_sd"]
        standard_error = running["standard_error"]
        t_critical = running["t_critical"]
        ci_half_width = running["ci_half_width"]
        ci_low = running["ci_low"]
        ci_high = running["ci_high"]
        cv = running["cv"]

        should_stop = bool(
            n >= config.min_replications
            and np.isfinite(ci_half_width)
            and ci_half_width <= config.target_ci_half_width_s
        )

        ci_text = f"{ci_half_width:.3f}s" if np.isfinite(ci_half_width) else "nan"
        cv_text = f"{cv:.4f}" if np.isfinite(cv) else "nan"
        pair_status = "pair_ok" if pair_completed else "pair_incomplete"
        print(
            f"[estimate rep {replication_id}/{config.max_replications}] "
            f"{pair_status} completed_pairs={n} ci_half={ci_text} cv={cv_text}",
            flush=True,
        )

        trace_rows.append(
            {
                "replication_id": replication_id,
                "n_completed_pairs": n,
                "sample_mean_diff": sample_mean,
                "sample_sd_diff": sample_sd,
                "standard_error": standard_error,
                "t_critical": t_critical,
                "ci_half_width": ci_half_width,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cv": cv,
                "pair_completed": pair_completed,
                "difference": diff,
                "error_a": error_a,
                "error_b": error_b,
                "stop_flag": should_stop,
            }
        )

        trace_df_live = pd.DataFrame(trace_rows)
        trace_df_live.to_csv(output_dir / "replication_stopping_trace.csv", index=False)
        pd.DataFrame(pair_attempt_rows).to_csv(output_dir / "replication_pair_attempts.csv", index=False)
        failures_live = pd.DataFrame(pair_attempt_rows)
        failures_live = failures_live[~failures_live["pair_completed"].astype(bool)]
        failures_live.to_csv(output_dir / "replication_pair_failures.csv", index=False)
        if pair_completed or should_stop or replication_id == config.max_replications:
            _write_live_stabilization_plots(
                trace_df=trace_df_live,
                output_dir=output_dir,
                ci_level=config.ci_level,
                target_half_width=config.target_ci_half_width_s,
            )

        if should_stop:
            stop_replication = replication_id
            print(
                "[estimate] stopping rule met "
                f"at replication={replication_id} "
                f"with ci_half={ci_text}",
                flush=True,
            )
            break

    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(output_dir / "replication_stopping_trace.csv", index=False)
    pair_attempt_df = pd.DataFrame(pair_attempt_rows)
    pair_attempt_df.to_csv(output_dir / "replication_pair_attempts.csv", index=False)
    pair_failure_df = pair_attempt_df[~pair_attempt_df["pair_completed"].astype(bool)]
    pair_failure_df.to_csv(output_dir / "replication_pair_failures.csv", index=False)
    _write_live_stabilization_plots(
        trace_df=trace_df,
        output_dir=output_dir,
        ci_level=config.ci_level,
        target_half_width=config.target_ci_half_width_s,
    )

    final_ci = float(trace_df.iloc[-1]["ci_half_width"]) if not trace_df.empty else float("nan")
    summary = {
        "required_replications": int(stop_replication),
        "completed_pairs": int(len(differences)),
        "target_ci_half_width_s": float(config.target_ci_half_width_s),
        "achieved_ci_half_width_s": final_ci,
        "ci_level": float(config.ci_level),
        "min_replications": int(config.min_replications),
        "max_replications": int(config.max_replications),
        "master_seed": int(config.master_seed),
        "policy_a": config.policy_a,
        "policy_b": config.policy_b,
        "load_factor": float(config.load_factor),
        "luggage_probability": float(config.luggage_probability),
        "cross_zone_violation_rate": float(config.cross_zone_violation_rate),
        "formal_stopping_criterion": "approach_1_ci_half_width",
        "supporting_diagnostic": "approach_2_running_cv",
        "stopping_rule_met": bool(
            np.isfinite(final_ci) and final_ci <= config.target_ci_half_width_s and len(differences) >= config.min_replications
        ),
    }

    with (output_dir / "required_replications.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate required paired replications using CI-width stopping.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_STUDY_CONFIG.master_seed)
    parser.add_argument("--policy-a", type=str, default=DEFAULT_STUDY_CONFIG.policy_a)
    parser.add_argument("--policy-b", type=str, default=DEFAULT_STUDY_CONFIG.policy_b)
    parser.add_argument("--load-factor", type=float, default=DEFAULT_STUDY_CONFIG.load_factor)
    parser.add_argument("--luggage-probability", type=float, default=DEFAULT_STUDY_CONFIG.luggage_probability)
    parser.add_argument("--cross-zone-violation-rate", type=float, default=DEFAULT_STUDY_CONFIG.cross_zone_violation_rate)
    parser.add_argument("--ci-level", type=float, default=DEFAULT_STUDY_CONFIG.ci_level)
    parser.add_argument("--target-ci-half-width", type=float, default=DEFAULT_STUDY_CONFIG.target_ci_half_width_s)
    parser.add_argument("--min-replications", type=int, default=DEFAULT_STUDY_CONFIG.min_replications)
    parser.add_argument("--max-replications", type=int, default=DEFAULT_STUDY_CONFIG.max_replications)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)
    summary = estimate_required_replications(config, output_dir)

    print("Replication estimation complete.")
    print(f"Required replications: {summary['required_replications']}")
    print(f"Completed pairs: {summary['completed_pairs']}")
    print(f"Achieved CI half-width: {summary['achieved_ci_half_width_s']}")


if __name__ == "__main__":
    main()
