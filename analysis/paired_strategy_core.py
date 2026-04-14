from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from calibration.calibration_config import (
    SHUFFLE_HIGH_S,
    SHUFFLE_LOW_S,
    SHUFFLE_MODE_S,
    SHUFFLE_MODEL,
    STOW_DIST,
    STOW_SCALE_S,
    STOW_SHAPE,
    STOW_UNIFORM_HIGH_S,
    STOW_UNIFORM_LOW_S,
    ShuffleConfig,
    StowConfig,
)
from manifest_generation.generate_passenger_manifest_run import (
    BUSINESS_LOAD_FACTOR,
    ECONOMY_LOAD_FACTOR,
    LUGGAGE_PROBABILITY,
    LOAD_FACTOR,
    NODES_FILE,
    EDGES_FILE,
    PREMIUM_LOAD_FACTOR,
    build_manifest,
    build_seat_map,
    load_input_files,
    resolve_graph_file,
)
from simulation import CabinEnvironment, BoardingSimulation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunConfig:
    replications: int = 100
    master_seed: int = 20260413
    batch_size: int = 10
    policy_a: str = "std"
    policy_b: str = "pyramid"
    normality_alpha: float = 0.05
    load_factor: float = LOAD_FACTOR
    luggage_probability: float = LUGGAGE_PROBABILITY
    cross_zone_violation_rate: float = 0.05


def replication_seed(master_seed: int, replication_id: int) -> int:
    return int(master_seed) + int(replication_id) - 1


def build_stow_config() -> StowConfig:
    return StowConfig(
        dist=STOW_DIST,
        shape_k=STOW_SHAPE,
        scale_lambda_s=STOW_SCALE_S,
        low_s=STOW_UNIFORM_LOW_S,
        high_s=STOW_UNIFORM_HIGH_S,
    )


def build_shuffle_config() -> ShuffleConfig:
    return ShuffleConfig(
        model=SHUFFLE_MODEL,
        low_s=SHUFFLE_LOW_S,
        high_s=SHUFFLE_HIGH_S,
        mode_s=SHUFFLE_MODE_S,
    )


def run_one_strategy(
    env: CabinEnvironment,
    manifest_df: pd.DataFrame,
    strategy: str,
    seed: int,
    shuffle_config: ShuffleConfig,
    cross_zone_violation_rate: float,
) -> Tuple[Dict[str, float], bool, str]:
    try:
        sim = BoardingSimulation(
            env=env,
            manifest_file=manifest_df,
            seed=seed,
            boarding_policy=strategy,
            shuffle_config=shuffle_config,
            cross_zone_violation_rate=cross_zone_violation_rate,
            log_summary=False,
        )
        metrics = sim.run_with_metrics(verbose=False, enforce_completion=False)
        completed = bool(metrics.get("completion_success", 0.0) > 0.5)
        return metrics, completed, ""
    except Exception as exc:  # noqa: BLE001
        return {}, False, f"{type(exc).__name__}: {exc}"


def flush_csvs(output_dir: Path, run_rows: List[Dict[str, object]], pair_rows: List[Dict[str, object]]) -> None:
    pd.DataFrame(run_rows).to_csv(output_dir / "paired_runs_long.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "paired_runs_pairs.csv", index=False)


def run_paired_experiment(config: RunConfig, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = resolve_graph_file(NODES_FILE)
    edges_path = resolve_graph_file(EDGES_FILE)
    nodes_df, edges_df = load_input_files(nodes_path, edges_path)
    seat_map_df = build_seat_map(nodes_df, edges_df)

    env = CabinEnvironment(nodes_path, edges_path)
    stow_config = build_stow_config()
    shuffle_config = build_shuffle_config()

    run_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    total_simulations = config.replications * 2
    simulation_counter = 0

    for replication_id in range(1, config.replications + 1):
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
        passenger_count = int(len(manifest_df))

        strategy_results: Dict[str, Dict[str, object]] = {}
        for strategy in (config.policy_a, config.policy_b):
            metrics, completed, error_message = run_one_strategy(
                env=env,
                manifest_df=manifest_df,
                strategy=strategy,
                seed=seed_i,
                shuffle_config=shuffle_config,
                cross_zone_violation_rate=config.cross_zone_violation_rate,
            )
            simulation_counter += 1
            total_seconds = float(metrics.get("total_seconds", np.nan))
            status = "ok" if completed else "failed"
            time_text = f"{total_seconds:.1f}s" if np.isfinite(total_seconds) else "nan"
            print(
                f"[sim {simulation_counter}/{total_simulations}] "
                f"rep {replication_id}/{config.replications} "
                f"strategy={strategy} seed={seed_i} status={status} time={time_text}"
            )
            run_rows.append(
                {
                    "replication_id": replication_id,
                    "scenario_seed": seed_i,
                    "strategy": strategy,
                    "total_boarding_time": total_seconds,
                    "load_factor": config.load_factor,
                    "luggage_probability": config.luggage_probability,
                    "cross_zone_violation_rate": config.cross_zone_violation_rate,
                    "passenger_count": passenger_count,
                    "completed": completed,
                    "error_message": error_message,
                }
            )
            strategy_results[strategy] = {
                "total_boarding_time": total_seconds,
                "completed": completed,
                "error_message": error_message,
            }

        a_result = strategy_results[config.policy_a]
        b_result = strategy_results[config.policy_b]
        both_completed = bool(a_result["completed"] and b_result["completed"])
        a_time = float(a_result["total_boarding_time"])
        b_time = float(b_result["total_boarding_time"])

        difference = a_time - b_time if both_completed else np.nan
        ratio = (b_time / a_time) if both_completed and a_time > 0 else np.nan
        relative_improvement = ((a_time - b_time) / a_time) if both_completed and a_time > 0 else np.nan

        pair_rows.append(
            {
                "replication_id": replication_id,
                "scenario_seed": seed_i,
                "boarding_time_zonal": a_time,
                "boarding_time_pyramid": b_time,
                "difference": difference,
                "ratio": ratio,
                "relative_improvement": relative_improvement,
                "pair_completed": both_completed,
                "zonal_error": a_result["error_message"],
                "pyramid_error": b_result["error_message"],
            }
        )

        if replication_id % config.batch_size == 0:
            flush_csvs(output_dir, run_rows, pair_rows)
            print(f"Flushed batch at replication {replication_id}/{config.replications}")

    flush_csvs(output_dir, run_rows, pair_rows)
    return pd.DataFrame(run_rows), pd.DataFrame(pair_rows)


def vargha_delaney_a_from_differences(differences: np.ndarray) -> float:
    wins = float(np.sum(differences > 0))
    ties = float(np.sum(differences == 0))
    total = float(len(differences))
    if total == 0:
        return float("nan")
    return (wins + 0.5 * ties) / total


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
        ax.hist(differences, bins=20, edgecolor="black", alpha=0.8)
        ax.set_title("Histogram of Paired Differences (zonal - pyramid)")
        ax.set_xlabel("Difference in total boarding time (s)")
        ax.set_ylabel("Frequency")
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

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(valid_pairs["replication_id"], valid_pairs["relative_improvement"], marker="o", linestyle="-")
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax.set_title("Relative Improvement by Replication")
        ax.set_xlabel("Replication id")
        ax.set_ylabel("Relative improvement")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_relative_improvement_by_replication.png", dpi=160)
        plt.close(fig)


def write_study_config(config: RunConfig, output_dir: Path) -> None:
    payload = asdict(config)
    payload["stow_config"] = asdict(build_stow_config())
    payload["shuffle_config"] = asdict(build_shuffle_config())
    payload["study_context"] = {
        "fixed_assumptions": [
            "aircraft layout and cabin topology",
            "seat map and class structure",
            "load factor",
            "luggage probability",
            "active boarding doors",
            "behavioral parameter settings",
            "simulation logic and completion condition",
        ],
        "independent_variable": "boarding strategy",
        "primary_dependent_variable": "total boarding time",
    }
    with (output_dir / "study_config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
