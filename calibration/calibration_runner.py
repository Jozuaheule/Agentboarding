from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calibration.calibration_config import (
    BASELINE_BOARDING_POLICY,
    CALIBRATION_CROSS_ZONE_VIOLATION_RATE,
    CALIBRATION_LOAD_FACTOR,
    CALIBRATION_LUGGAGE_PROBABILITY,
    CALIBRATION_N_MANIFESTS,
    SHUFFLE_HIGH_S,
    SHUFFLE_LOW_S,
    SHUFFLE_MODE_S,
    SHUFFLE_MODEL,
    STOW_SCALE_S,
    STOW_SHAPE,
    STOW_DIST,
    ShuffleConfig,
    StowConfig,
    candidate_parameter_grid,
)
from manifest_generation.generate_passenger_manifest_run import (
    BUSINESS_LOAD_FACTOR,
    ECONOMY_LOAD_FACTOR,
    EDGES_FILE,
    NODES_FILE,
    PREMIUM_LOAD_FACTOR,
    SEED,
    build_manifest,
    build_seat_map,
    load_input_files,
    resolve_graph_file,
    write_dataframe,
)
from simulation import CabinEnvironment, BoardingSimulation


RESULTS_DIR = ROOT_DIR / "calibration" / "results"
PER_RUN_CSV = RESULTS_DIR / "calibration_runs.csv"
SUMMARY_CSV = RESULTS_DIR / "calibration_summary.csv"
TOP3_CSV = RESULTS_DIR / "calibration_top3.csv"


def parameter_set_id(index: int, stow: StowConfig, shuffle: ShuffleConfig) -> str:
    if stow.dist.lower() == "weibull":
        stow_token = f"wb_k{stow.shape_k:.1f}_l{stow.scale_lambda_s:.0f}"
    else:
        stow_token = f"unif_{stow.low_s:.0f}_{stow.high_s:.0f}"
    if shuffle.model.lower() == "triangular":
        mode = shuffle.mode_s if shuffle.mode_s is not None else (shuffle.low_s + shuffle.high_s) / 2
        shuffle_token = f"tri_{shuffle.low_s:.0f}_{mode:.1f}_{shuffle.high_s:.0f}"
    else:
        shuffle_token = f"{shuffle.model.lower()}_{shuffle.low_s:.0f}_{shuffle.high_s:.0f}"
    return f"P{index:02d}_{stow_token}_{shuffle_token}"


def generate_manifest_for_candidate(
    seat_map_df: pd.DataFrame,
    candidate_id: str,
    manifest_id: int,
    stow_config: StowConfig,
) -> Path:
    manifest_dir = RESULTS_DIR / "manifests" / candidate_id
    manifest_dir.mkdir(parents=True, exist_ok=True)

    current_seed = None if SEED is None else SEED + manifest_id - 1
    rng = np.random.default_rng(current_seed)

    manifest_df = build_manifest(
        seat_map_df=seat_map_df,
        global_load_factor=CALIBRATION_LOAD_FACTOR,
        business_load_factor=BUSINESS_LOAD_FACTOR,
        premium_load_factor=PREMIUM_LOAD_FACTOR,
        economy_load_factor=ECONOMY_LOAD_FACTOR,
        rng=rng,
        luggage_probability=CALIBRATION_LUGGAGE_PROBABILITY,
        stow_config=stow_config,
    )

    output_path = manifest_dir / f"manifest_{manifest_id:02d}.xlsx"
    write_dataframe(manifest_df, output_path)
    return output_path


def stage1_screen(summary_df: pd.DataFrame) -> pd.DataFrame:
    screened = summary_df.copy()

    screened["screen_completion"] = screened["completion_rate"] >= 1.0
    screened["screen_time"] = screened["mean_total_seconds"].between(8 * 60, 45 * 60)
    screened["screen_stability"] = screened["std_total_seconds"] <= (
        screened["mean_total_seconds"] * 0.35
    )
    screened["screen_row_conflict"] = screened["mean_row_conflict_count"] <= 500
    screened["stage1_pass"] = (
        screened["screen_completion"]
        & screened["screen_time"]
        & screened["screen_stability"]
        & screened["screen_row_conflict"]
    )

    return screened


def _normalize(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum - minimum < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def rank_candidates(screened_df: pd.DataFrame) -> pd.DataFrame:
    survivors = screened_df.loc[screened_df["stage1_pass"]].copy()
    if survivors.empty:
        return survivors

    survivors["score"] = (
        0.55 * _normalize(survivors["mean_total_seconds"])
        + 0.20 * _normalize(survivors["std_total_seconds"])
        + 0.15 * _normalize(survivors["mean_avg_wait_seconds"])
        + 0.10 * _normalize(survivors["mean_row_conflict_count"])
    )

    return survivors.sort_values(["score", "mean_total_seconds"], ascending=[True, True])


def run_calibration() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nodes_path = resolve_graph_file(NODES_FILE)
    edges_path = resolve_graph_file(EDGES_FILE)
    nodes_df, edges_df = load_input_files(nodes_path, edges_path)
    seat_map_df = build_seat_map(nodes_df, edges_df)

    env = CabinEnvironment(nodes_path, edges_path)

    run_rows: List[Dict[str, float | int | str | bool]] = []

    parameter_grid = list(candidate_parameter_grid())
    for index, (stow_config, shuffle_config) in enumerate(parameter_grid, start=1):
        candidate_id = parameter_set_id(index, stow_config, shuffle_config)
        if shuffle_config.model == "triangular" and shuffle_config.mode_s is not None:
            shuffle_desc = (
                f"{shuffle_config.model} "
                f"({shuffle_config.low_s:.1f}-{shuffle_config.mode_s:.1f}-{shuffle_config.high_s:.1f}s)"
            )
        else:
            shuffle_desc = (
                f"{shuffle_config.model} "
                f"({shuffle_config.low_s:.1f}-{shuffle_config.high_s:.1f}s)"
            )
        print(
            f"Running {candidate_id}: stow={stow_config.dist} "
            f"(k={stow_config.shape_k:.1f}, scale={stow_config.scale_lambda_s:.1f}s), "
            f"shuffle={shuffle_desc}"
        )

        for manifest_id in range(1, CALIBRATION_N_MANIFESTS + 1):
            manifest_path = generate_manifest_for_candidate(
                seat_map_df=seat_map_df,
                candidate_id=candidate_id,
                manifest_id=manifest_id,
                stow_config=stow_config,
            )

            sim = BoardingSimulation(
                env,
                manifest_path,
                seed=SEED,
                boarding_policy=BASELINE_BOARDING_POLICY,
                shuffle_config=shuffle_config,
                cross_zone_violation_rate=CALIBRATION_CROSS_ZONE_VIOLATION_RATE,
            )
            metrics = sim.run_with_metrics(verbose=False, enforce_completion=False)

            run_rows.append(
                {
                    "parameter_set_id": candidate_id,
                    "manifest_id": manifest_id,
                    "boarding_policy": BASELINE_BOARDING_POLICY,
                    "load_factor": CALIBRATION_LOAD_FACTOR,
                    "luggage_probability": CALIBRATION_LUGGAGE_PROBABILITY,
                    "cross_zone_violation_rate": CALIBRATION_CROSS_ZONE_VIOLATION_RATE,
                    "stow_dist": stow_config.dist,
                    "stow_shape_k": stow_config.shape_k,
                    "stow_scale_lambda_s": stow_config.scale_lambda_s,
                    "shuffle_model": shuffle_config.model,
                    "shuffle_low_s": shuffle_config.low_s,
                    "shuffle_mode_s": shuffle_config.mode_s,
                    "shuffle_high_s": shuffle_config.high_s,
                    "total_ticks": int(metrics["total_ticks"]),
                    "total_seconds": metrics["total_seconds"],
                    "avg_boarding_seconds": metrics["avg_boarding_seconds"],
                    "avg_wait_seconds": metrics["avg_wait_seconds"],
                    "luggage_passengers": int(metrics["luggage_passengers"]),
                    "avg_assigned_stow_seconds": metrics["avg_stow_seconds"],
                    "head_on_count": int(metrics["head_on_count"]),
                    "row_conflict_count": int(metrics["row_conflict_count"]),
                    "seat_shuffle_starts": int(metrics["seat_shuffle_starts"]),
                    "seat_shuffle_finishes": int(metrics["seat_shuffle_finishes"]),
                    "boarding_completion_success": bool(metrics["completion_success"] > 0.5),
                }
            )

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(PER_RUN_CSV, index=False)

    grouped = runs_df.groupby("parameter_set_id", as_index=False)
    summary_df = grouped.agg(
        mean_total_seconds=("total_seconds", "mean"),
        std_total_seconds=("total_seconds", "std"),
        median_total_seconds=("total_seconds", "median"),
        min_total_seconds=("total_seconds", "min"),
        max_total_seconds=("total_seconds", "max"),
        mean_avg_wait_seconds=("avg_wait_seconds", "mean"),
        # Event-level row conflicts: starts of seat-shuffle episodes.
        mean_row_conflict_count=("seat_shuffle_starts", "mean"),
        # Tick-level row-conflict pressure, kept for diagnostics only.
        mean_row_conflict_ticks=("row_conflict_count", "mean"),
        completion_rate=("boarding_completion_success", "mean"),
    )

    parameter_meta = (
        runs_df[
            [
                "parameter_set_id",
                "stow_dist",
                "stow_shape_k",
                "stow_scale_lambda_s",
                "shuffle_model",
                "shuffle_low_s",
                "shuffle_mode_s",
                "shuffle_high_s",
                "cross_zone_violation_rate",
            ]
        ]
        .drop_duplicates(subset=["parameter_set_id"])
        .copy()
    )

    summary_df = summary_df.merge(parameter_meta, on="parameter_set_id", how="left")
    screened_df = stage1_screen(summary_df)
    ranked_df = rank_candidates(screened_df)
    top3_df = ranked_df.head(3).copy()

    screened_df.sort_values("parameter_set_id").to_csv(SUMMARY_CSV, index=False)
    top3_df.to_csv(TOP3_CSV, index=False)

    print("\nCalibration complete.")
    print(f"Per-run results: {PER_RUN_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Top 3 shortlist: {TOP3_CSV}")

    if top3_df.empty:
        print("No parameter set passed stage 1 screening. Consider relaxing thresholds.")
    else:
        print("\nTop 3 settings for face validation:")
        for _, row in top3_df.iterrows():
            print(
                f"  - {row['parameter_set_id']}: mean={row['mean_total_seconds']:.1f}s, "
                f"std={row['std_total_seconds']:.1f}s, row_conflicts={row['mean_row_conflict_count']:.1f}"
            )


def main() -> None:
    print("Calibration baseline")
    print(f"  Boarding policy   : {BASELINE_BOARDING_POLICY}")
    print(f"  Load factor       : {CALIBRATION_LOAD_FACTOR}")
    print(f"  Luggage prob      : {CALIBRATION_LUGGAGE_PROBABILITY}")
    print(f"  Violation rate    : {CALIBRATION_CROSS_ZONE_VIOLATION_RATE}")
    print(f"  Manifest count    : {CALIBRATION_N_MANIFESTS}")
    print(f"  Default stow      : {STOW_DIST} k={STOW_SHAPE}, scale={STOW_SCALE_S}s")
    if SHUFFLE_MODEL == "triangular":
        print(
            f"  Default shuffle   : {SHUFFLE_MODEL} "
            f"{SHUFFLE_LOW_S}-{SHUFFLE_MODE_S}-{SHUFFLE_HIGH_S}s"
        )
    else:
        print(f"  Default shuffle   : {SHUFFLE_MODEL} {SHUFFLE_LOW_S}-{SHUFFLE_HIGH_S}s")

    run_calibration()


if __name__ == "__main__":
    main()
