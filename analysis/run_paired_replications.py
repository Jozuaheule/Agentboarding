from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_strategy_core import RunConfig, run_paired_experiment, write_study_config
from analysis.study_config import DEFAULT_OUTPUT_DIR, DEFAULT_STUDY_CONFIG


def _required_context_mismatches(payload: dict, current_context: dict) -> list[str]:
    mismatches: list[str] = []
    for key, current_value in current_context.items():
        if key not in payload:
            mismatches.append(f"{key}: missing in required_replications.json")
            continue

        stored_value = payload[key]
        if isinstance(current_value, float):
            if not math.isclose(float(stored_value), current_value, rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append(f"{key}: required={stored_value}, current={current_value}")
        elif stored_value != current_value:
            mismatches.append(f"{key}: required={stored_value}, current={current_value}")

    return mismatches


def _resolve_replication_count(
    output_dir: Path,
    explicit_replications: int | None,
    current_context: dict,
) -> int:
    if explicit_replications is not None:
        return int(explicit_replications)

    required_path = output_dir / "required_replications.json"
    if required_path.exists():
        with required_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        mismatches = _required_context_mismatches(payload, current_context)
        if mismatches:
            mismatch_text = "\n".join(f"  - {item}" for item in mismatches)
            raise RuntimeError(
                "Existing required_replications.json does not match current scenario settings.\n"
                "Mismatches:\n"
                f"{mismatch_text}\n"
                "Re-run estimate_required_replications.py for this scenario or pass --replications explicitly."
            )

        required = int(payload.get("required_replications", DEFAULT_STUDY_CONFIG.default_replications))
        return max(required, 1)

    return int(DEFAULT_STUDY_CONFIG.default_replications)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired replications with deterministic seeding.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_STUDY_CONFIG.master_seed)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_STUDY_CONFIG.batch_size)
    parser.add_argument("--policy-a", type=str, default=DEFAULT_STUDY_CONFIG.policy_a)
    parser.add_argument("--policy-b", type=str, default=DEFAULT_STUDY_CONFIG.policy_b)
    parser.add_argument("--load-factor", type=float, default=DEFAULT_STUDY_CONFIG.load_factor)
    parser.add_argument("--luggage-probability", type=float, default=DEFAULT_STUDY_CONFIG.luggage_probability)
    parser.add_argument("--cross-zone-violation-rate", type=float, default=DEFAULT_STUDY_CONFIG.cross_zone_violation_rate)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    current_context = {
        "master_seed": int(args.master_seed),
        "policy_a": args.policy_a,
        "policy_b": args.policy_b,
        "load_factor": float(args.load_factor),
        "luggage_probability": float(args.luggage_probability),
        "cross_zone_violation_rate": float(args.cross_zone_violation_rate),
    }
    replications = _resolve_replication_count(output_dir, args.replications, current_context)

    config = RunConfig(
        replications=replications,
        master_seed=args.master_seed,
        batch_size=args.batch_size,
        policy_a=args.policy_a,
        policy_b=args.policy_b,
        load_factor=args.load_factor,
        luggage_probability=args.luggage_probability,
        cross_zone_violation_rate=args.cross_zone_violation_rate,
    )

    write_study_config(config, output_dir)
    runs_df, pairs_df = run_paired_experiment(config=config, output_dir=output_dir)

    failures_df = runs_df[~runs_df["completed"].astype(bool)].copy()
    failures_df.to_csv(output_dir / "run_failures.csv", index=False)

    print("Paired replications complete.")
    print(f"Replications attempted: {replications}")
    print(f"Run rows: {len(runs_df)}")
    print(f"Pair rows: {len(pairs_df)}")
    print(f"Completed pairs: {int(pairs_df['pair_completed'].sum()) if not pairs_df.empty else 0}")


if __name__ == "__main__":
    main()
