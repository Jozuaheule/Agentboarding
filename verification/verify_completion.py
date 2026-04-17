"""Completion-level verification experiments.

V10 checks single-run completion correctness and V11 checks repeated-run invariants.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation import BoardingSimulation, CabinEnvironment, EDGES_FILE, MANIFEST_FILE, NODES_FILE, SEED
try:
    from verification.verification_utils import (
        fail_result,
        fail_result_with_meta,
        pass_result,
        pass_result_with_meta,
    )
except ModuleNotFoundError:
    from verification_utils import (
        fail_result,
        fail_result_with_meta,
        pass_result,
        pass_result_with_meta,
    )


def run_completion_verification() -> list[dict]:
    """Run completion experiments (V10, V11) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")
        total_ticks = sim.run()

        seated = sum(1 for a in sim.agents if a.seated)
        assert seated == len(sim.agents)
        for agent in sim.agents:
            assert agent.position == agent.assigned_seat_node

        details = [
            "Full run reaches completion",
            "All passengers end seated",
            "All passengers end at assigned seat node",
            f"Completion ticks: {total_ticks}",
        ]

        results.append(
            pass_result_with_meta(
                "V10",
                "Completion-condition and final-state consistency",
                details,
                metadata={"kpis": {"event_counters": dict(sim.event_counters)}},
            )
        )
    except Exception as exc:
        results.append(
            fail_result(
                "V10",
                "Completion-condition and final-state consistency",
                [f"Exception: {exc}"],
            )
        )

    try:
        seed_list = [SEED, SEED + 1, SEED + 2]
        details = []
        per_seed_kpis = []

        for run_seed in seed_list:
            env = CabinEnvironment(NODES_FILE, EDGES_FILE)
            sim = BoardingSimulation(env, MANIFEST_FILE, seed=run_seed, boarding_policy="std")
            sim.run()

            positions = [a.position for a in sim.agents if a.position is not None]
            assert len(positions) == len(set(positions))
            assert all(a.position == a.assigned_seat_node for a in sim.agents)
            details.append(f"Seed {run_seed}: invariants hold at completion")
            per_seed_kpis.append(
                {
                    "seed": run_seed,
                    "event_counters": dict(sim.event_counters),
                    "total_ticks": sim.tick,
                }
            )

        avg_row_conflicts = sum(
            row["event_counters"].get("row_conflict_events", 0)
            for row in per_seed_kpis
        ) / max(len(per_seed_kpis), 1)

        results.append(
            pass_result_with_meta(
                "V11",
                "Repeated-run completion invariants",
                details,
                metadata={
                    "kpis": {
                        "per_seed": per_seed_kpis,
                        "averages": {
                            "row_conflict_events": avg_row_conflicts,
                        },
                    }
                },
            )
        )
    except Exception as exc:
        results.append(
            fail_result_with_meta(
                "V11",
                "Repeated-run completion invariants",
                [f"Exception: {exc}"],
                metadata={"kpis": {"per_seed": per_seed_kpis if 'per_seed_kpis' in locals() else []}},
            )
        )

    return results
