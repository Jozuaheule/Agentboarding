"""Spawning verification experiments.

V4 checks downstream-gated spawning behavior at door nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation import BoardingSimulation, CabinEnvironment, EDGES_FILE, MANIFEST_FILE, NODES_FILE, SEED
try:
    from verification.verification_utils import fail_result, pass_result
except ModuleNotFoundError:
    from verification_utils import fail_result, pass_result


def run_spawning_verification() -> list[dict]:
    """Run spawn-gating experiment (V4) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        non_empty_doors = [door for door, queue in sim.spawn_queues.items() if queue]
        assert non_empty_doors
        door_label = non_empty_doors[0]

        door_node = env.doors[door_label]
        first_agent = sim.spawn_queues[door_label][0]

        first_agent.position = door_node
        blocked_next = first_agent._best_aisle_advance(env, sim.occupied)
        first_agent.position = None
        assert blocked_next is not None

        sim.occupied.add(blocked_next)
        sim._spawn_next()
        assert first_agent.spawned is False

        sim.occupied.discard(blocked_next)
        sim._spawn_next()
        assert first_agent.spawned is True
        assert first_agent.position == door_node

        details = [
            "Passenger does not spawn when first downstream step is blocked",
            "Passenger spawns when downstream step becomes available",
            "Spawned passenger position equals assigned door node",
        ]

        results.append(
            pass_result(
                "V4",
                "Spawn-gating behavior",
                details,
            )
        )
    except Exception as exc:
        results.append(
            fail_result(
                "V4",
                "Spawn-gating behavior",
                [f"Exception: {exc}"],
            )
        )

    return results
