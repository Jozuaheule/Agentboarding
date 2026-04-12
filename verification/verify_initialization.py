"""Initialization verification experiments.

V3 validates passenger-agent initial states and spawn-queue consistency.
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


def run_initialization_verification() -> list[dict]:
    """Run initialization experiment (V3) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        details: list[str] = []

        assert len(sim.agents) > 0
        details.append("At least one passenger agent is created from manifest")

        for agent in sim.agents:
            assert agent.assigned_seat_node in env.seat_nodes
            assert agent.spawned is False
            assert agent.seated is False
            assert agent.position is None
            assert agent.intent == "none"
            assert agent.time_since_move == 0

            if agent.has_luggage:
                assert agent.luggage_status == "unstowed"
            else:
                assert agent.luggage_status == "none"

        details.append("All agents satisfy baseline initial-state checks")

        queued = sum(len(queue) for queue in sim.spawn_queues.values())
        assert queued == len(sim.agents)
        details.append("All agents are present in spawn queues")

        results.append(
            pass_result(
                "V3",
                "Passenger initialization and queue consistency",
                details,
            )
        )
    except Exception as exc:
        results.append(
            fail_result(
                "V3",
                "Passenger initialization and queue consistency",
                [f"Exception: {exc}"],
            )
        )

    return results
