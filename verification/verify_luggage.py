"""Luggage-state verification experiments.

V7 checks the state sequence unstowed -> stowing -> stowed.
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


def run_luggage_verification() -> list[dict]:
    """Run luggage transition experiment (V7) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        luggage_agent = next((a for a in sim.agents if a.has_luggage), None)
        assert luggage_agent is not None

        ap = luggage_agent._aisle_access_node(env)
        assert ap is not None

        luggage_agent.spawned = True
        luggage_agent.seated = False
        luggage_agent.position = ap
        luggage_agent.luggage_status = "unstowed"
        luggage_agent.remaining_stow_time = 0

        occupied = {ap}
        all_agents = {luggage_agent.pax_id: luggage_agent}
        agent_at = {ap: luggage_agent}

        luggage_agent.evaluate_intent(env, occupied, all_agents, agent_at, agent_at)
        assert luggage_agent.intent == "stow"

        next_positions: dict[str, str] = {}
        action = luggage_agent.execute_action(
            env,
            occupied,
            agent_at,
            all_agents,
            next_positions,
            sim.rng,
        )
        assert action == "startStow"
        assert luggage_agent.luggage_status == "stowing"

        while True:
            luggage_agent.intent = "stow"
            next_positions = {}
            action = luggage_agent.execute_action(
                env,
                occupied,
                agent_at,
                all_agents,
                next_positions,
                sim.rng,
            )
            if action == "stowComplete":
                break

        assert luggage_agent.luggage_status == "stowed"

        details = [
            "Passenger in stow zone selects stow intent",
            "First stow action transitions unstowed -> stowing",
            "Stow countdown progresses until stowComplete",
            "Final luggage state is stowed",
        ]
        results.append(pass_result("V7", "Luggage-state transition", details))
    except Exception as exc:
        results.append(
            fail_result(
                "V7",
                "Luggage-state transition",
                [f"Exception: {exc}"],
            )
        )

    return results
