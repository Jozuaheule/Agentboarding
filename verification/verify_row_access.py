"""Row-access verification experiments.

V8 checks clean row entry and V9 checks blocked-row resolution behavior.
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


def _find_clean_entry_agent(sim: BoardingSimulation, env: CabinEnvironment):
    """Pick an agent with a valid aisle-access node for clean-entry checks."""
    for agent in sim.agents:
        ap = agent._aisle_access_node(env)
        if ap is not None:
            return agent, ap
    return None, None


def _find_blocked_entry_agent(sim: BoardingSimulation, env: CabinEnvironment):
    """Pick an agent/path where at least one on-path blocker seat exists."""
    for agent in sim.agents:
        ap = agent._aisle_access_node(env)
        on_path = agent._on_path_nodes(env)
        if ap is not None and on_path:
            return agent, ap, on_path[0]
    return None, None, None


def run_row_access_verification() -> list[dict]:
    """Run row-access experiments (V8, V9) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        agent, ap = _find_clean_entry_agent(sim, env)
        assert agent is not None and ap is not None

        agent.spawned = True
        agent.seated = False
        agent.position = ap
        agent.has_luggage = False
        agent.luggage_status = "none"

        occupied = {ap}
        all_agents = {agent.pax_id: agent}
        agent_at = {ap: agent}

        agent.evaluate_intent(env, occupied, all_agents, agent_at, agent_at)
        assert agent.intent == "enterRow"

        next_positions: dict[str, str] = {}
        action = agent.execute_action(
            env,
            occupied,
            agent_at,
            all_agents,
            next_positions,
            sim.rng,
        )

        assert action.startswith("moveTo:")
        assert agent.position != ap

        details = [
            "Passenger at aisle-access node chooses enterRow when path is clear",
            "Enter-row action performs a legal row movement",
        ]
        results.append(pass_result("V8", "Row entry without blockers", details))
    except Exception as exc:
        results.append(
            fail_result(
                "V8",
                "Row entry without blockers",
                [f"Exception: {exc}"],
            )
        )

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        blocked, ap, blocker_node = _find_blocked_entry_agent(sim, env)
        assert blocked is not None and ap is not None and blocker_node is not None

        blocker = next(a for a in sim.agents if a.pax_id != blocked.pax_id)

        blocked.spawned = True
        blocked.seated = False
        blocked.position = ap
        blocked.has_luggage = False
        blocked.luggage_status = "none"
        blocked.seat_shuffle_delay = 0

        blocker.spawned = True
        blocker.seated = False
        blocker.position = blocker_node

        occupied = {ap, blocker_node}
        all_agents = {blocked.pax_id: blocked, blocker.pax_id: blocker}
        agent_at = {ap: blocked, blocker_node: blocker}

        blocked.evaluate_intent(env, occupied, all_agents, agent_at, agent_at)
        assert blocked.intent == "resolveSeatBlock"
        assert blocked.row_blocked is True

        next_positions: dict[str, str] = {}
        action1 = blocked.execute_action(
            env,
            occupied,
            agent_at,
            all_agents,
            next_positions,
            sim.rng,
        )
        assert action1 == "startShuffle"
        assert blocked.seat_shuffle_delay > 0

        blocked.intent = "resolveSeatBlock"
        blocked.row_blocked = True
        blocked.seat_shuffle_delay = 1
        next_positions = {}
        action2 = blocked.execute_action(
            env,
            occupied,
            agent_at,
            all_agents,
            next_positions,
            sim.rng,
        )
        assert action2 == "finishShuffle"
        assert blocked.position == blocked.assigned_seat_node

        details = [
            "Blocked row access is detected at aisle-access node",
            "Seat-block resolution starts shuffle delay",
            "After delay expiry, passenger reaches assigned seat",
        ]
        results.append(pass_result("V9", "Row-block detection and resolution", details))
    except Exception as exc:
        results.append(
            fail_result(
                "V9",
                "Row-block detection and resolution",
                [f"Exception: {exc}"],
            )
        )

    return results
