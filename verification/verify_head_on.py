"""Head-on conflict verification experiments.

V10 checks opposite-direction conflict detection, priority, and yielding behavior.
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


def _find_candidate_pair(env: CabinEnvironment) -> tuple[str, str] | tuple[None, None]:
    """Find adjacent aisle nodes suitable for constructing a head-on scenario."""
    for u in env.aisle_nodes:
        u_neighbors = env.neighbors(u)
        seat_neighbors = [n for n in u_neighbors if env.node_type(n) == "seat"]
        aisle_neighbors = [n for n in u_neighbors if env.node_type(n) == "aisle"]
        if not seat_neighbors:
            continue
        for v in aisle_neighbors:
            v_aisle_neighbors = [n for n in env.neighbors(v) if env.node_type(n) == "aisle" and n != u]
            if v_aisle_neighbors:
                return u, v
    return None, None


def run_head_on_verification() -> list[dict]:
    """Run head-on conflict experiment (V10) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        u, v = _find_candidate_pair(env)
        assert u is not None and v is not None

        p = sim.agents[0]
        q = sim.agents[1]

        p.spawned = True
        p.seated = False
        p.position = u
        p.has_luggage = False
        p.luggage_status = "none"
        p.assigned_aisle = env.aisle_type(u) or p.assigned_aisle
        p.seat_x = env.node_x(u) + 10

        q.spawned = True
        q.seated = False
        q.position = v
        q.has_luggage = True
        q.luggage_status = "unstowed"
        q.assigned_aisle = env.aisle_type(v) or q.assigned_aisle
        q.seat_x = env.node_x(v) - 10

        occupied = {u, v}
        all_agents = {p.pax_id: p, q.pax_id: q}
        agent_at = {u: p, v: q}

        p.evaluate_intent(env, occupied, all_agents, agent_at, agent_at)
        assert p.intent == "resolveHeadOn"
        assert p.head_on_conflict is True
        assert p.opponent_has_priority is True

        next_positions: dict[str, str] = {}
        action = p.execute_action(
            env,
            occupied,
            agent_at,
            all_agents,
            next_positions,
            sim.rng,
        )
        assert action.startswith("yield:") or action == "wait"

        details = [
            "Opposite-direction adjacent aisle agents trigger head-on conflict",
            "Priority assignment follows luggage rule (non-luggage yields)",
            "Resolution action is yielding (or safe wait if no refuge seat)",
        ]
        results.append(pass_result("V10", "Head-on conflict detection and resolution", details))
    except Exception as exc:
        results.append(
            fail_result(
                "V10",
                "Head-on conflict detection and resolution",
                [f"Exception: {exc}"],
            )
        )

    return results
