"""Movement and routing verification experiments.

V5 checks legal/occupancy-safe movement.
V6 checks progress toward aisle-access and assigned seat in an isolated scenario.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation import BoardingSimulation, CabinEnvironment, EDGES_FILE, MANIFEST_FILE, NODES_FILE, SEED

try:
    from verification.verification_utils import fail_result, pass_result
except ModuleNotFoundError:
    from verification_utils import fail_result, pass_result


def run_movement_verification() -> list[dict]:
    """Run movement and routing experiments (V5, V6) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        max_ticks = 180
        for _ in range(max_ticks):
            prev_positions = {agent.pax_id: agent.position for agent in sim.agents}
            done = sim.step()

            live_positions = [a.position for a in sim.agents if a.spawned and a.position is not None]
            assert len(live_positions) == len(set(live_positions))

            for pos in live_positions:
                assert pos in env.graph

            for agent in sim.agents:
                before = prev_positions[agent.pax_id]
                after = agent.position
                if before is None and after is not None:
                    spawn_node = env.doors.get(agent.assigned_spawn)
                    assert spawn_node is not None
                    assert after == spawn_node or after in env.neighbors(spawn_node)
                    continue
                if before is None or after is None or before == after:
                    continue

                if agent.last_action == "finishShuffle":
                    assert after == agent.assigned_seat_node
                else:
                    assert after in env.neighbors(before)

            if done:
                break

        details = [
            "All observed positions remain on valid graph nodes",
            "No duplicate occupancy observed per tick",
            "All non-shuffle moves follow graph successor edges",
            "Teleport move is only accepted for finishShuffle seat-resolution",
        ]
        results.append(pass_result("V5", "Movement legality and occupancy safety", details))
    except Exception as exc:
        results.append(
            fail_result(
                "V5",
                "Movement legality and occupancy safety",
                [f"Exception: {exc}", traceback.format_exc()],
            )
        )

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")

        agent = next(
            (
                a
                for a in sim.agents
                if (not a.has_luggage)
                and a.assigned_spawn in env.doors
                and a._aisle_access_node(env) is not None
                and env.aisle_type(env.doors[a.assigned_spawn]) == a.assigned_aisle
            ),
            next((a for a in sim.agents if not a.has_luggage), sim.agents[0]),
        )
        agent.has_luggage = False
        agent.luggage_status = "none"
        agent.spawned = True
        agent.seated = False

        start_node = env.doors.get(agent.assigned_spawn)
        assert start_node is not None
        agent.position = start_node

        ap = agent._aisle_access_node(env)
        assert ap is not None

        reached_ap = agent.position == ap
        reached_seat = False
        initial_ap_distance = agent._manhattan(agent.position, ap, env)
        min_ap_distance = initial_ap_distance
        max_steps = 350

        for _ in range(max_steps):
            cur = agent.position
            assert cur is not None
            target = ap if cur != ap else agent.assigned_seat_node
            d_before = agent._manhattan(cur, target, env)
            on_assigned_before = (
                env.node_type(cur) == "aisle"
                and env.aisle_type(cur) == agent.assigned_aisle
            )

            occupied = {cur}
            all_agents = {agent.pax_id: agent}
            agent_at = {cur: agent}

            agent.evaluate_intent(env, occupied, all_agents, agent_at, agent_at)
            next_positions: dict[str, str] = {}
            action = agent.execute_action(
                env,
                occupied,
                agent_at,
                all_agents,
                next_positions,
                sim.rng,
            )
            agent.last_action = action

            if action.startswith("moveTo:"):
                new_pos = agent.position
                assert new_pos is not None
                if on_assigned_before:
                    d_after = agent._manhattan(new_pos, target, env)
                    assert d_after <= d_before

            if agent.position == ap:
                reached_ap = True
            if agent.position is not None:
                min_ap_distance = min(
                    min_ap_distance,
                    agent._manhattan(agent.position, ap, env),
                )
            if agent.position == agent.assigned_seat_node:
                reached_seat = True
            if agent.seated:
                reached_seat = True
                break

        assert reached_seat is True
        assert reached_ap or min_ap_distance < initial_ap_distance

        details = [
            "Single isolated passenger reaches assigned seat in finite steps",
            "Passenger trajectory shows measurable progress toward row access",
            "On assigned aisle, move decisions do not increase distance to current target",
        ]
        results.append(pass_result("V6", "Routing progression toward access/seat", details))
    except Exception as exc:
        results.append(
            fail_result(
                "V6",
                "Routing progression toward access/seat",
                [f"Exception: {exc}", traceback.format_exc()],
            )
        )

    return results
