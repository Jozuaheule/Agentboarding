"""Environment-focused verification experiments.

V1 checks topology and door/aisle consistency.
V2 checks seat coordinate lookup consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation import CabinEnvironment, EDGES_FILE, NODES_FILE
try:
    from verification.verification_utils import fail_result, pass_result
except ModuleNotFoundError:
    from verification_utils import fail_result, pass_result


def run_environment_verification() -> list[dict]:
    """Run static environment experiments (V1, V2) and return PASS/FAIL records."""
    results: list[dict] = []

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        details: list[str] = []

        assert env.graph.number_of_nodes() > 0
        details.append("Graph contains nodes")

        assert env.graph.number_of_edges() > 0
        details.append("Graph contains edges")

        assert len(env.seat_nodes) > 0
        details.append("Seat-node partition is non-empty")

        assert len(env.aisle_nodes) > 0
        details.append("Aisle-node partition is non-empty")

        for label in ("F", "M", "R"):
            assert label in env.doors
            door_node = env.doors[label]
            assert door_node in env.aisle_nodes
        details.append("All doors F/M/R are mapped to aisle nodes")

        aisle_types = {env.aisle_type(nid) for nid in env.aisle_nodes}
        assert aisle_types.issubset({"L", "R"})
        details.append("Aisle type classification returns only L/R")

        results.append(
            pass_result(
                "V1",
                "Environment loading and topology consistency",
                details,
            )
        )
    except Exception as exc:
        results.append(
            fail_result(
                "V1",
                "Environment loading and topology consistency",
                [f"Exception: {exc}"],
            )
        )

    try:
        env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        details = []

        sampled = list(env.seat_nodes.items())[:15]
        assert sampled
        for seat_node, data in sampled:
            found = env.seat_node_at(int(data["x"]), int(data["y"]))
            assert found == seat_node
        details.append("Seat lookup maps known coordinates to the correct seat node")

        results.append(
            pass_result(
                "V2",
                "Seat lookup consistency",
                details,
            )
        )
    except Exception as exc:
        results.append(
            fail_result(
                "V2",
                "Seat lookup consistency",
                [f"Exception: {exc}"],
            )
        )

    return results
