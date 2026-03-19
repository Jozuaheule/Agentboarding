"""
Agent-Based Boarding Simulation for a Boeing 787 twin-aisle cabin.

Phase 2 features:
  - Loads the cabin graph (nodes_787.xlsx, edges_787.xlsx)
  - Loads a passenger manifest (generated_manifest.xlsx)
  - Spawns passengers one-by-one at their assigned door
  - Each tick, passengers advance along the shortest path toward their seat
  - Passengers with luggage stow it at their row's aisle node (blocking aisle)
  - When a passenger reaches their seat node, they sit down
  - Boarding completes when every passenger is seated

Simplifications:
  - No head-on conflict resolution
  - No row-blocking / squeeze maneuvers
  - No aisle switching
  - Single strategy: random boarding through middle door (Strategy 1)
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent / "Graph_and_manifest"
NODES_FILE = BASE_DIR / "nodes_787.xlsx"
EDGES_FILE = BASE_DIR / "edges_787.xlsx"
MANIFEST_FILE = BASE_DIR / "generated_manifest.xlsx"

SEED = 42
MAX_TICKS = 10_000          # safety cap
SPAWN_RATE = 1              # passengers spawned per door per tick
REPORT_EVERY = 50           # print progress every N ticks


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class CabinEnvironment:
    """Represents the aircraft cabin as a directed graph."""

    def __init__(self, nodes_file: Path, edges_file: Path) -> None:
        df_nodes = pd.read_excel(nodes_file)
        df_edges = pd.read_excel(edges_file)

        self.graph = nx.DiGraph()

        for _, row in df_nodes.iterrows():
            nid = row["id"]
            ntype = "seat" if row["type"] == "stand" else row["type"]
            self.graph.add_node(nid, x=int(row["x"]), y=int(row["y"]), type=ntype)

        for _, row in df_edges.iterrows():
            self.graph.add_edge(row["from"], row["to"], length=row["length"])

        # Build quick look-ups
        self.seat_nodes: Dict[str, dict] = {}      # id -> attrs
        self.aisle_nodes: Dict[str, dict] = {}
        for nid, data in self.graph.nodes(data=True):
            if data["type"] == "seat":
                self.seat_nodes[nid] = data
            elif data["type"] == "aisle":
                self.aisle_nodes[nid] = data

        # Identify door / spawn nodes.
        # Convention: the "M" (middle) door is around x ≈ 23 on the left aisle (y=3).
        # We pick the aisle node closest to x=23, y=3 as the middle door.
        # The "F" (front) door is the aisle node at x=0, y=3.
        # The "R" (rear) door is the aisle node near x=125, y=3.
        self.doors: Dict[str, str] = {}
        self._find_door("F", target_x=0, target_y=3)
        self._find_door("M", target_x=23, target_y=3)
        self._find_door("R", target_x=125, target_y=3)

        print(f"Environment loaded: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"  Seats: {len(self.seat_nodes)}, Aisle: {len(self.aisle_nodes)}")
        print(f"  Doors: {self.doors}")

    def _find_door(self, label: str, target_x: int, target_y: int) -> None:
        """Find the aisle node closest to (target_x, target_y)."""
        best_id, best_dist = None, float("inf")
        for nid, data in self.aisle_nodes.items():
            d = abs(data["x"] - target_x) + abs(data["y"] - target_y)
            if d < best_dist:
                best_dist = d
                best_id = nid
        if best_id is not None:
            self.doors[label] = best_id

    def seat_node_at(self, x: int, y: int) -> Optional[str]:
        """Return the seat node id at coordinates (x, y), or None."""
        for nid, data in self.seat_nodes.items():
            if data["x"] == x and data["y"] == y:
                return nid
        return None

    def node_x(self, node_id: str) -> int:
        """Return the x-coordinate of a node."""
        return self.graph.nodes[node_id]["x"]

    def node_type(self, node_id: str) -> str:
        """Return the type of a node."""
        return self.graph.nodes[node_id]["type"]

    def neighbors(self, node_id: str) -> List[str]:
        """Successors in the directed graph."""
        return list(self.graph.successors(node_id))


# ---------------------------------------------------------------------------
# Passenger Agent
# ---------------------------------------------------------------------------
class PassengerAgent:
    """A single passenger agent with BDI loop including luggage stowage."""

    def __init__(
        self,
        pax_id: str,
        assigned_seat_node: str,
        assigned_spawn: str,
        assigned_aisle: str,
        seat_x: int,
        has_luggage: bool = False,
        stow_duration: int = 0,
    ) -> None:
        self.pax_id = pax_id
        self.assigned_seat_node = assigned_seat_node
        self.assigned_spawn = assigned_spawn       # "F", "M", or "R"
        self.assigned_aisle = assigned_aisle        # "L" or "R"
        self.seat_x = seat_x                       # x-coordinate of assigned seat row

        # Luggage attributes
        self.has_luggage = has_luggage
        self.stow_duration = stow_duration

        # Internal state
        self.position: Optional[str] = None        # current node id (None = not yet spawned)
        self.seated: bool = False
        self.spawned: bool = False
        self.boarding_time: int = 0                 # ticks since spawn
        self.wait_count: int = 0                    # total ticks spent waiting

        # Luggage state: "none" | "unstowed" | "stowing" | "stowed"
        self.luggage_status: str = "unstowed" if has_luggage else "none"
        self.remaining_stow_time: int = 0

        # Cached shortest path to seat (list of node ids, excluding current)
        self._path_cache: Optional[List[str]] = None

    # --- Perception ----------------------------------------------------------
    def compute_path(self, env: CabinEnvironment) -> None:
        """Compute shortest path from current position to assigned seat."""
        if self.position is None or self.seated:
            self._path_cache = None
            return
        try:
            path = nx.shortest_path(
                env.graph, self.position, self.assigned_seat_node, weight="length"
            )
            self._path_cache = path[1:]  # drop current position
        except nx.NetworkXNoPath:
            self._path_cache = None

    def next_desired_node(self) -> Optional[str]:
        """Next node along the cached path."""
        if self._path_cache:
            return self._path_cache[0]
        return None

    def _at_seat_row_aisle(self, env: CabinEnvironment) -> bool:
        """Check if passenger is at an aisle node in the same row as their seat."""
        if self.position is None:
            return False
        return (
            env.node_type(self.position) == "aisle"
            and env.node_x(self.position) == self.seat_x
        )

    # --- Action generation ---------------------------------------------------
    def step(self, env: CabinEnvironment, occupied: Set[str]) -> Optional[str]:
        """
        Decide and execute one action. Returns the action description string.

        Actions:
          - "sit"            : passenger reaches seat, becomes seated
          - "move_to X"      : passenger moves to node X
          - "start_stow"     : begin luggage stowage (blocks aisle)
          - "continue_stow"  : continue stowing (N ticks remaining)
          - "wait"           : passenger cannot advance, waits
        """
        if self.seated or not self.spawned:
            return None

        self.boarding_time += 1

        # --- Priority 1: If at seat node -> sit ---
        if self.position == self.assigned_seat_node:
            self.seated = True
            occupied.discard(self.position)  # seats don't block aisle movement
            return "sit"

        # --- Priority 2: Continue ongoing luggage stowage ---
        if self.luggage_status == "stowing":
            self.remaining_stow_time -= 1
            if self.remaining_stow_time <= 0:
                self.luggage_status = "stowed"
                return "stow_complete"
            return f"continue_stow ({self.remaining_stow_time} left)"

        # --- Priority 3: Start luggage stowage at row aisle node ---
        if self.luggage_status == "unstowed" and self._at_seat_row_aisle(env):
            self.luggage_status = "stowing"
            self.remaining_stow_time = self.stow_duration
            return "start_stow"

        # --- Priority 4: Advance toward seat ---
        # Compute / recompute path if needed
        if self._path_cache is None or len(self._path_cache) == 0:
            self.compute_path(env)

        next_node = self.next_desired_node()

        if next_node is None:
            # No path found – wait
            self.wait_count += 1
            return "wait"

        # Check if next node is free
        if next_node not in occupied:
            # Move
            old = self.position
            occupied.discard(old)
            self.position = next_node
            occupied.add(next_node)
            self._path_cache = self._path_cache[1:]
            return f"move_to {next_node}"
        else:
            # Blocked – wait and recompute next tick
            self.wait_count += 1
            self._path_cache = None   # force recomputation
            return "wait"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class BoardingSimulation:
    """Runs the discrete-time boarding simulation."""

    def __init__(
        self,
        env: CabinEnvironment,
        manifest_file: Path,
        seed: int = SEED,
    ) -> None:
        self.env = env
        self.rng = random.Random(seed)
        self.tick = 0
        self.occupied: Set[str] = set()

        # Load manifest and create agents
        df = pd.read_excel(manifest_file)
        self.agents: List[PassengerAgent] = []
        missing_seats = 0

        for _, row in df.iterrows():
            seat_node = env.seat_node_at(int(row["x_coord"]), int(row["y_coord"]))
            if seat_node is None:
                missing_seats += 1
                continue

            has_lug = bool(row.get("has_luggage", False))
            # Scale stow_duration: manifest values are in seconds,
            # we convert to ticks (1 tick ≈ 1 second, but values are large,
            # so we scale down for reasonable simulation speed)
            raw_stow = int(row.get("stow_duration", 0)) if has_lug else 0
            stow_dur = max(1, raw_stow // 10) if has_lug else 0

            agent = PassengerAgent(
                pax_id=str(row["pax_id"]),
                assigned_seat_node=seat_node,
                assigned_spawn=str(row["preferred_spawn"]),
                assigned_aisle=str(row["assigned_aisle"]),
                seat_x=int(row["x_coord"]),
                has_luggage=has_lug,
                stow_duration=stow_dur,
            )
            self.agents.append(agent)

        if missing_seats:
            print(f"  Warning: {missing_seats} passengers skipped (seat node not found)")

        # Boarding queue: randomize order (Strategy 1: random through M door)
        self.rng.shuffle(self.agents)
        self.spawn_queue: deque[PassengerAgent] = deque(self.agents)

        print(f"Simulation created with {len(self.agents)} passengers")

    def _spawn_next(self) -> None:
        """Attempt to spawn the next passenger(s) from the queue."""
        spawned_this_tick = 0
        while self.spawn_queue and spawned_this_tick < SPAWN_RATE:
            agent = self.spawn_queue[0]
            # Use middle door for Strategy 1
            door_node = self.env.doors.get("M")
            if door_node is None:
                break
            if door_node in self.occupied:
                break  # door blocked, try next tick
            # Spawn
            agent.position = door_node
            agent.spawned = True
            self.occupied.add(door_node)
            self.spawn_queue.popleft()
            spawned_this_tick += 1

    def step(self) -> bool:
        """
        Execute one simulation tick.
        Returns True if boarding is complete (all seated), False otherwise.
        """
        self.tick += 1

        # 1. Spawn
        self._spawn_next()

        # 2. Shuffle agent update order for fairness
        active = [a for a in self.agents if a.spawned and not a.seated]
        self.rng.shuffle(active)

        # 3. Each active agent takes one action
        for agent in active:
            agent.step(self.env, self.occupied)

        # 4. Check completion
        all_seated = all(a.seated for a in self.agents)
        return all_seated

    def run(self) -> int:
        """Run until boarding complete or MAX_TICKS. Returns total ticks."""
        print(f"\n{'='*60}")
        print(f"  BOARDING SIMULATION START  (Strategy 1: Random / M door)")
        print(f"{'='*60}\n")

        while self.tick < MAX_TICKS:
            done = self.step()

            if self.tick % REPORT_EVERY == 0 or done:
                seated = sum(1 for a in self.agents if a.seated)
                spawned = sum(1 for a in self.agents if a.spawned)
                print(
                    f"  t={self.tick:>5d}  |  "
                    f"spawned={spawned:>3d}/{len(self.agents)}  |  "
                    f"seated={seated:>3d}/{len(self.agents)}  |  "
                    f"queue={len(self.spawn_queue)}"
                )
            if done:
                break

        # Summary
        total = self.tick
        seated_count = sum(1 for a in self.agents if a.seated)
        avg_boarding = (
            sum(a.boarding_time for a in self.agents if a.seated) / max(seated_count, 1)
        )
        avg_wait = (
            sum(a.wait_count for a in self.agents if a.seated) / max(seated_count, 1)
        )
        pax_with_luggage = sum(1 for a in self.agents if a.has_luggage)
        avg_stow = (
            sum(a.stow_duration for a in self.agents if a.has_luggage) / max(pax_with_luggage, 1)
        )

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total boarding time : {total} ticks")
        print(f"  Passengers seated   : {seated_count} / {len(self.agents)}")
        print(f"  Passengers w/ luggage: {pax_with_luggage} / {len(self.agents)}")
        print(f"  Avg boarding time   : {avg_boarding:.1f} ticks/passenger")
        print(f"  Avg wait time       : {avg_wait:.1f} ticks/passenger")
        print(f"  Avg stow duration   : {avg_stow:.1f} ticks (luggage pax only)")
        print(f"{'='*60}\n")

        # Sanity checks
        assert seated_count == len(self.agents), (
            f"Not all passengers seated! {seated_count}/{len(self.agents)}"
        )
        for a in self.agents:
            assert a.position == a.assigned_seat_node, (
                f"Passenger {a.pax_id} not at seat! "
                f"pos={a.position}, seat={a.assigned_seat_node}"
            )
        print("  ✅ All sanity checks passed.\n")
        return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    env = CabinEnvironment(NODES_FILE, EDGES_FILE)
    sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED)
    sim.run()


if __name__ == "__main__":
    main()
