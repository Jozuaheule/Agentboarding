"""
Agent-Based Boarding Simulation for a Boeing 787 twin-aisle cabin.

Fully aligned with the formal predicate model in ABMS_G20_Report-5.pdf (Ch. 3).

Implemented formal properties:
  IC1-IC6  Initial conditions
    B1-B6    Belief-state evolution
  O1-O6    Observation-based state characterisation
    I1-I3/I5 Intention selection
    A1-A10/A17 Action generation
  C1       System-level completion condition
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from calibration.calibration_config import (
    SHUFFLE_HIGH_S,
    SHUFFLE_LOW_S,
    SHUFFLE_MODE_S,
    SHUFFLE_MODEL,
    ShuffleConfig,
    seconds_to_ticks,
    validate_triangular_mode,
    ticks_to_seconds,
    validate_bounds,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "manifest_generation"
GRAPH_DIR = DATA_DIR / "graph"


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve required input file. Tried:\n - {tried}")


NODES_FILE = _resolve_existing_path(
    GRAPH_DIR / "nodes_787.xlsx",
    DATA_DIR / "nodes_787.xlsx",
)
EDGES_FILE = _resolve_existing_path(
    GRAPH_DIR / "edges_787.xlsx",
    DATA_DIR / "edges_787.xlsx",
)
MANIFEST_FILE = _resolve_existing_path(
    DATA_DIR / "generated_manifest_2.xlsx",
    PROJECT_ROOT / "generated_manifest_2.xlsx",
)

VISUALIZE_ONE = True  # Set False to batch-run all sequentially

SEED = 42
MAX_TICKS = 10_000          # safety cap
SPAWN_RATE = 1              # passengers spawned per door per tick
REPORT_EVERY = 50           # print progress every N ticks
K_OBS = 5                   # perception range in hops  (Nk neighbourhood)


# ---------------------------------------------------------------------------
# Environment  (EW = G = (N, E))
# ---------------------------------------------------------------------------
class CabinEnvironment:
    """Static aircraft cabin graph: nodes (seats, aisle), edges."""

    DOOR_TARGETS = {
        "F": (0, 0),
        "M": (15, 0),
        "R": (79, 1),
    }

    def __init__(self, nodes_file: Path, edges_file: Path) -> None:
        df_nodes = pd.read_excel(nodes_file)
        df_edges = pd.read_excel(edges_file)

        self.graph = nx.DiGraph()
        for _, row in df_nodes.iterrows():
            nid = row["id"]
            ntype = "seat" if row["type"] == "stand" else row["type"]
            self.graph.add_node(nid, x=int(row["x"]), y=int(row["y"]), type=ntype)
        for _, row in df_edges.iterrows():
            self.graph.add_edge(row["from"], row["to"], length=float(row["length"]))

        # Quick look-ups
        self.seat_nodes: Dict[str, dict] = {}
        self.aisle_nodes: Dict[str, dict] = {}
        for nid, data in self.graph.nodes(data=True):
            if data["type"] == "seat":
                self.seat_nodes[nid] = data
            elif data["type"] == "aisle":
                self.aisle_nodes[nid] = data

        self.doors: Dict[str, str] = {}
        for label, (target_x, target_y) in self.DOOR_TARGETS.items():
            self._find_door(label, target_x=target_x, target_y=target_y)

        # x_mid – longitudinal midpoint of the cabin (Property B8)
        all_x = [d["x"] for d in self.graph.nodes.values()]
        self.x_mid: int = (min(all_x) + max(all_x)) // 2

        # Hop-distance graph (undirected) for perception neighbourhood Nk
        self._hop_graph = self.graph.to_undirected()

        print(f"Environment loaded: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"  Seats: {len(self.seat_nodes)}, Aisle: {len(self.aisle_nodes)}")
        print(f"  Doors: {self.doors}  |  x_mid={self.x_mid}")
        # Debug: show door coordinates
        for label, door_id in self.doors.items():
            x, y = self.graph.nodes[door_id]["x"], self.graph.nodes[door_id]["y"]
            print(f"    Door {label}: node_id={door_id}, x={x}, y={y}")

    def _find_door(self, label: str, target_x: int, target_y: int) -> None:
        # Use the exact requested coordinate if present.
        for nid, data in self.aisle_nodes.items():
            if data["x"] == target_x and data["y"] == target_y:
                self.doors[label] = nid
                return

        raise ValueError(
            f"Door {label} target coordinate ({target_x}, {target_y}) not present in aisle nodes."
        )

    def seat_node_at(self, x: int, y: int) -> Optional[str]:
        for nid, data in self.seat_nodes.items():
            if data["x"] == x and data["y"] == y:
                return nid
        return None

    def node_x(self, nid: str) -> int:
        return self.graph.nodes[nid]["x"]

    def node_y(self, nid: str) -> int:
        return self.graph.nodes[nid]["y"]

    def node_type(self, nid: str) -> str:
        return self.graph.nodes[nid]["type"]

    def neighbors(self, nid: str) -> List[str]:
        return list(self.graph.successors(nid))

    def hop_distance(self, a: str, b: str) -> int:
        """Unweighted hop distance (for perception range Nk)."""
        try:
            return nx.shortest_path_length(self._hop_graph, a, b)
        except nx.NetworkXNoPath:
            return 9999

    def aisle_type(self, nid: str) -> Optional[str]:
        """Return 'L' or 'R' for aisle nodes based on y-coordinate."""
        if self.node_type(nid) != "aisle":
            return None
        return "L" if self.node_y(nid) <= 3 else "R"


# ---------------------------------------------------------------------------
# Passenger Agent  (Statep = ⟨Staticp, Internalp, Inputp, Outputp⟩)
# ---------------------------------------------------------------------------
class PassengerAgent:
    """Passenger agent with full BDI loop aligned to Ch. 3 formal properties."""

    def __init__(
        self,
        pax_id: str,
        assigned_seat_node: str,
        assigned_spawn: str,
        alternative_spawn: str,
        assigned_aisle: str,
        seat_x: int,
        seat_y: int,
        travel_class: str = "economy",
        has_luggage: bool = False,
        stow_duration: int = 0,
        shuffle_model: str = SHUFFLE_MODEL,
        shuffle_low_ticks: int = seconds_to_ticks(SHUFFLE_LOW_S, min_ticks=1),
        shuffle_high_ticks: int = seconds_to_ticks(SHUFFLE_HIGH_S, min_ticks=1),
        shuffle_mode_ticks: Optional[int] = (
            seconds_to_ticks(SHUFFLE_MODE_S, min_ticks=1)
            if SHUFFLE_MODE_S is not None
            else None
        ),
        zone_std: int = 1,
        zone_pyramid: int = 1,
    ) -> None:
        # === Staticp (time-invariant, Section 3.3.1) ===
        self.pax_id = pax_id
        self.assigned_seat_node = assigned_seat_node
        self.assigned_spawn = assigned_spawn
        self.alternative_spawn = alternative_spawn
        self.assigned_aisle = assigned_aisle
        self.seat_x = seat_x                        # xs
        self.seat_y = seat_y                        # ys
        self.travel_class = travel_class            # classp
        self.has_luggage = has_luggage              # hasLuggagep
        self.stow_duration = stow_duration          # stowDurationp
        self.shuffle_model = shuffle_model
        self.shuffle_low_ticks = max(1, int(shuffle_low_ticks))
        self.shuffle_high_ticks = max(self.shuffle_low_ticks, int(shuffle_high_ticks))
        self.shuffle_mode_ticks = (
            None
            if shuffle_mode_ticks is None
            else min(self.shuffle_high_ticks, max(self.shuffle_low_ticks, int(shuffle_mode_ticks)))
        )
        self.zone_std = zone_std                    # zoneSTDp
        self.zone_pyramid = zone_pyramid            # zonePyramidp

        # === Internalp – Beliefs ===
        self.position: Optional[str] = None         # observes(p, position(n))   [IC1]
        self.seated: bool = False                   # believes(p, seated)        [IC2]
        self.luggage_status: str = (                # believes(p, luggageStatus) [IC3]
            "unstowed" if has_luggage else "none"
        )
        self.remaining_stow_time: int = 0
        self.seat_shuffle_delay: int = 0            # Timer for penalty
        self.time_since_move: int = 0               # believes(p, timeSinceMove) [IC4]

        # Per-tick beliefs (reset each tick during evaluate_intent)
        self.row_blocker: bool = False              # believes(p, rowBlocker)     [B4]
        self.row_blocked: bool = False              # believes(p, rowBlocked)     [B5]
        self.row_shift_complete: bool = False       # believes(p, rowShiftComplete) [B6]

        # === Internalp – Intentions ===                               [IC5]
        self.intent: str = "wait"

        # === Outputp – Actions ===                                    [IC6]
        self.last_action: str = "wait"

        # === Bookkeeping ===
        self.spawned: bool = False
        self.boarding_time: int = 0
        self.wait_count: int = 0

    # -----------------------------------------------------------------------
    # Helper predicates
    # -----------------------------------------------------------------------
    def _dir(self, env: CabinEnvironment) -> int:
        """dir(p, t) = sign(xs − xm)  – intended travel direction."""
        if self.position is None:
            return 0
        return _sign(self.seat_x - env.node_x(self.position))

    def _ready_to_enter(self) -> bool:
        """readyToEnterp: no luggage OR luggage already stowed."""
        return (not self.has_luggage) or (self.luggage_status == "stowed")

    def _at_seat_row_aisle(self, env: CabinEnvironment) -> bool:
        """PDF B5/I5: True iff passenger is at their assigned aisle-access
        node ap.  Passengers always navigate to ap before entering the
        row, so this strict check is safe."""
        if self.position is None:
            return False
        ap = self._aisle_access_node(env)
        return ap is not None and self.position == ap

    def _on_assigned_aisle(self, env: CabinEnvironment) -> bool:
        """True if currently on an aisle node of the assigned aisle type."""
        if self.position is None:
            return False
        if env.node_type(self.position) != "aisle":
            return False
        return env.aisle_type(self.position) == self.assigned_aisle

    def _aisle_access_node(self, env: CabinEnvironment) -> Optional[str]:
        """ap: aisle-access node on assigned-aisle side for this pax's row."""
        for nid in env.aisle_nodes:
            if (env.node_x(nid) == self.seat_x
                    and env.aisle_type(nid) == self.assigned_aisle):
                return nid
        for nid in env.aisle_nodes:
            if env.node_x(nid) == self.seat_x:
                return nid
        return None

    def _on_path_nodes(self, env: CabinEnvironment) -> List[str]:
        """onPathp(n): seat nodes strictly between aisle-access and assigned seat."""
        ap = self._aisle_access_node(env)
        if ap is None:
            return []
        yap = env.node_y(ap)
        ys = self.seat_y
        result = []
        for nid, data in env.seat_nodes.items():
            if data["x"] == self.seat_x:
                yn = data["y"]
                if min(yap, ys) < yn < max(yap, ys):
                    result.append(nid)
        return result

    def _aisle_side_seat(self, env: CabinEnvironment) -> Optional[str]:
        """aisleSidep(n): seat adjacent to aisle-access in this row."""
        ap = self._aisle_access_node(env)
        if ap is None:
            return None
        for n in env.neighbors(ap):
            if env.node_type(n) == "seat" and env.node_x(n) == self.seat_x:
                return n
        return None

    def _manhattan(self, n1: str, n2: str, env: CabinEnvironment) -> int:
        return abs(env.node_x(n1) - env.node_x(n2)) + abs(env.node_y(n1) - env.node_y(n2))

    def _row_closer(self, n: str, env: CabinEnvironment, target_node: Optional[str] = None) -> bool:
        """O4: rowCloserIfMoveTo — |xs − xn| < |xs − xm|."""
        if self.position is None:
            return False
        tx = self.seat_x if target_node is None else env.node_x(target_node)
        return abs(tx - env.node_x(n)) < abs(tx - env.node_x(self.position))

    def _col_closer(self, n: str, env: CabinEnvironment, target_node: Optional[str] = None) -> bool:
        """O5: columnCloserIfMoveTo — |ys − yn| < |ys − ym|."""
        if self.position is None:
            return False
        ty = self.seat_y if target_node is None else env.node_y(target_node)
        return abs(ty - env.node_y(n)) < abs(ty - env.node_y(self.position))

    def _in_stow_zone(self, env: CabinEnvironment) -> bool:
        """A2: directStowTrigger — at aisle node within ±1 row of seat.
        PDF: |xn − xs| ≤ 1."""
        if self.position is None:
            return False
        return (
            env.node_type(self.position) == "aisle"
            and abs(env.node_x(self.position) - self.seat_x) <= 1
        )

    def _stow_trigger(self, env: CabinEnvironment) -> bool:
        """True if in stow zone with unstowed luggage (PDF A2)."""
        if self.position is None:
            return False
        if not self.has_luggage or self.luggage_status != "unstowed":
            return False
        return self._in_stow_zone(env)

    def _same_aisle_progress(
        self, env: CabinEnvironment, occupied: Set[str]
    ) -> bool:
        """Check whether _best_aisle_advance would return a valid target."""
        if self.position is None:
            return False
        return self._best_aisle_advance(env, occupied) is not None

    # -----------------------------------------------------------------------
    # Routing helpers (implements O4, O5 via coordinate predicates)
    # -----------------------------------------------------------------------
    def _best_aisle_advance(
        self, env: CabinEnvironment, occupied: Set[str]
    ) -> Optional[str]:
        """A5: Aisle advance with assigned-aisle discipline.
        Uses pure Manhattan distance ranking, NO DIJKSTRA."""
        if self.position is None:
            return None

        # --- Target is always ap (while in aisle) ---
        ap = self._aisle_access_node(env)
        target = (
            ap if ap is not None and self.position != ap
            else self.assigned_seat_node
        )
        cur_dist = self._manhattan(self.position, target, env)

        # --- On assigned aisle: coordinate-closer same-aisle (PDF A5) --
        if self._on_assigned_aisle(env):
            my_aisle = env.aisle_type(self.position)
            best, best_d = None, float("inf")
            for n in env.neighbors(self.position):
                if n in occupied or env.node_type(n) != "aisle":
                    continue
                if env.aisle_type(n) != my_aisle:
                    continue
                if not (self._row_closer(n, env, target) or self._col_closer(n, env, target)):
                    continue
                d = self._manhattan(n, target, env)
                if d > cur_dist: # Prevent lateral dead ends
                    continue
                if d < best_d:
                    best, best_d = n, d
            if best is not None:
                return best

        # --- Fallback (Galley Routing): Off-assigned-aisle ---
        # Prioritize colCloser to traverse cross-aisles
        best, best_d = None, float("inf")
        for n in env.neighbors(self.position):
            if n in occupied or env.node_type(n) == "seat":
                continue
            
            # Use pure colCloser property to cross the plane laterally
            if self._col_closer(n, env, target):
                d = self._manhattan(n, target, env)
                if d < best_d:
                    best, best_d = n, d
        
        return best

    def _best_row_step(
        self, env: CabinEnvironment, occupied: Set[str]
    ) -> Optional[str]:
        """A4: enter row — move to free neighbor that is columnCloser
        (PDF A4: columnCloserIfMoveTo). Uses purely mathematical manhattan distance."""
        if self.position is None:
            return None
        target = self.assigned_seat_node
        if self.position == target:
            return None

        best, best_d = None, float("inf")
        for n in env.neighbors(self.position):
            if n in occupied:
                continue
            if self._col_closer(n, env, target) or self._row_closer(n, env, target):
                d = self._manhattan(n, target, env)
                if d < best_d:
                    best, best_d = n, d
        return best

    def _seat_is_neighbor(self, env: CabinEnvironment) -> bool:
        """True if assigned seat is a direct graph neighbor."""
        if self.position is None:
            return False
        return self.assigned_seat_node in env.neighbors(self.position)

    def _any_free_aisle_neighbor(
        self, env: CabinEnvironment, occupied: Set[str]
    ) -> Optional[str]:
        """Return any free aisle neighbor (even if distance increases).
        Used to escape non-target seat nodes."""
        if self.position is None:
            return None
        best, best_d = None, float("inf")
        for n in env.neighbors(self.position):
            if n in occupied:
                continue
            if env.node_type(n) != "aisle":
                continue
            d = self._manhattan(n, self.assigned_seat_node, env)
            if d < best_d:
                best, best_d = n, d
        return best

    def _in_seat_row_not_target(self, env: CabinEnvironment) -> bool:
        """True if in a seat node at the assigned row but not the target seat."""
        if self.position is None:
            return False
        return (
            env.node_type(self.position) == "seat"
            and env.node_x(self.position) == self.seat_x
            and self.position != self.assigned_seat_node
        )

    # ===================================================================
    #  Phase A:  INTENTION EVALUATION   (Properties I1-I3/I5, B4-B6)
    # ===================================================================
    def evaluate_intent(
        self,
        env: CabinEnvironment,
        occupied: Set[str],
        all_agents: Dict[str, "PassengerAgent"],
        agent_at: Dict[str, "PassengerAgent"],
        all_agent_at: Dict[str, "PassengerAgent"],
    ) -> None:
        """Evaluate observations → beliefs → intention."""
        if not self.spawned or self.position is None:
            self.intent = "wait"
            return

        # R0: seated → terminal
        if self.seated:
            self.intent = "wait"
            return

        cur = self.position

        # Reset per-tick beliefs
        self.row_blocker = False
        self.row_blocked = False
        self.row_shift_complete = False

        # =================================================================
        #  A1:  Direct Seating — at assigned seat
        # =================================================================
        if cur == self.assigned_seat_node and self._ready_to_enter():
            self.intent = "sit"
            return

        # A1b:  Seat is direct neighbour and free → enter directly
        if (self._ready_to_enter()
                and self._seat_is_neighbor(env)
                and self.assigned_seat_node not in occupied):
            self.intent = "enterRow"
            return

        # =================================================================
        #  A2/A3:  Luggage stowage
        # =================================================================
        if self._stow_trigger(env):
            self.intent = "stow"
            return
        if self.luggage_status == "stowing":
            self.intent = "stow"
            return

        # =================================================================
        #  Continue row entry if already in a seat at our row
        # =================================================================
        if self._in_seat_row_not_target(env) and self._ready_to_enter():
            step = self._best_row_step(env, occupied)
            if step is not None:
                self.intent = "enterRow"
                return

        # =================================================================
        #  Return-to-aisle: in a non-target seat
        # =================================================================
        if (env.node_type(cur) == "seat"
                and cur != self.assigned_seat_node):
            # Try to return to aisle
            aisle_n = self._any_free_aisle_neighbor(env, occupied)
            if aisle_n is not None:
                self.intent = "advance"  # will use _best_aisle_advance or fallback
                return
            # No free aisle neighbor → wait
            self.intent = "wait"
            return

        # =================================================================
        #  B5 / I2:  Row-blocked (at aisle-access, ready, path occupied)
        # =================================================================
        if self._at_seat_row_aisle(env) and self._ready_to_enter():
            on_path = self._on_path_nodes(env)
            blockers = [n for n in on_path if n in all_agent_at]
            # Also check if the target seat itself is occupied by someone else
            seat_occupied = (
                self.assigned_seat_node in all_agent_at
                and all_agent_at[self.assigned_seat_node].pax_id != self.pax_id
            )
            if blockers or seat_occupied:
                self.row_blocked = True
                aisle_side = self._aisle_side_seat(env)
                deeper_clear = all(
                    n not in agent_at for n in on_path if n != aisle_side
                )
                if deeper_clear and aisle_side is not None:
                    self.row_shift_complete = True
                self.intent = "resolveSeatBlock"
                return

        # =================================================================
        #  B4 / I3:  Am I blocking someone else's row entry?
        # =================================================================
        if env.node_type(cur) == "seat":
            for other in agent_at.values():
                if other.pax_id == self.pax_id:
                    continue
                if not other._at_seat_row_aisle(env):
                    continue
                if not other._ready_to_enter():
                    continue
                # Blocking if on their path OR sitting in their target seat
                if cur in other._on_path_nodes(env) or cur == other.assigned_seat_node:
                    self.row_blocker = True
                    self.intent = "resolveSeatBlock"
                    return

        # =================================================================
        #  I5:  Enter row (at seat-row aisle, ready, no blockers)
        # =================================================================
        if (self._at_seat_row_aisle(env)
                and self._ready_to_enter()
                and cur != self.assigned_seat_node):
            self.intent = "enterRow"
            return

        # =================================================================
        #  I1:  Advance (same-aisle progress toward seat, PDF I1)
        # =================================================================
        if self._same_aisle_progress(env, occupied):
            self.intent = "advance"
            return

        # =================================================================
        #  A17:  Aisle switch — REMOVED
        #  Passengers are directed to their assigned aisle by the
        #  flight attendant at the door and stick with it.
        #  Cross-aisle routing is handled by apCloserIfMoveTo in
        #  _best_aisle_advance when the passenger has not yet
        #  reached their assigned aisle.
        # =================================================================

        # =================================================================
        #  A6:  Default wait
        # =================================================================
        self.intent = "wait"

    # ===================================================================
    #  Phase B:  ACTION EXECUTION   (Properties A1–A17)
    # ===================================================================
    def execute_action(
        self,
        env: CabinEnvironment,
        occupied: Set[str],
        agent_at: Dict[str, "PassengerAgent"],
        all_agents: Dict[str, "PassengerAgent"],
        next_positions: Dict[str, str],
        rng: random.Random,
    ) -> str:
        """Translate intent into physical state change."""
        if not self.spawned or self.position is None:
            return "wait"

        cur = self.position

        # ---- A1: sit ----
        if self.intent == "sit":
            if cur != self.assigned_seat_node:
                # Safety: shouldn't happen, but don't set seated if not at seat
                next_positions[self.pax_id] = cur
                return "wait"
            self.seated = True                      # B1
            self.time_since_move = 0
            # Seated agents don't block aisle
            return "sit"

        # ---- A2/A3: luggage stowage ----
        if self.intent == "stow":
            next_positions[self.pax_id] = cur
            if self.luggage_status == "unstowed":
                self.luggage_status = "stowing"
                self.remaining_stow_time = self.stow_duration
                self.time_since_move = 0
                return "startStow"
            else:
                self.remaining_stow_time -= 1
                if self.remaining_stow_time <= 0:
                    self.luggage_status = "stowed"
                    return "stowComplete"
                return "wait"

        # ---- A5: advance ----
        if self.intent == "advance":
            target = self._best_aisle_advance(env, occupied)
            # Fallback: if in a non-target seat, allow any free aisle neighbor
            if target is None and env.node_type(cur) == "seat" and cur != self.assigned_seat_node:
                target = self._any_free_aisle_neighbor(env, occupied)
            if target is not None and target not in next_positions.values():
                next_positions[self.pax_id] = target
                self.position = target
                self.time_since_move = 0
                return f"moveTo:{target}"
            next_positions[self.pax_id] = cur
            self.time_since_move += 1
            return "wait"

        # ---- A4: enter row ----
        if self.intent == "enterRow":
            target = self._best_row_step(env, occupied)
            if target is not None and target not in next_positions.values():
                next_positions[self.pax_id] = target
                self.position = target
                self.time_since_move = 0
                return f"moveTo:{target}"
            next_positions[self.pax_id] = cur
            self.time_since_move += 1
            return "wait"

        # ---- A7-A10/A15: seat-block resolution ----
        if self.intent == "resolveSeatBlock":
            if self.row_blocker:
                # The agent natively occupying the physical seat just sits and waits.
                # The burden of the delay is carried mathematically by the incoming passenger.
                next_positions[self.pax_id] = cur
                self.time_since_move += 1
                return "wait"

            elif self.row_blocked:
                # Time Penalty Execution
                if self.seat_shuffle_delay == 0:
                    if self.shuffle_model == "uniform":
                        self.seat_shuffle_delay = rng.randint(
                            self.shuffle_low_ticks,
                            self.shuffle_high_ticks,
                        )
                    elif self.shuffle_model == "triangular":
                        mode = (
                            self.shuffle_mode_ticks
                            if self.shuffle_mode_ticks is not None
                            else (self.shuffle_low_ticks + self.shuffle_high_ticks) / 2
                        )
                        sampled = rng.triangular(
                            self.shuffle_low_ticks,
                            self.shuffle_high_ticks,
                            mode,
                        )
                        sampled_ticks = int(round(sampled))
                        self.seat_shuffle_delay = min(
                            self.shuffle_high_ticks,
                            max(self.shuffle_low_ticks, sampled_ticks),
                        )
                    else:
                        raise ValueError(f"Unsupported shuffle model: {self.shuffle_model}")
                    next_positions[self.pax_id] = cur
                    return "startShuffle"
                
                self.seat_shuffle_delay -= 1
                if self.seat_shuffle_delay <= 0:
                    # Delay is complete. "Teleport" the passenger into the seat node.
                    target = self.assigned_seat_node
                    next_positions[self.pax_id] = target
                    self.position = target
                    self.time_since_move = 0
                    return "finishShuffle"
                else:
                    next_positions[self.pax_id] = cur
                    return "shufflingSeat"

            next_positions[self.pax_id] = cur
            self.time_since_move += 1
            return "wait"

        # ---- A6: default wait ----
        next_positions[self.pax_id] = cur
        self.time_since_move += 1
        return "wait"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class BoardingSimulation:
    """Discrete-time boarding simulation (C1 completion condition)."""

    def __init__(
        self,
        env: CabinEnvironment,
        manifest_file: Union[Path, pd.DataFrame],
        seed: int = SEED,
        boarding_policy: str = "random",
        shuffle_config: Optional[ShuffleConfig] = None,
        cross_zone_violation_rate: float = 0.05,
        log_summary: bool = True,
    ) -> None:
        self.env = env
        self.rng = random.Random(seed)
        self.tick = 0
        self.occupied: Set[str] = set()
        self.boarding_policy = boarding_policy
        allowed_policies = {"random", "std", "pyramid"}
        if self.boarding_policy not in allowed_policies:
            raise ValueError(
                f"Unsupported boarding_policy '{self.boarding_policy}'. "
                f"Supported values: {sorted(allowed_policies)}"
            )
        rate = float(cross_zone_violation_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"cross_zone_violation_rate must be between 0 and 1. Received: {cross_zone_violation_rate}"
            )
        self.cross_zone_violation_rate = rate
        self.log_summary = bool(log_summary)
        self.shuffle_config = self._normalize_shuffle_config(shuffle_config)
        self.shuffle_low_ticks = seconds_to_ticks(self.shuffle_config.low_s, min_ticks=1)
        self.shuffle_high_ticks = seconds_to_ticks(self.shuffle_config.high_s, min_ticks=1)
        self.shuffle_mode_ticks = (
            None
            if self.shuffle_config.mode_s is None
            else seconds_to_ticks(self.shuffle_config.mode_s, min_ticks=1)
        )
        self.event_counters: Dict[str, int] = {
            "spawned": 0,
            "moves": 0,
            "waits": 0,
            "seated": 0,
            "row_conflict_events": 0,
            "stow_start": 0,
            "stow_complete": 0,
            "seat_shuffle_start": 0,
            "seat_shuffle_finish": 0,
        }

        df = self._load_manifest_dataframe(manifest_file)
        
        # Enforce boarding policies
        if self.boarding_policy == "random":
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        elif self.boarding_policy == "std":
            df = df.sort_values(by="zone_std", ascending=True)
            df = df.groupby(["zone_std"], group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
        elif self.boarding_policy == "pyramid":
            df = df.sort_values(by="zone_pyramid", ascending=True)
            df = df.groupby(["zone_pyramid"], group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)

        df = self._apply_cross_zone_violations(df)

        self.agents: List[PassengerAgent] = []
        missing_seats = 0

        for _, row in df.iterrows():
            seat_node = env.seat_node_at(int(row["x_coord"]), int(row["y_coord"]))
            if seat_node is None:
                missing_seats += 1
                continue

            has_lug = bool(row.get("has_luggage", False))
            raw_stow_seconds = float(row.get("stow_duration", 0)) if has_lug else 0
            stow_dur = seconds_to_ticks(raw_stow_seconds, min_ticks=1) if has_lug else 0

            door_val = row.get("preferred_door")
            if pd.isna(door_val) or str(door_val).strip().upper() not in ["F", "M"]:
                pax_class = str(row.get("class", "economy")).lower()
                door_str = "F" if pax_class == "business" else "M"
            else:
                door_str = str(door_val).strip().upper()

            agent = PassengerAgent(
                pax_id=str(row["pax_id"]),
                assigned_seat_node=seat_node,
                assigned_spawn=door_str,
                alternative_spawn=door_str,

                assigned_aisle=str(row["assigned_aisle"]),
                seat_x=int(row["x_coord"]),
                seat_y=int(row["y_coord"]),
                travel_class=str(row.get("class", "economy")),
                has_luggage=has_lug,
                stow_duration=stow_dur,
                shuffle_model=self.shuffle_config.model,
                shuffle_low_ticks=self.shuffle_low_ticks,
                shuffle_high_ticks=self.shuffle_high_ticks,
                shuffle_mode_ticks=self.shuffle_mode_ticks,
                zone_std=int(row.get("zone_std", 1)),
                zone_pyramid=int(row.get("zone_pyramid", 1)),
            )
            self.agents.append(agent)

        if missing_seats and self.log_summary:
            print(f"  Warning: {missing_seats} passengers skipped (seat node not found)")

        self.agents_by_id: Dict[str, PassengerAgent] = {
            a.pax_id: a for a in self.agents
        }

        # Per-door spawn queues (IC1: two doors, bi-directional passengers)
        self.spawn_queues: Dict[str, deque] = {}
        for agent in self.agents:
            door = agent.assigned_spawn
            if door not in self.spawn_queues:
                self.spawn_queues[door] = deque()
            self.spawn_queues[door].append(agent)
        # Do not shuffle here! The dataframe was already sorted by zone
        # and shuffled within zones during policy application.
        # This preserves the exact group release sequence.

        if self.log_summary:
            print(f"Simulation created with {len(self.agents)} passengers")

    @staticmethod
    def _load_manifest_dataframe(manifest_file: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(manifest_file, pd.DataFrame):
            return manifest_file.copy()
        return pd.read_excel(manifest_file)

    def _apply_cross_zone_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.boarding_policy == "random" or self.cross_zone_violation_rate <= 0 or len(df) < 2:
            return df.reset_index(drop=True)

        ordered = df.reset_index(drop=True).copy()

        # Violations are applied within each effective door queue (same-door preservation).
        # Selection and insertion can happen anywhere in the per-door sequence,
        # not just at zone boundaries.
        if "preferred_door" in ordered.columns:
            effective_door = ordered["preferred_door"].astype(str).str.upper()
        else:
            effective_door = pd.Series([""] * len(ordered), index=ordered.index)
        fallback_door = np.where(
            ordered.get("class", "economy").astype(str).str.lower().eq("business"),
            "F",
            "M",
        )
        effective_door = np.where(effective_door.isin(["F", "M"]), effective_door, fallback_door)
        ordered["_effective_door"] = effective_door

        for door in ("F", "M"):
            door_indices = ordered.index[ordered["_effective_door"] == door].to_numpy()
            if len(door_indices) < 2:
                continue

            n = len(door_indices)
            n_violators = int(round(self.cross_zone_violation_rate * n))
            n_violators = max(0, min(n_violators, n - 1))
            if n_violators <= 0:
                continue

            door_rows = ordered.loc[door_indices].reset_index(drop=True)

            pick_scores = pd.to_numeric(door_rows.get("violation_pick_score"), errors="coerce")
            if pick_scores.isna().any():
                pick_scores = pick_scores.fillna(pd.Series([self.rng.random() for _ in range(n)]))
            violator_positions = np.argsort(pick_scores.to_numpy())[:n_violators]

            insert_scores = pd.to_numeric(door_rows.get("violation_insert_score"), errors="coerce")
            if insert_scores.isna().any():
                insert_scores = insert_scores.fillna(pd.Series([self.rng.random() for _ in range(n)]))
            target_positions = np.sort(np.argsort(insert_scores.to_numpy())[:n_violators])

            violator_df = door_rows.iloc[violator_positions].copy().sort_values("violation_insert_score", kind="mergesort")
            remaining_df = door_rows.drop(index=door_rows.index[violator_positions]).reset_index(drop=True)

            target_set = set(int(pos) for pos in target_positions)
            rebuilt_rows: List[pd.Series] = []
            v_ptr = 0
            r_ptr = 0
            for pos in range(n):
                if pos in target_set and v_ptr < len(violator_df):
                    rebuilt_rows.append(violator_df.iloc[v_ptr])
                    v_ptr += 1
                elif r_ptr < len(remaining_df):
                    rebuilt_rows.append(remaining_df.iloc[r_ptr])
                    r_ptr += 1
                elif v_ptr < len(violator_df):
                    rebuilt_rows.append(violator_df.iloc[v_ptr])
                    v_ptr += 1

            rebuilt_df = pd.DataFrame(rebuilt_rows).reset_index(drop=True)
            ordered.loc[door_indices, :] = rebuilt_df.to_numpy()

        return ordered.drop(columns=["_effective_door"]).reset_index(drop=True)

    @staticmethod
    def _normalize_shuffle_config(config: Optional[ShuffleConfig]) -> ShuffleConfig:
        if config is None:
            low_s, high_s = validate_bounds(SHUFFLE_LOW_S, SHUFFLE_HIGH_S, "shuffle")
            mode_s = (
                validate_triangular_mode(low_s, SHUFFLE_MODE_S, high_s, "shuffle triangular")
                if SHUFFLE_MODEL == "triangular" and SHUFFLE_MODE_S is not None
                else None
            )
            return ShuffleConfig(model=SHUFFLE_MODEL, low_s=low_s, high_s=high_s, mode_s=mode_s)

        model = config.model.strip().lower()
        if model not in {"uniform", "triangular"}:
            raise ValueError(f"Unsupported shuffle model: {config.model}")
        low_s, high_s = validate_bounds(config.low_s, config.high_s, "shuffle")
        mode_s = None
        if model == "triangular":
            if config.mode_s is None:
                raise ValueError("Triangular shuffle model requires mode_s")
            mode_s = validate_triangular_mode(low_s, config.mode_s, high_s, "shuffle triangular")
        return ShuffleConfig(model=model, low_s=low_s, high_s=high_s, mode_s=mode_s)

    def _spawn_next(self) -> None:
        """Spawn up to SPAWN_RATE passengers per door per tick.

        All passengers spawn at the door node (L-aisle).  A flight
        attendant directs them: R-assigned passengers are routed
        through the galley to the R-aisle by apCloserIfMoveTo.

        Gate: only spawn when the door node is free AND the first
        aisle step toward the passenger's ap is clear."""
        for door_label, queue in self.spawn_queues.items():
            door_node = self.env.doors.get(door_label)
            if door_node is None:
                continue
            spawned_this_tick = 0
            while queue and spawned_this_tick < SPAWN_RATE:
                if door_node in self.occupied:
                    break
                # Peek at the next passenger
                agent = queue[0]
                
                # Check if first step toward target is free (downstream gate)
                # We simulate their position at the door to see if their formal A5 advance 
                # mechanism returns a valid, unblocked next node.
                agent.position = door_node
                nh = agent._best_aisle_advance(self.env, self.occupied)
                agent.position = None

                if nh is None:
                    break  # downstream path blocked, wait in queue
                # Spawn
                queue.popleft()
                agent.position = door_node
                agent.spawned = True
                self.occupied.add(door_node)
                spawned_this_tick += 1
                self.event_counters["spawned"] += 1

    def _record_action_metrics(self, action: str, agent: PassengerAgent) -> None:
        if action.startswith("moveTo:"):
            self.event_counters["moves"] += 1
        elif action in {"wait", "shufflingSeat"}:
            self.event_counters["waits"] += 1
            agent.wait_count += 1
        elif action == "sit":
            self.event_counters["seated"] += 1
        elif action == "startStow":
            self.event_counters["stow_start"] += 1
        elif action == "stowComplete":
            self.event_counters["stow_complete"] += 1
        elif action == "startShuffle":
            self.event_counters["seat_shuffle_start"] += 1
        elif action == "finishShuffle":
            self.event_counters["seat_shuffle_finish"] += 1
    def step(self) -> bool:
        """One tick: Observe/Intent → Action/Commit."""
        self.tick += 1
        self._spawn_next()

        # Build lookups
        agent_at: Dict[str, PassengerAgent] = {}
        all_agent_at: Dict[str, PassengerAgent] = {}
        for a in self.agents:
            if a.spawned and a.position is not None:
                all_agent_at[a.position] = a
                if not a.seated:
                    agent_at[a.position] = a
        occupied = set(agent_at.keys())

        # Phase A: evaluate intentions
        active = [a for a in self.agents if a.spawned and not a.seated]
        self.rng.shuffle(active)
        for agent in active:
            agent.evaluate_intent(
                self.env, occupied, self.agents_by_id, agent_at, all_agent_at
            )
            if agent.intent == "resolveSeatBlock":
                self.event_counters["row_conflict_events"] += 1

        # Phase B: execute actions
        next_positions: Dict[str, str] = {}
        for agent in active:
            agent.boarding_time += 1
            action = agent.execute_action(
                self.env, occupied, agent_at,
                self.agents_by_id, next_positions, self.rng,
            )
            agent.last_action = action
            self._record_action_metrics(action, agent)

        # Sync occupied set
        self.occupied = set(next_positions.values())

        # C1: completion
        return all(a.seated for a in self.agents)

    def _compute_metrics(self, total_ticks: int) -> Dict[str, float]:
        seated_count = sum(1 for a in self.agents if a.seated)
        avg_boarding = (
            sum(a.boarding_time for a in self.agents if a.seated)
            / max(seated_count, 1)
        )
        avg_wait = (
            sum(a.wait_count for a in self.agents if a.seated)
            / max(seated_count, 1)
        )
        pax_lug = sum(1 for a in self.agents if a.has_luggage)
        avg_stow = (
            sum(a.stow_duration for a in self.agents if a.has_luggage)
            / max(pax_lug, 1)
        )
        completion_success = seated_count == len(self.agents)

        return {
            "total_ticks": float(total_ticks),
            "total_seconds": ticks_to_seconds(total_ticks),
            "seated_count": float(seated_count),
            "total_passengers": float(len(self.agents)),
            "completion_success": float(1 if completion_success else 0),
            "avg_boarding_ticks": float(avg_boarding),
            "avg_boarding_seconds": ticks_to_seconds(avg_boarding),
            "avg_wait_ticks": float(avg_wait),
            "avg_wait_seconds": ticks_to_seconds(avg_wait),
            "luggage_passengers": float(pax_lug),
            "avg_stow_ticks": float(avg_stow),
            "avg_stow_seconds": ticks_to_seconds(avg_stow),
            "row_conflict_count": float(self.event_counters["row_conflict_events"]),
            "seat_shuffle_starts": float(self.event_counters["seat_shuffle_start"]),
            "seat_shuffle_finishes": float(self.event_counters["seat_shuffle_finish"]),
        }

    def run(self, verbose: bool = True, enforce_completion: bool = True) -> int:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BOARDING SIMULATION START  (Strategy: {self.boarding_policy.upper()})")
            print(f"{'='*60}\n")

        while self.tick < MAX_TICKS:
            done = self.step()
            if verbose and (self.tick % REPORT_EVERY == 0 or done):
                seated = sum(1 for a in self.agents if a.seated)
                spawned = sum(1 for a in self.agents if a.spawned)
                print(
                    f"  t={self.tick:>5d}  |  "
                    f"spawned={spawned:>3d}/{len(self.agents)}  |  "
                    f"seated={seated:>3d}/{len(self.agents)}  |  "
                    f"queue={sum(len(q) for q in self.spawn_queues.values())}"
                )
            if done:
                break

        total = self.tick
        metrics = self._compute_metrics(total)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  RESULTS")
            print(f"{'='*60}")
            total_sec = metrics["total_seconds"]
            print(
                f"  Total boarding time : {total} ticks "
                f"({total_sec:.1f} s / {total_sec/60:.1f} m)"
            )
            print(
                f"  Passengers seated   : {int(metrics['seated_count'])} / "
                f"{int(metrics['total_passengers'])}"
            )
            print(
                f"  Passengers w/ lug.  : {int(metrics['luggage_passengers'])} / "
                f"{int(metrics['total_passengers'])}"
            )
            print(
                f"  Avg boarding time   : {metrics['avg_boarding_ticks']:.1f} ticks "
                f"({metrics['avg_boarding_seconds']:.1f} s) per pax"
            )
            print(
                f"  Avg wait time       : {metrics['avg_wait_ticks']:.1f} ticks "
                f"({metrics['avg_wait_seconds']:.1f} s) per pax"
            )
            print(
                f"  Avg stow duration   : {metrics['avg_stow_ticks']:.1f} ticks "
                f"({metrics['avg_stow_seconds']:.1f} s) (luggage pax)"
            )
            print(f"{'='*60}\n")

        if enforce_completion:
            seated_count = int(metrics["seated_count"])
            assert seated_count == len(self.agents), (
                f"Not all passengers seated! {seated_count}/{len(self.agents)}"
            )
            for a in self.agents:
                assert a.position == a.assigned_seat_node, (
                    f"Pax {a.pax_id} not at seat! "
                    f"pos={a.position}, seat={a.assigned_seat_node}"
                )
            if verbose:
                print("  [OK] All sanity checks passed.\n")
        return total

    def run_with_metrics(
        self,
        verbose: bool = False,
        enforce_completion: bool = False,
    ) -> Dict[str, float]:
        total = self.run(verbose=verbose, enforce_completion=enforce_completion)
        return self._compute_metrics(total)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sign(x: int) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    env = CabinEnvironment(NODES_FILE, EDGES_FILE)
    
    if VISUALIZE_ONE:
        print("\n[Mode] VISUALIZE_ONE active. Running single manifest...")
        print("\nRunning Strategy 1: Back-to-front zonal")
        sim1 = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="std")
        total1 = sim1.run()
        
        print("\nRunning Strategy 2: Modified reverse pyramid")
        sim2 = BoardingSimulation(env, MANIFEST_FILE, seed=SEED, boarding_policy="pyramid")
        total2 = sim2.run()
        
        total1_sec = ticks_to_seconds(total1)
        total2_sec = ticks_to_seconds(total2)
        
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON")
        print(f"{'='*60}")
        print(f"  Back-to-front (std)         : {total1} ticks ({total1_sec:.1f} s / {total1_sec/60:.1f} m)")
        print(f"  Reverse Pyramid (pyramid)   : {total2} ticks ({total2_sec:.1f} s / {total2_sec/60:.1f} m)")
        print(f"{'='*60}\n")
    else:
        import csv
        print("\n[Mode] BATCH active. Finding all manifests...")
        results = []
        i = 1
        while True:
            target_manifest = DATA_DIR / f"generated_manifest_{i}.xlsx"
            if not target_manifest.exists():
                target_manifest = PROJECT_ROOT / f"generated_manifest_{i}.xlsx"
            if not target_manifest.exists():
                break
            
            print(f"\n--- Batch: Evaluating Manifest {i} ---")
            sim1 = BoardingSimulation(env, target_manifest, seed=SEED, boarding_policy="std")
            total1 = sim1.run()
            
            sim2 = BoardingSimulation(env, target_manifest, seed=SEED, boarding_policy="pyramid")
            total2 = sim2.run()
            
            results.append({
                "manifest_id": i,
                "std_ticks": total1,
                "std_seconds": ticks_to_seconds(total1),
                "pyramid_ticks": total2,
                "pyramid_seconds": ticks_to_seconds(total2)
            })
            i += 1
            
        csv_out = DATA_DIR / "simulation_batch_results.csv"
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["manifest_id", "std_ticks", "std_seconds", "pyramid_ticks", "pyramid_seconds"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\nBatch Complete! Executed {len(results)} distinct scenarios. Data exported linearly to {csv_out}")


if __name__ == "__main__":
    main()
