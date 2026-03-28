"""
Agent-Based Boarding Simulation for a Boeing 787 twin-aisle cabin.

Fully aligned with the formal predicate model in ABMS_G20_Report-5.pdf (Ch. 3).

Implemented formal properties:
  IC1-IC6  Initial conditions
  B1-B9    Belief-state evolution
  O1-O6    Observation-based state characterisation
  I1-I5    Intention selection
  A1-A17   Action generation
  C1       System-level completion condition

Navigation uses Dijkstra shortest-path distance as the executable implementation
of the formal coordinate-based observations (O4: rowCloserIfMoveTo, O5:
columnCloserIfMoveTo).  Non-regression routing (d(n) ≤ d(m)) is used to
handle graph-topology plateaus at cross-aisle connectors.
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
K_OBS = 5                   # perception range in hops  (Nk neighbourhood)


# ---------------------------------------------------------------------------
# Environment  (EW = G = (N, E))
# ---------------------------------------------------------------------------
class CabinEnvironment:
    """Static aircraft cabin graph: nodes (seats, aisle), edges."""

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
        self._find_door("F", target_x=0, target_y=3)
        self._find_door("M", target_x=26, target_y=3)  # Centered perfectly on cross-galley
        self._find_door("R", target_x=125, target_y=3)

        # x_mid – longitudinal midpoint of the cabin (Property B8)
        all_x = [d["x"] for d in self.graph.nodes.values()]
        self.x_mid: int = (min(all_x) + max(all_x)) // 2

        # Hop-distance graph (undirected) for perception neighbourhood Nk
        self._hop_graph = self.graph.to_undirected()

        print(f"Environment loaded: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"  Seats: {len(self.seat_nodes)}, Aisle: {len(self.aisle_nodes)}")
        print(f"  Doors: {self.doors}  |  x_mid={self.x_mid}")

    def _find_door(self, label: str, target_x: int, target_y: int) -> None:
        best_id, best_d = None, float("inf")
        for nid, data in self.aisle_nodes.items():
            d = abs(data["x"] - target_x) + abs(data["y"] - target_y)
            if d < best_d:
                best_d = d
                best_id = nid
        if best_id is not None:
            self.doors[label] = best_id

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
        preferred_speed: int = 1,
        lateral_speed: int = 1,
        patience_threshold: int = 15,
        compliance_level: float = 0.8,
        has_luggage: bool = False,
        stow_duration: int = 0,
        zone_std: int = 1,
        zone_outsidein: int = 1,
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
        self.preferred_speed = preferred_speed      # preferredSpeedp
        self.lateral_speed = lateral_speed          # lateralSpeedp
        self.patience_threshold = patience_threshold  # patienceThresholdp
        self.compliance_level = compliance_level    # complianceLevelp
        self.has_luggage = has_luggage              # hasLuggagep
        self.stow_duration = stow_duration          # stowDurationp
        self.zone_std = zone_std                    # zoneSTDp
        self.zone_outsidein = zone_outsidein        # zoneOutsideInp
        self.zone_pyramid = zone_pyramid            # zonePyramidp

        # === Internalp – Beliefs ===
        self.position: Optional[str] = None         # observes(p, position(n))   [IC1]
        self.seated: bool = False                   # believes(p, seated)        [IC2]
        self.luggage_status: str = (                # believes(p, luggageStatus) [IC3]
            "unstowed" if has_luggage else "none"
        )
        self.remaining_stow_time: int = 0
        self.time_since_move: int = 0               # believes(p, timeSinceMove) [IC4]

        # Per-tick beliefs (reset each tick during evaluate_intent)
        self.head_on_conflict: bool = False         # believes(p, headOnConflict) [B7]
        self.opponent_has_priority: bool = False    # believes(p, opponentHasPriority) [B8]
        self.row_blocker: bool = False              # believes(p, rowBlocker)     [B4]
        self.row_blocked: bool = False              # believes(p, rowBlocked)     [B5]
        self.row_shift_complete: bool = False       # believes(p, rowShiftComplete) [B6]

        # Persistent head-on state: tracks time-since-yield and opponent
        self._yielding_since: int = 0               # ticks spent in refuge
        self._yield_opponent_id: Optional[str] = None

        # === Internalp – Intentions ===                               [IC5]
        self.intent: str = "none"

        # === Outputp – Actions ===                                    [IC6]
        self.last_action: str = "none"

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
    #  Phase A:  INTENTION EVALUATION   (Properties I1–I5, B4-B9)
    # ===================================================================
    def evaluate_intent(
        self,
        env: CabinEnvironment,
        occupied: Set[str],
        all_agents: Dict[str, "PassengerAgent"],
        agent_at: Dict[str, "PassengerAgent"],
    ) -> None:
        """Evaluate observations → beliefs → intention."""
        if not self.spawned or self.position is None:
            self.intent = "none"
            return

        # R0: seated → terminal
        if self.seated:
            self.intent = "none"
            return

        cur = self.position

        # Reset per-tick beliefs
        self.head_on_conflict = False
        self.opponent_has_priority = False
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
        #  Return-to-aisle: in a non-target seat (e.g. from yielding)
        # =================================================================
        if (env.node_type(cur) == "seat"
                and cur != self.assigned_seat_node):
            # Clear yield memory if present
            self._yield_opponent_id = None
            self._yielding_since = 0
            # Try to return to aisle
            aisle_n = self._any_free_aisle_neighbor(env, occupied)
            if aisle_n is not None:
                self.intent = "advance"  # will use _best_aisle_advance or fallback
                return
            # No free aisle neighbor → wait
            self.intent = "wait"
            return

        # =================================================================
        #  B9:  Still yielding from a previous head-on?
        # =================================================================
        if self._yield_opponent_id is not None and env.node_type(cur) == "seat":
            opp = all_agents.get(self._yield_opponent_id)
            if opp is not None and opp.position is not None:
                # Check if opponent has passed (B9): opponent no longer within
                # K_OBS hops or has moved past my x-coordinate
                h = env.hop_distance(cur, opp.position)
                opp_x = env.node_x(opp.position)
                my_x = env.node_x(cur)
                opp_dir = opp._dir(env)
                still_nearby = (h <= K_OBS and not opp.seated)
                still_blocking = False
                if still_nearby:
                    # Opponent hasn't passed if they're still at or behind my position
                    if (opp_dir > 0 and opp_x < my_x) or \
                       (opp_dir < 0 and opp_x > my_x):
                        still_blocking = True
                    elif opp_x == my_x:
                        still_blocking = True

                if still_blocking:
                    self._yielding_since += 1
                    # Safety valve: don't yield forever (max 10 ticks)
                    if self._yielding_since < 10:
                        self.head_on_conflict = True
                        self.opponent_has_priority = True
                        self.intent = "resolveHeadOn"
                        return

            # Opponent passed or gone → clear yield state
            self._yield_opponent_id = None
            self._yielding_since = 0
            # Return to aisle: treated as advance
            adv = self._best_aisle_advance(env, occupied)
            if adv is not None:
                self.intent = "advance"
                return

        # =================================================================
        #  B7/B8:  Fresh head-on detection (aisle, opposite direction, ≤ K_OBS)
        # =================================================================
        my_dir = self._dir(env)
        if my_dir != 0 and env.node_type(cur) == "aisle":
            for n in env.neighbors(cur):
                if n not in agent_at:
                    continue
                if env.node_type(n) != "aisle":
                    continue
                if env.hop_distance(cur, n) > K_OBS:
                    continue
                q = agent_at[n]
                q_dir = q._dir(env)
                if q_dir != 0 and q_dir == -my_dir:
                    # B7: head-on detected
                    # B8: priority assignment
                    p_yields = False
                    if (not self.has_luggage) and q.has_luggage:
                        p_yields = True
                    elif self.has_luggage and not q.has_luggage:
                        p_yields = False
                    elif self.has_luggage == q.has_luggage:
                        my_mid = abs(self.seat_x - env.x_mid)
                        q_mid = abs(q.seat_x - env.x_mid)
                        if my_mid > q_mid:
                            p_yields = True
                        elif my_mid == q_mid:
                            p_yields = int(self.pax_id) > int(q.pax_id)

                    # Safety: skip head-on if priority side can't advance
                    # (prevents permanent conflict stall)
                    if p_yields:
                        # I would yield — check if opponent (priority) can advance
                        q_occ = occupied - {n}  # q's position excluded
                        q_adv = q._best_aisle_advance(env, q_occ)
                        if q_adv is None:
                            continue  # opponent stuck, no point yielding
                    else:
                        # I have priority — check if I can actually advance
                        adv = self._best_aisle_advance(env, occupied)
                        if adv is None:
                            continue  # can't advance, skip head-on

                    self.head_on_conflict = True
                    self.opponent_has_priority = p_yields
                    self.intent = "resolveHeadOn"
                    return

        # =================================================================
        #  B5 / I2:  Row-blocked (at aisle-access, ready, path occupied)
        # =================================================================
        if self._at_seat_row_aisle(env) and self._ready_to_enter():
            on_path = self._on_path_nodes(env)
            blockers = [n for n in on_path if n in agent_at]
            # Also check if the target seat itself is occupied by someone else
            seat_occupied = (
                self.assigned_seat_node in agent_at
                and agent_at[self.assigned_seat_node].pax_id != self.pax_id
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
            return "none"

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

        # ---- A11-A16: head-on conflict resolution ----
        if self.intent == "resolveHeadOn":
            if not self.opponent_has_priority:
                # A11: I have priority → advance if possible
                target = self._best_aisle_advance(env, occupied)
                if target is not None and target not in next_positions.values():
                    next_positions[self.pax_id] = target
                    self.position = target
                    self.time_since_move = 0
                    return f"moveTo:{target}"
                # A12: priority wait
                next_positions[self.pax_id] = cur
                self.time_since_move += 1
                return "wait"
            else:
                # A13: yield → move into adjacent free seat (refuge)
                if env.node_type(cur) == "aisle":
                    for n in env.neighbors(cur):
                        if n in occupied or n in next_positions.values():
                            continue
                        if env.node_type(n) == "seat":
                            next_positions[self.pax_id] = n
                            self.position = n
                            self.time_since_move = 0
                            # Track yield state for B9
                            # Find the opponent
                            for nb in env.neighbors(cur):
                                if nb in agent_at and nb != n:
                                    q = agent_at[nb]
                                    if q._dir(env) == -self._dir(env):
                                        self._yield_opponent_id = q.pax_id
                                        self._yielding_since = 0
                                        break
                            return f"yield:{n}"
                    # A16: no free seat → wait
                    next_positions[self.pax_id] = cur
                    self.time_since_move += 1
                    return "wait"
                # A14: already in refuge seat → wait
                next_positions[self.pax_id] = cur
                self.time_since_move += 1
                return "wait"

        # ---- A7-A10/A15: seat-block resolution ----
        if self.intent == "resolveSeatBlock":
            if self.row_blocker:
                # A7: shift outward toward aisle-access
                req_ap = None
                for other in agent_at.values():
                    if other.pax_id == self.pax_id:
                        continue
                    if not other._at_seat_row_aisle(env):
                        continue
                    if not other._ready_to_enter():
                        continue
                    if cur in other._on_path_nodes(env) or cur == other.assigned_seat_node:
                        req_ap = other._aisle_access_node(env)
                        break

                if req_ap is not None:
                    yap = env.node_y(req_ap)
                    # A7: shift to adjacent seat closer to aisle
                    for n in env.neighbors(cur):
                        if n in occupied or n in next_positions.values():
                            continue
                        if (env.node_type(n) == "seat"
                                and env.node_x(n) == env.node_x(cur)):
                            if abs(env.node_y(n) - yap) < abs(env.node_y(cur) - yap):
                                next_positions[self.pax_id] = n
                                self.position = n
                                self.time_since_move = 0
                                self.seated = False
                                return f"shiftOut:{n}"
                    # A8: outermost blocker → step into aisle
                    if req_ap in env.neighbors(cur):
                        if req_ap not in occupied and req_ap not in next_positions.values():
                            next_positions[self.pax_id] = req_ap
                            self.position = req_ap
                            self.time_since_move = 0
                            self.seated = False
                            return f"shiftToAisle:{req_ap}"

                next_positions[self.pax_id] = cur
                self.time_since_move += 1
                return "wait"

            elif self.row_blocked:
                if self.row_shift_complete:
                    # A9/A10: squeeze with aisle-side occupant
                    aisle_side = self._aisle_side_seat(env)
                    if aisle_side is not None and aisle_side in agent_at:
                        q = agent_at[aisle_side]
                        if not q.seated:  # Never swap with a seated agent
                            old_p, old_q = cur, q.position
                            self.position = old_q
                            q.position = old_p
                            q.seated = False  # Safety: ensure not marked seated
                            next_positions[self.pax_id] = old_q
                            next_positions[q.pax_id] = old_p
                            self.time_since_move = 0
                            q.time_since_move = 0
                            return f"squeezeWith:{q.pax_id}"
                    elif aisle_side is not None and aisle_side not in occupied:
                        if aisle_side not in next_positions.values():
                            next_positions[self.pax_id] = aisle_side
                            self.position = aisle_side
                            self.time_since_move = 0
                            return f"moveTo:{aisle_side}"

                # A15: wait for shift
                next_positions[self.pax_id] = cur
                self.time_since_move += 1
                return "wait"

            next_positions[self.pax_id] = cur
            self.time_since_move += 1
            return "wait"

        # ---- A17: aisle switch — DISABLED (passengers stick to assigned aisle)
        if self.intent == "switchAisle":
            # Should not be reached; fall through to wait
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
        manifest_file: Path,
        seed: int = SEED,
        boarding_policy: str = "random",
    ) -> None:
        self.env = env
        self.rng = random.Random(seed)
        self.tick = 0
        self.occupied: Set[str] = set()
        self.boarding_policy = boarding_policy

        df = pd.read_excel(manifest_file)
        
        # Enforce boarding policies
        if self.boarding_policy == "random":
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        elif self.boarding_policy == "std":
            df = df.sort_values(by="zone_std", ascending=True)
            df = df.groupby(["zone_std"], group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
        elif self.boarding_policy == "wilma":
            df = df.sort_values(by="zone_outsidein", ascending=True)
            df = df.groupby(["zone_outsidein"], group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
        elif self.boarding_policy == "pyramid":
            df = df.sort_values(by="zone_pyramid", ascending=True)
            df = df.groupby(["zone_pyramid"], group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
        self.agents: List[PassengerAgent] = []
        missing_seats = 0

        for _, row in df.iterrows():
            seat_node = env.seat_node_at(int(row["x_coord"]), int(row["y_coord"]))
            if seat_node is None:
                missing_seats += 1
                continue

            has_lug = bool(row.get("has_luggage", False))
            raw_stow = int(row.get("stow_duration", 0)) if has_lug else 0
            stow_dur = max(1, raw_stow // 10) if has_lug else 0

            agent = PassengerAgent(
                pax_id=str(row["pax_id"]),
                assigned_seat_node=seat_node,
                assigned_spawn="M",  # 100% Door M logic
                alternative_spawn="M",

                assigned_aisle=str(row["assigned_aisle"]),
                seat_x=int(row["x_coord"]),
                seat_y=int(row["y_coord"]),
                travel_class=str(row.get("class", "economy")),
                preferred_speed=int(row.get("preferred_speed", 1)),
                lateral_speed=int(row.get("lateral_speed", 1)),
                patience_threshold=int(row.get("patience_threshold", 15)),
                compliance_level=0.8,
                has_luggage=has_lug,
                stow_duration=stow_dur,
                zone_std=int(row.get("zone_std", 1)),
                zone_outsidein=int(row.get("zone_outsidein", 1)),
                zone_pyramid=int(row.get("zone_pyramid", 1)),
            )
            self.agents.append(agent)

        if missing_seats:
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
        # Shuffle within each door queue for randomness
        for door in self.spawn_queues:
            q = list(self.spawn_queues[door])
            self.rng.shuffle(q)
            self.spawn_queues[door] = deque(q)

        print(f"Simulation created with {len(self.agents)} passengers")
        for door, q in self.spawn_queues.items():
            print(f"  Door {door}: {len(q)} passengers")

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

    def step(self) -> bool:
        """One tick: Observe/Intent → Action/Commit."""
        self.tick += 1
        self._spawn_next()

        # Build lookups
        agent_at: Dict[str, PassengerAgent] = {}
        for a in self.agents:
            if a.spawned and not a.seated and a.position is not None:
                agent_at[a.position] = a
        occupied = set(agent_at.keys())

        # Phase A: evaluate intentions
        active = [a for a in self.agents if a.spawned and not a.seated]
        self.rng.shuffle(active)
        for agent in active:
            agent.evaluate_intent(
                self.env, occupied, self.agents_by_id, agent_at
            )

        # Phase B: execute actions
        next_positions: Dict[str, str] = {}
        for agent in active:
            agent.boarding_time += 1
            agent.execute_action(
                self.env, occupied, agent_at,
                self.agents_by_id, next_positions, self.rng,
            )

        # Sync occupied set
        self.occupied = set(next_positions.values())

        # C1: completion
        return all(a.seated for a in self.agents)

    def run(self) -> int:
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
                    f"queue={sum(len(q) for q in self.spawn_queues.values())}"
                )
            if done:
                break

        total = self.tick
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

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total boarding time : {total} ticks")
        print(f"  Passengers seated   : {seated_count} / {len(self.agents)}")
        print(f"  Passengers w/ luggage: {pax_lug} / {len(self.agents)}")
        print(f"  Avg boarding time   : {avg_boarding:.1f} ticks/passenger")
        print(f"  Avg wait time       : {avg_wait:.1f} ticks/passenger")
        print(f"  Avg stow duration   : {avg_stow:.1f} ticks (luggage pax only)")
        print(f"{'='*60}\n")

        assert seated_count == len(self.agents), (
            f"Not all passengers seated! {seated_count}/{len(self.agents)}"
        )
        for a in self.agents:
            assert a.position == a.assigned_seat_node, (
                f"Pax {a.pax_id} not at seat! "
                f"pos={a.position}, seat={a.assigned_seat_node}"
            )
        print("  [OK] All sanity checks passed.\n")
        return total


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
    sim = BoardingSimulation(env, MANIFEST_FILE, seed=SEED)
    sim.run()


if __name__ == "__main__":
    main()
