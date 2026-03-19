# Simulation Phases — ABM Boarding Simulation

> **Date:** 2026-03-19  
> **Authors:** Group 20  
> **Status:** ✅ Working baseline

---

## Overview

This phase delivers a **minimal but working** agent-based simulation of the Boeing 787 twin-aisle passenger boarding process, together with a **real-time pygame visualiser**. It translates the formal model from the ABMS G20 report into runnable Python code.

### What's included

| File | Purpose |
|---|---|
| `simulation.py` | Core ABM simulation engine |
| `visualiser.py` | Pygame real-time animation |
| `Graph_and_manifest/nodes_787.xlsx` | Cabin graph nodes (pre-existing) |
| `Graph_and_manifest/edges_787.xlsx` | Cabin graph edges (pre-existing) |
| `Graph_and_manifest/generated_manifest.xlsx` | Passenger manifest (pre-existing) |
| `Graph_and_manifest/generate_passenger_manifest_run.py` | Manifest generator (pre-existing) |
| `Graph_and_manifest/visualize_787_layout.py` | Static graph plot (pre-existing) |

### What's simplified in Phase 1

This is intentionally minimal to validate the data pipeline and core loop:

- ❌ No luggage stowage (added in Phase 2)
- ❌ No head-on conflict resolution
- ❌ No row-blocking / squeeze maneuvers
- ❌ No aisle switching
- ❌ Single boarding strategy only (random through middle door)
- ✅ Passengers spawn → advance along shortest path → sit down

---

## Data: The Cabin Graph

The aircraft cabin is represented as a **directed graph** loaded from Excel files:

- **527 nodes** — 291 seat nodes + 236 aisle nodes
- **1500 directed edges** — feasible one-step movements
- **Coordinate system**: `x ∈ [0, 125]` (longitudinal), `y ∈ [0, 10]` (lateral)

Three doors are identified by their approximate coordinates:

| Door | Label | Location | Node ID |
|---|---|---|---|
| Front | `F` | x≈0, y=3 | `aisle_0_3` |
| Middle | `M` | x≈22, y=3 | `aisle_22_3` |
| Rear | `R` | x≈125, y=3 | `aisle_125_3` |

The **manifest** contains 248 passengers (85% load factor) with seat assignments, class, spawn preference, aisle assignment, and boarding zones.

---

## `simulation.py` — Architecture

The simulation consists of three classes:

### 1. `CabinEnvironment`

Loads the graph and provides spatial lookups.

```
CabinEnvironment(nodes_file, edges_file)
├── self.graph          # NetworkX DiGraph
├── self.seat_nodes     # dict: node_id → {x, y, type}
├── self.aisle_nodes    # dict: node_id → {x, y, type}
├── self.doors          # dict: "F"/"M"/"R" → node_id
├── seat_node_at(x, y)  # find seat by coordinates
└── neighbors(node_id)  # graph successors
```

### 2. `PassengerAgent`

Each passenger is an autonomous agent with a **simplified BDI loop**:

```
PassengerAgent(pax_id, assigned_seat_node, assigned_spawn, assigned_aisle)

Static attributes:
├── pax_id, assigned_seat_node, assigned_spawn, assigned_aisle

Internal state (beliefs):
├── position        # current node (None = not yet spawned)
├── seated          # bool
├── spawned         # bool
├── boarding_time   # ticks since spawn
└── wait_count      # total ticks spent waiting

Key methods:
├── compute_path(env)       # shortest path via NetworkX
├── next_desired_node()     # next node on path
└── step(env, occupied)     # perceive → decide → act
```

**Decision logic in `step()`:**

```
if at_seat_node        → action = "sit"
elif next_node is free → action = "move_to(next)"
else                   → action = "wait" (recompute path next tick)
```

### 3. `BoardingSimulation`

Orchestrates the tick loop:

```
BoardingSimulation(env, manifest_file, seed)
├── self.agents         # list of PassengerAgent
├── self.spawn_queue    # deque (randomized order)
├── self.occupied       # set of occupied node IDs
│
├── step() → bool       # one tick; returns True when all seated
└── run()  → int        # full run; returns total ticks
```

**Each tick:**

1. **Spawn**: pop next passenger from queue onto door node (if door is free)
2. **Shuffle**: randomize agent update order (fairness)
3. **Step**: each active (spawned, not seated) agent executes one action
4. **Check**: if all agents seated → boarding complete

### Occupancy tracking

- `occupied: Set[str]` tracks which nodes currently have a passenger standing on them
- A passenger moving from node A to B: removes A from `occupied`, adds B
- When a passenger sits down, their node is removed from `occupied` (seats don't block aisles)
- Spawn is blocked if the door node is currently occupied

---

## `visualiser.py` — Architecture

The visualiser imports the simulation module and drives it tick-by-tick with a pygame render loop.

### `BoardingVisualiser`

```
BoardingVisualiser()
├── env              # CabinEnvironment (shared with simulation)
├── sim              # BoardingSimulation instance
├── node_pos         # dict: node_id → (screen_x, screen_y)
├── pax_class        # dict: pax_id → travel class (for coloring)
│
├── run()            # main pygame loop
├── _draw_edges()    # gray lines between connected nodes
├── _draw_nodes()    # colored circles for each node type
├── _draw_passengers()  # animated passenger dots
└── _draw_hud()      # stats overlay, progress bar, legend
```

### Rendering details

| Element | Visual |
|---|---|
| Aisle nodes | Small dark gray circles |
| Seat nodes | Small dark blue circles |
| Door nodes | Red outlined circles with label |
| Moving passengers | Colored circle with white border + glow |
| Seated passengers | Smaller, dimmer circle |
| Business class | 🟡 Gold |
| Premium Economy | 🟢 Teal |
| Economy class | 🔵 Blue |

### Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / Resume |
| `↑` | Speed up (fewer ms between ticks) |
| `↓` | Slow down (more ms between ticks) |
| `R` | Restart simulation from scratch |
| `Q` / `ESC` | Quit |

---

## Running

### Prerequisites

```bash
# Create and activate virtual environment (only once)
python3 -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl networkx matplotlib pygame-ce
```

### Run the simulation (console only)

```bash
source .venv/bin/activate
python simulation.py
```

Expected output:
```
Total boarding time : 491 ticks
Passengers seated   : 248 / 248
Avg boarding time   : 37.4 ticks/passenger
Avg wait time       : 1.3 ticks/passenger
✅ All sanity checks passed.
```

### Run the visualiser

```bash
source .venv/bin/activate
python visualiser.py
```

---

## Key Design Decisions

1. **NetworkX shortest path** for movement: each agent computes `nx.shortest_path()` to its seat. If the next node is blocked, the path is recomputed next tick. This is simple but correct.

2. **Random update order**: agents are shuffled each tick to avoid systematic bias from iteration order.

3. **Single occupancy**: each node can hold at most one passenger. This naturally creates queues in the aisles without explicit queue logic.

4. **Seats don't block**: when a passenger sits, their node is removed from `occupied`. This lets other passengers pass by on adjacent aisle nodes.

---

## Phase 2 — Planned Extensions

The following features correspond to the full report specification and will be added incrementally:

- [x] Luggage stowage (see Phase 2 below)
- [ ] Head-on conflict resolution (priority rules, yielding into seats)
- [ ] Row-blocking / squeeze maneuvers
- [ ] Patience-based aisle switching
- [ ] All 6 boarding strategies from the report
- [ ] Multiple replications for statistical comparison
- [ ] Non-compliance modeling

---

# Phase 2 — Luggage Stowage & Visual Improvements

> **Date:** 2026-03-19  
> **Status:** ✅ Working

## Overview

Phase 2 adds **luggage stowage** to the agent behaviour and improves the pygame visualiser to clearly show **occupied seats** and **stowing passengers**.

## Changes to `simulation.py`

### New agent attributes (from manifest)

| Attribute | Type | Description |
|---|---|---|
| `has_luggage` | `bool` | Whether the passenger carries luggage |
| `stow_duration` | `int` | Ticks needed to stow (manifest ÷ 10) |
| `seat_x` | `int` | X-coordinate of assigned seat row |

### New agent states

```
luggage_status: "none" → (no luggage, skip stowing)
                "unstowed" → "stowing" → "stowed" → (proceed to seat)
```

### Stowage logic in `step()`

The agent's decision priorities (in order):

```
1. At seat node?           → sit
2. Currently stowing?      → continue_stow (decrement timer)
3. At row aisle + unstowed? → start_stow (begin blocking aisle)
4. Otherwise               → advance toward seat (or wait if blocked)
```

**Key behaviour:** When a passenger stows luggage, they remain on the aisle node and it stays in the `occupied` set. Other passengers:
- **Cannot walk through** the stowing passenger (same node blocked)
- **Can reroute** through the other aisle (twin-aisle layout) via shortest-path recomputation
- **Wait** if no alternative path exists

### `_at_seat_row_aisle()` helper

Detects when a passenger has reached an aisle node at the same x-coordinate as their assigned seat — the trigger for stowing.

## Changes to `visualiser.py`

### Passenger rendering

| State | Visual |
|---|---|
| Moving | Class-colored circle + white border + glow effect |
| Stowing | Class-colored circle + **orange ring** + small orange luggage square |
| Seated | Class-colored filled circle + **bright inner dot** on seat node |

### HUD additions

- **"Stowing" counter** in the stats row
- **Stowing legend entry** with orange ring indicator

## Results comparison

| Metric | Phase 1 (no luggage) | Phase 2 (with luggage) |
|---|---|---|
| Total boarding time | 491 ticks | **510 ticks** (+4%) |
| Passengers w/ luggage | 0 | 142 / 248 (57%) |
| Avg boarding time | 37.4 ticks/pax | 38.0 ticks/pax |
| Avg wait time | 1.3 ticks/pax | 1.5 ticks/pax |
| Avg stow duration | — | 5.7 ticks |

The modest increase (+4%) is because the twin-aisle layout allows blocked passengers to reroute through the other aisle rather than waiting.

---

## Future Phases

- [ ] Head-on conflict resolution (priority rules, yielding into seats)
- [ ] Row-blocking / squeeze maneuvers
- [ ] Patience-based aisle switching
- [ ] All 6 boarding strategies from the report
- [ ] Multiple replications for statistical comparison
- [ ] Non-compliance modeling
