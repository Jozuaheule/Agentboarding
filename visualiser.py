"""
Pygame Visualiser for the Agent-Based Boarding Simulation.

Shows the full boarding process animated on the cabin graph.

Controls:
  SPACE  - pause / resume
  UP     - speed up  (fewer ms per tick)
  DOWN   - slow down (more ms per tick)
  R      - restart simulation
  Q/ESC  - quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pygame
import pygame.freetype

# Import simulation components from the existing module
from simulation import (
    BASE_DIR,
    EDGES_FILE,
    MANIFEST_FILE,
    NODES_FILE,
    SEED,
    BoardingSimulation,
    CabinEnvironment,
)

# ---------------------------------------------------------------------------
# Visual settings
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Boeing 787 Boarding Simulation"

# Layout (pixels)
MARGIN_LEFT = 80
MARGIN_TOP = 120
MARGIN_RIGHT = 40
MARGIN_BOTTOM = 100

NODE_RADIUS = 5
PAX_RADIUS = 7

# Timing
INITIAL_TICK_DELAY_MS = 80  # ms between simulation ticks
MIN_DELAY = 5
MAX_DELAY = 500
DELAY_STEP = 10

# Colors
BG_COLOR = (18, 18, 24)
GRID_COLOR = (40, 40, 55)
TEXT_COLOR = (220, 220, 230)
TEXT_DIM = (130, 130, 150)
ACCENT = (80, 180, 255)
ACCENT2 = (255, 160, 60)

NODE_COLORS = {
    "aisle": (55, 55, 70),
    "seat": (45, 80, 130),
    "galley": (70, 70, 50),
}

EDGE_COLOR = (35, 35, 48)

# Passenger colors by class
PAX_COLORS = {
    "business": (255, 200, 60),
    "premium_economy": (60, 200, 160),
    "economy": (100, 160, 255),
}
PAX_SEATED_ALPHA = 120  # dimmer when seated
PAX_DEFAULT = (200, 200, 200)

DOOR_COLOR = (255, 80, 80)
DOOR_RADIUS = 9


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------
class BoardingVisualiser:
    """Pygame-based visualiser for the boarding simulation."""

    def __init__(self) -> None:
        pygame.init()
        pygame.freetype.init()

        # Set up environment & simulation
        self.env = CabinEnvironment(NODES_FILE, EDGES_FILE)
        self.sim: Optional[BoardingSimulation] = None
        self._reset_simulation()

        # Compute coordinate mapping
        self._compute_layout()

        # Create window
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )
        pygame.display.set_caption(WINDOW_TITLE)

        # Fonts
        self.font_big = pygame.freetype.SysFont("Helvetica,Arial", 22)
        self.font_med = pygame.freetype.SysFont("Helvetica,Arial", 16)
        self.font_sm = pygame.freetype.SysFont("Helvetica,Arial", 13)

        # State
        self.running = True
        self.paused = False
        self.tick_delay = INITIAL_TICK_DELAY_MS
        self.done = False

        self.clock = pygame.time.Clock()
        self.last_tick_time = pygame.time.get_ticks()

        # Precompute passenger class for coloring
        self._load_pax_classes()

    def _reset_simulation(self) -> None:
        self.sim = BoardingSimulation(self.env, MANIFEST_FILE, seed=SEED)
        self.done = False

    def _load_pax_classes(self) -> None:
        """Map pax_id -> travel class for coloring."""
        import pandas as pd
        df = pd.read_excel(MANIFEST_FILE)
        self.pax_class: Dict[str, str] = {}
        for _, row in df.iterrows():
            self.pax_class[str(row["pax_id"])] = str(row["class"])

    def _compute_layout(self) -> None:
        """Map graph coordinates to screen pixels."""
        xs = [d["x"] for _, d in self.env.graph.nodes(data=True)]
        ys = [d["y"] for _, d in self.env.graph.nodes(data=True)]

        self.data_x_min, self.data_x_max = min(xs), max(xs)
        self.data_y_min, self.data_y_max = min(ys), max(ys)

        data_w = self.data_x_max - self.data_x_min or 1
        data_h = self.data_y_max - self.data_y_min or 1

        # Cap to reasonable screen size; cabin is very wide (125×10)
        max_win_w = 1600
        max_win_h = 650
        available_w = max_win_w - MARGIN_LEFT - MARGIN_RIGHT
        available_h = max_win_h - MARGIN_TOP - MARGIN_BOTTOM

        # Scale uniformly: fit in available area while keeping aspect ratio
        scale_x = available_w / data_w
        scale_y = available_h / data_h
        self.scale = min(scale_x, scale_y)

        self.draw_w = int(data_w * self.scale)
        self.draw_h = int(data_h * self.scale)

        self.win_w = self.draw_w + MARGIN_LEFT + MARGIN_RIGHT
        self.win_h = self.draw_h + MARGIN_TOP + MARGIN_BOTTOM

        # Precompute node screen positions
        self.node_pos: Dict[str, Tuple[int, int]] = {}
        for nid, data in self.env.graph.nodes(data=True):
            sx = MARGIN_LEFT + int((data["x"] - self.data_x_min) * self.scale)
            sy = MARGIN_TOP + int((data["y"] - self.data_y_min) * self.scale)
            self.node_pos[nid] = (sx, sy)

    def _coord_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        data_w = self.data_x_max - self.data_x_min or 1
        data_h = self.data_y_max - self.data_y_min or 1
        sx = MARGIN_LEFT + int((x - self.data_x_min) / data_w * self.draw_w)
        sy = MARGIN_TOP + int((y - self.data_y_min) / data_h * self.draw_h)
        return sx, sy

    # --- Drawing ---

    def _draw_edges(self) -> None:
        for u, v in self.env.graph.edges():
            p1 = self.node_pos.get(u)
            p2 = self.node_pos.get(v)
            if p1 and p2:
                pygame.draw.aaline(self.screen, EDGE_COLOR, p1, p2)

    def _draw_nodes(self) -> None:
        for nid, data in self.env.graph.nodes(data=True):
            pos = self.node_pos.get(nid)
            if not pos:
                continue
            color = NODE_COLORS.get(data.get("type", "aisle"), NODE_COLORS["aisle"])
            pygame.draw.circle(self.screen, color, pos, NODE_RADIUS)

        # Highlight doors
        for label, door_id in self.env.doors.items():
            pos = self.node_pos.get(door_id)
            if pos:
                pygame.draw.circle(self.screen, DOOR_COLOR, pos, DOOR_RADIUS, 2)
                self.font_sm.render_to(
                    self.screen, (pos[0] - 4, pos[1] - DOOR_RADIUS - 16),
                    label, DOOR_COLOR
                )

    def _draw_passengers(self) -> None:
        if not self.sim:
            return
        for agent in self.sim.agents:
            if not agent.spawned or agent.position is None:
                continue
            pos = self.node_pos.get(agent.position)
            if not pos:
                continue

            travel_class = self.pax_class.get(agent.pax_id, "economy")
            base_color = PAX_COLORS.get(travel_class, PAX_DEFAULT)

            if agent.seated:
                # Seated: solid filled circle on the seat node
                pygame.draw.circle(self.screen, base_color, pos, PAX_RADIUS - 1)
                # Inner dot to mark "occupied"
                inner = tuple(min(c + 80, 255) for c in base_color)
                pygame.draw.circle(self.screen, inner, pos, 2)

            elif agent.luggage_status == "stowing":
                # Stowing luggage: draw passenger + orange ring indicator
                pygame.draw.circle(self.screen, base_color, pos, PAX_RADIUS)
                pygame.draw.circle(self.screen, (255, 140, 30), pos, PAX_RADIUS + 3, 2)
                # Small luggage icon (filled square offset)
                lx, ly = pos[0] + PAX_RADIUS + 2, pos[1] - PAX_RADIUS
                pygame.draw.rect(self.screen, (255, 140, 30), (lx, ly, 5, 5))

            else:
                # Moving: draw full with a glow
                glow_surf = pygame.Surface((PAX_RADIUS * 6, PAX_RADIUS * 6), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow_surf,
                    (*base_color, 40),
                    (PAX_RADIUS * 3, PAX_RADIUS * 3),
                    PAX_RADIUS * 3,
                )
                self.screen.blit(
                    glow_surf,
                    (pos[0] - PAX_RADIUS * 3, pos[1] - PAX_RADIUS * 3),
                )
                pygame.draw.circle(self.screen, base_color, pos, PAX_RADIUS)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, PAX_RADIUS, 1)

    def _draw_hud(self) -> None:
        """Draw the heads-up display with stats and controls."""
        if not self.sim:
            return

        seated = sum(1 for a in self.sim.agents if a.seated)
        spawned = sum(1 for a in self.sim.agents if a.spawned)
        stowing = sum(1 for a in self.sim.agents if a.luggage_status == "stowing")
        total = len(self.sim.agents)
        queue_len = len(self.sim.spawn_queue)

        # Title bar
        self.font_big.render_to(self.screen, (20, 20), WINDOW_TITLE, ACCENT)

        # Stats row
        y = 55
        stats = [
            f"Tick: {self.sim.tick}",
            f"Spawned: {spawned}/{total}",
            f"Seated: {seated}/{total}",
            f"Stowing: {stowing}",
            f"Queue: {queue_len}",
            f"Speed: {1000 // self.tick_delay:.0f} tps" if self.tick_delay > 0 else "Speed: MAX",
        ]
        x = 20
        for s in stats:
            self.font_med.render_to(self.screen, (x, y), s, TEXT_COLOR)
            x += 180

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 20, 82, self.win_w - 40, 12
        progress = seated / total if total > 0 else 0
        pygame.draw.rect(self.screen, GRID_COLOR, (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            grad_color = (
                int(ACCENT[0] + (ACCENT2[0] - ACCENT[0]) * progress),
                int(ACCENT[1] + (ACCENT2[1] - ACCENT[1]) * progress),
                int(ACCENT[2] + (ACCENT2[2] - ACCENT[2]) * progress),
            )
            pygame.draw.rect(
                self.screen, grad_color, (bar_x, bar_y, fill_w, bar_h), border_radius=6
            )
        pct_text = f"{progress * 100:.0f}%"
        self.font_sm.render_to(
            self.screen, (bar_x + bar_w + 8, bar_y - 1), pct_text, TEXT_COLOR
        )

        # Status
        if self.done:
            self.font_med.render_to(
                self.screen, (20, self.win_h - 70),
                f"✓ BOARDING COMPLETE in {self.sim.tick} ticks", ACCENT2
            )
        elif self.paused:
            self.font_med.render_to(
                self.screen, (20, self.win_h - 70), "▐▐  PAUSED", ACCENT
            )

        # Legend
        legend_y = self.win_h - 40
        x = 20
        for cls, color in PAX_COLORS.items():
            pygame.draw.circle(self.screen, color, (x + 6, legend_y + 6), 6)
            label = cls.replace("_", " ").title()
            self.font_sm.render_to(self.screen, (x + 18, legend_y), label, TEXT_DIM)
            x += 150

        # Stowing indicator in legend
        pygame.draw.circle(self.screen, (150, 150, 150), (x + 6, legend_y + 6), 6)
        pygame.draw.circle(self.screen, (255, 140, 30), (x + 6, legend_y + 6), 9, 2)
        self.font_sm.render_to(self.screen, (x + 20, legend_y), "Stowing", TEXT_DIM)
        x += 100

        # Controls
        controls = "SPACE: pause  |  ↑↓: speed  |  R: restart  |  Q: quit"
        self.font_sm.render_to(
            self.screen, (self.win_w - 420, self.win_h - 40), controls, TEXT_DIM
        )

    # --- Main loop ---

    def run(self) -> None:
        while self.running:
            self._handle_events()

            # Advance simulation
            now = pygame.time.get_ticks()
            if (
                not self.paused
                and not self.done
                and now - self.last_tick_time >= self.tick_delay
            ):
                self.done = self.sim.step()
                self.last_tick_time = now

            # Render
            self.screen.fill(BG_COLOR)
            self._draw_edges()
            self._draw_nodes()
            self._draw_passengers()
            self._draw_hud()

            pygame.display.flip()
            self.clock.tick(120)  # cap at 120 fps

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.tick_delay = max(self.tick_delay - DELAY_STEP, MIN_DELAY)
                elif event.key == pygame.K_DOWN:
                    self.tick_delay = min(self.tick_delay + DELAY_STEP, MAX_DELAY)
                elif event.key == pygame.K_r:
                    self._reset_simulation()
                    self.paused = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    vis = BoardingVisualiser()
    vis.run()


if __name__ == "__main__":
    main()
