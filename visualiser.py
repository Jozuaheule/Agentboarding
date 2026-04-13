"""
Pygame Visualiser for the Agent-Based Boarding Simulation.

Shows the full boarding process animated on the cabin graph.

Controls:
  SPACE  - pause / resume
    RIGHT   - speed up  (fewer ms per tick)
    LEFT  - slow down (more ms per tick)
  R      - restart simulation
  Q/ESC  - quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pygame
import pygame.freetype
from calibration.calibration_config import ticks_to_seconds

# Import simulation components from the existing module
from simulation import (
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

MAX_WINDOW_WIDTH = 1800
MAX_WINDOW_HEIGHT = 900
AIRCRAFT_SCALE_FACTOR = 0.78
STRATEGY_GAP = 180

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

# Render low-y graph nodes near the lower side of the panel.
FLIP_Y_AXIS = True


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
        self.sim_std: Optional[BoardingSimulation] = None
        self.sim_pyramid: Optional[BoardingSimulation] = None
        self._reset_simulation()

        # Compute coordinate mapping
        self._compute_layout()

        # Create window
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE | pygame.DOUBLEBUF
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
        self.sim_std = BoardingSimulation(self.env, MANIFEST_FILE, seed=SEED, boarding_policy="std")
        self.sim_pyramid = BoardingSimulation(self.env, MANIFEST_FILE, seed=SEED, boarding_policy="pyramid")
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
        available_w = MAX_WINDOW_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
        available_h = MAX_WINDOW_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

        # Scale uniformly: fit in available area while keeping aspect ratio
        scale_x = available_w / data_w
        scale_y = available_h / data_h
        self.scale = min(scale_x, scale_y) * AIRCRAFT_SCALE_FACTOR

        self.draw_w = int(data_w * self.scale)
        self.draw_h = int(data_h * self.scale)
        self.y_offset = self.draw_h + STRATEGY_GAP

        self.win_w = self.draw_w + MARGIN_LEFT + MARGIN_RIGHT
        self.win_h = self.draw_h * 2 + STRATEGY_GAP + MARGIN_TOP + MARGIN_BOTTOM

        # Precompute node screen positions
        self.node_pos: Dict[str, Tuple[int, int]] = {}
        for nid, data in self.env.graph.nodes(data=True):
            sx = MARGIN_LEFT + int((data["x"] - self.data_x_min) * self.scale)
            if FLIP_Y_AXIS:
                sy = MARGIN_TOP + int((self.data_y_max - data["y"]) * self.scale)
            else:
                sy = MARGIN_TOP + int((data["y"] - self.data_y_min) * self.scale)
            self.node_pos[nid] = (sx, sy)

    def _coord_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        data_w = self.data_x_max - self.data_x_min or 1
        data_h = self.data_y_max - self.data_y_min or 1
        sx = MARGIN_LEFT + int((x - self.data_x_min) / data_w * self.draw_w)
        if FLIP_Y_AXIS:
            sy = MARGIN_TOP + int((self.data_y_max - y) / data_h * self.draw_h)
        else:
            sy = MARGIN_TOP + int((y - self.data_y_min) / data_h * self.draw_h)
        return sx, sy

    # --- Drawing ---

    def _draw_edges(self, offset_y: int = 0) -> None:
        for u, v in self.env.graph.edges():
            p1 = self.node_pos.get(u)
            p2 = self.node_pos.get(v)
            if p1 and p2:
                pygame.draw.aaline(self.screen, EDGE_COLOR, (p1[0], p1[1] + offset_y), (p2[0], p2[1] + offset_y))

    def _draw_nodes(self, offset_y: int = 0) -> None:
        for nid, data in self.env.graph.nodes(data=True):
            p = self.node_pos.get(nid)
            if not p:
                continue
            pos = (p[0], p[1] + offset_y)
            color = NODE_COLORS.get(data.get("type", "aisle"), NODE_COLORS["aisle"])
            pygame.draw.circle(self.screen, color, pos, NODE_RADIUS)

        # Highlight doors
        for label, door_id in self.env.doors.items():
            p = self.node_pos.get(door_id)
            if p:
                pos = (p[0], p[1] + offset_y)
                pygame.draw.circle(self.screen, DOOR_COLOR, pos, DOOR_RADIUS, 2)
                label_rect = self.font_sm.get_rect(label)
                self.font_sm.render_to(
                    self.screen,
                    (pos[0] - label_rect.width // 2, pos[1] + DOOR_RADIUS + 3),
                    label, DOOR_COLOR
                )

    def _draw_passengers(self, sim: BoardingSimulation, offset_y: int = 0) -> None:
        if not sim:
            return
        for agent in sim.agents:
            if not agent.spawned or agent.position is None:
                continue
            p = self.node_pos.get(agent.position)
            if not p:
                continue
            pos = (p[0], p[1] + offset_y)

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
                # Add conflict glows based on intent
                glow_color = None
                if agent.intent == "resolveHeadOn":
                    glow_color = (255, 50, 50, 100) # Red glow
                elif agent.intent == "resolveSeatBlock":
                    if getattr(agent, "seat_shuffle_delay", 0) > 0:
                        glow_color = (255, 255, 50, 150) # Strong Yellow glow targeting penalty
                    else:
                        glow_color = (180, 50, 255, 100) # Purple glow
                elif agent.intent == "switchAisle":
                    glow_color = (50, 200, 255, 100) # Cyan glow
                elif agent.intent == "advance":
                    glow_color = (*base_color, 40)   # Default glow
                
                if glow_color:
                    glow_surf = pygame.Surface((PAX_RADIUS * 6, PAX_RADIUS * 6), pygame.SRCALPHA)
                    pygame.draw.circle(
                        glow_surf,
                        glow_color,
                        (PAX_RADIUS * 3, PAX_RADIUS * 3),
                        PAX_RADIUS * 3,
                    )
                    self.screen.blit(
                        glow_surf,
                        (pos[0] - PAX_RADIUS * 3, pos[1] - PAX_RADIUS * 3),
                    )
                    
                pygame.draw.circle(self.screen, base_color, pos, PAX_RADIUS)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, PAX_RADIUS, 1)

    def _draw_hud_for_sim(self, sim: BoardingSimulation, offset_y: int, title: str) -> None:
        """Draw HUD components for a particular simulation."""
        if not sim:
            return

        total_shuffles = int(sim.event_counters.get("seat_shuffle_start", 0))
        total_stowed = int(sim.event_counters.get("stow_complete", 0))
        seated = sum(1 for a in sim.agents if a.seated)
        total = len(sim.agents)
        is_done = all(a.seated for a in sim.agents)
        completion_time = self._format_elapsed_time(ticks_to_seconds(sim.tick))
        title_text = f"{title}  COMPLETED ({completion_time})" if is_done else title
        
        # Section Title
        self.font_big.render_to(self.screen, (20, 20 + offset_y), title_text, ACCENT)

        # Totals row (non-live counters only)
        y = 50 + offset_y
        stats = [
            f"TOTAL SHUFFLES: {total_shuffles}",
            f"TOTAL STOWED: {total_stowed}",
        ]
        x = 20
        for s in stats:
            self.font_med.render_to(self.screen, (x, y), s, TEXT_COLOR)
            x += 300

        # Completion progress bar
        bar_x, bar_y, bar_w, bar_h = 20, 74 + offset_y, self.win_w - 40, 10
        progress = seated / total if total > 0 else 0
        pygame.draw.rect(self.screen, GRID_COLOR, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            grad_color = (
                int(ACCENT[0] + (ACCENT2[0] - ACCENT[0]) * progress),
                int(ACCENT[1] + (ACCENT2[1] - ACCENT[1]) * progress),
                int(ACCENT[2] + (ACCENT2[2] - ACCENT[2]) * progress),
            )
            pygame.draw.rect(
                self.screen, grad_color, (bar_x, bar_y, fill_w, bar_h), border_radius=5
            )

    def _draw_global_hud(self) -> None:
        """Global HUD like legend and controls, placed at bottom."""
        # Single centered top counter for both strategies
        current_tick = max(
            self.sim_std.tick if self.sim_std else 0,
            self.sim_pyramid.tick if self.sim_pyramid else 0,
        )
        elapsed_time = self._format_elapsed_time(ticks_to_seconds(current_tick))
        top_text = f"Tick: {current_tick}  |  Time: {elapsed_time}"
        top_rect = self.font_big.get_rect(top_text)
        self.font_big.render_to(
            self.screen,
            ((self.win_w - top_rect.width) // 2, 16),
            top_text,
            TEXT_COLOR,
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

        # Seat-shuffle action symbol in legend (yellow halo only)
        halo_cx, halo_cy = x + 6, legend_y + 6
        halo = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(halo, (255, 255, 80, 85), (18, 18), 14)
        self.screen.blit(halo, (halo_cx - 18, halo_cy - 18))
        pygame.draw.circle(self.screen, (150, 150, 150), (halo_cx, halo_cy), 5)
        pygame.draw.circle(self.screen, (255, 255, 80), (halo_cx, halo_cy), 12, 3)
        self.font_sm.render_to(self.screen, (x + 20, legend_y), "Shuffle", TEXT_DIM)
        x += 100

        # Controls & Speed
        if self.paused:
            self.font_med.render_to(
                self.screen, (20, self.win_h - 70), "▐▐  PAUSED", ACCENT
            )
        controls = "SPACE: pause  |  <- ->: speed  |  R: restart  |  Q: quit"
        self.font_sm.render_to(
            self.screen, (self.win_w - 420, self.win_h - 40), controls, TEXT_DIM
        )
        speed_txt = f"Speed: {1000 // self.tick_delay:.0f} tps" if self.tick_delay > 0 else "Speed: MAX"
        self.font_sm.render_to(self.screen, (self.win_w - 420, self.win_h - 22), speed_txt, TEXT_DIM)

    @staticmethod
    def _format_elapsed_time(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

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
                done1 = self.sim_std.step() if self.sim_std and not all(a.seated for a in self.sim_std.agents) else True
                done2 = self.sim_pyramid.step() if self.sim_pyramid and not all(a.seated for a in self.sim_pyramid.agents) else True
                self.done = done1 and done2
                self.last_tick_time = now

            # Render
            self.screen.fill(BG_COLOR)
            
            # Draw Strategy 1 (Top)
            self._draw_edges(offset_y=0)
            self._draw_nodes(offset_y=0)
            self._draw_passengers(self.sim_std, offset_y=0)
            self._draw_hud_for_sim(self.sim_std, offset_y=0, title="Strategy 1: Back-to-front Zonal")

            # Draw Strategy 2 (Bottom)
            self._draw_edges(offset_y=self.y_offset)
            self._draw_nodes(offset_y=self.y_offset)
            self._draw_passengers(self.sim_pyramid, offset_y=self.y_offset)
            self._draw_hud_for_sim(self.sim_pyramid, offset_y=self.y_offset, title="Strategy 2: Modified Reverse Pyramid")
            
            self._draw_global_hud()

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
                elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                    self.tick_delay = max(self.tick_delay - DELAY_STEP, MIN_DELAY)
                elif event.key in (pygame.K_LEFT, pygame.K_DOWN):
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
