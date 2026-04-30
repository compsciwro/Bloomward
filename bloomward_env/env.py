# ============================================================
# bloomward_env/env.py
#
# BloomwardEnv — Gymnasium-compatible single-agent RL
# environment for the Bloomward hex strategy game.
#
# Gymnasium API:
#   obs, info = env.reset(seed=seed)
#   obs, reward, terminated, truncated, info = env.step(action)
#
# Observation space (Dict):
#   board          : Box(int8, shape=(n_tiles,))  encoded tile values 0–5
#   current_flower : Discrete(3)                  0=sunflower 1=tulip 2=blossom
#   season         : Discrete(4)                  0=spring … 3=winter
#   spirit_count   : Box(int8,  shape=(1,))
#   turn           : Box(int16, shape=(1,))
#
# Action space:
#   Discrete(n_tiles) — tile index to place the current flower on
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .board import HexBoard
from .constants import (
    BOARD_RADIUS, MAX_TURNS,
    FLOWER_TYPES, FLOWER_WEIGHTS, FLOWER_NAMES,
    SEASON_NAMES,
    SPREAD_EVERY_N_TURNS,
    REWARD_VALID_PLACEMENT, REWARD_INVALID_ACTION,
    REWARD_COMBO, REWARD_WIN, REWARD_LOSS, REWARD_TURN_PENALTY,
)
from .rules import (
    detect_combos, activate_spirit,
    spread_corruption, get_season, spread_rate_for_season,
    check_terminal,
)


class BloomwardEnv(gym.Env):
    """
    Single-agent Bloomward environment.

    The agent places flowers on a hex board, forms combos to
    activate spirits, and tries to protect the Sacred Core from
    creeping corruption.

    Later extension point
    ---------------------
    The human-vs-AI game mode will be built on top of this
    single-agent version once training is stable.  The board
    encoding, spirit system, and corruption logic are already
    designed to support distinguishing player ownership if needed.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 2}

    def __init__(self, render_mode=None,
                 board_radius=BOARD_RADIUS,
                 max_turns=MAX_TURNS):
        super().__init__()

        self.render_mode  = render_mode
        self.board_radius = board_radius
        self.max_turns    = max_turns

        # Build board object (stateless until reset())
        self._board   = HexBoard(radius=board_radius)
        self._n_tiles = self._board.num_tiles()

        # ── Action space ─────────────────────────────────────────────
        self.action_space = spaces.Discrete(self._n_tiles)

        # ── Observation space ─────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0, high=5,
                shape=(self._n_tiles,),
                dtype=np.int8,
            ),
            "current_flower": spaces.Discrete(3),   # 0/1/2 → types 1/2/3
            "season":         spaces.Discrete(4),
            "spirit_count":   spaces.Box(
                low=0, high=20,
                shape=(1,),
                dtype=np.int8,
            ),
            "turn": spaces.Box(
                low=0, high=max_turns,
                shape=(1,),
                dtype=np.int16,
            ),
        })

        # Game state — initialised properly in reset()
        self._turn            = 0
        self._spirit_count    = 0
        self._current_flower  = FLOWER_TYPES[0]
        self._activated_combos = set()   # tracks already-counted triangles
        self._skip_corruption  = False   # set True by Spirit of Rain

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset to the initial game state.

        Parameters
        ----------
        seed : int or None — seeds the Gymnasium RNG (self.np_random)

        Returns
        -------
        observation : dict
        info        : dict
        """
        # Seed the Gymnasium built-in RNG — this is what SB3 expects
        super().reset(seed=seed)

        # Rebuild board
        self._board.reset()

        # Reset episode state
        self._turn             = 0
        self._spirit_count     = 0
        self._activated_combos = set()
        self._skip_corruption  = False
        self._current_flower   = self._draw_flower()

        obs  = self._make_observation()
        info = {
            "turn":   self._turn,
            "season": get_season(self._turn),
        }

        if self.render_mode in ("human", "ansi"):
            self.render()

        return obs, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int):
        """
        Apply one action.

        Parameters
        ----------
        action : int — tile index (into self._board.tiles)

        Returns
        -------
        observation : dict
        reward      : float
        terminated  : bool
        truncated   : bool
        info        : dict
        """
        reward = 0.0
        info   = {}

        # ── Validate the action ───────────────────────────────────────
        tile = self._board.tile_at_index(action)

        if tile is None or not tile.is_placeable():
            # Invalid — penalise and return without advancing the turn
            reward                 = float(REWARD_INVALID_ACTION)
            info["invalid_action"] = True
            info["turn"]           = self._turn
            info["season"]         = get_season(self._turn)
            return self._make_observation(), reward, False, False, info

        info["invalid_action"] = False

        # ── Place the flower ──────────────────────────────────────────
        tile.flower    = self._current_flower
        placed_coord   = (tile.q, tile.r)
        reward        += float(REWARD_VALID_PLACEMENT)

        # ── Combo detection ───────────────────────────────────────────
        # Only check triangles that contain the newly placed tile,
        # and skip any triangle that was already counted this episode.
        new_combos = detect_combos(
            self._board,
            last_placed_coord=placed_coord,
            activated_combos=self._activated_combos,
        )

        spirits_activated  = []
        self._skip_corruption = False   # reset Rain flag each step

        for combo in new_combos:
            # Mark this triangle as activated so it isn't counted again
            self._activated_combos.add(combo["key"])

            self._spirit_count += 1
            reward             += float(REWARD_COMBO)

            # Resolve the spirit effect
            spirit_result = activate_spirit(
                self._board, combo, self.np_random
            )
            spirits_activated.append(spirit_result)

            # Spirit of Rain sets a flag to skip corruption this turn
            if spirit_result.get("skip_corruption", False):
                self._skip_corruption = True

        info["combos"]            = len(new_combos)
        info["spirits_activated"] = spirits_activated

        # ── Advance turn ──────────────────────────────────────────────
        self._turn += 1
        reward     += float(REWARD_TURN_PENALTY)   # efficiency nudge

        # ── Corruption spread ─────────────────────────────────────────
        newly_corrupted = []
        if self._turn % SPREAD_EVERY_N_TURNS == 0:
            if self._skip_corruption:
                # Spirit of Rain delays spread this activation
                info["corruption_skipped"] = True
            else:
                season   = get_season(self._turn)
                spread_n = spread_rate_for_season(season)
                newly_corrupted = spread_corruption(
                    self._board, spread_n, self.np_random
                )

        info["corruption_spread"] = newly_corrupted

        # ── Season update ─────────────────────────────────────────────
        info["season"] = get_season(self._turn)

        # ── Draw next flower ──────────────────────────────────────────
        self._current_flower = self._draw_flower()

        # ── Win / loss / truncation check ─────────────────────────────
        terminated, truncated, reason = check_terminal(
            self._board, self._spirit_count,
            self._turn, self.max_turns,
        )

        if terminated:
            if reason == "win":
                reward += float(REWARD_WIN)
            else:
                reward += float(REWARD_LOSS)

        info["reason"]       = reason
        info["turn"]         = self._turn
        info["spirit_count"] = self._spirit_count

        obs = self._make_observation()

        if self.render_mode in ("human", "ansi"):
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # render()
    # ------------------------------------------------------------------

    def render(self):
        """
        Render the current game state.

        render_mode="ansi"  → returns a string
        render_mode="human" → prints to console (or opens Pygame if installed)
        """
        season_name  = SEASON_NAMES[get_season(self._turn)]
        flower_name  = FLOWER_NAMES.get(self._current_flower, "?")
        corrupt_count = self._board.count_corrupted()

        lines = [
            "=" * 48,
            f"  BLOOMWARD  |  Turn: {self._turn}  |  Season: {season_name}",
            f"  Next flower : {flower_name}",
            f"  Spirits     : {self._spirit_count}",
            f"  Corrupted   : {corrupt_count}",
            "-" * 48,
            "  O=Sacred  S=Sunflower  T=Tulip  B=Blossom",
            "  X=Corrupt  .=Empty",
            "",
            self._board.render_text(),
            "=" * 48,
        ]

        output = "\n".join(lines)

        if self.render_mode == "ansi":
            return output
        else:   # "human"
            # Try Pygame if available, fall back to text
            try:
                self._render_pygame()
            except Exception:
                print(output)

    def _render_pygame(self):
        """Pygame visual render — opened on demand."""
        import pygame, math

        COLOURS = {
            "bg":      (30, 30, 30),
            "sacred":  (255, 215, 0),
            "fertile": (60, 120, 60),
            "corrupt": (80, 40, 40),
            1:         (255, 220, 50),   # sunflower
            2:         (220, 80, 180),   # tulip
            3:         (180, 120, 255),  # blossom
            "text":    (220, 220, 220),
            "outline": (0, 0, 0),
        }
        HEX_SIZE       = 36
        WIDTH, HEIGHT  = 700, 600
        CX, CY         = WIDTH // 2, HEIGHT // 2 - 20

        if not hasattr(self, "_pg_screen"):
            pygame.init()
            self._pg_screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Bloomward")
            self._pg_clock  = pygame.time.Clock()
            self._pg_font   = pygame.font.SysFont("Arial", 13)
            self._pg_hfont  = pygame.font.SysFont("Arial", 14, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self._pg_screen.fill(COLOURS["bg"])

        def hex_to_pixel(q, r):
            x = CX + HEX_SIZE * (1.5 * q)
            y = CY + HEX_SIZE * (3 ** 0.5 * (r + q / 2))
            return int(x), int(y)

        def draw_hex(colour, x, y, size):
            pts = [(x + size * math.cos(math.radians(60 * i)),
                    y + size * math.sin(math.radians(60 * i)))
                   for i in range(6)]
            pygame.draw.polygon(self._pg_screen, colour, pts)
            pygame.draw.polygon(self._pg_screen, COLOURS["outline"], pts, 2)

        labels = {1: "Su", 2: "Tu", 3: "Bl"}

        for tile in self._board.tiles:
            px, py = hex_to_pixel(tile.q, tile.r)
            if tile.tile_type == "sacred":
                col = COLOURS["sacred"]
            elif tile.corrupted:
                col = COLOURS["corrupt"]
            elif tile.flower:
                col = COLOURS[tile.flower]
            else:
                col = COLOURS["fertile"]

            draw_hex(col, px, py, HEX_SIZE - 2)

            if tile.flower:
                lbl = self._pg_font.render(
                    labels[tile.flower], True, COLOURS["outline"])
                self._pg_screen.blit(lbl, (px - 10, py - 7))
            elif tile.tile_type == "sacred":
                lbl = self._pg_font.render("CORE", True, COLOURS["outline"])
                self._pg_screen.blit(lbl, (px - 14, py - 7))

        hud = (f"Turn:{self._turn}  "
               f"Season:{SEASON_NAMES[get_season(self._turn)]}  "
               f"Next:{FLOWER_NAMES.get(self._current_flower,'?')}  "
               f"Spirits:{self._spirit_count}  "
               f"Corrupt:{self._board.count_corrupted()}")
        self._pg_screen.blit(
            self._pg_hfont.render(hud, True, COLOURS["text"]),
            (10, HEIGHT - 35))

        pygame.display.flip()
        self._pg_clock.tick(self.metadata["render_fps"])

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def close(self):
        """Release resources."""
        try:
            import pygame
            if hasattr(self, "_pg_screen"):
                pygame.quit()
                del self._pg_screen
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_flower(self) -> int:
        """
        Draw a flower type using weighted probabilities.
        Uses self.np_random (seeded by reset()) for reproducibility.
        Weights: Sunflower 50%, Tulip 30%, Blossom 20%.
        """
        r = self.np_random.random()   # uniform [0, 1)
        cumulative = 0.0
        for flower, weight in zip(FLOWER_TYPES, FLOWER_WEIGHTS):
            cumulative += weight
            if r < cumulative:
                return flower
        return FLOWER_TYPES[-1]       # fallback (should not happen)

    def _make_observation(self) -> dict:
        """
        Build the observation dict matching self.observation_space.

        Note: current_flower is stored as Discrete index 0/1/2,
        mapped from internal flower type 1/2/3.
        """
        board_array  = np.array(self._board.to_array(), dtype=np.int8)
        flower_index = self._current_flower - 1   # 1→0, 2→1, 3→2
        season       = get_season(self._turn)

        return {
            "board":          board_array,
            "current_flower": int(flower_index),
            "season":         int(season),
            "spirit_count":   np.array([self._spirit_count], dtype=np.int8),
            "turn":           np.array([self._turn],         dtype=np.int16),
        }
