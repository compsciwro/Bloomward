# -*- coding: utf-8 -*-
# ============================================================
# bloomward_env/env.py
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .board import HexBoard
from .constants import (
    BOARD_RADIUS, MAX_TURNS,
    FLOWER_TYPES, FLOWER_WEIGHTS, FLOWER_NAMES,
    FLOWER_TULIP,
    SEASON_NAMES,
    SPREAD_EVERY_N_TURNS,
    REWARD_VALID_PLACEMENT, REWARD_INVALID_ACTION,
    REWARD_COMBO, REWARD_NEAR_COMBO,
    REWARD_WIN, REWARD_LOSS, REWARD_TURN_PENALTY,
    WIN_SCORE_TARGET,
)
from .rules import (
    detect_combos, activate_spirit,
    get_spirit_targets, apply_spirit_at_target,
    spread_corruption, get_season, spread_rate_for_season,
    check_terminal,
)

_SPIRIT_NAMES = {
    1: "Spirit of Renewal",
    2: "Spirit of Rain",
    3: "Spirit of Blossom",
}


class BloomwardEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 2}

    def __init__(self, render_mode=None,
                 board_radius=BOARD_RADIUS,
                 max_turns=MAX_TURNS):
        super().__init__()

        self.render_mode  = render_mode
        self.board_radius = board_radius
        self.max_turns    = max_turns

        self._board   = HexBoard(radius=board_radius)
        self._n_tiles = self._board.num_tiles()

        self.action_space = spaces.Discrete(self._n_tiles)

        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0, high=5,
                shape=(self._n_tiles,),
                dtype=np.int8,
            ),
            "current_flower": spaces.Discrete(3),
            "season":         spaces.Discrete(4),
            "spirit_count":   spaces.Box(low=0, high=40, shape=(1,), dtype=np.int8),
            "spirit_count_p1": spaces.Box(low=0, high=40, shape=(1,), dtype=np.int8),
            "spirit_count_p2": spaces.Box(low=0, high=40, shape=(1,), dtype=np.int8),
            "turn":           spaces.Box(low=0, high=max_turns, shape=(1,), dtype=np.int16),
        })

        self._turn              = 0
        self._spirit_count      = 0
        self._spirit_count_p1   = 0
        self._spirit_count_p2   = 0
        self._current_flower    = FLOWER_TYPES[0]
        self._activated_combos  = set()
        self._skip_corruption   = False

        self._pending_spirit = None
        self._pending_reward = 0.0
        self._pending_info   = {}
        self._pending_player = 0

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._board.reset()
        self._turn             = 0
        self._spirit_count     = 0
        self._spirit_count_p1  = 0
        self._spirit_count_p2  = 0
        self._activated_combos = set()
        self._skip_corruption  = False
        self._current_flower   = self._draw_flower()

        self._pending_spirit = None
        self._pending_reward = 0.0
        self._pending_info   = {}
        self._pending_player = 0

        obs  = self._make_observation()
        info = {"turn": self._turn, "season": get_season(self._turn)}

        if self.render_mode in ("human", "ansi"):
            self.render()

        return obs, info

    # ------------------------------------------------------------------
    # action_masks()
    # ------------------------------------------------------------------

    def action_masks(self):
        mask = np.zeros(self._n_tiles, dtype=bool)
        if self._pending_spirit is not None:
            for idx in self._pending_spirit["valid_targets"]:
                mask[idx] = True
        else:
            for idx in self._board.placeable_indices():
                mask[idx] = True
        return mask

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int, player: int = 0):
        """
        Apply one action.

        Parameters
        ----------
        action : int   tile index for placement or spirit target
        player : int   0 = Player 1 (human), 1 = Player 2 (agent)
        """
        if self._pending_spirit is not None:
            return self._resolve_spirit_target(action)

        reward = 0.0
        info   = {}

        tile = self._board.tile_at_index(action)
        if tile is None or not tile.is_placeable():
            reward                 = float(REWARD_INVALID_ACTION)
            info["invalid_action"] = True
            info["turn"]           = self._turn
            info["season"]         = get_season(self._turn)
            return self._make_observation(), reward, False, False, info

        info["invalid_action"] = False

        tile.flower   = self._current_flower
        placed_coord  = (tile.q, tile.r)
        placed_flower = self._current_flower
        reward       += float(REWARD_VALID_PLACEMENT)

        new_combos = detect_combos(
            self._board,
            last_placed_coord=placed_coord,
            activated_combos=self._activated_combos,
        )

        if len(new_combos) == 0:
            neighbors = self._board.get_neighbours_of(placed_coord[0], placed_coord[1])
            matching  = [n for n in neighbors if n.flower == placed_flower]
            if len(matching) >= 1:
                reward += float(REWARD_NEAR_COMBO)

        spirits_activated     = []
        self._skip_corruption = False

        for combo in new_combos:
            self._activated_combos.add(combo["key"])
            self._spirit_count += 1
            reward             += float(REWARD_COMBO)

            # Track per-player spirit count
            if player == 0:
                self._spirit_count_p1 += 1
            else:
                self._spirit_count_p2 += 1

            if combo["flower"] == FLOWER_TULIP:
                spirit_result = activate_spirit(self._board, combo, self.np_random)
                spirits_activated.append(spirit_result)
                if spirit_result.get("skip_corruption", False):
                    self._skip_corruption = True
            else:
                valid_targets = get_spirit_targets(self._board, combo)
                if valid_targets:
                    self._pending_spirit = {
                        "combo":         combo,
                        "valid_targets": valid_targets,
                    }
                    self._pending_reward = reward
                    self._pending_player = player
                    self._pending_info   = {
                        "spirits_activated": spirits_activated,
                        "combos":            len(new_combos),
                        "skip_corruption":   self._skip_corruption,
                        "invalid_action":    False,
                    }
                    spirit_name = _SPIRIT_NAMES.get(combo["flower"], "Spirit")
                    return (
                        self._make_observation(),
                        reward,
                        False,
                        False,
                        {
                            "spirit_pending":    True,
                            "spirit_name":       spirit_name,
                            "valid_targets":     valid_targets,
                            "combos":            len(new_combos),
                            "spirits_activated": spirits_activated,
                            "invalid_action":    False,
                        },
                    )
                else:
                    spirits_activated.append({
                        "spirit":     _SPIRIT_NAMES.get(combo["flower"], "Spirit"),
                        "no_targets": True,
                    })

        info["combos"]            = len(new_combos)
        info["spirits_activated"] = spirits_activated

        return self._complete_turn(reward, info)

    # ------------------------------------------------------------------
    # _resolve_spirit_target()
    # ------------------------------------------------------------------

    def _resolve_spirit_target(self, target_idx: int):
        pending = self._pending_spirit
        reward  = self._pending_reward
        info    = dict(self._pending_info)

        spirit_result = apply_spirit_at_target(
            self._board, pending["combo"], target_idx
        )
        info.setdefault("spirits_activated", []).append(spirit_result)
        info["spirit_resolved"] = True
        info["invalid_action"]  = False

        self._skip_corruption = info.pop("skip_corruption", False)

        self._pending_spirit = None
        self._pending_reward = 0.0
        self._pending_info   = {}
        self._pending_player = 0

        return self._complete_turn(reward, info)

    # ------------------------------------------------------------------
    # _complete_turn()
    # ------------------------------------------------------------------

    def _complete_turn(self, reward: float, info: dict):
        self._turn += 1
        reward     += float(REWARD_TURN_PENALTY)

        newly_corrupted = []
        if self._turn % SPREAD_EVERY_N_TURNS == 0:
            if self._skip_corruption:
                info["corruption_skipped"] = True
            else:
                season   = get_season(self._turn)
                spread_n = spread_rate_for_season(season)
                newly_corrupted = spread_corruption(
                    self._board, spread_n, self.np_random
                )

        info["corruption_spread"] = newly_corrupted
        info["season"]            = get_season(self._turn)

        self._current_flower = self._draw_flower()

        terminated, truncated, reason = check_terminal(
            self._board, self._spirit_count,
            self._turn, self.max_turns,
        )

        if terminated:
            if reason == "win":
                reward += float(REWARD_WIN)
                # Determine winner by per-player spirit count
                if self._spirit_count_p1 >= self._spirit_count_p2:
                    info["winner"] = 0
                else:
                    info["winner"] = 1
            else:
                reward += float(REWARD_LOSS)
                info["winner"] = None

        info["reason"]          = reason
        info["turn"]            = self._turn
        info["spirit_count"]    = self._spirit_count
        info["spirit_count_p1"] = self._spirit_count_p1
        info["spirit_count_p2"] = self._spirit_count_p2

        obs = self._make_observation()

        if self.render_mode in ("human", "ansi"):
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # render()
    # ------------------------------------------------------------------

    def render(self):
        season_name   = SEASON_NAMES[get_season(self._turn)]
        flower_name   = FLOWER_NAMES.get(self._current_flower, "?")
        corrupt_count = self._board.count_corrupted()

        lines = [
            "=" * 48,
            f"  BLOOMWARD  |  Turn: {self._turn}  |  Season: {season_name}",
            f"  Next flower : {flower_name}",
            f"  Spirits P1  : {self._spirit_count_p1}",
            f"  Spirits P2  : {self._spirit_count_p2}",
            f"  Corrupted   : {corrupt_count}",
            "-" * 48,
            self._board.render_text(),
            "=" * 48,
        ]

        output = "\n".join(lines)
        if self.render_mode == "ansi":
            return output
        else:
            try:
                self._render_pygame()
            except Exception:
                print(output)

    def _render_pygame(self):
        import pygame, math

        COLOURS = {
            "bg":      (30, 30, 30),
            "sacred":  (255, 215, 0),
            "fertile": (60, 120, 60),
            "corrupt": (80, 40, 40),
            1:         (255, 220, 50),
            2:         (220, 80, 180),
            3:         (180, 120, 255),
            "text":    (220, 220, 220),
            "outline": (0, 0, 0),
        }
        HEX_SIZE      = 36
        WIDTH, HEIGHT = 700, 600
        CX, CY        = WIDTH // 2, HEIGHT // 2 - 20

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
                lbl = self._pg_font.render(labels[tile.flower], True, COLOURS["outline"])
                self._pg_screen.blit(lbl, (px - 10, py - 7))

        hud = (f"Turn:{self._turn}  "
               f"Season:{SEASON_NAMES[get_season(self._turn)]}  "
               f"P1:{self._spirit_count_p1}  P2:{self._spirit_count_p2}  "
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
        r = self.np_random.random()
        cumulative = 0.0
        for flower, weight in zip(FLOWER_TYPES, FLOWER_WEIGHTS):
            cumulative += weight
            if r < cumulative:
                return flower
        return FLOWER_TYPES[-1]

    def _make_observation(self) -> dict:
        board_array  = np.array(self._board.to_array(), dtype=np.int8)
        flower_index = self._current_flower - 1
        season       = get_season(self._turn)
        return {
            "board":           board_array,
            "current_flower":  int(flower_index),
            "season":          int(season),
            "spirit_count":    np.array([self._spirit_count],    dtype=np.int8),
            "spirit_count_p1": np.array([self._spirit_count_p1], dtype=np.int8),
            "spirit_count_p2": np.array([self._spirit_count_p2], dtype=np.int8),
            "turn":            np.array([self._turn],            dtype=np.int16),
        }
