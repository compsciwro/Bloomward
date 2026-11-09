# -*- coding: utf-8 -*-
"""
Bloomward - game_screen.py
Main game screen with updated UI assets.
Human (Player 1) vs trained MaskablePPO agent (Player 2).
"""

import pygame
import math
import numpy as np
import os

from effects import EffectsManager, bloom_scale, BLOOM_TIME

# ── RL stack ──────────────────────────────────────────────────────────────────
try:
    from bloomward_env import BloomwardEnv
    from bloomward_env.constants import SPREAD_EVERY_N_TURNS, WIN_SCORE_TARGET
    from bloomward_env.rules import get_season, spread_rate_for_season
    ENV_OK = True
except ImportError:
    SPREAD_EVERY_N_TURNS, WIN_SCORE_TARGET = 8, 60
    ENV_OK = False
    print("[game_screen] BloomwardEnv not found - running in demo mode.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bloomward_ppo.zip")

try:
    from sb3_contrib import MaskablePPO
    if os.path.exists(MODEL_PATH):
        AGENT = MaskablePPO.load(MODEL_PATH)
        AGENT_OK = True
        print(f"[game_screen] Trained agent loaded from {MODEL_PATH}")
    else:
        AGENT, AGENT_OK = None, False
        print(f"[game_screen] NO MODEL at {MODEL_PATH} - agent is RANDOM.")
except ImportError:
    AGENT, AGENT_OK = None, False
    print("[game_screen] sb3-contrib missing - agent is RANDOM.")

# ── Colours ───────────────────────────────────────────────────────────────────
C_STATUS_BG  = (45, 90, 35)
C_HOVER      = (215, 215, 80)
C_OUTLINE    = (25, 25, 25)
C_WHITE      = (255, 255, 255)
C_DARK       = (50, 25, 5)
C_LIGHT      = (255, 245, 220)

C_PANEL      = (255, 248, 230, 225)   # parchment panels (with alpha)
C_PANEL_EDGE = (120, 80, 40)
C_FERTILE    = (76, 153, 0)
C_P1         = (50, 110, 220)          # player colours: bars + last-move rings
C_P2         = (150, 70, 200)
C_COMBO      = (255, 235, 90)          # combo flash
C_SPREAD     = (140, 50, 190)          # newly corrupted flash (violet = decay)
FLASH_MS     = 2600

C_TARGET_RENEWAL = (220, 100, 60)

# Spirit of Renewal targeting (cleanse a corrupted tile)
C_RENEW_LINE  = (255, 79, 216)    # #FF4FD8 primary outline
C_RENEW_GLOW  = (255, 193, 240)   # #FFC1F0 soft outer glow
RENEW_PULSE_S = 1.5               # seconds per glow cycle
TARGET_DIM_ALPHA = 34             # darken non-target tiles while targeting (0 = off, ~13% darker)
C_TARGET_BLOSSOM = (80, 210, 100)
C_TARGET_HOVER   = (255, 240, 60)

SPIRIT_COLOURS = {
    "Spirit of Renewal": (255, 220, 80),
    "Spirit of Rain":    (100, 200, 255),
    "Spirit of Blossom": (180, 255, 150),
}
SPIRIT_DEFAULT_COLOUR = (255, 240, 180)

# ── Tile colour map ───────────────────────────────────────────────────────────
TILE_COLOR = {
    0: (76, 153, 0),
    1: (255, 200, 0),
    2: (230, 80, 110),
    3: (180, 150, 255),
    4: (101, 55, 0),
    5: (245, 230, 175),
}

FLOWER_COLOR  = {0: (255, 200, 0), 1: (230, 80, 110), 2: (180, 150, 255)}
FLOWER_EDGE   = {0: (170, 110, 0), 1: (150, 40, 70), 2: (110, 80, 190)}
FLOWER_CENTRE = {0: (120, 70, 20), 1: (255, 230, 120), 2: (255, 250, 210)}
FLOWER_NAME  = {0: "Sunflower", 1: "Tulip", 2: "Blossom"}
SEASON_NAME  = {0: "Spring", 1: "Rain", 2: "Autumn", 3: "Winter"}

OBS_BOARD      = "board"
OBS_FLOWER     = "current_flower"
OBS_SEASON     = "season"
OBS_SPIRITS    = "spirit_count"
OBS_SPIRITS_P1 = "spirit_count_p1"
OBS_SPIRITS_P2 = "spirit_count_p2"

BOARD_RADIUS = 4
HEX_SIZE     = 38

ASSET_DIR   = os.path.join(os.path.dirname(__file__), "assets")
FLOWER_FILE = {0: "flower_sunflower.png", 1: "flower_tulip.png", 2: "flower_blossom.png"}
TREE_FILE   = "sacred_tree.png"
GRASS_FILE  = "fertile_grass.png"
VICTORY_CARD_FILE = "victory_card.png"
PLAY_AGAIN_FILE   = "btnplayagain.png"
HOME_BTN_FILE     = "btnhomescreen.png"
VICTORY_DIM_ALPHA = 165            # ~65% black over the finished board
GRAVEL_FILE = "corrupted_gravel.png"
TREE_SCALE  = 1.0   # Sacred Tree tile size relative to a normal tile
NEIGHBOUR_FLOWER_SCALE = 0.8   # flowers beside the tree are drawn a little smaller


def _ring_dist(q, r):
    return max(abs(q), abs(r), abs(q + r))


def _build_tile_list():
    tiles = []
    for q in range(-BOARD_RADIUS, BOARD_RADIUS + 1):
        r_min = max(-BOARD_RADIUS, -q - BOARD_RADIUS)
        r_max = min(BOARD_RADIUS,  -q + BOARD_RADIUS)
        for r in range(r_min, r_max + 1):
            tiles.append((q, r))
    return tiles


_TILES        = _build_tile_list()
_COORD_TO_IDX = {c: i for i, c in enumerate(_TILES)}


def _hex_to_pixel(q, r, size, cx, cy):
    x = cx + size * 1.5 * q
    y = cy + size * math.sqrt(3) * (r + q / 2.0)
    return int(x), int(y)


def _pixel_to_hex(px, py, size, cx, cy):
    px -= cx; py -= cy
    fq =  (2 / 3 * px) / size
    fr = (-1 / 3 * px + math.sqrt(3) / 3 * py) / size
    fz = -fq - fr
    q, r, z = round(fq), round(fr), round(fz)
    dq, dr, dz = abs(q - fq), abs(r - fr), abs(z - fz)
    if dq > dr and dq > dz:
        q = -r - z
    elif dr > dz:
        r = -q - z
    return q, r


def _hex_corners(cx, cy, size):
    return [(cx + size * math.cos(math.radians(60 * i)),
             cy + size * math.sin(math.radians(60 * i))) for i in range(6)]


class _ImgBtn:
    def __init__(self, img, x, y):
        self.img  = img
        self.rect = img.get_rect(topleft=(x, y)) if img else pygame.Rect(x, y, 180, 48)

    def draw(self, surf):
        if self.img:
            surf.blit(self.img, self.rect)
        else:
            pygame.draw.rect(surf, (75, 105, 40), self.rect, border_radius=8)

    def clicked(self, events):
        return any(
            e.type == pygame.MOUSEBUTTONDOWN and e.button == 1
            and self.rect.collidepoint(e.pos)
            for e in events
        )


class GameScreen:
    AGENT_DELAY_MS  = 900
    TARGET_DELAY_MS = 600
    NOTIFY_DURATION = 2800

    def __init__(self, screen, assets):
        self.screen = screen
        self.assets = assets
        self.sw, self.sh = screen.get_size()

        # Board centre — shifted right to leave room for flower card on left
        self.cx = self.sw // 2 + 40
        self.cy = self.sh // 2 + 35

        # Fonts
        self.f_sm     = pygame.font.SysFont("Georgia", 15)
        self.f_md     = pygame.font.SysFont("Georgia", 17, bold=True)
        self.f_lg     = pygame.font.SysFont("Georgia", 26, bold=True)
        self.f_notify = pygame.font.SysFont("Georgia", 16, bold=True)
        self.f_idx    = pygame.font.SysFont("Georgia", 11)
        self.f_card   = pygame.font.SysFont("Georgia", 16, bold=True)
        self.f_card_sm = pygame.font.SysFont("Georgia", 13)

        # Prepare flower card image (scale to fit left panel)
        raw_card = assets.get("flower_card")
        self.card_img = None
        self.card_w, self.card_h = 310, 485
        if raw_card:
            self.card_img = pygame.transform.smoothscale(
                raw_card, (self.card_w, self.card_h))

        # Logo
        raw_logo = assets.get("logo")
        self.logo_img = None
        if raw_logo:
            ratio = 60 / raw_logo.get_height()
            self.logo_img = pygame.transform.smoothscale(
                raw_logo, (int(raw_logo.get_width() * ratio), 60))

        # Right-side buttons — stacked with 72px gap
        bx, by, gap = self.sw - 230, 260, 90
        self.b_rules = _ImgBtn(assets.get("btn_rules"), bx, by)
        self.b_home  = _ImgBtn(assets.get("btn_home"),  bx, by + gap)
        self.b_exit  = _ImgBtn(assets.get("btn_exit"),  bx, by + gap * 2)

        self.fx = EffectsManager()
        self._prev_ticks = pygame.time.get_ticks()
        self._load_art()

        self.hover_idx  = None
        self.debug_mode = False
        self._reset()

    # ── Artwork ───────────────────────────────────────────────────────────────
    def _load_img(self, name):
        path = os.path.join(ASSET_DIR, name)
        if not os.path.exists(path):
            print(f"[game_screen] {name} not found - using drawn shapes instead.")
            return None
        return pygame.image.load(path).convert_alpha()

    @staticmethod
    def _fit(img, box):
        """Scale img to fit inside a box x box square, keeping its proportions."""
        ratio = box / max(img.get_width(), img.get_height())
        return pygame.transform.smoothscale(
            img, (max(1, int(img.get_width() * ratio)), max(1, int(img.get_height() * ratio))))

    def _load_victory_art(self):
        """Victory card and its two buttons, scaled once and positioned relative to the card."""
        self.victory = None
        self.fx_ui = EffectsManager()
        self._ui_spark = 0.0
        card = self._load_img(VICTORY_CARD_FILE)
        play = self._load_img(PLAY_AGAIN_FILE)
        home = self._load_img(HOME_BTN_FILE)
        if not (card and play and home):
            return                       # missing art: the old text overlay is used instead
        card = card.subsurface(card.get_bounding_rect(min_alpha=40)).copy()
        ch = int(self.sh * 0.92)
        cw = int(card.get_width() * ch / card.get_height())
        card = pygame.transform.smoothscale(card, (cw, ch))
        rect = card.get_rect(center=(self.sw // 2, self.sh // 2))

        def button(img, centre_y):
            img = img.subsurface(img.get_bounding_rect(min_alpha=128)).copy()
            w = int(cw * 0.36)                                      # both buttons the same width
            h = int(img.get_height() * w / img.get_width())
            normal = pygame.transform.smoothscale(img, (w, h))
            hover = pygame.transform.smoothscale(img, (int(w * 1.04), int(h * 1.04)))
            hover.fill((28, 28, 18), special_flags=pygame.BLEND_RGB_ADD)   # slightly brighter
            r = normal.get_rect(center=(rect.centerx, rect.y + int(ch * centre_y)))
            return {"normal": normal, "hover": hover, "rect": r}

        self.victory = {
            "card": card, "rect": rect,
            "play": button(play, 0.655),        # positions inside the empty parchment area
            "home": button(home, 0.79),
            "score_y": rect.y + int(ch * 0.548),  # just below the twig ornaments
            "font": pygame.font.SysFont("Georgia", max(18, int(ch * 0.04)), bold=True),
            "dim": None,
        }
        dim = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, VICTORY_DIM_ALPHA))
        self.victory["dim"] = dim
        self.fx_ui = EffectsManager()           # sparkles drawn above the card
        self._ui_spark = 0.0

    def _victory_showing(self):
        return self.game_over and self.winner == 0 and self.victory is not None

    def _hex_texture(self, name):
        """Load a hex-shaped tile image, crop it to the hexagon, scale it to the
        board's tile size (corner distance = HEX_SIZE) and trim it to the hex shape.
        The images are regular flat-topped hexagons, so scaling keeps their shape."""
        img = self._load_img(name)
        if img is None:
            return None
        bbox = img.get_bounding_rect(min_alpha=128)          # the hexagon inside the PNG
        img = img.subsurface(bbox).copy()
        r = HEX_SIZE + 0.5                                     # +0.5 px hides seams between tiles
        w, h = int(round(2 * r)), int(round(math.sqrt(3) * r))
        tile = pygame.transform.smoothscale(img, (w, h))
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), _hex_corners(w / 2, h / 2, r))
        tile.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        return tile

    def _hex_overlay(self, colour, alpha):
        """Translucent hex used for hover and targeting highlights on textured tiles."""
        key = (colour, alpha)
        if key not in self._overlay_cache:
            size = HEX_SIZE * 2 + 2
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.polygon(s, (*colour, alpha), _hex_corners(size / 2, size / 2, HEX_SIZE - 2))
            self._overlay_cache[key] = s
        return self._overlay_cache[key]

    def _load_art(self):
        """Flower images for the board and the card, and the Sacred Tree clipped to its hex.
        Anything missing falls back to the drawn shapes, so the game still runs."""
        self.flower_src, self.flower_board, self.flower_card = {}, {}, {}
        self._bloom_cache = {}
        for k, name in FLOWER_FILE.items():
            img = self._load_img(name)
            if img:
                self.flower_src[k]   = img
                self.flower_board[k] = self._fit(img, int(HEX_SIZE * 1.55))
                self.flower_card[k]  = self._fit(img, 120)

        # Sacred Tree artwork clipped to its hex tile
        self._load_victory_art()

        # Terrain tiles: scaled once to the exact hex size of the board and cached
        self.tex_grass  = self._hex_texture(GRASS_FILE)
        self.tex_gravel = self._hex_texture(GRAVEL_FILE)
        self._overlay_cache = {}

        self.tree_big = None
        self.tree_shine = None
        tree_src = self._load_img(TREE_FILE)
        if tree_src:
            R = int(HEX_SIZE * TREE_SCALE)              # same corner distance as the terrain tiles
            size = R * 2 + 8
            c = size / 2
            ratio = (R * 2.15) / min(tree_src.get_width(), tree_src.get_height())
            art = pygame.transform.smoothscale(
                tree_src, (int(tree_src.get_width() * ratio), int(tree_src.get_height() * ratio)))
            badge = pygame.Surface((size, size), pygame.SRCALPHA)
            badge.fill((250, 238, 190, 255))
            badge.blit(art, art.get_rect(center=(c, c + R * 0.08)))
            mask = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.polygon(mask, (255, 255, 255, 255), _hex_corners(c, c, R))
            badge.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            pygame.draw.polygon(badge, C_OUTLINE, _hex_corners(c, c, R), 2)   # same outline as other tiles
            self.tree_big = badge
            # Hex-shaped light used for the shimmer across the tree
            self.tree_shine = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.polygon(self.tree_shine, (255, 220, 120, 255), _hex_corners(c, c, R - 2))

        self.tree_tile = None
        tree = self._load_img(TREE_FILE)
        if tree:
            # Cover the hex, then cut the picture to the hex shape
            size = (HEX_SIZE - 2) * 2
            ratio = size / min(tree.get_width(), tree.get_height())
            tree = pygame.transform.smoothscale(
                tree, (int(tree.get_width() * ratio), int(tree.get_height() * ratio)))
            tile = pygame.Surface((size, size), pygame.SRCALPHA)
            tile.fill((245, 230, 175, 255))
            tile.blit(tree, tree.get_rect(center=(size // 2, size // 2)))
            mask = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.polygon(mask, (255, 255, 255, 255),
                                _hex_corners(size / 2, size / 2, size / 2))
            tile.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.tree_tile = tile

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset(self):
        if ENV_OK:
            self.env = BloomwardEnv(render_mode=None)
            self.obs, self.info = self.env.reset()
        else:
            self.env = None
            self.obs = self._demo_obs()
            self.info = {}

        self.winner               = None
        if getattr(self, "fx_ui", None):
            self.fx_ui.clear()
        self.round_num            = 1
        self.turn_count           = 0
        self.game_over            = False
        self.result_msg           = ""
        self.log: list[str]       = []
        self._notifications: list = []
        self._agent_waiting       = False
        self._agent_tick          = 0
        self.spirits_p1           = 0
        self.spirits_p2           = 0

        self._targeting            = False
        self._target_spirit_name   = ""
        self._target_hint          = ""
        self._target_highlight_col = C_TARGET_RENEWAL
        self._valid_target_idxs    = set()

        # Visual feedback (purely cosmetic; never touches the environment)
        if hasattr(self, "fx"):
            self.fx.clear()
        self._blooms        = {}          # tile index -> tick when its flower was placed
        self._hover_spark   = 0.0
        self._renewal_drawn = set()
        self._last_move     = {}          # player -> tile index of last placement
        self._flash_combo   = (set(), 0)  # (tile indices, expiry tick)
        self._flash_spread  = (set(), 0)

    def _demo_obs(self):
        board = np.zeros(len(_TILES), dtype=np.int8)
        for i, (q, r) in enumerate(_TILES):
            d = _ring_dist(q, r)
            if d == 0:
                board[i] = 5
            elif d == BOARD_RADIUS:
                board[i] = 4
        return {
            OBS_BOARD:      board,
            OBS_FLOWER:     np.array([0]),
            OBS_SEASON:     np.array([0]),
            OBS_SPIRITS:    np.array([0], dtype=np.int8),
            OBS_SPIRITS_P1: np.array([0], dtype=np.int8),
            OBS_SPIRITS_P2: np.array([0], dtype=np.int8),
        }

    # ── Obs helpers ───────────────────────────────────────────────────────────
    def _s(self, key, default=0):
        v = self.obs.get(key)
        if v is None:
            return default
        return int(v) if np.isscalar(v) else int(np.asarray(v).flat[0])

    @property
    def board(self):
        return self.obs.get(OBS_BOARD, np.zeros(len(_TILES), dtype=np.int8))

    @property
    def cur_flower(self):  return self._s(OBS_FLOWER, 0)
    @property
    def season(self):      return self._s(OBS_SEASON, 0)
    @property
    def corruption(self):  return int(np.sum(self.board == 4))

    def _get_sp1(self):
        v = self._s(OBS_SPIRITS_P1, -1)
        return v if v >= 0 else self.spirits_p1

    def _get_sp2(self):
        v = self._s(OBS_SPIRITS_P2, -1)
        return v if v >= 0 else self.spirits_p2

    # ── Notifications ─────────────────────────────────────────────────────────
    def _add_notification(self, msg, colour):
        self._notifications.append((msg, pygame.time.get_ticks() + self.NOTIFY_DURATION, colour))
        if len(self._notifications) > 2:
            self._notifications.pop(0)

    def _notify_spirits(self, spirits_activated, player):
        for sr in spirits_activated:
            if sr.get("no_targets"):
                continue
            name   = sr.get("spirit", "Spirit")
            colour = SPIRIT_COLOURS.get(name, SPIRIT_DEFAULT_COLOUR)
            who    = "P1" if player == 0 else "P2"
            if player == 0:
                self.spirits_p1 += 1
            else:
                self.spirits_p2 += 1
            if sr.get("cleansed_tile"):
                q, r = sr["cleansed_tile"]
                msg = f"[{who}] {name}! Cleansed tile at ({q},{r})."
            elif sr.get("skip_corruption"):
                msg = f"[{who}] {name}! Corruption skipped."
            elif sr.get("protected_tiles") is not None:
                msg = f"[{who}] {name}! Protected {len(sr['protected_tiles'])} tile(s)."
            else:
                msg = f"[{who}] {name} activated!"
            self._add_notification(msg, colour)
            self._log(msg)

    # ── Update ────────────────────────────────────────────────────────────────
    def _update_victory(self, events):
        v = self.victory
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if v["play"]["rect"].collidepoint(e.pos):
                    self._reset()                    # normal env reset, fresh game vs the agent
                    return None
                if v["home"]["rect"].collidepoint(e.pos):
                    return "home"
                return None                          # one click, one action at most
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                self._reset()
                return None
        return None

    def _victory_burst(self):
        v = self.victory
        r = v["rect"]
        self.fx_ui.emit(r.centerx, r.y + r.h * 0.18, 30, [(255, 230, 140), (255, 245, 200), (255, 190, 220)],
                        speed=(60, 160), life=(0.9, 1.6), size=(2, 4), drag=1.8, rise=-20, twinkle=True)

    def _tick_effects(self):
        now = pygame.time.get_ticks()
        dt = min(0.05, max(0.0, (now - self._prev_ticks) / 1000.0))   # clamp after pauses
        self._prev_ticks = now
        self.fx.update(dt)
        cx, cy = self._tile_px(_COORD_TO_IDX[(0, 0)])
        self.fx.tree_fireflies(dt, cx, cy - 8, HEX_SIZE * 1.1, HEX_SIZE * 0.7)
        for idx in [i for i, t in self._blooms.items() if now - t > BLOOM_TIME * 1000]:
            del self._blooms[idx]
        if self._victory_showing():
            self.fx_ui.update(dt)
            self._ui_spark -= dt
            if self._ui_spark <= 0:
                self._ui_spark = 0.07
                r = self.victory["rect"]
                import random as _r
                spots = [  # around the tree crown, the ribbon ends and the corner flowers
                    (r.centerx + _r.uniform(-0.17, 0.17) * r.w, r.y + _r.uniform(0.02, 0.24) * r.h),
                    (r.x + _r.uniform(0.18, 0.82) * r.w, r.y + _r.uniform(0.27, 0.45) * r.h),
                    (r.x + _r.choice([_r.uniform(0.04, 0.2), _r.uniform(0.8, 0.96)]) * r.w,
                     r.y + _r.uniform(0.2, 0.95) * r.h),
                ]
                x, y = _r.choice(spots)
                self.fx_ui.emit(x, y, 1, [(255, 236, 150), (255, 250, 220), (255, 200, 230)],
                                speed=(3, 12), life=(0.8, 1.5), size=(2, 4), rise=-14,
                                twinkle=True, drag=0.5)
        self._hover_spark -= dt
        if (self._renewal_targeting() and self._human_targeting()
                and self.hover_idx in self._valid_target_idxs and self._hover_spark <= 0):
            self._hover_spark = 0.18
            x, y = self._tile_px(self.hover_idx)
            self.fx.emit(x, y, 1, [C_RENEW_GLOW, (255, 255, 255)], speed=(5, 15),
                         life=(0.5, 0.8), rise=-20, twinkle=True, spread=18)

    def _tile_px(self, idx):
        q, r = _TILES[idx]
        return _hex_to_pixel(q, r, HEX_SIZE, self.cx, self.cy)

    def update(self, events):
        self._tick_effects()
        if self._victory_showing():
            return self._update_victory(events)       # nothing reaches the board or side buttons
        if self.b_home.clicked(events):  return "home"
        if self.b_exit.clicked(events):  return "exit"
        if self.b_rules.clicked(events): return "rules"

        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r and self.game_over:
                    self._reset()
                if e.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode

        if self.game_over:
            return None

        is_agent = self._agent_waiting or (self.turn_count % 2 == 1)
        if is_agent:
            self._handle_agent()
        else:
            self._handle_human(events)
        return None

    # ── Human ─────────────────────────────────────────────────────────────────
    def _handle_human(self, events):
        mx, my = pygame.mouse.get_pos()
        hq, hr = _pixel_to_hex(mx, my, HEX_SIZE, self.cx, self.cy)
        self.hover_idx = _COORD_TO_IDX.get((hq, hr))

        if self._targeting:
            for e in events:
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    coord = _pixel_to_hex(e.pos[0], e.pos[1], HEX_SIZE, self.cx, self.cy)
                    idx   = _COORD_TO_IDX.get(coord)
                    if idx is not None and idx in self._valid_target_idxs:
                        self._targeting = False
                        self._do_step(idx, "You", 0)
            return

        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                coord = _pixel_to_hex(e.pos[0], e.pos[1], HEX_SIZE, self.cx, self.cy)
                idx   = _COORD_TO_IDX.get(coord)
                if idx is not None:
                    self._do_step(idx, "You", 0)

        # ── Agent ─────────────────────────────────────────────────────────────────
    def _handle_agent(self):
        # Spirit targeting phase
        if self._targeting:
            if not self._agent_waiting:
                self._agent_waiting = True
                self._agent_tick = pygame.time.get_ticks()
                return
            if pygame.time.get_ticks() - self._agent_tick < self.TARGET_DELAY_MS:
                return
            self._agent_waiting = False

            if not self._valid_target_idxs:
                self._targeting = False
                return

            masks = self.env.action_masks()
            if AGENT_OK and AGENT is not None:
                action = int(AGENT.predict(self.obs, deterministic=True,
                                           action_masks=masks)[0])
                if not masks[action]:
                    action = int(np.random.choice(list(self._valid_target_idxs)))
            else:
                action = int(np.random.choice(list(self._valid_target_idxs)))

            self._targeting = False
            self._do_step(action, "Agent", 1)
            return

        # Placement phase
        if not self._agent_waiting:
            self._agent_waiting = True
            self._agent_tick = pygame.time.get_ticks()
            return
        if pygame.time.get_ticks() - self._agent_tick < self.AGENT_DELAY_MS:
            return
        self._agent_waiting = False

        if self.env is None:
            self.turn_count += 1
            return

        masks = self.env.action_masks()
        if not masks.any():
            self._log("Agent has no valid moves.")
            self.turn_count += 1
            return

        if AGENT_OK and AGENT is not None:
            action = int(AGENT.predict(self.obs, deterministic=True,
                                       action_masks=masks)[0])
            if not masks[action]:
                action = int(np.random.choice(np.where(masks)[0]))
        else:
            action = int(np.random.choice(np.where(masks)[0]))

        self._do_step(action, "Agent", 1)

    # ── Step ──────────────────────────────────────────────────────────────────
    def _do_step(self, action, label, player, _retry=0):
        if self.env is not None:
            is_placement  = self.env._pending_spirit is None
            placed_flower = self.cur_flower
            combos_before = set(self.env._activated_combos)

            obs, reward, terminated, truncated, info = self.env.step(action, player=player)
            self.obs  = obs
            self.info = info

            if info.get("invalid_action"):
                self._log(f"{label} -> INVALID tile {action}")
                if player == 1 and _retry < 5:
                    m     = self.env.action_masks()
                    valid = np.where(m)[0]
                    valid = valid[valid != action]
                    if len(valid):
                        self._do_step(int(np.random.choice(valid)), label, player, _retry + 1)
                return

            now = pygame.time.get_ticks()
            if is_placement:
                self._last_move[player] = action
                self._blooms[action] = now
                self.fx.placement(*self._tile_px(action), placed_flower)
            new_combos = self.env._activated_combos - combos_before
            if new_combos:
                idxs = {_COORD_TO_IDX[c] for key in new_combos for c in key if c in _COORD_TO_IDX}
                self._flash_combo = (idxs, now + FLASH_MS)
                for key in new_combos:
                    tiles = [_COORD_TO_IDX[c] for c in key if c in _COORD_TO_IDX]
                    centres = [self._tile_px(i) for i in tiles]
                    kind = int(self.board[tiles[0]]) if tiles else 0
                    if kind == 1:
                        self.fx.spirit_renewal(centres)
                    elif kind == 2:
                        self.fx.spirit_rain(centres)
                    elif kind == 3:
                        self.fx.spirit_blossom(centres)
            if not info.get("spirit_pending"):
                # Targeted spirits: only once the target has been resolved
                for sr in info.get("spirits_activated", []):
                    if sr.get("cleansed_tile") is not None:
                        idx = _COORD_TO_IDX.get(tuple(sr["cleansed_tile"]))
                        if idx is not None:
                            self.fx.cleanse(*self._tile_px(idx))
                    if sr.get("protected_tiles"):
                        pts = [self._tile_px(_COORD_TO_IDX[tuple(c)])
                               for c in sr["protected_tiles"] if tuple(c) in _COORD_TO_IDX]
                        if pts:
                            self.fx.protect(pts)
            spread = info.get("corruption_spread") or []
            if spread:
                spread_idxs = {_COORD_TO_IDX[c] for c in spread if c in _COORD_TO_IDX}
                self._flash_spread = (spread_idxs, now + FLASH_MS)
                for idx in spread_idxs:
                    x, y = self._tile_px(idx)
                    self.fx.corruption(x, y, _hex_corners(x, y, HEX_SIZE - 2))
                self._log(f"Corruption spread to {len(spread)} tile(s).")

            if info.get("spirit_pending"):
                self._targeting            = True
                self._valid_target_idxs    = set(info.get("valid_targets", []))
                sname                      = info.get("spirit_name", "Spirit")
                self._target_spirit_name   = sname
                self._target_hint          = ("Choose a corrupt tile to cleanse"
                                              if "Renewal" in sname else
                                              "Choose a fertile tile to protect")
                self._target_highlight_col = (C_TARGET_RENEWAL if "Renewal" in sname
                                              else C_TARGET_BLOSSOM)
                self._log(f"{label} placed a {FLOWER_NAME.get(placed_flower, 'flower')}: "
                          f"{sname}!")
                return

            self.turn_count += 1
            if self.turn_count % 2 == 0:
                self.round_num += 1
            if is_placement:
                msg = f"{label} placed a {FLOWER_NAME.get(placed_flower, 'flower')}"
            else:
                msg = f"{label} used the spirit"
            if self.debug_mode:
                msg += f"  (tile {action}, {reward:+.1f})"
            self._log(msg)
            spirits = info.get("spirits_activated", [])
            if spirits:
                self._notify_spirits(spirits, player)
            if terminated or truncated:
                self._finish(info)
        else:
            b = self.board.copy()
            if b[action] == 0:
                b[action] = self.cur_flower + 1
                self.obs[OBS_BOARD] = b
                self._log(f"{label} placed {FLOWER_NAME.get(self.cur_flower,'?')} on tile {action}")
            self.turn_count += 1

    def _finish(self, info):
        self.game_over = True
        reason = info.get("reason", "?")
        winner = info.get("winner", None)

        if reason == "win_p1":
            winner = 0
        elif reason == "win_p2":
            winner = 1
        elif reason == "win" and winner is None:
            winner = 0 if self.spirits_p1 >= self.spirits_p2 else 1

        sp1, sp2 = self._get_sp1(), self._get_sp2()
        self.winner = winner
        if winner == 0 and self.victory:
            self._victory_burst()

        if winner == 0:
            self.result_msg = f"You Win!  (Your spirits: {sp1}  Agent spirits: {sp2})  Press R to play again."
        elif winner == 1:
            self.result_msg = f"Agent Wins!  (Agent spirits: {sp2}  Your spirits: {sp1})  Press R to play again."
        elif reason == "loss_sacred_core":
            self.result_msg = "The Sacred Tree has fallen. Everyone loses.  Press R to play again."
        else:
            self.result_msg = f"Game Over ({reason}).  Press R to play again."

    def _log(self, msg):
        self.log.append(msg)
        if len(self.log) > 5:
            self.log.pop(0)

    # ── Draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        # Background
        if not hasattr(self, "_bg_scaled"):                  # scale the background once
            bg = self.assets.get("game_bg") or self.assets.get("bg")
            self._bg_scaled = pygame.transform.smoothscale(bg, (self.sw, self.sh)) if bg else None
        if self._bg_scaled:
            self.screen.blit(self._bg_scaled, (0, 0))
        else:
            self.screen.fill((200, 160, 160))

        self._draw_status_bar()
        self._draw_logo()
        self._draw_score_panel()
        self._draw_board()
        self._draw_targeting_banner()
        self._draw_flower_card()
        self._draw_buttons()
        self._draw_notifications()
        self._draw_log()

        if self._victory_showing():
            self._draw_victory()
        elif self.game_over:
            self._draw_overlay(self.result_msg)       # agent win / loss: existing overlay

    def _draw_victory(self):
        v = self.victory
        self.screen.blit(v["dim"], (0, 0))
        self.screen.blit(v["card"], v["rect"])

        # Final spirit counts from the game state (same values as the score panel)
        f, brown = v["font"], (75, 40, 18)
        left  = f.render(f"Your spirits: {self._get_sp1()}", True, brown)
        right = f.render(f"Agent spirits: {self._get_sp2()}", True, brown)
        gap = int(v["rect"].w * 0.07)
        y = v["score_y"]
        cx = v["rect"].centerx
        self.screen.blit(left, left.get_rect(midright=(cx - gap // 2, y)))
        self.screen.blit(right, right.get_rect(midleft=(cx + gap // 2, y)))
        d = max(4, gap // 7)                              # small diamond between the counts
        pygame.draw.polygon(self.screen, (120, 140, 40),
                            [(cx, y - d), (cx + d, y), (cx, y + d), (cx - d, y)])

        mouse = pygame.mouse.get_pos()
        for key in ("play", "home"):
            b = v[key]
            if b["rect"].collidepoint(mouse):
                self.screen.blit(b["hover"], b["hover"].get_rect(center=b["rect"].center))
            else:
                self.screen.blit(b["normal"], b["rect"])

        self.fx_ui.draw_under(self.screen)
        self.fx_ui.draw_over(self.screen)

    def _draw_status_bar(self):
        pygame.draw.rect(self.screen, C_STATUS_BG, (110, 5, self.sw - 115, 38))
        is_agent = self._agent_waiting or (self.turn_count % 2 == 1)
        if self._targeting:
            cp = "Choose a target tile!"
        elif is_agent:
            cp = "Agent (P2)"
        else:
            cp = "You (P1)"

        txt = (f"Turn: {cp}        Round {self.round_num}        "
               f"Season: {SEASON_NAME.get(self.season, '?')}")
        self.screen.blit(self.f_md.render(txt, True, C_LIGHT), (120, 12))

        spread = self._spread_forecast()
        if spread:
            n, tiles = spread
            when = "next turn" if n == 1 else f"in {n} turns"
            st = self.f_md.render(f"Corruption spreads {when} (+{tiles})", True, (255, 200, 170))
            self.screen.blit(st, st.get_rect(right=self.sw - 20, top=12))

    def _spread_forecast(self):
        """(turns until corruption next spreads, tiles it will spread)."""
        if self.env is None:
            return None
        t = self.env._turn
        n = SPREAD_EVERY_N_TURNS - (t % SPREAD_EVERY_N_TURNS)
        return n, spread_rate_for_season(get_season(t + n))

    def _panel(self, rect):
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        box.fill(C_PANEL)
        self.screen.blit(box, rect.topleft)
        pygame.draw.rect(self.screen, C_PANEL_EDGE, rect, 2, border_radius=8)

    def _draw_score_panel(self):
        """Race-to-target bars: score = spirits x 10 - corrupted tiles."""
        rect = pygame.Rect(self.sw - 250, 52, 230, 196)
        self._panel(rect)
        x0 = rect.x + 14
        title = self.f_md.render(f"Race to {WIN_SCORE_TARGET}", True, C_DARK)
        self.screen.blit(title, (x0, rect.y + 8))
        rule = self.f_idx.render("score = spirits x 10 - corrupted tiles", True, (90, 60, 30))
        self.screen.blit(rule, (x0, rect.y + 32))

        rows = [("You (P1)", self._get_sp1(), C_P1), ("Agent (P2)", self._get_sp2(), C_P2)]
        bar_w = rect.w - 28
        for k, (name, sp, col) in enumerate(rows):
            y = rect.y + 56 + k * 64
            score = sp * 10 - self.corruption
            self.screen.blit(self.f_sm.render(name, True, C_DARK), (x0, y))
            val = self.f_sm.render(f"{score} / {WIN_SCORE_TARGET}", True, C_DARK)
            self.screen.blit(val, val.get_rect(right=x0 + bar_w, top=y))
            bar = pygame.Rect(x0, y + 22, bar_w, 14)
            pygame.draw.rect(self.screen, (225, 210, 185), bar, border_radius=7)
            frac = max(0.0, min(1.0, score / WIN_SCORE_TARGET))
            if frac > 0:
                pygame.draw.rect(self.screen, col, (bar.x, bar.y, max(8, int(bar.w * frac)), bar.h),
                                 border_radius=7)
            pygame.draw.rect(self.screen, C_PANEL_EDGE, bar, 1, border_radius=7)
            info = self.f_idx.render(f"{sp} spirits  -  {self.corruption} corrupted", True, (90, 60, 30))
            self.screen.blit(info, (x0, y + 40))

    def _draw_logo(self):
        if self.logo_img:
            self.screen.blit(self.logo_img, (5, 0))

    def _draw_board(self):
        bd = self.board
        is_human = not (self._agent_waiting or (self.turn_count % 2 == 1))

        for i, (q, r) in enumerate(_TILES):
            px, py   = _hex_to_pixel(q, r, HEX_SIZE, self.cx, self.cy)
            corners  = _hex_corners(px, py, HEX_SIZE - 2)
            tile_val = int(bd[i])

            is_tgt   = self._targeting and i in self._valid_target_idxs
            is_hov   = i == self.hover_idx

            if is_tgt and is_hov:
                col = C_TARGET_HOVER
            elif is_tgt:
                col = self._target_highlight_col
            elif is_hov and tile_val == 0 and is_human and not self._targeting and not self.game_over:
                col = C_HOVER
            else:
                col = TILE_COLOR.get(tile_val, (76, 153, 0))

            if tile_val in (1, 2, 3) and not is_tgt:
                col = C_FERTILE   # flowers sit on fertile soil

            tex = (self.tex_gravel if tile_val == 4 else
                   self.tex_grass if tile_val in (0, 1, 2, 3) else None)
            if tex:
                self.screen.blit(tex, tex.get_rect(center=(px, py)))
                # Highlights are laid over the artwork so the terrain still shows
                renewal_tgt = is_tgt and self._renewal_targeting()
                if renewal_tgt:
                    ov = None                      # drawn later by _draw_renewal_targets
                elif is_tgt and is_hov:
                    ov = self._hex_overlay(C_TARGET_HOVER, 150)
                elif is_tgt:
                    ov = self._hex_overlay(self._target_highlight_col, 115)
                elif col == C_HOVER:
                    ov = self._hex_overlay(C_HOVER, 140)
                else:
                    ov = None
                if ov:
                    self.screen.blit(ov, ov.get_rect(center=(px, py)))
                if is_tgt and not renewal_tgt:
                    pygame.draw.polygon(self.screen, C_OUTLINE, corners, 3)
            else:
                pygame.draw.polygon(self.screen, col, corners)
                pygame.draw.polygon(self.screen, C_OUTLINE, corners, 3 if is_tgt else 2)

        # Tile-level effects (corruption darkening, placement glows), then the tree
        self.fx.draw_under(self.screen)
        self._draw_tree()

        # Flowers. On the six tiles touching the Sacred Tree, the flower moves outward
        # and shrinks a little so it sits fully in the part of the tile left visible.
        now = pygame.time.get_ticks()
        for i, (q, r) in enumerate(_TILES):
            tile_val = int(bd[i])
            px, py = _hex_to_pixel(q, r, HEX_SIZE, self.cx, self.cy)
            if tile_val in (1, 2, 3):
                start = self._blooms.get(i)
                scale = bloom_scale((now - start) / 1000.0) if start is not None else 1.0
                if _ring_dist(q, r) == 1 and self.tree_big and TREE_SCALE > 1.0:
                    px, py = self._nudge_out(px, py)
                    scale *= NEIGHBOUR_FLOWER_SCALE
                img = self._flower_img(tile_val - 1, scale)
                if img:
                    self.screen.blit(img, img.get_rect(center=(px, py)))
                else:
                    self._draw_flower_at(px, py, int((HEX_SIZE // 2 - 4) * scale), tile_val - 1)
            if self.debug_mode:
                s = self.f_idx.render(str(i), True, C_WHITE if tile_val == 4 else C_DARK)
                self.screen.blit(s, s.get_rect(center=(px, py)))

        self._draw_highlights()
        if self._targeting:
            self._dim_non_targets()
            if self._renewal_targeting():
                self._draw_renewal_targets()
        self.fx.draw_over(self.screen)

    # ── Spirit targeting visuals (driven only by the env's valid_targets) ────────
    def _renewal_targeting(self):
        return self._targeting and "Renewal" in self._target_spirit_name

    def _human_targeting(self):
        return self._targeting and not (self._agent_waiting or self.turn_count % 2 == 1)

    def _dim_non_targets(self):
        """Slightly darken every tile that is not a valid target (Sacred Tree excluded)."""
        if TARGET_DIM_ALPHA <= 0:
            return
        dim = self._hex_overlay((0, 0, 0), TARGET_DIM_ALPHA)
        bd = self.board
        for i in range(len(_TILES)):
            if i in self._valid_target_idxs or int(bd[i]) == 5:
                continue
            self.screen.blit(dim, dim.get_rect(center=self._tile_px(i)))

    def _renewal_glow(self):
        """Soft magenta halo following the hex outline (built once, then cached)."""
        if getattr(self, "_renew_glow_surf", None) is None:
            size = HEX_SIZE * 2 + 28
            c = size / 2
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            for width, alpha in ((16, 45), (11, 80), (7, 130), (4, 190)):
                pygame.draw.polygon(s, (*C_RENEW_GLOW, alpha), _hex_corners(c, c, HEX_SIZE), width)
            self._renew_glow_surf = s
        return self._renew_glow_surf

    def _draw_renewal_targets(self):
        """Magenta outlines with a gentle 1.5 s pulse on every tile the env says can be cleansed."""
        t = pygame.time.get_ticks() / 1000.0
        w = 0.5 + 0.5 * math.sin(t * math.tau / RENEW_PULSE_S)
        glow = self._renewal_glow()
        hover_ok = self._human_targeting()
        self._renewal_drawn = set()                       # for testing: what was highlighted
        for idx in self._valid_target_idxs:
            x, y = self._tile_px(idx)
            hov = hover_ok and idx == self.hover_idx
            glow.set_alpha(int((150 + 105 * w) if hov else (70 + 110 * w)))
            self.screen.blit(glow, glow.get_rect(center=(x, y)))
            corners = _hex_corners(x, y, HEX_SIZE - 2)
            if hov:
                line = tuple(min(255, c + 60) for c in C_RENEW_LINE)
                pygame.draw.polygon(self.screen, line, corners, 5)
                pygame.draw.polygon(self.screen, C_WHITE, _hex_corners(x, y, HEX_SIZE - 7), 2)
            else:
                # Never fades below a clearly visible magenta
                k = 0.8 + 0.2 * w
                line = tuple(int(c * k) for c in C_RENEW_LINE)
                pygame.draw.polygon(self.screen, line, corners, 4)
            self._renewal_drawn.add(idx)

    def _nudge_out(self, px, py):
        """Move a point on a tile next to the Sacred Tree into the middle of the
        strip of that tile that the tree badge does not cover."""
        ox, oy = self._tile_px(_COORD_TO_IDX[(0, 0)])
        dx, dy = px - ox, py - oy
        d = math.hypot(dx, dy) or 1.0
        apothem = HEX_SIZE * math.sqrt(3) / 2
        badge_edge = apothem * TREE_SCALE          # how far the badge reaches
        outer_edge = d + apothem                   # far edge of the neighbour tile
        target = (badge_edge + outer_edge) / 2
        return int(ox + dx / d * target), int(oy + dy / d * target)

    def _flower_img(self, flower, scale):
        """Board flower at a given scale; scaled copies are cached, never distorted."""
        if flower not in self.flower_board:
            return None
        if abs(scale - 1.0) < 0.01:
            return self.flower_board[flower]
        key = (flower, round(scale * 50))
        if key not in self._bloom_cache:
            self._bloom_cache[key] = self._fit(self.flower_src[flower],
                                               int(HEX_SIZE * 1.55 * key[1] / 50))
        return self._bloom_cache[key]

    def _draw_tree(self):
        """Sacred Tree: slow golden glow (about 3.5 s per cycle) behind a stationary tree."""
        x, y = self._tile_px(_COORD_TO_IDX[(0, 0)])
        t = pygame.time.get_ticks() / 1000.0
        for colour, radius, lo, hi, shift in (((255, 205, 90), 96, 90, 185, 0.0),
                                              ((170, 240, 190), 74, 30, 85, 1.2),
                                              ((255, 245, 170), 56, 70, 160, 0.6)):
            w = 0.5 + 0.5 * math.sin(t * math.tau / 3.5 + shift)
            spr = self.fx.glow_sprite(colour, radius)
            spr.set_alpha(int(lo + (hi - lo) * w))
            self.screen.blit(spr, spr.get_rect(center=(x, y)))
        if self.tree_big:
            self.screen.blit(self.tree_big, self.tree_big.get_rect(center=(x, y)))
            # Shiny: a soft golden light that brightens and fades on the tree itself
            shine = int(8 + 42 * (0.5 + 0.5 * math.sin(t * math.tau / 3.5 + 0.6)))
            self.tree_shine.set_alpha(shine)
            self.screen.blit(self.tree_shine, self.tree_shine.get_rect(center=(x, y)))
        elif self.tree_tile:
            self.screen.blit(self.tree_tile, self.tree_tile.get_rect(center=(x, y)))
        else:
            self._draw_sacred_tree(x, y)

    def _draw_highlights(self):
        now   = pygame.time.get_ticks()
        pulse = 0.5 + 0.5 * math.sin(now / 140.0)

        def ring(idx, colour, width):
            q, r = _TILES[idx]
            px, py = _hex_to_pixel(q, r, HEX_SIZE, self.cx, self.cy)
            pygame.draw.polygon(self.screen, colour, _hex_corners(px, py, HEX_SIZE - 3), width)

        # Last placement by each player
        for player, idx in self._last_move.items():
            if idx is not None and int(self.board[idx]) in (1, 2, 3):
                ring(idx, C_P1 if player == 0 else C_P2, 4)

        # Combo just formed: pulsing gold outline
        idxs, until = self._flash_combo
        if now < until:
            for idx in idxs:
                ring(idx, C_COMBO, 3 + int(3 * pulse))

        # Tiles corruption just reached: pulsing red outline
        idxs, until = self._flash_spread
        if now < until:
            for idx in idxs:
                ring(idx, C_SPREAD, 3 + int(3 * pulse))

    def _draw_flower_at(self, cx, cy, radius, flower):
        """Six petals with a darker edge and a centre, so flowers read as flowers."""
        fc, edge, centre = (FLOWER_COLOR.get(flower, (180, 180, 180)),
                            FLOWER_EDGE.get(flower, C_OUTLINE),
                            FLOWER_CENTRE.get(flower, C_WHITE))
        pr = max(3, int(radius * 0.48))
        for k in range(6):
            ang = math.radians(60 * k + 30)
            p = (int(cx + radius * 0.62 * math.cos(ang)), int(cy + radius * 0.62 * math.sin(ang)))
            pygame.draw.circle(self.screen, fc, p, pr)
            pygame.draw.circle(self.screen, edge, p, pr, 1)
        pygame.draw.circle(self.screen, centre, (cx, cy), max(3, int(radius * 0.38)))
        pygame.draw.circle(self.screen, edge, (cx, cy), max(3, int(radius * 0.38)), 1)

    def _draw_sacred_tree(self, cx, cy):
        pygame.draw.rect(self.screen, (110, 70, 30), (cx - 3, cy - 2, 6, 15))
        for dx, dy, rad in ((0, -10, 9), (-8, -4, 7), (8, -4, 7)):
            pygame.draw.circle(self.screen, (60, 140, 50), (cx + dx, cy + dy), rad)
            pygame.draw.circle(self.screen, (35, 90, 30), (cx + dx, cy + dy), rad, 1)

    def _draw_targeting_banner(self):
        if not self._targeting:
            return
        bw, bh = 540, 52
        bx, by = self.sw // 2 - bw // 2, 55
        box = pygame.Surface((bw, bh), pygame.SRCALPHA)
        box.fill((255, 160, 40, 235))
        self.screen.blit(box, (bx, by))
        pygame.draw.rect(self.screen, (160, 90, 20), (bx, by, bw, bh), 2, border_radius=7)
        t = self.f_md.render(f"{self._target_spirit_name} Activated!", True, (40, 20, 5))
        h = self.f_sm.render(self._target_hint, True, (70, 35, 5))
        self.screen.blit(t, t.get_rect(centerx=self.sw // 2, top=by + 5))
        self.screen.blit(h, h.get_rect(centerx=self.sw // 2, top=by + 30))

    def _draw_flower_card(self):
        """Draw the ornate flower card on the left, with the flower inside the hex frame."""
        card_x = 20
        card_y = 115

        if self.card_img:
            self.screen.blit(self.card_img, (card_x, card_y))
            # Draw flower inside the hex frame on the card
            # The hex frame is roughly centred at 50% x, 48% y of the card
            fcx = card_x + self.card_w // 2
            fcy = card_y + int(self.card_h * 0.5)
            self._draw_flower_graphic(fcx, fcy, radius=30)
            # Flower name below the frame
            fl  = self.cur_flower
            nm  = self.f_card.render(FLOWER_NAME.get(fl, "?"), True, (60, 38, 10))
            self.screen.blit(nm, nm.get_rect(centerx=card_x + self.card_w // 2,
                                              top=card_y + int(self.card_h * 0.68)))
            # Hint text
            is_agent = self._agent_waiting or (self.turn_count % 2 == 1)
            if self._targeting and not is_agent:
                hint = "Pick a highlighted tile"
            elif self._targeting:
                hint = "Agent choosing..."
            elif is_agent:
                hint = "Agent is thinking..."
            else:
                hint = "Click a green tile"
            hs = self.f_sm.render(hint, True, (80, 50, 25))
            self.screen.blit(hs, hs.get_rect(centerx=card_x + self.card_w // 2,
                                              top=card_y + int(self.card_h * 0.83)))
        else:
            # Fallback plain card
            pygame.draw.rect(self.screen, (240, 220, 200), (card_x, card_y, 175, 275), border_radius=12)
            pygame.draw.rect(self.screen, (80, 50, 30), (card_x, card_y, 175, 275), 3, border_radius=12)
            fcx = card_x + 87
            fcy = card_y + 138
            self._draw_flower_graphic(fcx, fcy, radius=36)
            fl  = self.cur_flower
            nm  = self.f_md.render(FLOWER_NAME.get(fl, "?"), True, C_DARK)
            self.screen.blit(nm, nm.get_rect(centerx=card_x + 87, top=card_y + 195))

    def _draw_flower_graphic(self, cx, cy, radius=36):
        """Draw a simple flower at (cx, cy)."""
        img = self.flower_card.get(self.cur_flower)
        if img:
            self.screen.blit(img, img.get_rect(center=(cx, cy)))
        else:
            self._draw_flower_at(cx, cy, int(radius * 1.4), self.cur_flower)

    def _draw_buttons(self):
        for b in (self.b_rules, self.b_home, self.b_exit):
            b.draw(self.screen)

    def _draw_notifications(self):
        now = pygame.time.get_ticks()
        self._notifications = [(m, e, c) for m, e, c in self._notifications if e > now]
        if not self._notifications:
            return

        nw, nh = 500, 38
        nx     = self.sw // 2 - nw // 2
        ny_top = 65

        for i, (msg, _, colour) in enumerate(self._notifications):
            ny  = ny_top + i * (nh + 6)
            box = pygame.Surface((nw, nh), pygame.SRCALPHA)
            box.fill((*colour, 220))
            self.screen.blit(box, (nx, ny))
            pygame.draw.rect(self.screen, (80, 50, 20), (nx, ny, nw, nh), 2, border_radius=6)
            txt = self.f_notify.render(msg, True, (40, 20, 5))
            self.screen.blit(txt, txt.get_rect(midleft=(nx + 12, ny + nh // 2)))

    def _draw_log(self):
        if not self.log:
            return
        rect = pygame.Rect(20, self.sh - 118, 310, 110)
        self._panel(rect)
        for i, msg in enumerate(self.log):
            self.screen.blit(self.f_card_sm.render(msg, True, C_DARK),
                             (rect.x + 10, rect.y + 6 + i * 20))

    def _draw_overlay(self, msg):
        ov = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 155))
        self.screen.blit(ov, (0, 0))
        s  = self.f_lg.render(msg, True, (255, 255, 180))
        s2 = self.f_sm.render("Press R to restart   |   Home button to return to menu",
                               True, (210, 210, 170))
        self.screen.blit(s,  s.get_rect(center=(self.sw // 2, self.sh // 2)))
        self.screen.blit(s2, s2.get_rect(center=(self.sw // 2, self.sh // 2 + 46)))