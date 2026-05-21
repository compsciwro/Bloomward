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

# ── RL stack ──────────────────────────────────────────────────────────────────
try:
    from bloomward_env import BloomwardEnv
    ENV_OK = True
except ImportError:
    ENV_OK = False
    print("[game_screen] BloomwardEnv not found - running in demo mode.")

try:
    from sb3_contrib import MaskablePPO
    AGENT, AGENT_OK = None, False   # skip until retrained
except ImportError:
    AGENT, AGENT_OK = None, False

# ── Colours ───────────────────────────────────────────────────────────────────
C_STATUS_BG  = (45, 90, 35)
C_HOVER      = (215, 215, 80)
C_OUTLINE    = (25, 25, 25)
C_WHITE      = (255, 255, 255)
C_DARK       = (50, 25, 5)
C_LIGHT      = (255, 245, 220)

C_TARGET_RENEWAL = (220, 100, 60)
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

FLOWER_COLOR = {0: (255, 200, 0), 1: (230, 80, 110), 2: (180, 150, 255)}
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

        self.hover_idx  = None
        self.debug_mode = False
        self._reset()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset(self):
        if ENV_OK:
            self.env = BloomwardEnv(render_mode=None)
            self.obs, self.info = self.env.reset()
        else:
            self.env = None
            self.obs = self._demo_obs()
            self.info = {}

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
        if len(self._notifications) > 4:
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
    def update(self, events):
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
        if self._targeting:
            if not self._agent_waiting:
                self._agent_waiting = True
                self._agent_tick = pygame.time.get_ticks()
                return
            if pygame.time.get_ticks() - self._agent_tick < self.TARGET_DELAY_MS:
                return
            self._agent_waiting = False
            if self._valid_target_idxs:
                self._targeting = False
                self._do_step(int(np.random.choice(list(self._valid_target_idxs))), "Agent", 1)
            else:
                self._targeting = False
            return

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

        if AGENT_OK and AGENT is not None:
            masks  = self.env.action_masks() if hasattr(self.env, "action_masks") else None
            action = int(AGENT.predict(self.obs, deterministic=True, action_masks=masks)[0])
        else:
            valid  = np.where(self.env.action_masks())[0] if hasattr(self.env, "action_masks") else np.where(self.board == 0)[0]
            action = int(np.random.choice(valid)) if len(valid) else 0

        self._do_step(action, "Agent", 1)

    # ── Step ──────────────────────────────────────────────────────────────────
    def _do_step(self, action, label, player):
        if self.env is not None:
            obs, reward, terminated, truncated, info = self.env.step(action, player=player)
            self.obs  = obs
            self.info = info

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
                self._log(f"{sname} activated - choose a target!")
                return

            self.turn_count += 1
            if self.turn_count % 2 == 0:
                self.round_num += 1
            self._log(f"{label} -> tile {action}  ({reward:+.1f})")
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
        if winner is None and reason == "win":
            winner = 0 if self.spirits_p1 >= self.spirits_p2 else 1
        sp1, sp2 = self._get_sp1(), self._get_sp2()
        if reason == "win":
            if winner == 0:
                self.result_msg = f"You Win!  (Your spirits: {sp1}  Agent spirits: {sp2})  Press R to play again."
            else:
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
        bg = self.assets.get("game_bg") or self.assets.get("bg")
        if bg:
            self.screen.blit(pygame.transform.smoothscale(bg, (self.sw, self.sh)), (0, 0))
        else:
            self.screen.fill((200, 160, 160))

        self._draw_status_bar()
        self._draw_logo()
        self._draw_board()
        self._draw_targeting_banner()
        self._draw_flower_card()
        self._draw_buttons()
        self._draw_notifications()
        self._draw_log()

        if self.game_over:
            self._draw_overlay(self.result_msg)

    def _draw_status_bar(self):
        pygame.draw.rect(self.screen, C_STATUS_BG, (110, 5, self.sw - 115, 38))
        is_agent = self._agent_waiting or (self.turn_count % 2 == 1)
        if self._targeting:
            cp = "Choose a target tile!"
        elif is_agent:
            cp = "Agent (P2)"
        else:
            cp = "You (P1)"

        sp1_score = self._get_sp1() * 10 - self.corruption
        sp2_score = self._get_sp2() * 10 - self.corruption

        txt = (f"Turn: {cp}          Round: {self.round_num}          "
               f"Season: {SEASON_NAME.get(self.season, '?')}          "
               f"Corruption: {self.corruption}          "
               f"P1 Score: {sp1_score}          P2 Score: {sp2_score}")
        self.screen.blit(self.f_md.render(txt, True, C_LIGHT), (120, 12))

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

            pygame.draw.polygon(self.screen, col, corners)
            pygame.draw.polygon(self.screen, C_OUTLINE, corners, 3 if is_tgt else 2)

            if tile_val in (1, 2, 3):
                fc = TILE_COLOR[tile_val]
                r2 = HEX_SIZE // 2 - 5
                for k in range(6):
                    ang = math.radians(60 * k)
                    pygame.draw.circle(self.screen, fc,
                                       (int(px + r2 * math.cos(ang)), int(py + r2 * math.sin(ang))),
                                       r2 // 2 - 1)
                pygame.draw.circle(self.screen, fc, (px, py), r2 - 4)
                pygame.draw.circle(self.screen, C_WHITE, (px, py), r2 // 2 - 2)

            if self.debug_mode:
                s = self.f_idx.render(str(i), True, C_WHITE if tile_val == 4 else C_DARK)
                self.screen.blit(s, s.get_rect(center=(px, py)))

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
        fl = self.cur_flower
        fc = FLOWER_COLOR.get(fl, (180, 180, 180))
        for k in range(6):
            ang = math.radians(60 * k)
            pygame.draw.circle(self.screen, fc,
                               (int(cx + radius * math.cos(ang)),
                                int(cy + radius * math.sin(ang))),
                               radius // 2)
        pygame.draw.circle(self.screen, fc, (cx, cy), int(radius * 0.6))
        pygame.draw.circle(self.screen, C_WHITE, (cx, cy), int(radius * 0.25))

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
        lx, ly = 25, self.sh - 120
        for i, msg in enumerate(self.log):
            self.screen.blit(self.f_sm.render(msg, True, C_DARK), (lx, ly + i * 22))

    def _draw_overlay(self, msg):
        ov = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 155))
        self.screen.blit(ov, (0, 0))
        s  = self.f_lg.render(msg, True, (255, 255, 180))
        s2 = self.f_sm.render("Press R to restart   |   Home button to return to menu",
                               True, (210, 210, 170))
        self.screen.blit(s,  s.get_rect(center=(self.sw // 2, self.sh // 2)))
        self.screen.blit(s2, s2.get_rect(center=(self.sw // 2, self.sh // 2 + 46)))
