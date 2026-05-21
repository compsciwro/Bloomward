# -*- coding: utf-8 -*-
"""
Bloomward - rules_screen.py
Rules screen using game_bg.png background and btnback.png button.
"""

import pygame
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _load(name):
    path = os.path.join(ASSETS_DIR, name)
    return pygame.image.load(path).convert_alpha() if os.path.exists(path) else None


def _scale_to_width(img, w):
    if img is None:
        return None
    ratio = w / img.get_width()
    return pygame.transform.smoothscale(img, (w, int(img.get_height() * ratio)))


RULES = [
    ("BLOOMWARD — HOW TO PLAY", True),
    ("", False),
    ("OBJECTIVE", True),
    ("Race to the highest score before the board is consumed.", False),
    ("Score = (spirits activated x 10) minus corruption tiles.", False),
    ("First to reach a score of 120 wins.", False),
    ("Protect the Sacred Tree at the centre of the board.", False),
    ("", False),
    ("YOUR TURN", True),
    ("1. A flower is automatically drawn for you at the start of your turn.", False),
    ("2. Click any green (fertile) tile to place your flower.", False),
    ("3. If 3 matching flowers form a triangle on the grid, a spirit activates!", False),
    ("", False),
    ("FLOWERS AND SPIRITS", True),
    ("Sunflower  ->  Spirit of Renewal  (heals and cleanses corrupted tiles)", False),
    ("Tulip      ->  Spirit of Rain     (growth and support effects)", False),
    ("Blossom    ->  Spirit of Blossom  (defence and corruption resistance)", False),
    ("9-flower Sunflower cluster  ->  Spirit of Rebalance  (major board reset)", False),
    ("", False),
    ("FLOWER DRAW RATES", True),
    ("50% Sunflower   |   30% Tulip   |   20% Blossom", False),
    ("", False),
    ("CORRUPTION", True),
    ("Corruption starts at the outer ring and spreads inward each round.", False),
    ("Flowers on tiles protect those tiles from being corrupted.", False),
    ("If corruption reaches the Sacred Tree, all players lose immediately.", False),
    ("", False),
    ("SEASONS  (4 rounds each)", True),
    ("Spring  — 2 connected matching flowers can activate a spirit.", False),
    ("Rain    — Sunflowers cannot be placed.", False),
    ("Autumn  — Corruption spreads faster.", False),
    ("Winter  — Frozen tiles appear and cannot be used.", False),
    ("", False),
    ("WIN CONDITION", True),
    ("Score = spirits x 10 minus current corruption count.", False),
    ("First player to reach a score of 120 wins.", False),
    ("The win is checked after every action.", False),
    ("", False),
    ("LOSE CONDITION", True),
    ("Corruption reaches the Sacred Tree.", False),
    ("No valid moves remain for any player.", False),
    ("", False),
    ("CONTROLS", True),
    ("Click a green tile to place your flower.", False),
    ("Press D to toggle debug mode (shows tile indices).", False),
    ("Press R after game over to restart.", False),
    ("ESC or Back button to return.", False),
]

C_H     = (50, 25, 5)
C_B     = (60, 35, 10)
C_PANEL = (255, 245, 225, 210)


class RulesScreen:
    def __init__(self, screen, assets, from_screen="home"):
        self.screen      = screen
        self.assets      = assets
        self.from_screen = from_screen
        self.sw, self.sh = screen.get_size()

        self.f_h   = pygame.font.SysFont("Georgia", 17, bold=True)
        self.f_b   = pygame.font.SysFont("Georgia", 15)

        # Load back button
        self.btn_back = _scale_to_width(_load("btnback.png"), 200)
        bw = self.btn_back.get_width() if self.btn_back else 200
        bh = self.btn_back.get_height() if self.btn_back else 55
        self.back_rect = pygame.Rect(self.sw // 2 - bw // 2, self.sh - bh - 15, bw, bh)

        self.panel_x = self.sw // 2 - 370
        self.panel_w = 740
        self.scroll  = 0
        self.hover_back = False

    def update(self, events):
        mx, my = pygame.mouse.get_pos()
        self.hover_back = self.back_rect.collidepoint(mx, my)

        for e in events:
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                return "back"
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1 and self.back_rect.collidepoint(e.pos):
                    return "back"
                if e.button == 4:
                    self.scroll = max(0, self.scroll - 28)
                if e.button == 5:
                    self.scroll += 28
        return None

    def draw(self):
        # Background — use game_bg, fall back to regular bg
        bg = self.assets.get("game_bg") or self.assets.get("bg")
        if bg:
            self.screen.blit(
                pygame.transform.smoothscale(bg, (self.sw, self.sh)), (0, 0))
        else:
            self.screen.fill((210, 185, 175))

        # Semi-transparent panel
        panel_h = self.sh - self.back_rect.height - 30
        panel   = pygame.Surface((self.panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(C_PANEL)
        self.screen.blit(panel, (self.panel_x, 15))
        pygame.draw.rect(self.screen, (160, 120, 60),
                         (self.panel_x, 15, self.panel_w, panel_h), 2, border_radius=10)

        # Rules text
        # Set a clip region so text never draws outside the panel
        clip_rect = pygame.Rect(self.panel_x, 15, self.panel_w, panel_h - 10)
        self.screen.set_clip(clip_rect)

        y = 30 - self.scroll
        for text, is_header in RULES:
            if y > panel_h:
                break
            if y > 10:
                font   = self.f_h if is_header else self.f_b
                colour = C_H if is_header else C_B
                surf   = font.render(text, True, colour)
                self.screen.blit(surf, (self.panel_x + 28, y))
            y += 30 if is_header else 24

        # Remove the clip so rest of screen draws normally
        self.screen.set_clip(None)

        # Back button
        if self.btn_back:
            img = self.btn_back.copy()
            if self.hover_back:
                img.fill((20, 20, 10, 0), special_flags=pygame.BLEND_RGB_ADD)
            self.screen.blit(img, self.back_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (75, 105, 40), self.back_rect, border_radius=8)
            f = pygame.font.SysFont("Georgia", 18, bold=True)
            s = f.render("Back", True, (255, 245, 210))
            self.screen.blit(s, s.get_rect(center=self.back_rect.center))
