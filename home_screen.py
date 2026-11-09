# -*- coding: utf-8 -*-
"""
Bloomward - home_screen.py
Main menu using the new Bloomward UI assets.
"""

import pygame
import os
import random

from effects import EffectsManager

# Sparkle colours chosen to show up on the cream background
HOME_GOLD = [(225, 150, 20), (240, 175, 40), (210, 135, 15)]
HOME_PINK = [(230, 80, 145), (240, 115, 170)]

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _load(name):
    path = os.path.join(ASSETS_DIR, name)
    return pygame.image.load(path).convert_alpha() if os.path.exists(path) else None


def _scale_to_width(img, w):
    if img is None:
        return None
    ratio = w / img.get_width()
    return pygame.transform.smoothscale(img, (w, int(img.get_height() * ratio)))


class HomeScreen:
    def __init__(self, screen, assets):
        self.screen = screen
        self.assets = assets
        self.sw, self.sh = screen.get_size()

        # Home-specific background with the Sacred Tree
        self.bg = _load("home_bg.png")

        # Buttons from assets dict (pre-scaled to 280px in main.py)
        self.btn_imgs = {
            "play":  assets.get("hbtn_play"),
            "rules": assets.get("hbtn_rules"),
            "exit":  assets.get("hbtn_exit"),
        }
        self.btn_labels = {"play": "Play Game", "rules": "Rules", "exit": "Exit"}
        self.f_btn = pygame.font.SysFont("Georgia", 22, bold=True)

        self._build_rects()
        self.hover = None

        # Sparkles (visual only)
        self.fx = EffectsManager()
        self._prev_ticks = pygame.time.get_ticks()
        self._t_tree = self._t_air = self._t_petal = self._t_btn = 0.0
        self._bg_scaled = None

    def _build_rects(self):
        cx  = self.sw // 2
        bw  = 280
        sample = next((v for v in self.btn_imgs.values() if v), None)
        bh  = sample.get_height() if sample else 60
        gap = 16
        y0  = int(self.sh * 0.50)

        self.rects = {
            "play":  pygame.Rect(cx - bw // 2, y0,                   bw, bh),
            "rules": pygame.Rect(cx - bw // 2, y0 + bh + gap,        bw, bh),
            "exit":  pygame.Rect(cx - bw // 2, y0 + (bh + gap) * 2,  bw, bh),
        }

    def _tick_sparkles(self):
        now = pygame.time.get_ticks()
        dt = min(0.05, max(0.0, (now - self._prev_ticks) / 1000.0))
        self._prev_ticks = now
        self.fx.update(dt)
        w, h = self.sw, self.sh

        # Twinkles around the Sacred Tree in the logo
        self._t_tree -= dt
        if self._t_tree <= 0:
            self._t_tree = 0.12
            self.fx.emit(random.uniform(w * 0.38, w * 0.62), random.uniform(h * 0.06, h * 0.36), 1,
                         HOME_GOLD + HOME_PINK[:1], speed=(2, 8), life=(1.0, 1.8),
                         size=(2, 4), rise=-8, twinkle=True, drag=0.4, shape="star")

        # Golden motes drifting up through the air, mostly at the sides
        self._t_air -= dt
        if self._t_air <= 0:
            self._t_air = 0.22
            x = random.choice([random.uniform(0, w * 0.32), random.uniform(w * 0.68, w)])
            self.fx.emit(x, random.uniform(h * 0.3, h), 1, HOME_GOLD, speed=(2, 6),
                         life=(3.0, 5.0), size=(2, 3), rise=-14, twinkle=True, drag=0.0, sway=8,
                         shape="star")

        # A few pink petal sparkles floating down
        self._t_petal -= dt
        if self._t_petal <= 0:
            self._t_petal = 0.6
            self.fx.emit(random.uniform(0, w), -10, 1, HOME_PINK, speed=(0, 4), life=(5.0, 7.0),
                         size=(3, 4), rise=22, drag=0.0, sway=20)

        # Sparkles along the edge of the hovered button
        self._t_btn -= dt
        if self.hover and self._t_btn <= 0:
            self._t_btn = 0.05
            r = self.rects[self.hover]
            edge = random.choice(["top", "bottom", "left", "right"])
            x = (random.uniform(r.left, r.right) if edge in ("top", "bottom")
                 else (r.left + 8 if edge == "left" else r.right - 8))
            y = (random.uniform(r.top + 6, r.bottom - 6) if edge in ("left", "right")
                 else (r.top + 4 if edge == "top" else r.bottom - 4))
            self.fx.emit(x, y, 1, HOME_GOLD, speed=(5, 18),
                         life=(0.5, 0.9), size=(2, 3), rise=-12, twinkle=True, shape="star")

    def update(self, events):
        self._tick_sparkles()
        mx, my = pygame.mouse.get_pos()
        self.hover = next(
            (n for n, r in self.rects.items() if r.collidepoint(mx, my)), None)
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for n, r in self.rects.items():
                    if r.collidepoint(e.pos):
                        return n
        return None

    def draw(self):
        # Background
        bg = self.bg or self.assets.get("bg")
        if bg and self._bg_scaled is None:                   # scale once, not every frame
            self._bg_scaled = pygame.transform.smoothscale(bg, (self.sw, self.sh))
        if self._bg_scaled:
            self.screen.blit(self._bg_scaled, (0, 0))
        else:
            self.screen.fill((210, 185, 175))

        # Buttons
        for name, rect in self.rects.items():
            img = self.btn_imgs.get(name)
            if img:
                if name == self.hover:
                    bright = img.copy()
                    bright.fill((25, 25, 10, 0), special_flags=pygame.BLEND_RGB_ADD)
                    self.screen.blit(bright, rect.topleft)
                else:
                    self.screen.blit(img, rect.topleft)
            else:
                # Fallback plain button
                c = (100, 130, 55) if name == self.hover else (75, 105, 40)
                pygame.draw.rect(self.screen, c, rect, border_radius=10)
                pygame.draw.rect(self.screen, (200, 170, 80), rect, 2, border_radius=10)
                ls = self.f_btn.render(self.btn_labels[name], True, (255, 245, 210))
                self.screen.blit(ls, ls.get_rect(center=rect.center))

        self.fx.draw_over(self.screen)