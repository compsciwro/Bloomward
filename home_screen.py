# -*- coding: utf-8 -*-
"""
Bloomward - home_screen.py
Main menu using the new Bloomward UI assets.
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

    def update(self, events):
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
        if bg:
            self.screen.blit(
                pygame.transform.smoothscale(bg, (self.sw, self.sh)), (0, 0))
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
