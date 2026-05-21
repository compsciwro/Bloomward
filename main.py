# -*- coding: utf-8 -*-
"""
Bloomward - main.py
Entry point. Loads all shared assets and manages screen transitions.
"""

import pygame
import sys
import os

SCREEN_W, SCREEN_H = 1280, 720
FPS      = 60
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _load(name):
    path = os.path.join(ASSETS_DIR, name)
    return pygame.image.load(path).convert_alpha() if os.path.exists(path) else None


def _scale_to_width(img, target_w):
    if img is None:
        return None
    ratio = target_w / img.get_width()
    return pygame.transform.smoothscale(img, (target_w, int(img.get_height() * ratio)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Bloomward")
    clock = pygame.time.Clock()

    # ── Shared assets ─────────────────────────────────────────────────────────
    assets = {
        # Backgrounds
        "bg":          _load("background.png"),       # original fallback
        "game_bg":     _load("game_bg.png"),           # new game screen bg

        # Flower card panel
        "flower_card": _load("flower_card.png"),

        # Logo (top-left of game screen)
        "logo":        _load("logo_img.png"),

        # In-game buttons (right side panel)
        "btn_draw":    _scale_to_width(_load("btndrawflower.png"), 180),
        "btn_rules":   _scale_to_width(_load("btnrules.png"),      180),
        "btn_home":    _scale_to_width(_load("btnhomescreen.png"), 180),
        "btn_exit":    _scale_to_width(_load("btn_exit.png"),      180),

        # Home screen buttons (wider)
        "hbtn_play":   _scale_to_width(_load("btnplaygame.png"),   280),
        "hbtn_rules":  _scale_to_width(_load("btnrules.png"),      280),
        "hbtn_exit":   _scale_to_width(_load("btn_exit.png"),      280),
    }

    from home_screen import HomeScreen
    from game_screen import GameScreen
    from rules_screen import RulesScreen

    current = "home"
    home  = HomeScreen(screen, assets)
    game  = None
    rules = None

    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if current == "home":
            result = home.update(events)
            home.draw()
            if result == "play":
                game = GameScreen(screen, assets)
                current = "game"
            elif result == "rules":
                rules = RulesScreen(screen, assets, from_screen="home")
                current = "rules"
            elif result == "exit":
                pygame.quit()
                sys.exit()

        elif current == "game":
            result = game.update(events)
            game.draw()
            if result == "home":
                current = "home"
            elif result == "rules":
                rules = RulesScreen(screen, assets, from_screen="game")
                current = "rules"
            elif result == "exit":
                pygame.quit()
                sys.exit()

        elif current == "rules":
            result = rules.update(events)
            rules.draw()
            if result == "back":
                current = rules.from_screen

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
