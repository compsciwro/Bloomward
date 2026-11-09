# -*- coding: utf-8 -*-
"""
Bloomward - effects.py
Visual effects only: particles, glows, rings and tile tints.

Nothing in this module reads or changes the game environment. The game
screen tells the EffectsManager what has ALREADY happened on the board
(a flower placed, a spirit activated, corruption spread) and the manager
draws an animation for it. All timing uses delta time in seconds.
"""

import math
import random
import pygame


# ── Colour palettes ───────────────────────────────────────────────────────────
GOLD     = [(255, 215, 90), (255, 236, 150), (255, 196, 60)]
MINT     = [(170, 240, 190), (200, 255, 210)]
SUN      = [(255, 215, 70), (255, 190, 40), (255, 240, 150)]
PINK     = [(255, 150, 190), (240, 110, 150), (255, 205, 222)]
LAVENDER = [(200, 170, 255), (222, 202, 255), (170, 130, 230)]
RAIN     = [(130, 190, 255), (180, 220, 255), (255, 170, 205)]
CLEANSE  = [(170, 240, 120), (220, 240, 120), (255, 225, 110)]
VIOLET   = [(150, 70, 200), (110, 40, 160), (190, 120, 230)]

FLOWER_PRESET = {
    # flower index -> (sparkle colours, glow colour, style)
    0: (SUN,      (255, 210, 80),  "sunburst"),
    1: (PINK,     (255, 170, 200), "petals"),
    2: (LAVENDER, (205, 175, 255), "float"),
}


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2


def bloom_scale(age):
    """Flower size over its placement animation (age in seconds).
    0-0.2 s: appears at ~70%   0.2-0.5 s: grows to 110%   0.5-0.9 s: settles to 100%."""
    if age < 0.2:
        return 0.70 + 0.05 * (age / 0.2)
    if age < 0.5:
        return 0.75 + 0.35 * ease_out_cubic((age - 0.2) / 0.3)
    if age < 0.9:
        return 1.10 - 0.10 * ease_in_out_sine((age - 0.5) / 0.4)
    return 1.0


BLOOM_TIME = 0.9


# ── Effect objects ────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "age", "life", "size", "colour",
                 "gravity", "drag", "twinkle", "phase", "tag", "sway", "shape")

    def __init__(self, x, y, vx, vy, life, size, colour,
                 gravity=0.0, drag=1.5, twinkle=False, tag=None, sway=0.0, delay=0.0,
                 shape="dot"):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.age, self.life = -delay, life
        self.size, self.colour = size, colour
        self.gravity, self.drag = gravity, drag
        self.twinkle, self.sway, self.tag = twinkle, sway, tag
        self.phase = random.random() * math.tau
        self.shape = shape

    def update(self, dt):
        self.age += dt
        if self.age < 0:
            return
        damp = max(0.0, 1.0 - self.drag * dt)
        self.vx *= damp
        self.vy = self.vy * damp + self.gravity * dt
        self.x += (self.vx + self.sway * math.sin(self.age * 4 + self.phase)) * dt
        self.y += self.vy * dt

    @property
    def alive(self):
        return self.age < self.life

    def alpha(self):
        if self.age < 0:
            return 0
        t = self.age / self.life
        a = min(1.0, t / 0.15) if t < 0.15 else (1.0 if t < 0.6 else max(0.0, (1 - t) / 0.4))
        if self.twinkle:
            a *= 0.55 + 0.45 * math.sin(self.age * 9 + self.phase)
        return int(255 * max(0.0, a))


class Ring:
    """A circle that expands and fades."""
    def __init__(self, x, y, colour, r0, r1, duration, width=3, peak=200, delay=0.0):
        self.x, self.y, self.colour = x, y, colour
        self.r0, self.r1, self.duration = r0, r1, duration
        self.width, self.peak, self.age = width, peak, -delay

    def update(self, dt):
        self.age += dt

    @property
    def alive(self):
        return self.age < self.duration


class Glow:
    """A soft radial light that fades out (or pulses, for ambient use)."""
    def __init__(self, x, y, colour, radius, duration, peak=170, delay=0.0):
        self.x, self.y, self.colour = x, y, colour
        self.radius, self.duration, self.peak, self.age = radius, duration, peak, -delay

    def update(self, dt):
        self.age += dt

    @property
    def alive(self):
        return self.age < self.duration


class TileTint:
    """A hexagon-shaped colour wash that fades (used to darken newly corrupted tiles)."""
    def __init__(self, points, colour, duration, peak=150):
        self.points, self.colour = points, colour
        self.duration, self.peak, self.age = duration, peak, 0.0

    def update(self, dt):
        self.age += dt

    @property
    def alive(self):
        return self.age < self.duration


# ── Manager ───────────────────────────────────────────────────────────────────
class EffectsManager:
    MAX_PARTICLES = 320
    MAX_FIREFLIES = 14

    def __init__(self):
        self._dot_cache  = {}
        self._star_cache = {}
        self._glow_cache = {}
        self.clear()

    def clear(self):
        self.particles, self.rings, self.glows, self.tints = [], [], [], []
        self._firefly_timer = 0.0

    def count(self):
        return len(self.particles) + len(self.rings) + len(self.glows) + len(self.tints)

    # ── Cached sprites ────────────────────────────────────────────────────────
    def _dot(self, colour, size):
        """Soft glowing dot: bright core with a faint halo."""
        key = (colour, size)
        if key not in self._dot_cache:
            r = size * 3
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            for i in range(r, 0, -1):
                a = int(255 * (1 - i / r) ** 1.5)
                pygame.draw.circle(s, (*colour, a), (r, r), i)
            pygame.draw.circle(s, (255, 255, 240, 230), (r, r), max(1, size // 2))
            self._dot_cache[key] = s
        return self._dot_cache[key]

    def _star(self, colour, size):
        """Four-point twinkle star; reads well on light backgrounds."""
        key = (colour, size)
        if key not in self._star_cache:
            r = size * 3
            s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            c = r + 1
            pygame.draw.circle(s, (*colour, 55), (c, c), int(r * 0.6))
            inner = max(1, size * 0.45)
            pts = []
            for k in range(8):
                ang = k * math.pi / 4 - math.pi / 2
                rad = r if k % 2 == 0 else inner
                pts.append((c + rad * math.cos(ang), c + rad * math.sin(ang)))
            pygame.draw.polygon(s, (*colour, 255), pts)
            pygame.draw.circle(s, (255, 250, 230, 255), (c, c), max(1, size // 2))
            self._star_cache[key] = s
        return self._star_cache[key]

    def glow_sprite(self, colour, radius):
        """Radial gradient used for glows. Cached, so it is built once."""
        key = (colour, radius)
        if key not in self._glow_cache:
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for i in range(radius, 0, -2):
                a = int(255 * (1 - i / radius) ** 1.05)
                pygame.draw.circle(s, (*colour, a), (radius, radius), i)
            self._glow_cache[key] = s
        return self._glow_cache[key]

    # ── Primitives ────────────────────────────────────────────────────────────
    def emit(self, x, y, n, colours, speed=(20, 60), angles=None, life=(0.5, 1.0),
             size=(2, 3), gravity=0.0, drag=1.5, rise=0.0, twinkle=False,
             spread=0.0, sway=0.0, tag=None, delay=0.0, shape="dot"):
        for k in range(n):
            if len(self.particles) >= self.MAX_PARTICLES:
                return
            ang = (angles[k % len(angles)] if angles else random.uniform(0, math.tau))
            sp = random.uniform(*speed)
            px = x + random.uniform(-spread, spread)
            py = y + random.uniform(-spread, spread)
            self.particles.append(Particle(
                px, py, math.cos(ang) * sp, math.sin(ang) * sp + rise,
                random.uniform(*life), random.randint(*size), random.choice(colours),
                gravity=gravity, drag=drag, twinkle=twinkle, sway=sway, tag=tag, delay=delay,
                shape=shape))

    def ring(self, *a, **k):
        self.rings.append(Ring(*a, **k))

    def glow(self, *a, **k):
        self.glows.append(Glow(*a, **k))

    def tint(self, *a, **k):
        self.tints.append(TileTint(*a, **k))

    # ── Presets: flower placement ─────────────────────────────────────────────
    def placement(self, x, y, flower):
        colours, glow_col, style = FLOWER_PRESET.get(flower, FLOWER_PRESET[0])
        self.glow(x, y, glow_col, 34, BLOOM_TIME, peak=150)
        if style == "sunburst":
            rays = [i * math.tau / 8 for i in range(8)]
            self.emit(x, y, 8, colours, speed=(70, 90), angles=rays, life=(0.45, 0.6),
                      size=(2, 2), drag=3.0)
            self.emit(x, y, 5, colours, speed=(15, 40), life=(0.6, 0.9), rise=-18, twinkle=True)
        elif style == "petals":
            self.emit(x, y, 7, colours, speed=(20, 45), life=(0.6, 0.9), rise=-10, twinkle=True)
            self.emit(x, y, 4, colours, speed=(5, 15), life=(0.8, 1.1), size=(3, 4),
                      rise=-30, sway=18, drag=0.8)
        else:
            self.emit(x, y, 10, colours, speed=(10, 35), life=(0.7, 1.0), rise=-28,
                      drag=1.0, twinkle=True)

    # ── Presets: spirits ──────────────────────────────────────────────────────
    def spirit_renewal(self, centres):
        cx, cy = _centroid(centres)
        for (x, y) in centres:
            self.glow(x, y, (255, 215, 90), 40, 1.1, peak=190)
            self.emit(x, y, 6, GOLD, speed=(15, 45), life=(0.7, 1.1), rise=-20, twinkle=True)
        self.ring(cx, cy, (255, 215, 90), 12, 110, 1.0, width=4, peak=210)

    def spirit_rain(self, centres):
        cx, cy = _centroid(centres)
        for (x, y) in centres:
            self.emit(x, y, 5, RAIN, speed=(15, 40), life=(0.7, 1.0), rise=-15, twinkle=True)
        for d in (0.0, 0.25, 0.5):
            self.ring(cx, cy, (140, 195, 255), 10, 95, 0.9, width=2, peak=170, delay=d)
        # A few droplets fall briefly from above the combo
        self.emit(cx, cy - 70, 9, RAIN[:2], speed=(0, 10), angles=[math.pi / 2],
                  life=(0.6, 0.8), size=(2, 3), gravity=260, drag=0.2, spread=45)

    def spirit_blossom(self, centres):
        cx, cy = _centroid(centres)
        for (x, y) in centres:
            self.glow(x, y, (205, 175, 255), 36, 1.0, peak=170)
            self.emit(x, y, 6, LAVENDER, speed=(10, 30), life=(0.8, 1.2), rise=-22, twinkle=True)
        self.ring(cx, cy, (200, 165, 255), 10, 85, 1.1, width=3, peak=190)

    def cleanse(self, x, y):
        """Corrupted tile restored by the Spirit of Renewal."""
        self.glow(x, y, (190, 240, 140), 42, 1.3, peak=200)
        self.ring(x, y, (200, 240, 120), 6, 48, 0.9, width=3, peak=200)
        self.emit(x, y, 12, CLEANSE, speed=(10, 35), life=(0.8, 1.2), rise=-30,
                  twinkle=True, spread=10)

    def protect(self, centres):
        """Tiles protected by the Spirit of Blossom."""
        for (x, y) in centres:
            self.glow(x, y, (215, 195, 255), 30, 1.2, peak=150)
            self.ring(x, y, (205, 175, 255), 8, 34, 1.0, width=2, peak=170)

    # ── Presets: corruption ───────────────────────────────────────────────────
    def corruption(self, x, y, hex_points):
        self.tint(hex_points, (25, 0, 35), 1.4, peak=170)
        self.glow(x, y, (120, 40, 170), 38, 1.2, peak=170)
        self.ring(x, y, (130, 50, 180), 6, 40, 0.9, width=3, peak=200)
        self.emit(x, y, 7, VIOLET, speed=(8, 25), life=(0.8, 1.2), gravity=20,
                  drag=1.0, spread=12)

    # ── Ambient: Sacred Tree fireflies ────────────────────────────────────────
    def tree_fireflies(self, dt, x, y, rx, ry):
        self._firefly_timer -= dt
        alive = sum(1 for p in self.particles if p.tag == "firefly")
        if self._firefly_timer <= 0 and alive < self.MAX_FIREFLIES:
            self._firefly_timer = random.uniform(0.25, 0.5)
            ang = random.uniform(0, math.tau)
            d = random.uniform(0.2, 1.0)
            self.particles.append(Particle(
                x + math.cos(ang) * rx * d, y + math.sin(ang) * ry * d + ry * 0.2,
                random.uniform(-4, 4), random.uniform(-16, -7),
                random.uniform(2.5, 4.5), random.randint(3, 4),
                random.choice(GOLD + MINT[:1]), drag=0.0, twinkle=True,
                sway=6, tag="firefly"))

    # ── Frame update / draw ───────────────────────────────────────────────────
    def update(self, dt):
        for group in (self.particles, self.rings, self.glows, self.tints):
            for e in group:
                e.update(dt)
        self.particles = [p for p in self.particles if p.alive]
        self.rings     = [r for r in self.rings if r.alive]
        self.glows     = [g for g in self.glows if g.alive]
        self.tints     = [t for t in self.tints if t.alive]

    def draw_under(self, surf):
        """Effects that belong on the tile surface (tints, glows)."""
        for t in self.tints:
            a = int(t.peak * (1 - t.age / t.duration))
            xs = [p[0] for p in t.points]; ys = [p[1] for p in t.points]
            x0, y0 = int(min(xs)), int(min(ys))
            s = pygame.Surface((int(max(xs)) - x0 + 2, int(max(ys)) - y0 + 2), pygame.SRCALPHA)
            pygame.draw.polygon(s, (*t.colour, a), [(px - x0, py - y0) for px, py in t.points])
            surf.blit(s, (x0, y0))
        for g in self.glows:
            if g.age < 0:
                continue
            t = g.age / g.duration
            a = g.peak * (min(1.0, t / 0.15) if t < 0.15 else (1 - t) / 0.85)
            spr = self.glow_sprite(g.colour, g.radius)
            spr.set_alpha(int(max(0, a)))
            surf.blit(spr, spr.get_rect(center=(int(g.x), int(g.y))))

    def draw_over(self, surf):
        """Effects that float above the flowers (rings, particles)."""
        for r in self.rings:
            if r.age < 0:
                continue
            t = r.age / r.duration
            rad = int(r.r0 + (r.r1 - r.r0) * ease_out_cubic(t))
            a = int(r.peak * (1 - t))
            if rad <= r.width or a <= 0:
                continue
            s = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*r.colour, a), (rad + 2, rad + 2), rad, r.width)
            surf.blit(s, (int(r.x) - rad - 2, int(r.y) - rad - 2))
        for p in self.particles:
            a = p.alpha()
            if a <= 0:
                continue
            spr = self._star(p.colour, p.size) if p.shape == "star" else self._dot(p.colour, p.size)
            spr.set_alpha(a)
            surf.blit(spr, spr.get_rect(center=(int(p.x), int(p.y))))


def _centroid(points):
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))