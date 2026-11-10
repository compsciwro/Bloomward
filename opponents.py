# -*- coding: utf-8 -*-
"""
Bloomward - opponents.py
Players the v2 agent trains against and is measured against.

None of these is the final agent. The three rule players are fixed yardsticks;
ModelPlayer lets a saved copy of the agent act as an opponent (self-play).
Every player has act(env, player) and returns a legal tile index.
"""

import numpy as np

from agent_features import build_obs, completes_combo, completing_flowers
from bloomward_env.constants import FLOWER_TYPES, FLOWER_WEIGHTS


class RandomPlayer:
    """Uniform random choice among legal tiles."""
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, env, player):
        return int(self.rng.choice(np.flatnonzero(env.action_masks())))


class ComboPlayer(RandomPlayer):
    """Completes a triangle whenever it can; otherwise random."""
    name = "combo"

    def act(self, env, player):
        legal = np.flatnonzero(env.action_masks())
        if env._pending_spirit is None:
            wins = [i for i in legal if completes_combo(env, int(i), env._current_flower)]
            if wins:
                return int(self.rng.choice(wins))
        return int(self.rng.choice(legal))


def gift_chance(env, idx):
    """Chance the NEXT player can complete a triangle if the current flower goes on idx
    (weighted by how likely each flower is to be drawn)."""
    board = env._board
    tile = board.tile_at_index(idx)
    tile.flower = env._current_flower
    try:
        open_flowers = set()
        for j in board.placeable_indices():
            open_flowers |= completing_flowers(env, board.tiles[j])
            if len(open_flowers) == len(FLOWER_TYPES):
                break
        return sum(w for f, w in zip(FLOWER_TYPES, FLOWER_WEIGHTS) if f in open_flowers)
    finally:
        tile.flower = None


class CautiousPlayer(RandomPlayer):
    """Completes a triangle when it can; otherwise picks the move that leaves the
    opponent the smallest chance of completing one next turn."""
    name = "cautious"

    def act(self, env, player):
        legal = [int(i) for i in np.flatnonzero(env.action_masks())]
        if env._pending_spirit is not None:
            return int(self.rng.choice(legal))
        wins = [i for i in legal if completes_combo(env, i, env._current_flower)]
        if wins:
            return int(self.rng.choice(wins))
        scored = [(gift_chance(env, i), i) for i in legal]
        best = min(s for s, _ in scored)
        return int(self.rng.choice([i for s, i in scored if s <= best + 1e-9]))


class ModelPlayer:
    """A trained v2 model as a player. deterministic=False adds variety for self-play."""
    name = "model"

    def __init__(self, model, deterministic=True):
        self.model, self.deterministic = model, deterministic

    def act(self, env, player):
        action, _ = self.model.predict(build_obs(env, player), deterministic=self.deterministic,
                                       action_masks=env.action_masks())
        return int(action)
