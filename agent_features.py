# -*- coding: utf-8 -*-
"""
Bloomward - agent_features.py
Turns the game state into the inputs (observation) for the v2 agent.

Everything is read from the existing BloomwardEnv; nothing in the game is
changed. The observation is written from the point of view of the player
about to move ("my score" / "opponent's score"), so the same agent can play
as Player 1 or Player 2.

Per tile (TILE_FEATS numbers for each of the 61 tiles):
   0-5   what is on the tile (empty, sunflower, tulip, blossom, corrupted, sacred)
   6-8   share of its 6 neighbours holding a sunflower / tulip / blossom
   9-11  would a sunflower / tulip / blossom placed here complete a triangle?
   12    share of its neighbours that are corrupted
   13    protected by the Spirit of Blossom
   14    is this tile a legal choice right now (same as the action mask)

Whole board (GLOBAL_FEATS numbers):
   0-2   current flower          3-6   season
   7     my score / 60           8     opponent's score / 60
   9     my spirits / 10         10    opponent's spirits / 10
   11    corrupted tiles / 36    12    turn / max turns
   13    turns until corruption spreads / 8
   14    am I choosing a spirit target?   15-16  which spirit (Renewal, Blossom)
"""

import numpy as np
from gymnasium import spaces

from bloomward_env.rules import _triangles_containing, get_season
from bloomward_env.constants import (
    FLOWER_TYPES, FLOWER_SUNFLOWER, FLOWER_BLOSSOM,
    SPREAD_EVERY_N_TURNS, WIN_SCORE_TARGET, MAX_TURNS, TILE_SACRED,
)

TILE_FEATS   = 15
GLOBAL_FEATS = 17
_F_INDEX = {f: k for k, f in enumerate(FLOWER_TYPES)}


def observation_space(n_tiles):
    return spaces.Dict({
        "tiles":   spaces.Box(0.0, 1.0, (n_tiles * TILE_FEATS,), np.float32),
        "globals": spaces.Box(-1.0, 1.0, (GLOBAL_FEATS,), np.float32),
    })


def completing_flowers(env, tile):
    """Flower types that would complete a not-yet-used triangle if placed on this tile."""
    done = set()
    for tri in _triangles_containing(env._board, tile.q, tile.r):
        a, b = [t for t in tri if t is not tile]
        if a.flower is not None and a.flower == b.flower:
            key = frozenset((t.q, t.r) for t in tri)
            if key not in env._activated_combos:
                done.add(a.flower)
    return done


def completes_combo(env, idx, flower):
    tile = env._board.tile_at_index(idx)
    return tile is not None and tile.is_placeable() and flower in completing_flowers(env, tile)


def adjacency(env):
    """61 x 61 table: 1 where two tiles touch. Used by the tile policy network."""
    board = env._board
    n = len(board.tiles)
    A = np.zeros((n, n), np.float32)
    for i, t in enumerate(board.tiles):
        for nb in board.get_neighbours_of(t.q, t.r):
            A[i, board.index_of[(nb.q, nb.r)]] = 1.0
    return A


def scores(env, player):
    """(my score, opponent's score), using the game's own rule: spirits x 10 - corrupted."""
    corrupted = env._board.count_corrupted()
    sp = (env._spirit_count_p1, env._spirit_count_p2)
    return sp[player] * 10 - corrupted, sp[1 - player] * 10 - corrupted


def build_obs(env, player):
    """Observation for `player` (0 = Player 1, 1 = Player 2) from the current game state."""
    board = env._board
    n = len(board.tiles)
    mask = env.action_masks()
    t = np.zeros((n, TILE_FEATS), np.float32)

    for i, tile in enumerate(board.tiles):
        if tile.tile_type == TILE_SACRED:
            kind = 5
        elif tile.corrupted:
            kind = 4
        elif tile.flower is not None:
            kind = 1 + _F_INDEX[tile.flower]
        else:
            kind = 0
        t[i, kind] = 1.0
        t[i, 13] = float(tile.protected)
        t[i, 14] = float(mask[i])
        if kind == 0:                                   # empty fertile tile
            nbs = board.get_neighbours_of(tile.q, tile.r)
            for nb in nbs:
                if nb.flower is not None:
                    t[i, 6 + _F_INDEX[nb.flower]] += 1 / 6
                if nb.corrupted:
                    t[i, 12] += 1 / 6
            for f in completing_flowers(env, tile):
                t[i, 9 + _F_INDEX[f]] = 1.0

    g = np.zeros(GLOBAL_FEATS, np.float32)
    g[_F_INDEX[env._current_flower]] = 1.0
    g[3 + get_season(env._turn)] = 1.0
    mine, theirs = scores(env, player)
    sp = (env._spirit_count_p1, env._spirit_count_p2)
    g[7], g[8] = mine / WIN_SCORE_TARGET, theirs / WIN_SCORE_TARGET
    g[9], g[10] = sp[player] / 10, sp[1 - player] / 10
    g[11] = board.count_corrupted() / 36
    g[12] = env._turn / MAX_TURNS
    g[13] = (SPREAD_EVERY_N_TURNS - env._turn % SPREAD_EVERY_N_TURNS) / SPREAD_EVERY_N_TURNS
    pending = env._pending_spirit
    if pending is not None:
        g[14] = 1.0
        g[15] = float(pending["combo"]["flower"] == FLOWER_SUNFLOWER)
        g[16] = float(pending["combo"]["flower"] == FLOWER_BLOSSOM)
    return {"tiles": t.ravel(), "globals": np.clip(g, -1.0, 1.0)}
