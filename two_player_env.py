# -*- coding: utf-8 -*-
"""
Bloomward - two_player_env.py
Training environment for the v2 agent: the real game, played against an opponent.

This wraps the existing BloomwardEnv without changing it. One step is:
    the agent moves  ->  (if it formed a targeted spirit, it picks the target next step)
    ->  the opponent takes its whole turn  ->  the agent sees the new board.

Reward (from the agent's point of view, using the game's own score rule):
    change in MY score / 10            my spirit = +1, tile cleansed = +0.1,
                                       corruption spreading = -0.1 per tile
    - 0.5 for each spirit the OPPONENT gains
    + 10  if I win      - 10 if the opponent wins
    -  5  if the garden is lost (no moves left, or the Sacred Tree falls)

Each game the agent is randomly Player 1 or Player 2, and the opponent is drawn
from a mix: random, combo rule, cautious rule, and saved copies of the agent.
"""

import glob
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent_features import build_obs, observation_space, scores
from bloomward_env.env import BloomwardEnv
from opponents import RandomPlayer, ComboPlayer, CautiousPlayer, ModelPlayer

WIN_BONUS, LOSS_PENALTY, GARDEN_LOST_PENALTY = 10.0, -10.0, -5.0
OPP_SPIRIT_PENALTY = 0.5

# Opponent mix. "self" is only used once saved checkpoints exist; until then its
# share goes to the combo player.
OPPONENT_MIX = {"random": 0.15, "combo": 0.35, "cautious": 0.25, "self": 0.25}


class TwoPlayerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, checkpoint_dir=None, opponent_mix=None):
        super().__init__()
        self.game = BloomwardEnv(render_mode=None)
        n = self.game._n_tiles
        self.action_space = spaces.Discrete(n)
        self.observation_space = observation_space(n)
        self.checkpoint_dir = checkpoint_dir
        self.mix = dict(opponent_mix or OPPONENT_MIX)
        self._model_cache = {}
        self.seat, self.opponent, self.opp_name = 0, None, "random"

    # ── Opponents ─────────────────────────────────────────────────────────────
    def _pick_opponent(self):
        rng = self.np_random
        names = list(self.mix)
        probs = np.array([self.mix[k] for k in names], float)
        name = names[int(rng.choice(len(names), p=probs / probs.sum()))]
        seed = int(rng.integers(1 << 31))
        if name == "self":
            files = (sorted(glob.glob(os.path.join(self.checkpoint_dir, "bloomward_v2_*.zip")))
                     if self.checkpoint_dir else [])
            if not files:
                name = "combo"
            else:
                path = files[int(rng.integers(len(files)))]
                if path not in self._model_cache:
                    from sb3_contrib import MaskablePPO
                    if len(self._model_cache) > 6:
                        self._model_cache.clear()
                    self._model_cache[path] = MaskablePPO.load(path, device="cpu")
                return ModelPlayer(self._model_cache[path], deterministic=False), "self"
        cls = {"random": RandomPlayer, "combo": ComboPlayer, "cautious": CautiousPlayer}[name]
        return cls(seed), name

    def _opponent_turn(self):
        """Opponent places a flower and, if needed, chooses its spirit target."""
        info, term, trunc = {}, False, False
        for _ in range(2):
            a = self.opponent.act(self.game, 1 - self.seat)
            _, _, term, trunc, info = self.game.step(a, player=1 - self.seat)
            if term or trunc or not info.get("spirit_pending"):
                break
        return term, trunc, info

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=int(self.np_random.integers(1 << 31)))
        self.seat = int(self.np_random.integers(2))
        self.opponent, self.opp_name = self._pick_opponent()
        if self.seat == 1:                       # agent is Player 2: opponent opens
            self._opponent_turn()
        return build_obs(self.game, self.seat), {}

    def action_masks(self):
        return self.game.action_masks()

    def _progress(self):
        mine, _ = scores(self.game, self.seat)
        opp_spirits = (self.game._spirit_count_p2 if self.seat == 0
                       else self.game._spirit_count_p1)
        return mine, opp_spirits

    def step(self, action):
        before_score, before_opp = self._progress()
        _, _, term, trunc, info = self.game.step(int(action), player=self.seat)

        if info.get("invalid_action"):
            return build_obs(self.game, self.seat), -0.2, False, False, info

        if not (term or trunc) and not info.get("spirit_pending"):
            term, trunc, info = self._opponent_turn()

        after_score, after_opp = self._progress()
        reward = (after_score - before_score) / 10.0 - OPP_SPIRIT_PENALTY * (after_opp - before_opp)

        if term or trunc:
            reason = info.get("reason", "")
            won = reason == f"win_p{self.seat + 1}"
            lost = reason == f"win_p{2 - self.seat}"
            if won:
                reward += WIN_BONUS
            elif lost:
                reward += LOSS_PENALTY
            elif term:
                reward += GARDEN_LOST_PENALTY
            info = dict(info, agent_win=int(won), opp_win=int(lost),
                        opponent=self.opp_name, seat=self.seat)
        return build_obs(self.game, self.seat), float(reward), term, trunc, info
