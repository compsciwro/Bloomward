#!/usr/bin/env python3
# ============================================================
# evaluate_two_player.py
#
# Measures an agent the way Bloomward is actually played: two
# players alternating on one board, exactly like game_screen.py.
# Works for the original model (bloomward_ppo.zip) and v2 models.
#
# Opponents (see opponents.py): random, combo, cautious.
# Every matchup is played from both seats (agent as P1 and as P2)
# on the same seeded games.
#
# Usage:
#   py evaluate_two_player.py --model bloomward_ppo.zip
#   py evaluate_two_player.py --model models/v2/bloomward_v2_final.zip
#   py evaluate_two_player.py --select models/v2
#        Checks every checkpoint on VALIDATION games, picks the best,
#        then tests that one on separate TEST games.
#
# Results are saved to results/two_player_*.csv
# ============================================================

import argparse
import csv
import glob
import math
import os
from datetime import datetime

import numpy as np
from sb3_contrib import MaskablePPO

from agent_features import build_obs, completes_combo
from bloomward_env.env import BloomwardEnv
from opponents import RandomPlayer, ComboPlayer, CautiousPlayer, gift_chance

VALIDATION_SEED = 80_000     # used only to choose a checkpoint
TEST_SEED       = 90_000     # used only for the final reported numbers
OPPONENTS = {"random": RandomPlayer, "combo": ComboPlayer, "cautious": CautiousPlayer}
MAX_ACTIONS = 600            # safety cap per game


class AgentPlayer:
    """A trained model. Uses v2 inputs if the model was trained with them,
    otherwise the original environment observation."""
    name = "agent"

    def __init__(self, model):
        self.model = model
        self.v2 = "tiles" in model.observation_space.spaces

    def act(self, env, player):
        obs = build_obs(env, player) if self.v2 else env._make_observation()
        action, _ = self.model.predict(obs, deterministic=True, action_masks=env.action_masks())
        return int(action)


def play(agent, opponent, seed, agent_seat):
    """One full game. Returns what happened, from the agent's point of view."""
    env = BloomwardEnv(render_mode=None)
    env.reset(seed=seed)
    players = {agent_seat: agent, 1 - agent_seat: opponent}
    mover, actions, invalid = 0, 0, 0
    chances = taken = moves = 0
    gift = 0.0
    info = {}

    while actions < MAX_ACTIONS:
        is_agent = mover == agent_seat
        placing = env._pending_spirit is None
        can = []
        if is_agent and placing:
            can = [int(i) for i in np.flatnonzero(env.action_masks())
                   if completes_combo(env, int(i), env._current_flower)]
        a = players[mover].act(env, mover)
        if is_agent and placing and env._board.tile_at_index(a).is_placeable():
            moves += 1
            chances += bool(can)
            taken += a in can
            gift += gift_chance(env, a)
        _, _, terminated, truncated, info = env.step(a, player=mover)
        actions += 1
        if info.get("invalid_action"):
            invalid += is_agent
            continue
        if terminated or truncated:
            break
        if not info.get("spirit_pending"):       # a pending spirit: same player picks its target
            mover = 1 - mover

    reason = info.get("reason", "cap")
    spirits = (env._spirit_count_p1, env._spirit_count_p2)
    me, opp = agent_seat, 1 - agent_seat
    return dict(
        seed=seed, seat=f"P{me + 1}", reason=reason,
        win=int(reason == f"win_p{me + 1}"), opp_win=int(reason == f"win_p{opp + 1}"),
        garden_lost=int(reason in ("loss_no_valid_moves", "loss_sacred_core")),
        agent_spirits=spirits[me], opp_spirits=spirits[opp],
        score_diff=(spirits[me] - spirits[opp]) * 10,
        corrupted=env._board.count_corrupted(), turns=env._turn, invalid=invalid,
        combo_chances=chances, combo_taken=taken, gift=gift / max(1, moves),
    )


def wilson(k, n, z=1.96):
    """95% confidence interval for a win rate."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def summarise(rows):
    n = len(rows)
    wins = sum(r["win"] for r in rows)
    lo, hi = wilson(wins, n)
    chances = sum(r["combo_chances"] for r in rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return dict(
        games=n, win_rate=100 * wins / n, win_ci_low=lo, win_ci_high=hi,
        opp_win_rate=100 * mean("opp_win"), garden_lost_rate=100 * mean("garden_lost"),
        avg_score_diff=mean("score_diff"), agent_spirits=mean("agent_spirits"),
        opp_spirits=mean("opp_spirits"), corrupted=mean("corrupted"), turns=mean("turns"),
        invalid=mean("invalid"),
        combo_take_rate=100 * sum(r["combo_taken"] for r in rows) / chances if chances else 0.0,
        gift_chance=100 * mean("gift"),
    )


def evaluate(model, opponents, games, seed_start):
    """All matchups for one model. Returns (summary rows, per-game rows)."""
    agent = AgentPlayer(model)
    summary, per_game = [], []
    for name in opponents:
        for seat in (0, 1):
            rows = [play(agent, OPPONENTS[name](seed_start + g + 1), seed_start + g, seat)
                    for g in range(games)]
            s = summarise(rows)
            label = dict(opponent=name, seat=f"P{seat + 1}")
            summary.append({**label, **{k: round(v, 2) for k, v in s.items()}})
            per_game += [{**label, **r} for r in rows]
    return summary, per_game


def print_table(summary):
    for r in summary:
        print(f"  vs {r['opponent']:8s} as {r['seat']}: win {r['win_rate']:5.1f}% "
              f"({r['win_ci_low']:.0f}-{r['win_ci_high']:.0f})  opp win {r['opp_win_rate']:5.1f}%  "
              f"garden lost {r['garden_lost_rate']:5.1f}%  spirits {r['agent_spirits']:.1f} v "
              f"{r['opp_spirits']:.1f}  takes combos {r['combo_take_rate']:5.1f}%  "
              f"gifts {r['gift_chance']:4.1f}%")


def save(name, rows, extra):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"two_player_{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        rows = [{**extra, **r} for r in rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bloomward_ppo.zip")
    ap.add_argument("--select", metavar="DIR", help="pick the best checkpoint in DIR, then test it")
    ap.add_argument("--games", type=int, default=200, help="test games per opponent per seat")
    ap.add_argument("--val-games", type=int, default=50)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.model

    if args.select:
        files = sorted(glob.glob(os.path.join(args.select, "*.zip")), key=os.path.getmtime)
        print(f"Validating {len(files)} checkpoints ({args.val_games} games x 2 seats "
              f"vs combo and cautious, seeds from {VALIDATION_SEED})")
        table = []
        for path in files:
            s, _ = evaluate(MaskablePPO.load(path, device="cpu"), ["combo", "cautious"],
                            args.val_games, VALIDATION_SEED)
            score = float(np.mean([r["win_rate"] for r in s]))
            table.append(dict(checkpoint=os.path.basename(path), validation_win_rate=round(score, 2)))
            print(f"  {os.path.basename(path):34s} {score:5.1f}%")
        best = max(table, key=lambda r: r["validation_win_rate"])
        model_path = os.path.join(args.select, best["checkpoint"])
        save(f"validation_{stamp}", table, {})
        print(f"\nBest on validation: {best['checkpoint']} ({best['validation_win_rate']}%)")

    print(f"\nTesting {model_path}: {args.games} games per opponent per seat, seeds from {TEST_SEED}")
    summary, games = evaluate(MaskablePPO.load(model_path, device="cpu"),
                              list(OPPONENTS), args.games, TEST_SEED)
    print_table(summary)
    extra = dict(model=os.path.basename(model_path))
    path = save(f"summary_{stamp}", summary, extra)
    save(f"games_{stamp}", games, extra)
    print(f"\nSaved {path}")
    if args.select:
        print(f"To use this agent in the game, set MODEL_PATH in game_screen.py to:\n  {model_path}")


if __name__ == "__main__":
    main()
