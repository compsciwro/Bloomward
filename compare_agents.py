#!/usr/bin/env python3
# ============================================================
# compare_agents.py
#
# Compares agents on Bloomward using the same game rules, the
# same evaluation games, and a recorded training setup.
#
# Agents:
#   random      uniform random choice among LEGAL tiles
#   greedy      tries every legal tile on a copy of the game and
#               keeps the one with the highest immediate reward
#   dqn         DQN, epsilon greedy exploration, no action mask
#   ppo         PPO, no action mask
#   maskppo     MaskablePPO, illegal tiles masked out
#   a2c         A2C, no action mask (optional)
#
# How to use:
#   1. Edit AGENTS_TO_RUN and REWARD_TAG below.
#   2. py compare_agents.py
#
# Reward experiment: change the reward values in constants.py,
# set REWARD_TAG to name that version, and run maskppo only.
# The CSV records the actual reward values used for every row.
#
# Output (appended, so separate runs build one table):
#   results/comparison_summary.csv   one row per agent
#   results/comparison_episodes.csv  one row per evaluation game
# Trained models: models/compare/<agent>_<tag>_<run id>.zip
# (new file every run, so nothing is overwritten; bloomward_ppo.zip
#  is never touched)
# ============================================================

import copy
import csv
import json
import math
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import bloomward_env.env as env_module
from bloomward_env.env import BloomwardEnv


# ============================================================
# Experiment settings (edit here)
# ============================================================
 
AGENTS_TO_RUN   = ["random", "greedy", "dqn", "ppo", "maskppo"]  # add "a2c" if wanted
REWARD_TAG      = "v3_final"  # name of the reward version currently in constants.py
SMOKE_TEST      = True       # True = tiny run to check the script works (not for the report)

TRAIN_TIMESTEPS = 1_000_000   # same budget for every learning agent
TRAIN_SEED      = 0           # seeds network init, exploration and training games
EVAL_EPISODES   = 100
EVAL_SEED_START = 50_000      # evaluation game i uses seed 50_000 + i
MAX_STEPS       = 300         # step cap per game (counts illegal moves too)

# Same network size for every learning agent: two hidden layers of 64.
# Each model gets its own deep copy of its settings (see train()),
# because A2C adds its optimiser into policy_kwargs in place, which
# would otherwise leak into PPO's settings.
NET = dict(net_arch=[64, 64])

# Written out in full so they are recorded in the CSV, rather than
# relying on library defaults. PPO and MaskablePPO use identical
# values, so the only difference between them is the action mask.
HYPERPARAMS = {
    "dqn": dict(learning_rate=1e-4, buffer_size=100_000, learning_starts=10_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=5_000,
                exploration_fraction=0.2, exploration_initial_eps=1.0,
                exploration_final_eps=0.05, policy_kwargs=NET),
    "a2c": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
                ent_coef=0.01, vf_coef=0.5, policy_kwargs=NET),
    "ppo": dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                vf_coef=0.5, policy_kwargs=NET),
}
HYPERPARAMS["maskppo"] = HYPERPARAMS["ppo"]

ALGORITHMS = {"dqn": DQN, "a2c": A2C, "ppo": PPO, "maskppo": MaskablePPO}
ALL_AGENTS = ["random", "greedy", "dqn", "ppo", "maskppo", "a2c"]

REWARD_NAMES = [
    "REWARD_VALID_PLACEMENT", "REWARD_INVALID_ACTION", "REWARD_COMBO",
    "REWARD_NEAR_COMBO", "REWARD_WIN", "REWARD_LOSS", "REWARD_TURN_PENALTY",
    "REWARD_CLEANSE", "REWARD_CORRUPTION_SPREAD",
]


# ============================================================
# 1. Wrapper: step cap and end-of-game stats
# ============================================================

class StatsWrapper(gym.Wrapper):
    """
    Step cap: an illegal move does not advance the turn, so an
    unmasked agent could repeat one illegal tile forever. After
    MAX_STEPS the game is cut off and labelled "step_cap", which is
    reported separately from real losses.
    """

    def __init__(self, env, max_steps=MAX_STEPS):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        if not (terminated or truncated) and self.steps >= self.max_steps:
            truncated = True
            info["reason"] = "step_cap"

        if terminated or truncated:
            game = self.env.unwrapped
            reason = info.get("reason", "unknown")
            info["win"] = 1 if reason in ("win_p1", "win_p2") else 0
            info["spirits"] = game._spirit_count_p1
            info["corrupted"] = game._board.count_corrupted()
            info["score"] = info["spirits"] * 10 - info["corrupted"]
            info["turns"] = game._turn       # completed turns (legal moves)
            info["steps"] = self.steps       # all actions, including illegal ones

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return self.env.unwrapped.action_masks()


# ============================================================
# 2. Environment builder (same game for every agent)
# ============================================================

def make_env(masked, monitor_path=None):
    env = BloomwardEnv(render_mode=None)
    env = StatsWrapper(env)
    if masked:
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return Monitor(env, filename=monitor_path,
                   info_keywords=("win", "score", "spirits", "corrupted", "turns"))


# ============================================================
# 3. Training
# ============================================================

def train(kind, timesteps, tag, run_id):
    os.makedirs("models/compare", exist_ok=True)
    os.makedirs("logs/compare", exist_ok=True)
    name = f"{kind}_{tag}_{run_id}"   # run_id makes every file unique

    env = make_env(masked=(kind == "maskppo"), monitor_path=f"logs/compare/{name}")
    model = ALGORITHMS[kind]("MultiInputPolicy", env, seed=TRAIN_SEED,
                             device="cpu", verbose=0, **copy.deepcopy(HYPERPARAMS[kind]))

    print(f"  Training {kind} for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps)
    minutes = (time.time() - start) / 60

    model.save(f"models/compare/{name}")
    train_episodes = len(env.get_episode_rewards())
    env.close()
    print(f"  Done: {train_episodes:,} training episodes in {minutes:.1f} min")
    print(f"  Saved models/compare/{name}.zip")
    return model, train_episodes, minutes, f"models/compare/{name}.zip"


# ============================================================
# 4. Evaluation
# ============================================================

def random_action(env, rng):
    """Uniform random choice among legal tiles."""
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def greedy_action(env, rng):
    """
    Try each legal tile on a deep copy of the game and keep the one
    with the highest immediate reward. Ties are broken uniformly at
    random. The copy has its own RNG state, so simulating moves does
    not change the real game's flower draws or corruption.
    """
    game = env.unwrapped
    best, best_reward = [], -math.inf
    for a in np.flatnonzero(game.action_masks()):
        sim = copy.deepcopy(game)
        _, r, _, _, _ = sim.step(int(a))
        if r > best_reward + 1e-9:
            best, best_reward = [int(a)], r
        elif abs(r - best_reward) <= 1e-9:
            best.append(int(a))
    return int(rng.choice(best))


def evaluate(kind, model, n_episodes):
    """
    Game i always uses seed EVAL_SEED_START + i, so every agent plays
    the same set of games. Random and greedy get their own RNG per
    game (also seeded), kept separate from the game's RNG, so the
    agent's choices never change the game's flower draws.
    Learned agents act deterministically (their best action).
    """
    env = StatsWrapper(BloomwardEnv(render_mode=None))
    rows = []

    for i in range(n_episodes):
        seed = EVAL_SEED_START + i
        obs, _ = env.reset(seed=seed)
        agent_rng = np.random.default_rng(seed)
        total_reward, invalid = 0.0, 0

        while True:
            if kind == "random":
                action = random_action(env, agent_rng)
            elif kind == "greedy":
                action = greedy_action(env, agent_rng)
            elif kind == "maskppo":
                action, _ = model.predict(obs, deterministic=True,
                                          action_masks=env.action_masks())
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            invalid += int(info.get("invalid_action", False))

            if terminated or truncated:
                rows.append(dict(
                    seed=seed, reason=info["reason"], win=info["win"],
                    score=info["score"], spirits=info["spirits"],
                    corrupted=info["corrupted"], turns=info["turns"],
                    steps=info["steps"], invalid_actions=invalid,
                    reward=round(total_reward, 2),
                ))
                break

    env.close()
    return rows


# ============================================================
# 5. Summary and CSV
# ============================================================

def wilson_ci(wins, n, z=1.96):
    """95% confidence interval for a win rate. Unlike the simple
    +/- formula, it stays sensible at 0% and 100%."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def summarise(rows):
    n = len(rows)
    reasons = [r["reason"] for r in rows]
    wins = sum(r["win"] for r in rows)
    lo, hi = wilson_ci(wins, n)

    def mean(k): return float(np.mean([r[k] for r in rows]))
    def std(k): return float(np.std([r[k] for r in rows]))
    def pct(*names): return 100 * sum(x in names for x in reasons) / n

    s = dict(
        win_rate=100 * wins / n, win_ci_low=lo, win_ci_high=hi,
        loss_no_valid_moves=pct("loss_no_valid_moves"),
        loss_sacred_core=pct("loss_sacred_core"),
        trunc_max_turns=pct("truncated_max_turns"),
        trunc_step_cap=pct("step_cap"),
        avg_score=mean("score"), std_score=std("score"),
        avg_turns=mean("turns"), avg_steps=mean("steps"),
        avg_invalid_actions=mean("invalid_actions"),
        avg_reward=mean("reward"), std_reward=std("reward"),
        avg_spirits=mean("spirits"), avg_corrupted=mean("corrupted"),
    )
    # Sanity check: every game must end in exactly one category.
    total = (s["win_rate"] + s["loss_no_valid_moves"] + s["loss_sacred_core"]
             + s["trunc_max_turns"] + s["trunc_step_cap"])
    assert abs(total - 100) < 1e-6, f"Outcome categories add to {total}%, not 100%"
    return {k: round(v, 2) for k, v in s.items()}


def append_csv(path, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(kind, s):
    print(f"\n  {kind}")
    print(f"    Win rate          : {s['win_rate']:.1f}%  (95% CI {s['win_ci_low']:.1f} to {s['win_ci_high']:.1f})")
    print(f"    Losses            : no valid moves {s['loss_no_valid_moves']:.0f}%, "
          f"sacred core {s['loss_sacred_core']:.0f}%")
    print(f"    Truncated         : turn limit {s['trunc_max_turns']:.0f}%, "
          f"step cap {s['trunc_step_cap']:.0f}%")
    print(f"    Avg score         : {s['avg_score']:.1f}  (std {s['std_score']:.1f})")
    print(f"    Avg length        : {s['avg_turns']:.1f} turns, {s['avg_steps']:.1f} steps")
    print(f"    Avg invalid moves : {s['avg_invalid_actions']:.1f}")
    print(f"    Avg reward        : {s['avg_reward']:+.1f}  (std {s['std_reward']:.1f})")


# ============================================================
# Main
# ============================================================

def main():
    timesteps = 3_000 if SMOKE_TEST else TRAIN_TIMESTEPS
    episodes  = 5 if SMOKE_TEST else EVAL_EPISODES
    tag       = "smoke" if SMOKE_TEST else REWARD_TAG
    out_dir   = "results/smoke" if SMOKE_TEST else "results"
    os.makedirs(out_dir, exist_ok=True)

    for kind in AGENTS_TO_RUN:
        if kind not in ALL_AGENTS:
            raise SystemExit(f"Unknown agent '{kind}'. Choose from: {' '.join(ALL_AGENTS)}")

    rewards = {k: getattr(env_module, k) for k in REWARD_NAMES}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run {run_id} | rewards '{tag}' | {episodes} eval games | "
          f"WIN_SCORE_TARGET={env_module.WIN_SCORE_TARGET}")

    for kind in AGENTS_TO_RUN:
        print(f"\n[{kind}]")
        learns = kind in ALGORITHMS
        model, train_eps, minutes, model_path = None, 0, 0.0, ""
        if learns:
            model, train_eps, minutes, model_path = train(kind, timesteps, tag, run_id)

        print(f"  Evaluating over {episodes} games...")
        rows = evaluate(kind, model, episodes)
        summary = summarise(rows)
        print_summary(kind, summary)

        config = dict(
            run_id=run_id, agent=kind, reward_tag=tag, model_file=model_path,
            train_timesteps=timesteps if learns else 0,
            train_episodes=train_eps, train_minutes=round(minutes, 1),
            train_seed=TRAIN_SEED if learns else "",
            eval_episodes=episodes, eval_seed_start=EVAL_SEED_START,
            max_steps=MAX_STEPS, win_score_target=env_module.WIN_SCORE_TARGET,
            max_turns=env_module.MAX_TURNS,
            hyperparams=json.dumps(HYPERPARAMS[kind]) if learns else "",
            rewards=json.dumps(rewards),
        )
        append_csv(f"{out_dir}/comparison_summary.csv", [{**config, **summary}])
        append_csv(f"{out_dir}/comparison_episodes.csv",
                   [{"run_id": run_id, "agent": kind, "reward_tag": tag, **r} for r in rows])

    print(f"\nSaved to {out_dir}/comparison_summary.csv and comparison_episodes.csv")


if __name__ == "__main__":
    main()