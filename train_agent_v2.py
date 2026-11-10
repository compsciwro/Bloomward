#!/usr/bin/env python3
# ============================================================
# train_agent_v2.py
#
# Trains the v2 Bloomward agent: still MaskablePPO, but with
#   - per-tile observations          (agent_features.py)
#   - a network that scores each tile (tile_policy.py)
#   - a reward that follows the win rule, and an opponent to play
#     against                         (two_player_env.py)
#
# Saves a checkpoint every CHECKPOINT_EVERY steps to models/v2/.
# It never touches bloomward_ppo.zip.
#
# Usage:   py train_agent_v2.py
# ============================================================

import os
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agent_features import adjacency
from tile_policy import TilePolicy
from two_player_env import TwoPlayerEnv

# ---------------- Settings (edit here) ----------------
SMOKE_TEST       = False       # True = 2 minute run to check everything works
TOTAL_TIMESTEPS  = 1_500_000
N_ENVS           = 4            # games played in parallel (set to 1 if it gives trouble)
CHECKPOINT_EVERY = 50_000
SEED             = 0
MODEL_DIR        = os.path.join("models", "v2")
LOG_DIR          = os.path.join("logs", "v2")

if SMOKE_TEST:                  # tiny run, kept apart from the real checkpoints
    TOTAL_TIMESTEPS, CHECKPOINT_EVERY = 12_000, 4_000
    MODEL_DIR, LOG_DIR = os.path.join("models", "v2_smoke"), os.path.join("logs", "v2_smoke")

HYPERPARAMS = dict(learning_rate=3e-4, n_steps=2048 // N_ENVS, batch_size=64, n_epochs=10,
                   gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5)


def make_env(rank):
    def _init():
        env = TwoPlayerEnv(checkpoint_dir=MODEL_DIR)
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return Monitor(env, filename=os.path.join(LOG_DIR, f"env{rank}"),
                       info_keywords=("agent_win", "opp_win", "opponent", "seat"))
    return _init


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    fns = [make_env(i) for i in range(N_ENVS)]
    env = SubprocVecEnv(fns) if N_ENVS > 1 else DummyVecEnv(fns)

    model = MaskablePPO(
        TilePolicy, env, seed=SEED, device="cpu", verbose=1,
        policy_kwargs=dict(adjacency=adjacency(TwoPlayerEnv().game)),
        **HYPERPARAMS)

    checkpoints = CheckpointCallback(save_freq=max(1, CHECKPOINT_EVERY // N_ENVS),
                                     save_path=MODEL_DIR, name_prefix="bloomward_v2")
    print(f"Training v2 agent for {TOTAL_TIMESTEPS:,} timesteps on {N_ENVS} parallel games...")
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoints)
    model.save(os.path.join(MODEL_DIR, "bloomward_v2_final"))
    env.close()
    print(f"Done in {(time.time() - start) / 60:.0f} min. Checkpoints are in {MODEL_DIR}")


if __name__ == "__main__":       # required for parallel games on Windows
    main()
