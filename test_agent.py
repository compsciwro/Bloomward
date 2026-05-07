#!/usr/bin/env python3
# ============================================================
# test_agent.py
#
# Load a trained PPO model and watch it play one episode.
# Run this after train_agent.py has produced bloomward_ppo.zip
#
# Usage:  python test_agent.py
# ============================================================

import time
from sb3_contrib import MaskablePPO
from bloomward_env.env import BloomwardEnv
from bloomward_env.constants import SEASON_NAMES


def main(render_mode="human", delay=0.4):
    print("Loading trained model...")
    model = MaskablePPO.load("bloomward_ppo")

    env = BloomwardEnv(render_mode=render_mode)
    env.metadata["render_fps"] = 2

    obs, info = env.reset(seed=0)
    env.render()

    total_reward = 0.0
    steps        = 0

    print("\nWatching trained agent play...\n")

    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps        += 1
        if render_mode == "human":
            time.sleep(delay)
        if terminated or truncated:
            print(f"\nEpisode ended: {info.get('reason', 'unknown')}")
            break
    else:
        info["reason"] = "truncated"

    env.render()

    print(f"\n{'='*45}")
    print(f"  Trained Agent Summary")
    print(f"{'='*45}")
    print(f"  Steps taken  : {steps}")
    print(f"  Total reward : {total_reward:.1f}")
    print(f"  Spirits      : {env._spirit_count}")
    print(f"  Corrupted    : {env._board.count_corrupted()}")
    print(f"  Result       : {info.get('reason', '?')}")
    print(f"{'='*45}\n")

    env.close()


if __name__ == "__main__":
    main()
