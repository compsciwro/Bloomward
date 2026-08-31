#!/usr/bin/env python3
# ============================================================
# evaluate_agent.py
#
# Compares trained PPO agent vs random agent across N episodes.
# Use the output for the results section of your Honours report.
#
# Usage:
#   python evaluate_agent.py              ← 50 episodes each
#   python evaluate_agent.py --episodes 100
#
# Results are also saved automatically to results/eval_<timestamp>.txt
# ============================================================

import argparse
import numpy as np
import sys
from datetime import datetime
import os
from sb3_contrib import MaskablePPO
from bloomward_env.env import BloomwardEnv


# ============================================================
# Tee — duplicates print() output to both the console and a file
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# Core evaluation function
# ============================================================

def evaluate(model_or_none, n_episodes: int, seed_offset: int = 0):
    """
    Run n_episodes and collect statistics.

    Parameters
    ----------
    model_or_none : PPO model, or None for a random agent
    n_episodes    : number of episodes to run
    seed_offset   : offset applied to episode seeds for variety

    Returns
    -------
    dict of lists with per-episode metrics
    """
    results = {
        "total_reward":    [],
        "episode_length":  [],
        "win":             [],
        "win_p1":          [],
        "win_p2":          [],
        "loss":            [],
        "truncated":       [],
        "invalid_actions": [],
        "spirits":         [],
        "corrupted_final": [],
    }

    env = BloomwardEnv(render_mode=None)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)

        ep_reward   = 0.0
        ep_steps    = 0
        ep_invalid  = 0

        while True:
            if model_or_none is None:
                # Random agent — sample only from valid tiles to avoid
                # infinite invalid-action loops as the board fills up
                valid = env._board.placeable_indices()
                if valid:
                    action = int(env.np_random.choice(valid))
                else:
                    action = env.action_space.sample()
            else:
                # Trained agent
                action, _ = model_or_none.predict(obs, deterministic=True, action_masks=env.action_masks())
                action    = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward  += reward
            ep_steps   += 1

            if info.get("invalid_action", False):
                ep_invalid += 1

            if terminated or truncated:
                reason = info.get("reason", "ongoing")
                results["total_reward"].append(ep_reward)
                results["episode_length"].append(ep_steps)
                results["invalid_actions"].append(ep_invalid)
                results["spirits"].append(env._spirit_count)
                results["corrupted_final"].append(env._board.count_corrupted())

                # FIX: check_terminal() returns "win_p1" / "win_p2",
                # never the bare string "win" — this used to always
                # evaluate to 0 and silently hide every win.
                results["win"].append(1 if reason in ("win_p1", "win_p2") else 0)
                results["win_p1"].append(1 if reason == "win_p1" else 0)
                results["win_p2"].append(1 if reason == "win_p2" else 0)

                results["loss"].append(
                    1 if reason in ("loss_sacred_core",
                                    "loss_no_valid_moves") else 0)
                results["truncated"].append(
                    1 if reason == "truncated_max_turns" else 0)
                print(f"  ep {ep+1:>3}/{n_episodes}  steps={ep_steps:>4}  "
                      f"reward={ep_reward:+.0f}  {reason}")
                break

    env.close()

    return results


# ============================================================
# Summary printer
# ============================================================

def print_summary(label: str, results: dict):
    """Print a formatted results table."""
    n = len(results["total_reward"])

    def mean(key):
        return np.mean(results[key])

    def pct(key):
        return 100 * mean(key)

    print(f"\n{'='*50}")
    print(f"  {label}  ({n} episodes)")
    print(f"{'='*50}")
    print(f"  Avg total reward    : {mean('total_reward'):+.1f}")
    print(f"  Avg episode length  : {mean('episode_length'):.1f} steps")
    print(f"  Win rate            : {pct('win'):.1f}%")
    print(f"    - P1 win rate     : {pct('win_p1'):.1f}%")
    print(f"    - P2 win rate     : {pct('win_p2'):.1f}%")
    print(f"  Loss rate           : {pct('loss'):.1f}%")
    print(f"  Truncation rate     : {pct('truncated'):.1f}%")
    print(f"  Avg invalid actions : {mean('invalid_actions'):.1f}")
    print(f"  Avg spirits         : {mean('spirits'):.2f}")
    print(f"  Avg corrupted (end) : {mean('corrupted_final'):.1f}")
    print(f"{'='*50}")


# ============================================================
# Main
# ============================================================

def main(n_episodes: int = 50):

    # ── Set up automatic file logging ──────────────────────────
    # Every print() below is duplicated to both the terminal and
    # a timestamped file in results/, so nothing is lost again
    # if the terminal window gets closed.
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/eval_{timestamp}.txt"

    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print(f"\nEvaluating over {n_episodes} episodes each...\n")

        # ── Random agent ─────────────────────────────────────────
        print("Running random agent...")
        random_results = evaluate(None, n_episodes, seed_offset=0)
        print_summary("Random Agent", random_results)

        # ── Trained PPO agent ─────────────────────────────────────
        print("\nLoading trained PPO model...")
        try:
            model = MaskablePPO.load("bloomward_ppo")
            print("Running trained agent...")
            trained_results = evaluate(model, n_episodes, seed_offset=1000)
            print_summary("Trained PPO Agent", trained_results)

            # ── Improvement summary ───────────────────────────────
            r_mean = np.mean(random_results["total_reward"])
            t_mean = np.mean(trained_results["total_reward"])
            improvement = t_mean - r_mean

            r_win = 100 * np.mean(random_results["win"])
            t_win = 100 * np.mean(trained_results["win"])

            print(f"\n{'='*50}")
            print(f"  Improvement Summary")
            print(f"{'='*50}")
            print(f"  Random avg reward  : {r_mean:+.1f}")
            print(f"  Trained avg reward : {t_mean:+.1f}")
            print(f"  Improvement        : {improvement:+.1f}")
            print(f"  Random win rate    : {r_win:.1f}%")
            print(f"  Trained win rate   : {t_win:.1f}%")
            print(f"  Invalid reduction  : "
                  f"{np.mean(random_results['invalid_actions']):.1f} → "
                  f"{np.mean(trained_results['invalid_actions']):.1f}")
            print(f"{'='*50}\n")

        except FileNotFoundError:
            print("\nboomward_ppo.zip not found.")
            print("Run 'python train_agent.py' first to train the model.")

        print(f"\nResults saved to: {log_path}\n")

    finally:
        # Always restore stdout and close the file, even if something
        # above raised an exception.
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes per agent")
    args = parser.parse_args()
    main(n_episodes=args.episodes)