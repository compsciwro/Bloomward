#!/usr/bin/env python3
# ============================================================
# demo_random_agent.py
#
# Runs one episode with a random agent and prints statistics.
# Use this to confirm the environment is working before training.
#
# Usage:  python demo_random_agent.py
# ============================================================

from bloomward_env.env import BloomwardEnv
from bloomward_env.constants import SEASON_NAMES, FLOWER_NAMES


def run_demo(seed: int = 42, verbose: bool = True):
    env = BloomwardEnv(render_mode="ansi")
    obs, info = env.reset(seed=seed)

    print(f"\n{'='*55}")
    print(f"  Bloomward — Random Agent Demo  (seed={seed})")
    print(f"{'='*55}")
    print(f"  Board tiles  : {env._n_tiles}")
    print(f"  Max turns    : {env.max_turns}")
    print(f"  Obs in space : {env.observation_space.contains(obs)}")
    print(f"{'='*55}\n")

    print(env.render())

    total_reward  = 0.0
    invalid_count = 0
    combo_count   = 0

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get("invalid_action", False):
            invalid_count += 1
        combo_count += info.get("combos", 0)

        if verbose and step % 10 == 0:
            flower = FLOWER_NAMES.get(env._current_flower, "?")
            season = SEASON_NAMES[info["season"]]
            print(
                f"  Step {step:3d} | "
                f"action={action:2d} | "
                f"reward={reward:+6.1f} | "
                f"season={season:7s} | "
                f"spirits={env._spirit_count:2d} | "
                f"invalid={info.get('invalid_action', False)}"
            )

        if terminated or truncated:
            reason = info.get("reason", "unknown")
            print(f"\n  >>> Episode ended at step {step+1}: {reason} <<<\n")
            print(env.render())
            break

    in_space = env.observation_space.contains(obs)

    print(f"\n{'='*55}")
    print(f"  Episode Summary")
    print(f"{'='*55}")
    print(f"  Total reward    : {total_reward:.1f}")
    print(f"  Invalid actions : {invalid_count}")
    print(f"  Combos formed   : {combo_count}")
    print(f"  Final spirits   : {env._spirit_count}")
    print(f"  Obs in space ✓  : {in_space}")
    print(f"{'='*55}\n")

    assert in_space, "ERROR: final observation is outside observation_space!"

    env.close()
    return {
        "total_reward":  total_reward,
        "invalid_count": invalid_count,
        "combo_count":   combo_count,
        "spirit_count":  env._spirit_count,
    }


if __name__ == "__main__":
    run_demo(seed=42, verbose=True)
