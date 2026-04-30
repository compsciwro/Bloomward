#!/usr/bin/env python3
# ============================================================
# train_agent.py
#
# Train a PPO agent on the Bloomward environment using
# Stable-Baselines3.
#
# Usage:  python train_agent.py
# ============================================================

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from bloomward_env.env import BloomwardEnv


def main():
    # ── 1. Create and validate the environment ────────────────
    print("Creating environment...")
    env = BloomwardEnv(render_mode=None)

    print("Running environment check...")
    check_env(env, warn=True)
    print("check_env passed!\n")

    # ── 2. Wrap with Monitor to record episode statistics ─────
    # Monitor saves reward/length data to a CSV in ./logs/
    env = Monitor(env, filename="./logs/bloomward_monitor")

    # ── 3. Create the PPO agent ───────────────────────────────
    # MultiInputPolicy is required for Dict observation spaces.
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="./logs/",
    )

    # ── 4. Train ──────────────────────────────────────────────
    print("Training started — watch ep_rew_mean rise above the random baseline.")
    print("Random agent baseline: approx -430 total reward\n")

    model.learn(total_timesteps=100_000)

    # ── 5. Save ───────────────────────────────────────────────
    model.save("bloomward_ppo")
    print("\nTraining complete!")
    print("Model saved → bloomward_ppo.zip")
    print("\nNext steps:")
    print("  python test_agent.py        ← watch the trained agent play")
    print("  python evaluate_agent.py    ← compare trained vs random")
    print("  tensorboard --logdir logs/  ← view training graphs")

    env.close()


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    main()
