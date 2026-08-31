#!/usr/bin/env python3
# ============================================================
# train_agent.py
#
# Train a PPO agent on the Bloomward environment using
# Stable-Baselines3.
#
# Usage:  python train_agent.py
# ============================================================
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from bloomward_env.env import BloomwardEnv


def main():
    # ── 1. Create and validate the environment ────────────────
    print("Creating environment...")
    env = BloomwardEnv(render_mode=None)
    print("Running environment check...")
    check_env(env, warn=True)
    print("check_env passed!\n")

    # ── 2. Wrap with ActionMasker then Monitor ────────────────
    env = ActionMasker(env, lambda e: e.action_masks())
    env = Monitor(env, filename="./logs/bloomward_monitor")

    # ── 3. Create the PPO agent ───────────────────────────────
    # MultiInputPolicy is required for Dict observation spaces.
    #
    # learning_rate: linearly decays from 0.0003 to 0 over training.
    #   Fixed-rate training let 10M-step updates stay large even
    #   after the policy had already converged around 5M steps,
    #   which likely knocked a good solution off course (30% win
    #   rate at 5M dropped to 20% at 10M with a fixed rate). A
    #   decaying schedule lets late-stage updates fine-tune rather
    #   than destabilize.
    #
    # ent_coef: small entropy bonus (default is 0.0) to discourage
    #   the policy from collapsing onto one strategy too early.
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=lambda progress_remaining: 0.0003 * progress_remaining,
        ent_coef=0.01,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="./logs/",
    )

    # ── 4. Train ──────────────────────────────────────────────
    print("Training started — watch ep_rew_mean rise above the random baseline.")
    print("Random agent baseline: approx -430 total reward\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="bloomward_ppo"
    )

    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)

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
    os.makedirs("checkpoints", exist_ok=True)
    main()