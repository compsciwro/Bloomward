# ============================================================
# bloomward_env/opponent.py
#
# Placeholder for future opponent logic.
#
# Planned extensions (in order of development):
#
# 1. RandomOpponent
#    Chooses a random valid tile each turn.
#    Used as a baseline for evaluating the trained RL agent.
#
# 2. RuleBasedOpponent
#    Heuristic placement: prefers tiles that form or block combos.
#    Useful for testing before RL training is complete.
#
# 3. TrainedAgentOpponent
#    Wraps a saved PPO model.
#    Used in the human-vs-AI game mode.
#    Example:
#        model = PPO.load("bloomward_ppo")
#        action, _ = model.predict(obs, deterministic=True)
#
# This file will be populated when the single-agent prototype
# is stable and human-vs-AI mode development begins.
# ============================================================
