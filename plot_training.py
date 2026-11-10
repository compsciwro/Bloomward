# plot_training.py
# Training convergence graph for the MaskablePPO agent (report Section 6.4).
# Method: sliding window over the episode rewards, as in the Q-learning practical.
# Each point is the average total reward of the last WINDOW training episodes.
# Data: the Monitor logs written during training (one row per finished episode).

import os

from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np

log_dir = "./logs/v2/"
WINDOW  = 1000    # episodes in the sliding window
CUT     = None    # None = show the whole run. e.g. 6000 = show only the first 6,000 episodes.

# x = episode number, y = total reward of that episode
x, y = ts2xy(load_results(log_dir), "episodes")
x = x + 1

def sliding_window_average(values, window):
    """Average of each run of `window` consecutive values."""
    return np.convolve(values, np.ones(window) / window, mode="valid")

average = sliding_window_average(y, WINDOW)

plt.rcParams.update({"font.family": "serif", "font.size": 10})
plt.figure(figsize=(7, 4))
plt.plot(x[WINDOW - 1:], average, color="deeppink", linewidth=1.2)
plt.axhline(0, color="grey", linewidth=0.8, linestyle="--")
plt.xlabel("Training episode")
plt.ylabel(f"Average total reward (last {WINDOW} episodes)")
plt.title(f"MaskablePPO Training Reward Across Episodes\n({WINDOW}-Episode Moving Average)")
plt.grid(alpha=0.3)
if CUT:
    plt.xlim(0, CUT)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/training_convergence.png", dpi=300)
plt.savefig("results/training_convergence.pdf")
print("Saved results/training_convergence.png and .pdf")

# ---- Numbers to quote in the report ----
print(f"\n{len(y):,} episodes in total; episode rewards ranged from {y.min():.1f} to {y.max():.1f}")
print(f"{WINDOW}-episode average: first {average[0]:.2f}, highest {average.max():.2f} "
      f"(episode {int(x[WINDOW - 1 + average.argmax()]):,}), last {average[-1]:.2f}")
print("Average reward in each tenth of training:")
for i, part in enumerate(np.array_split(y, 10)):
    print(f"  {i * 10:3d}-{(i + 1) * 10:3d}%: {part.mean():6.2f}")
last = average[-len(average) // 5:]
print(f"In the last fifth of training the {WINDOW}-episode average moved between "
      f"{last.min():.2f} and {last.max():.2f}")

plt.show()