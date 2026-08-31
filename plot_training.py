from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np

log_dir = "./logs/"

x, y = ts2xy(load_results(log_dir), "timesteps")

window = 100

def plot_on(ax, x_data, y_data, title):
    ax.plot(x_data, y_data, alpha=0.3, label="Episode reward")
    if len(y_data) >= window:
        y_smooth = np.convolve(y_data, np.ones(window)/window, mode="valid")
        x_smooth = x_data[window-1:]
        ax.plot(x_smooth, y_smooth, label=f"{window}-episode moving average", linewidth=2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Full run
plot_on(ax1, x, y, "Full Training Run (5M steps)")

# Early-training zoom -- first 500k steps only, own auto-scaled y-axis
mask = x <= 500_000
x_early = x[mask]
y_early = y[mask]
plot_on(ax2, x_early, y_early, "Early Training (first 500k steps)")

plt.tight_layout()
plt.savefig("results/training_convergence.png", dpi=150)
plt.show()
print("Saved to results/training_convergence.png")