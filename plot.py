"""Step 10: plot training progress, curriculum vs. baseline.

    python plot.py   ->  figures/learning_curves.png

Left: success rate over the last 100 training episodes, on whatever stage the agent
is on at that point. Right: which stage each run is on.
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

import config

COLORS = {"baseline": "#e8a33d", "curriculum": "#2f6fdb"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for agent, color in COLORS.items():
    runs = [pd.read_csv(p) for p in sorted(glob.glob(f"{config.RESULTS_DIR}/{agent}_s*/eval_curve.csv"))]
    for i, df in enumerate(runs):
        label = f"{agent} ({len(runs)} seeds)" if i == 0 else None
        ax1.plot(df.timesteps, df.train_success, color=color, lw=1.5, alpha=0.8, label=label)
        ax2.plot(df.timesteps, df.stage + (i - 1) * 0.03, color=color, lw=1.5, alpha=0.8, label=label)

ax1.axhline(config.ADVANCE_THRESHOLD, color="gray", ls="--", lw=1)
ax1.text(0, config.ADVANCE_THRESHOLD + 0.02, "advance threshold (80%)", color="gray", fontsize=9)
ax1.set(title="Training success rate (last 100 episodes, current stage)",
        xlabel="training steps", ylabel="success rate", ylim=(-0.02, 1.02))
ax1.legend(loc="center right")
ax1.grid(alpha=0.3)
ax2.set(title="Curriculum stage", xlabel="training steps", ylabel="stage", yticks=[1, 2, 3, 4])
ax2.legend(loc="center right")
ax2.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(config.FIGURES_DIR, exist_ok=True)
fig.savefig(os.path.join(config.FIGURES_DIR, "learning_curves.png"), dpi=150)
print("saved figures/learning_curves.png")
