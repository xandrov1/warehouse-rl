"""Short MaskablePPO run on the warehouse environment: proves training, logging and
saving work end to end on our own env. Not a result; the real runs are Step 8."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

import config
from env import WarehouseEnv

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--clutter", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=config.SEEDS[0])
args = parser.parse_args()

run_dir = os.path.join(config.RESULTS_DIR, "smoke")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)

env = Monitor(WarehouseEnv(clutter_density=args.clutter), os.path.join(run_dir, "monitor"),
              info_keywords=("event",))
env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
model = MaskablePPO("MlpPolicy", env, seed=args.seed, verbose=0,
                    tensorboard_log=os.path.join(run_dir, "tb"))
model.learn(total_timesteps=args.steps, progress_bar=False)
model.save(os.path.join(run_dir, "model"))

log = pd.read_csv(os.path.join(run_dir, "monitor.monitor.csv"), skiprows=1)
log["success"] = (log["event"] == "delivered").astype(float)
log["steps_done"] = log["l"].cumsum()
roll = log[["r", "success"]].rolling(config.SUCCESS_WINDOW, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(log["steps_done"], roll["r"])
ax1.set(xlabel="environment steps", ylabel="episode return",
        title=f"Return (mean of last {config.SUCCESS_WINDOW})")
ax2.plot(log["steps_done"], roll["success"] * 100)
ax2.set(xlabel="environment steps", ylabel="success %", ylim=(0, 100),
        title=f"Delivery success (last {config.SUCCESS_WINDOW})")
fig.suptitle(f"MaskablePPO smoke run: {args.steps:,} steps, {args.clutter:.0%} clutter, seed {args.seed}")
fig.tight_layout()
out = os.path.join(config.FIGURES_DIR, "smoke_training.png")
fig.savefig(out, dpi=150)

first, last = log.head(config.SUCCESS_WINDOW), log.tail(config.SUCCESS_WINDOW)
print(f"{len(log)} episodes. First {config.SUCCESS_WINDOW}: return {first['r'].mean():.2f}, "
      f"success {first['success'].mean():.0%}. Last {config.SUCCESS_WINDOW}: return "
      f"{last['r'].mean():.2f}, success {last['success'].mean():.0%}.")
print(f"Saved model to {run_dir}/model.zip and plot to {out}. Training smoke test OK.")
