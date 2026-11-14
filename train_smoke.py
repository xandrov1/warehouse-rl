"""Short MaskablePPO run on the warehouse env (no curriculum) to make sure training works end to end."""
import os

import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor, load_results

import config
from env.warehouse_env import WarehouseEnv

STEPS = 300_000
run_dir = os.path.join(config.RESULTS_DIR, "smoke")
os.makedirs(run_dir, exist_ok=True)

env = Monitor(WarehouseEnv(), run_dir, info_keywords=("event",))
env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
model = MaskablePPO("MlpPolicy", env, seed=config.SEEDS[0], verbose=0)
model.learn(total_timesteps=STEPS)
model.save(os.path.join(run_dir, "model"))

log = load_results(run_dir)
log["success"] = (log["event"] == "delivered") * 100.0
smooth = log[["r", "success"]].rolling(config.SUCCESS_WINDOW, min_periods=1).mean()
steps = log["l"].cumsum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(steps, smooth["r"])
ax1.set(xlabel="environment steps", ylabel="episode return", title="Return (last 100 episodes)")
ax2.plot(steps, smooth["success"])
ax2.set(xlabel="environment steps", ylabel="success %", ylim=(0, 100), title="Delivery success (last 100)")
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "smoke_training.png"), dpi=150)
print(f"{len(log)} episodes, last 100: return {smooth['r'].iloc[-1]:.2f}, "
      f"success {smooth['success'].iloc[-1]:.0f}%")
