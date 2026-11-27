"""Step 5: sanity checks before training.

    python sanity_check.py

1. Gymnasium's and Stable-Baselines3's environment checkers.
2. The scripted expert on 800 episodes: it should always deliver, in the shortest path.
3. A random agent on the same episodes: it should almost never deliver.
"""
import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check

from env.scripted import expert_action, random_action
from env.warehouse_env import WarehouseEnv

gym_check(WarehouseEnv(), skip_render_check=True)
sb3_check(WarehouseEnv())
print("check_env (Gymnasium and SB3): OK")

rng = np.random.default_rng(0)
for name, policy in [("expert", expert_action), ("random", lambda e: random_action(e, rng))]:
    env = WarehouseEnv()
    wins = shortest = 0
    rewards = []
    for ep in range(800):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy(env))
            done = term or trunc
        wins += info["is_success"]
        shortest += info["is_success"] and info["episode_steps"] == info["shortest_path"] + 2
        rewards.append(env.episode_reward)
    print(f"{name:6s}: delivered {wins}/800, shortest path {shortest}/800, "
          f"mean reward {np.mean(rewards):+.2f}")
