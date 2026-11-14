"""Step 5 checks: env passes check_env, renders, and a random vs. scripted robot behave as expected."""
import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from env.warehouse_env import WarehouseEnv, scripted_action

for density in (0.0, 0.2):
    gym_check_env(WarehouseEnv(clutter_density=density), skip_render_check=True)
    sb3_check_env(WarehouseEnv(clutter_density=density))
env = WarehouseEnv(render_mode="rgb_array")
env.reset(seed=0)
assert env.render().shape == (636, 600, 3)
env.close()
print("check_env and rendering OK.")

N = 200
print(f"\n{N} episodes each:")
print(f"{'agent':10}{'clutter':>8}{'success':>9}{'wrong drop':>12}{'return':>9}{'steps':>8}")
for scripted in (False, True):
    for density in (0.0, 0.05, 0.10, 0.20):
        env = WarehouseEnv(clutter_density=density)
        rng = np.random.default_rng(0)
        events, returns, steps = [], [], []
        for ep in range(N):
            env.reset(seed=1000 + ep)
            done, total = False, 0.0
            while not done:
                mask = env.action_masks()
                a = scripted_action(env) if scripted else rng.choice(np.flatnonzero(mask))
                assert mask[a]
                _, r, term, trunc, info = env.step(a)
                total += r
                done = term or trunc
            if scripted:
                assert info["event"] == "delivered" and info["steps"] == info["shortest_path"]
            events.append(info["event"])
            returns.append(total)
            steps.append(info["steps"])
        print(f"{'scripted' if scripted else 'random':10}{density:>8.0%}"
              f"{events.count('delivered') / N:>9.0%}{events.count('wrong_drop') / N:>12.0%}"
              f"{np.mean(returns):>9.2f}{np.mean(steps):>8.1f}")

print("\nScripted robot took the shortest path every episode. Step 5 OK.")
