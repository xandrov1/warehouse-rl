"""Step 5 checks: the environment passes Gymnasium/SB3 checks, action masks are
correct, and a random agent and a scripted shortest-path agent behave as expected."""
import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import config
from env import WarehouseEnv
from env.scripted import scripted_action
from env.warehouse_env import DROP, PICK

# 1. API checks
for density in (0.0, 0.2):
    env = WarehouseEnv(clutter_density=density)
    gym_check_env(env, skip_render_check=True)
    sb3_check_env(env, warn=True)
print("Gymnasium and SB3 check_env passed (0% and 20% clutter).")

# 2. Episodes: random agent (masked) vs scripted shortest path
def run(policy, density, episodes, seed):
    env = WarehouseEnv(clutter_density=density)
    rng = np.random.default_rng(seed)
    stats = {"success": 0, "wrong_drop": 0, "returns": [], "steps": [], "optimal": 0}
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        assert env.observation_space.contains(obs)
        done, total = False, 0.0
        while not done:
            mask = env.action_masks()
            # masks must match the rules exactly
            assert mask[PICK] == ((not env.carrying) and env.robot in env.map.pickup)
            assert mask[DROP] == (env.carrying and env.map.is_delivery_cell(*env.robot))
            a = policy(env, mask, rng)
            assert mask[a], "policy chose a masked action"
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        stats["success"] += info["event"] == "delivered"
        stats["wrong_drop"] += info["event"] == "wrong_drop"
        stats["optimal"] += info["event"] == "delivered" and info["steps"] == info["shortest_path"]
        stats["returns"].append(total)
        stats["steps"].append(info["steps"])
    return stats

random_policy = lambda env, mask, rng: int(rng.choice(np.flatnonzero(mask)))
scripted_policy = lambda env, mask, rng: scripted_action(env)

N = 200
print(f"\n{N} episodes each, max {config.MAX_EPISODE_STEPS} steps:")
print(f"{'agent':10} {'clutter':>7} {'success':>8} {'wrong drop':>10} {'mean return':>12} {'mean steps':>10}")
for name, policy in (("random", random_policy), ("scripted", scripted_policy)):
    for density in (0.0, 0.05, 0.10, 0.20):
        s = run(policy, density, N, seed=1000)
        print(f"{name:10} {density:>7.0%} {s['success'] / N:>8.0%} {s['wrong_drop'] / N:>10.0%} "
              f"{np.mean(s['returns']):>12.2f} {np.mean(s['steps']):>10.1f}")
        if name == "scripted":
            assert s["success"] == N and s["optimal"] == N, "scripted agent must always take the shortest path"

print("\nScripted agent delivered every episode in exactly the shortest path. Step 5 OK.")
