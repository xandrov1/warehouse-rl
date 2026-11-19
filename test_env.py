"""Step 3 checks: scripted episodes that test every rule of the environment."""
import numpy as np

import config
from env.warehouse_env import DOWN, DROP, LEFT, PICK, RIGHT, UP, WarehouseEnv
from env.warehouse_map import NEIGHBORS


def path_moves(env, target):
    """Shortest list of move actions from the robot to target (uses BFS from target)."""
    dist = env.map.bfs(target)
    x, y = env.robot
    moves = []
    while (x, y) != target:
        for action, (dx, dy) in zip((UP, DOWN, LEFT, RIGHT), NEIGHBORS):
            nx, ny = x + dx, y + dy
            if env.map.is_free(nx, ny) and dist[ny, nx] == dist[y, x] - 1:
                moves.append(action)
                x, y = nx, ny
                break
    return moves


def nearest_pickup(env):
    dist = env.map.bfs(env.robot)
    return min(env.map.pickup, key=lambda p: dist[p[1], p[0]])


def best_pickup(env):
    """Pickup cell on the shortest start -> pickup -> goal route."""
    d_start, d_goal = env.map.bfs(env.robot), env.map.bfs(env.goal)
    return min(env.map.pickup, key=lambda p: d_start[p[1], p[0]] + d_goal[p[1], p[0]])


def run(env, actions):
    total, out = 0.0, None
    for a in actions:
        assert env.action_masks()[a], f"action {a} was masked at {env.robot}"
        out = env.step(a)
        total += out[1]
    return total, out


env = WarehouseEnv(render_mode="ansi")

# 1. Same seed gives the same episode
o1, _ = env.reset(seed=42)
g1, r1 = env.goal, env.robot
o2, _ = env.reset(seed=42)
assert np.array_equal(o1, o2) and env.goal == g1 and env.robot == r1
print("1. Reproducible resets: OK")
print(env.render())

# 2. A perfect episode: go to pickup, PICK, go to goal, DROP
for seed in range(200):
    env.reset(seed=seed)
    pickup = best_pickup(env)
    total, _ = run(env, path_moves(env, pickup))
    assert env.action_masks()[PICK] and not env.action_masks()[DROP]
    total2, _ = run(env, [PICK] + path_moves(env, env.goal))
    assert env.action_masks()[DROP] and not env.action_masks()[PICK]
    total3, (obs, reward, term, trunc, info) = run(env, [DROP])
    steps = info["episode_steps"]
    assert term and not trunc and info["is_success"] and not info["wrong_drop"]
    assert steps == info["shortest_path"] + 2, "path should be shortest moves + PICK + DROP"
    assert abs((total + total2 + total3) - (config.R_DELIVER + config.R_STEP * steps)) < 1e-6
print(f"2. Perfect delivery, 200 episodes: OK (reward = +10 - 0.01 x steps, success flagged)")

# 3. Wrong drop: carry the object to a different delivery cell
env.reset(seed=7)
run(env, path_moves(env, nearest_pickup(env)) + [PICK])
wrong = next(c for c in env.map.delivery_cells if c != env.goal)
_, (_, reward, term, _, info) = run(env, path_moves(env, wrong) + [DROP])
assert term and info["wrong_drop"] and not info["is_success"]
assert abs(reward - (config.R_WRONG_DROP + config.R_STEP)) < 1e-6
print("3. Wrong drop: OK (-5, episode ends)")

# 4. Collision: walk into the top wall
env.reset(seed=3)
run(env, path_moves(env, (env.robot[0], 0)))  # go to the top row
before = env.robot
_, reward, term, _, info = env.step(UP)
assert env.robot == before and info["collisions"] == 1 and not term
assert abs(reward - (config.R_COLLISION + config.R_STEP)) < 1e-6
print("4. Collision: OK (-1, robot stays put)")

# 5. Masks: PICK only in the pickup zone, DROP only when carrying on a delivery cell
for seed in range(50):
    env.reset(seed=seed)
    m = env.action_masks()
    assert m[:4].all() and not m[DROP]
    assert m[PICK] == (env.robot in env.map.pickup)
print("5. Action masks: OK")

# 6. Observations stay inside the observation space under random play
rng = np.random.default_rng(0)
for seed in range(20):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        assert env.observation_space.contains(obs), obs
        valid = np.flatnonzero(env.action_masks())
        obs, _, term, trunc, _ = env.step(rng.choice(valid))
        done = term or trunc
print("6. Observations in range under random play: OK")

# 7. Step limit ends the episode
env.reset(seed=1)
for _ in range(env.settings["max_steps"]):
    _, _, term, trunc, info = env.step(DOWN if env.robot[1] == 0 else UP)  # bounce, never deliver
assert trunc and not term and not info["is_success"]
print(f"7. Truncation at {env.settings['max_steps']} steps: OK")

# 8. Navigation task (early curriculum stages): empty floor, short goal distance
nav = WarehouseEnv(shelves=False, clutter_density=0.0, task="navigate", max_goal_dist=4)
for seed in range(100):
    nav.reset(seed=seed)
    assert 1 <= nav.shortest_path <= 4
    assert not nav.action_masks()[PICK] and not nav.action_masks()[DROP]
    _, (_, _, term, _, info) = run(nav, path_moves(nav, nav.goal))
    assert term and info["is_success"]
print("8. Navigation task with max goal distance 4: OK")

# 9. Settings can be changed between episodes (used by the curriculum)
env.set_settings(clutter_density=0.05, max_steps=100)
env.reset(seed=0)
assert env.settings["max_steps"] == 100 and env.map.n_clutter < 20
print("9. set_settings: OK")

print("\nAll Step 3 checks passed.")