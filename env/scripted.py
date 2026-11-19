"""Scripted agents for testing and demos. They are NOT the learning robot.

expert_action  uses the full map (BFS) to always take the ideal move. Good for checking the environment and for comparison videos.
random_action  picks any valid action at random. Shows what an untrained robot does.
"""
import numpy as np

from env.warehouse_env import DOWN, DROP, LEFT, PICK, RIGHT, UP
from env.warehouse_map import NEIGHBORS


def step_toward(env, target):
    """First move on a shortest path from the robot to target."""
    dist = env.map.bfs(target)
    x, y = env.robot
    for action, (dx, dy) in zip((UP, DOWN, LEFT, RIGHT), NEIGHBORS):
        nx, ny = x + dx, y + dy
        if env.map.is_free(nx, ny) and dist[ny, nx] == dist[y, x] - 1:
            return action
    raise RuntimeError("target unreachable")


def best_pickup(env):
    """Pickup cell on the shortest robot -> pickup -> goal route."""
    d_robot, d_goal = env.map.bfs(env.robot), env.map.bfs(env.goal)
    return min(env.map.pickup, key=lambda p: d_robot[p[1], p[0]] + d_goal[p[1], p[0]])


def expert_action(env):
    if env.settings["task"] == "navigate":
        return step_toward(env, env.goal)
    if not env.carrying:
        return PICK if env.robot in env.map.pickup else step_toward(env, best_pickup(env))
    return DROP if env.robot == env.goal else step_toward(env, env.goal)


def random_action(env, rng=np.random.default_rng()):
    return int(rng.choice(np.flatnonzero(env.action_masks())))