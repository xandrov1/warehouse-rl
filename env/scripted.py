"""Scripted shortest-path robot. Not a learner: it proves every episode is solvable
and gives the best possible score to compare trained agents against."""
from env.warehouse_env import DOWN, DROP, LEFT, PICK, RIGHT, UP
from env.warehouse_map import NEIGHBORS


def scripted_action(env):
    m = env.map
    if env.carrying:
        if env.robot == env.goal:
            return DROP
        dist = env._goal_dist
    else:
        if env.robot in m.pickup:
            return PICK
        # head for the pickup cell on the shortest whole trip (robot -> pickup -> goal)
        here = m.bfs(env.robot)
        via = min(sorted(m.pickup), key=lambda c: here[c[1], c[0]] + env._goal_dist[c[1], c[0]])
        dist = m.bfs(via)
    x, y = env.robot
    best, best_a = dist[y, x], UP
    for a, (dx, dy) in zip((UP, DOWN, LEFT, RIGHT), NEIGHBORS):
        nx, ny = x + dx, y + dy
        if m.is_free(nx, ny) and 0 <= dist[ny, nx] < best:
            best, best_a = dist[ny, nx], a
    return best_a
