"""Gymnasium env: walk to the pickup zone, PICK, walk to the order's delivery cell, DROP.

Actions: 0 up, 1 down, 2 left, 3 right, 4 PICK, 5 DROP
Obs: [0:8] free cells to the nearest obstacle in 8 directions, [8:10] (dx, dy) to the
target (pickup before PICK, goal after), [10] carrying. All scaled by the grid size.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from env.render import Renderer
from env.warehouse_map import NEIGHBORS, WarehouseMap

PICK, DROP = 4, 5
RAYS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, clutter_density=0.0, render_mode=None):
        self.clutter_density = clutter_density
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        low = np.array([0] * 8 + [-1, -1, 0], dtype=np.float32)
        self.observation_space = spaces.Box(low, np.ones(11, dtype=np.float32))
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m = self.map = WarehouseMap(clutter_density=self.clutter_density, rng=self.np_random)
        cells = sorted(m.delivery_cells)
        self.goal = cells[self.np_random.integers(len(cells))]
        starts = [c for c in m.free_cells() if c not in m.pickup and c != self.goal]
        self.robot = starts[self.np_random.integers(len(starts))]
        self.carrying = False
        self.steps = self.collisions = 0

        # best pickup cell for the whole trip, used for the path-length metric and scripted robot
        to_goal, from_start = m.bfs(self.goal), m.bfs(self.robot)
        via = min(sorted(m.pickup), key=lambda c: from_start[c[1], c[0]] + to_goal[c[1], c[0]])
        self.shortest_path = int(from_start[via[1], via[0]] + to_goal[via[1], via[0]]) + 2
        self.dist = {False: m.bfs(via), True: to_goal}

        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info(None)

    def step(self, action):
        mask = self.action_masks()
        reward, terminated, event = config.R_STEP, False, None
        self.steps += 1

        if action < PICK:
            dx, dy = NEIGHBORS[action]
            nxt = (self.robot[0] + dx, self.robot[1] + dy)
            if self.map.is_free(*nxt):
                self.robot = nxt
            else:
                reward += config.R_COLLISION
                self.collisions += 1
                event = "collision"
        elif action == PICK and mask[PICK]:
            self.carrying, event = True, "pick"
        elif action == DROP and mask[DROP]:
            terminated = True
            if self.robot == self.goal:
                reward, event = reward + config.R_DELIVER, "delivered"
            else:
                reward, event = reward + config.R_WRONG_DROP, "wrong_drop"

        truncated = not terminated and self.steps >= config.MAX_EPISODE_STEPS
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info(event)

    def action_masks(self):
        mask = np.ones(6, dtype=bool)
        mask[PICK] = not self.carrying and self.robot in self.map.pickup
        mask[DROP] = self.carrying and self.map.is_delivery_cell(*self.robot)
        return mask

    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = Renderer(self.map.width, self.map.height, self.render_mode == "human")
        return self.renderer.draw(self)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def _obs(self):
        m, (x0, y0) = self.map, self.robot
        size = max(m.width, m.height)
        rays = []
        for dx, dy in RAYS:
            n = 1
            while m.is_free(x0 + n * dx, y0 + n * dy):
                n += 1
            rays.append((n - 1) / size)
        if self.carrying:
            tx, ty = self.goal
        else:
            tx, ty = min(m.pickup, key=lambda c: (abs(c[0] - x0) + abs(c[1] - y0), c))
        return np.array(rays + [(tx - x0) / size, (ty - y0) / size, float(self.carrying)],
                        dtype=np.float32)

    def _info(self, event):
        return {"event": event, "steps": self.steps, "collisions": self.collisions,
                "shortest_path": self.shortest_path}


def scripted_action(env):
    """Follows the BFS shortest path. Not a learner, just proof every episode is solvable."""
    mask = env.action_masks()
    if mask[PICK]:
        return PICK
    if mask[DROP] and env.robot == env.goal:
        return DROP
    dist, (x, y) = env.dist[env.carrying], env.robot
    moves = [(dist[y + dy, x + dx], a) for a, (dx, dy) in enumerate(NEIGHBORS)
             if env.map.is_free(x + dx, y + dy)]
    return min(moves)[1]
