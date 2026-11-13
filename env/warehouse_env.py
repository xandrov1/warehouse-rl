"""Gymnasium environment: one robot picks up an object and delivers it to one shelf spot.

Actions:      0 up, 1 down, 2 left, 3 right, 4 PICK, 5 DROP
Observation:  11 floats
              [0:8]  free cells to the nearest obstacle in 8 directions
                     (N, NE, E, SE, S, SW, W, NW), divided by the grid size
              [8:10] (dx, dy) to the current target, divided by the grid size
                     (target = nearest pickup cell before PICK, the order's
                     delivery cell after)
              [10]   1.0 if carrying, else 0.0
Rewards:      config.R_DELIVER on the correct drop (ends the episode),
              config.R_WRONG_DROP on any other delivery cell (ends the episode),
              config.R_COLLISION for bumping into a wall, shelf or clutter,
              config.R_STEP every step.
Action masks: PICK only on a pickup cell while empty-handed,
              DROP only on a delivery cell while carrying. Moves are always allowed.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from env.warehouse_map import NEIGHBORS, WarehouseMap

UP, DOWN, LEFT, RIGHT, PICK, DROP = range(6)
ACTION_NAMES = ["up", "down", "left", "right", "pick", "drop"]
RAYS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


class WarehouseEnv(gym.Env):
    """Args:
        clutter_density: fraction of open floor covered by clutter (see WarehouseMap)
        max_steps:       episode is truncated after this many steps
        render_mode:     None, "human" (Pygame window) or "rgb_array"
        map_kwargs:      passed through to WarehouseMap (width, height, segment_len, ...)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, clutter_density=0.0, max_steps=config.MAX_EPISODE_STEPS,
                 render_mode=None, **map_kwargs):
        if map_kwargs.get("shelves") is False:
            raise ValueError("the delivery task needs shelves; navigation-only stages come in Step 6")
        self.clutter_density = clutter_density
        self.max_steps = max_steps
        self.map_kwargs = map_kwargs
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        low = np.array([0.0] * 8 + [-1.0, -1.0, 0.0], dtype=np.float32)
        high = np.ones(11, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.map = None
        self._renderer = None

    # gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = WarehouseMap(clutter_density=self.clutter_density,
                                rng=self.np_random, **self.map_kwargs)
        m = self.map
        deliveries = sorted(m.delivery_cells)
        self.goal = deliveries[self.np_random.integers(len(deliveries))]
        starts = [c for c in m.free_cells() if c not in m.pickup and c != self.goal]
        self.robot = starts[self.np_random.integers(len(starts))]
        self.carrying = False
        self.steps = 0
        self.collisions = 0
        self._goal_dist = m.bfs(self.goal)
        self.shortest_path = self._shortest_task_length()
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        m = self.map
        reward = config.R_STEP
        terminated = False
        event = None
        self.steps += 1

        if action in (UP, DOWN, LEFT, RIGHT):
            dx, dy = NEIGHBORS[action]
            nx, ny = self.robot[0] + dx, self.robot[1] + dy
            if m.is_free(nx, ny):
                self.robot = (nx, ny)
            else:
                reward += config.R_COLLISION
                self.collisions += 1
                event = "collision"
        elif action == PICK:
            if not self.carrying and self.robot in m.pickup:
                self.carrying = True
                event = "pick"
        elif action == DROP:
            if self.carrying and m.is_delivery_cell(*self.robot):
                terminated = True
                if self.robot == self.goal:
                    reward += config.R_DELIVER
                    event = "delivered"
                else:
                    reward += config.R_WRONG_DROP
                    event = "wrong_drop"

        truncated = not terminated and self.steps >= self.max_steps
        info = self._info()
        info["event"] = event
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

    def action_masks(self):
        mask = np.ones(6, dtype=bool)
        mask[PICK] = (not self.carrying) and self.robot in self.map.pickup
        mask[DROP] = self.carrying and self.map.is_delivery_cell(*self.robot)
        return mask

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from env.render import Renderer
            self._renderer = Renderer(self.map.width, self.map.height,
                                      human=self.render_mode == "human",
                                      fps=self.metadata["render_fps"])
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # helpers
    def target(self):
        """Nearest pickup cell (by grid distance) before PICK, the goal after."""
        if self.carrying:
            return self.goal
        rx, ry = self.robot
        return min(self.map.pickup, key=lambda c: (abs(c[0] - rx) + abs(c[1] - ry), c))

    def _obs(self):
        m = self.map
        size = max(m.width, m.height)
        rays = []
        x0, y0 = self.robot
        for dx, dy in RAYS:
            n, x, y = 0, x0 + dx, y0 + dy
            while m.is_free(x, y):
                n += 1
                x, y = x + dx, y + dy
            rays.append(n / size)
        tx, ty = self.target()
        obs = rays + [(tx - x0) / size, (ty - y0) / size, float(self.carrying)]
        return np.array(obs, dtype=np.float32)

    def _shortest_task_length(self):
        """Fewest actions to finish from the start: walk to pickup, PICK, walk to goal, DROP."""
        from_start = self.map.bfs(self.robot)
        best = min(from_start[y, x] + self._goal_dist[y, x] for (x, y) in self.map.pickup)
        return int(best) + 2

    def _info(self):
        return {"carrying": self.carrying, "steps": self.steps,
                "collisions": self.collisions, "shortest_path": self.shortest_path}
