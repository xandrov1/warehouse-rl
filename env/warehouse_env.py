"""Gymnasium environment: a robot picks up an object and delivers it to one delivery cell.

Actions:      0 up, 1 down, 2 left, 3 right, 4 PICK, 5 DROP
Observation:  11 numbers, all roughly in [-1, 1]
                [0:8]  distance to the nearest obstacle in 8 directions
                       (N, NE, E, SE, S, SW, W, NW), divided by the grid size
                [8:10] dx, dy to the current target, divided by the grid size
                [10]   1.0 if carrying an object, else 0.0
              The target is the nearest pickup cell while empty-handed,
              and the order's delivery cell while carrying.

Tasks:
    "full"      pick up at the pickup zone, then drop at the goal delivery cell
    "navigate"  just reach the goal cell (PICK and DROP are always masked);
                used by the early curriculum stages
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from env.warehouse_map import NEIGHBORS, WarehouseMap

UP, DOWN, LEFT, RIGHT, PICK, DROP = range(6)
MOVE_DELTAS = NEIGHBORS  # same order as UP, DOWN, LEFT, RIGHT
RAY_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    def __init__(self, width=15, height=15, shelves=True, clutter_density=0.20,
                 task="full", max_goal_dist=None, max_order_dist=None, max_steps=200,
                 render_mode=None):
        super().__init__()
        assert task in ("full", "navigate")
        assert shelves or task == "navigate", "the full task needs shelves"
        self.settings = dict(width=width, height=height, shelves=shelves,
                             clutter_density=clutter_density, task=task,
                             max_goal_dist=max_goal_dist, max_order_dist=max_order_dist,
                             max_steps=max_steps)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = None

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)

        self.map = None
        self.robot = None
        self.goal = None
        self.carrying = False

    # curriculum
    def set_settings(self, **kwargs):
        """Change the task settings. Takes effect at the next reset().
        The curriculum (Step 7) calls this to move between stages."""
        unknown = set(kwargs) - set(self.settings)
        assert not unknown, f"unknown settings: {unknown}"
        self.settings.update(kwargs)

    # gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.settings
        if options:
            self.set_settings(**options)

        self.map = WarehouseMap(width=s["width"], height=s["height"], shelves=s["shelves"],
                                clutter_density=s["clutter_density"], rng=self.np_random)
        self._place_robot_and_goal()

        self.carrying = False
        self.steps = 0
        self.collisions = 0
        self.picked_at_step = None
        self.episode_reward = 0.0
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        s = self.settings
        reward = config.R_STEP
        terminated = False
        self.steps += 1
        outcome = None  # "success", "wrong_drop", or None

        if action in (UP, DOWN, LEFT, RIGHT):
            dx, dy = MOVE_DELTAS[action]
            nx, ny = self.robot[0] + dx, self.robot[1] + dy
            if self.map.is_free(nx, ny):
                self.robot = (nx, ny)
                if s["task"] == "navigate" and self.robot == self.goal:
                    reward += config.R_DELIVER
                    terminated, outcome = True, "success"
            else:
                reward += config.R_COLLISION  # robot stays where it is
                self.collisions += 1

        elif action == PICK:
            # Only reachable if the mask allowed it: empty-handed, in the pickup zone
            if self._can_pick():
                self.carrying = True
                self.picked_at_step = self.steps

        elif action == DROP:
            # Only reachable if the mask allowed it: carrying, on a delivery cell
            if self._can_drop():
                self.carrying = False
                terminated = True
                if self.robot == self.goal:
                    reward += config.R_DELIVER
                    outcome = "success"
                else:
                    reward += config.R_WRONG_DROP
                    outcome = "wrong_drop"

        self.episode_reward += reward
        truncated = (not terminated) and self.steps >= s["max_steps"]
        info = self._info()
        if terminated or truncated:
            info.update(is_success=outcome == "success",
                        wrong_drop=outcome == "wrong_drop",
                        episode_steps=self.steps,
                        shortest_path=self.shortest_path)
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def action_masks(self):
        """Boolean array of valid actions, read by MaskablePPO.
        Moves are always allowed (bumping into things is penalized, not masked)."""
        mask = np.ones(6, dtype=bool)
        mask[PICK] = self._can_pick()
        mask[DROP] = self._can_drop()
        return mask

    def render(self):
        """ansi: text picture. human: Pygame window (drawn automatically every step).
        rgb_array: image as a numpy array, used for saving GIFs and videos."""
        if self.render_mode is None or self.map is None:
            return None
        if self.render_mode == "ansi":
            return self.map.ascii(robot=self.robot, goal=self.goal)
        if self.renderer is not None and self.renderer.grid_size != (self.map.width, self.map.height):
            self.renderer.close()  # map size changed between stages
            self.renderer = None
        if self.renderer is None:
            from env.renderer import Renderer
            self.renderer = Renderer(self.map.width, self.map.height,
                                     self.render_mode, self.metadata["render_fps"])
        return self.renderer.draw(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # helpers
    def _can_pick(self):
        return (self.settings["task"] == "full" and not self.carrying
                and self.robot in self.map.pickup)

    def _can_drop(self):
        return self.carrying and self.map.is_delivery_cell(*self.robot)

    def _place_robot_and_goal(self):
        """Pick a goal and a start cell, honoring max_goal_dist if it is set.
        Also stores the shortest possible number of moves for this episode."""
        s = self.settings
        rng = self.np_random
        m = self.map

        if s["task"] == "full":
            goal_choices = list(m.delivery_cells)
            if s["max_order_dist"] is not None:
                # full task: limit how far the order's delivery cell is from the pickup zone
                pick_dist = self._dist_to_pickup()
                near = [c for c in goal_choices if 0 <= pick_dist[c[1], c[0]] <= s["max_order_dist"]]
                goal_choices = near or goal_choices
        else:
            # navigate: delivery cells if there are shelves, otherwise any open cell
            goal_choices = list(m.delivery_cells) or [c for c in m.free_cells()
                                                     if c not in m.pickup]
        self.goal = goal_choices[rng.integers(len(goal_choices))]
        goal_dist = m.bfs(self.goal)

        starts = [c for c in m.free_cells() if c != self.goal and c not in m.pickup]
        if s["max_goal_dist"] is not None:
            if s["task"] == "navigate":
                starts = [c for c in starts if goal_dist[c[1], c[0]] <= s["max_goal_dist"]]
            else:
                # full task: limit how far the robot starts from the pickup zone
                pick_dist = self._dist_to_pickup()
                starts = [c for c in starts if pick_dist[c[1], c[0]] <= s["max_goal_dist"]]
        if not starts:  # max_goal_dist too small for this map: fall back to any start
            starts = [c for c in m.free_cells() if c != self.goal and c not in m.pickup]
        self.robot = starts[rng.integers(len(starts))]

        # Shortest number of moves to finish the task (for the path-length metric)
        if s["task"] == "navigate":
            self.shortest_path = int(goal_dist[self.robot[1], self.robot[0]])
        else:
            start_dist = m.bfs(self.robot)
            self.shortest_path = int(min(start_dist[y, x] + goal_dist[y, x]
                                         for (x, y) in m.pickup))

    def _dist_to_pickup(self):
        """Shortest distance from every cell to the nearest pickup cell."""
        dists = [self.map.bfs(p) for p in self.map.pickup]
        d = np.min(np.where(np.stack(dists) < 0, 10**6, np.stack(dists)), axis=0)
        return np.where(d >= 10**6, -1, d)

    def _target(self):
        if self.settings["task"] == "full" and not self.carrying:
            x, y = self.robot
            return min(self.map.pickup, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        return self.goal

    def _ray(self, dx, dy):
        """Cells until the first shelf, clutter, or wall in this direction (1 = adjacent)."""
        x, y = self.robot
        dist = 0
        while True:
            x, y = x + dx, y + dy
            dist += 1
            if not self.map.is_free(x, y):
                return dist

    def _obs(self):
        size = max(self.map.width, self.map.height)
        rays = [self._ray(dx, dy) / size for dx, dy in RAY_DIRS]
        tx, ty = self._target()
        dx = (tx - self.robot[0]) / self.map.width
        dy = (ty - self.robot[1]) / self.map.height
        return np.array(rays + [dx, dy, float(self.carrying)], dtype=np.float32)

    def _info(self):
        return {"collisions": self.collisions, "carrying": self.carrying}