"""Pygame drawing for the warehouse. Used by WarehouseEnv.render().

Colors match Figure 2 in the paper:
    blue circle = robot (a small orange box on it = carrying)
    green = goal delivery cell, light green = other delivery cells
    gray = shelf, orange = pickup zone, brown = clutter
"""
import os

import numpy as np

from env.warehouse_map import CLUTTER, SHELF

CELL = 40          # pixels per grid cell
HUD = 36           # height of the status bar under the grid

C_FLOOR = (255, 255, 255)
C_GRID = (213, 217, 222)
C_SHELF = (91, 100, 112)
C_CLUTTER = (176, 122, 60)
C_PICKUP = (232, 163, 61)
C_DELIVERY = (220, 240, 228)
C_GOAL = (46, 158, 91)
C_ROBOT = (47, 111, 219)
C_TEXT = (40, 44, 52)
C_HUD_BG = (244, 246, 249)


class Renderer:
    def __init__(self, width, height, mode, fps):
        if mode != "human":
            # draw off-screen, so rgb_array works without a display (e.g. on a server)
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # imported here so the env works even where pygame is missing
        self.pg = pygame
        pygame.display.init()  # only what what's needed (pygame.init also starts audio)
        pygame.font.init()
        self.mode, self.fps = mode, fps
        self.grid_size = (width, height)
        self.size = (width * CELL, height * CELL + HUD)
        if mode == "human":
            self.window = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Warehouse RL")
            self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface(self.size)
        self.font = pygame.font.SysFont("arial", 16)

    def draw(self, env):
        pg, m, s = self.pg, env.map, self.canvas
        s.fill(C_FLOOR)

        # cells
        for y in range(m.height):
            for x in range(m.width):
                color = None
                if m.grid[y, x] == SHELF:
                    color = C_SHELF
                elif m.grid[y, x] == CLUTTER:
                    color = C_CLUTTER
                elif (x, y) in m.pickup:
                    color = C_PICKUP
                elif (x, y) == env.goal:
                    color = C_GOAL
                elif (x, y) in m.delivery_cells:
                    color = C_DELIVERY
                rect = pg.Rect(x * CELL, y * CELL, CELL, CELL)
                if color:
                    pg.draw.rect(s, color, rect)
                pg.draw.rect(s, C_GRID, rect, 1)

        # robot
        rx, ry = env.robot
        center = (rx * CELL + CELL // 2, ry * CELL + CELL // 2)
        pg.draw.circle(s, C_ROBOT, center, int(CELL * 0.36))
        if env.carrying:
            box = pg.Rect(0, 0, CELL // 3, CELL // 3)
            box.center = center
            pg.draw.rect(s, C_PICKUP, box)
            pg.draw.rect(s, (255, 255, 255), box, 1)

        # status bar
        top = m.height * CELL
        pg.draw.rect(s, C_HUD_BG, pg.Rect(0, top, self.size[0], HUD))
        status = "carrying" if env.carrying else "empty-handed"
        if env.settings["task"] == "navigate":
            status = "navigate"
        text = (f"step {env.steps}/{env.settings['max_steps']}   "
                f"reward {env.episode_reward:+.2f}   {status}   "
                f"collisions {env.collisions}")
        s.blit(self.font.render(text, True, C_TEXT), (10, top + 9))

        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
                    raise SystemExit
            self.window.blit(s, (0, 0))
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        # rgb_array: (height, width, 3) uint8 image
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2)).copy()

    def close(self):
        self.pg.display.quit()
        self.pg.quit()