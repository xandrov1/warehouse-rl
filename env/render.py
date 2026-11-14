"""Pygame drawing for WarehouseEnv, in a window ("human") or off-screen ("rgb_array")."""
import os

import numpy as np
import pygame

from env.warehouse_map import CLUTTER, SHELF

CELL, HUD = 40, 36
COLORS = {"floor": (236, 236, 230), "line": (210, 210, 204), "shelf": (120, 84, 52),
          "clutter": (176, 140, 92), "pickup": (150, 200, 240), "delivery": (243, 230, 196),
          "goal": (90, 190, 110), "robot": (220, 70, 60), "carry": (250, 210, 60)}


class Renderer:
    def __init__(self, width, height, human=False):
        if not human:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.human = human
        size = (width * CELL, height * CELL + HUD)
        self.screen = pygame.display.set_mode(size) if human else pygame.Surface(size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def cell_color(self, m, x, y):
        if m.grid[y, x] == SHELF:
            return COLORS["shelf"]
        if m.grid[y, x] == CLUTTER:
            return COLORS["clutter"]
        if (x, y) in m.pickup:
            return COLORS["pickup"]
        return COLORS["delivery"] if (x, y) in m.delivery_cells else COLORS["floor"]

    def draw(self, env):
        s, m = self.screen, env.map
        s.fill((250, 250, 248))
        for y in range(m.height):
            for x in range(m.width):
                r = pygame.Rect(x * CELL, HUD + y * CELL, CELL, CELL)
                pygame.draw.rect(s, self.cell_color(m, x, y), r)
                pygame.draw.rect(s, COLORS["line"], r, 1)

        gx, gy = env.goal
        pygame.draw.rect(s, COLORS["goal"], (gx * CELL + 4, HUD + gy * CELL + 4, CELL - 8, CELL - 8), 4)
        center = (env.robot[0] * CELL + CELL // 2, HUD + env.robot[1] * CELL + CELL // 2)
        pygame.draw.circle(s, COLORS["robot"], center, CELL // 2 - 5)
        if env.carrying:
            pygame.draw.rect(s, COLORS["carry"], (center[0] - 7, center[1] - 7, 14, 14))

        status = "carrying" if env.carrying else "going to pickup"
        hud = f"step {env.steps}   {status}   collisions {env.collisions}"
        s.blit(self.font.render(hud, True, (40, 40, 40)), (10, 10))

        if self.human:
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(8)
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def close(self):
        pygame.quit()
