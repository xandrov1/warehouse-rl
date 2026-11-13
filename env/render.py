"""Pygame drawing for WarehouseEnv. Works in a window ("human") or off-screen ("rgb_array")."""
import os

import numpy as np

from env.warehouse_map import CLUTTER, SHELF

CELL = 40
HUD = 36
COLORS = {
    "floor": (236, 236, 230),
    "grid": (210, 210, 204),
    "shelf": (120, 84, 52),
    "clutter": (176, 140, 92),
    "pickup": (150, 200, 240),
    "delivery": (243, 230, 196),
    "goal": (90, 190, 110),
    "robot": (220, 70, 60),
    "carry": (250, 210, 60),
    "text": (40, 40, 40),
    "hud": (250, 250, 248),
}


class Renderer:
    def __init__(self, width, height, human=False, fps=8):
        if not human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        self.human, self.fps = human, fps
        self.size = (width * CELL, height * CELL + HUD)
        if human:
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Warehouse RL")
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface(self.size)
        self.font = pygame.font.Font(None, 24)

    def draw(self, env):
        pg, s, m = self.pg, self.screen, env.map
        s.fill(COLORS["hud"])
        for y in range(m.height):
            for x in range(m.width):
                r = pg.Rect(x * CELL, HUD + y * CELL, CELL, CELL)
                if m.grid[y, x] == SHELF:
                    color = COLORS["shelf"]
                elif m.grid[y, x] == CLUTTER:
                    color = COLORS["clutter"]
                elif (x, y) in m.pickup:
                    color = COLORS["pickup"]
                elif (x, y) in m.delivery_cells:
                    color = COLORS["delivery"]
                else:
                    color = COLORS["floor"]
                pg.draw.rect(s, color, r)
                pg.draw.rect(s, COLORS["grid"], r, 1)
                if m.grid[y, x] == CLUTTER:
                    pg.draw.line(s, COLORS["shelf"], r.topleft, r.bottomright, 2)
                    pg.draw.line(s, COLORS["shelf"], r.topright, r.bottomleft, 2)

        gx, gy = env.goal
        pg.draw.rect(s, COLORS["goal"], pg.Rect(gx * CELL + 4, HUD + gy * CELL + 4, CELL - 8, CELL - 8), 4)

        rx, ry = env.robot
        center = (rx * CELL + CELL // 2, HUD + ry * CELL + CELL // 2)
        pg.draw.circle(s, COLORS["robot"], center, CELL // 2 - 5)
        if env.carrying:
            pg.draw.rect(s, COLORS["carry"], pg.Rect(center[0] - 7, center[1] - 7, 14, 14))

        status = "carrying" if env.carrying else "going to pickup"
        hud = f"step {env.steps}   {status}   collisions {env.collisions}"
        s.blit(self.font.render(hud, True, COLORS["text"]), (10, 10))

        if self.human:
            pg.event.pump()
            pg.display.flip()
            self.clock.tick(self.fps)
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2))

    def close(self):
        if self.human:
            self.pg.display.quit()
        self.pg.quit()
