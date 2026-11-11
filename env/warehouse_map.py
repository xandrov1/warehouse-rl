"""Warehouse grid map: layout, clutter placement, delivery cells, and shortest-path distances.

Coordinates are (x, y) with x = column, y = row, and y growing downward.
The grid is stored as grid[y, x]. Anything outside the grid counts as a wall.
"""
from collections import deque

import numpy as np

FREE, SHELF, CLUTTER = 0, 1, 2
NEIGHBORS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right


class WarehouseMap:
    """One randomly generated warehouse layout.

    Args:
        width, height:    grid size in cells
        shelves:          False gives an empty floor (used for navigation-only stages)
        clutter_density:  fraction of open floor cells to fill with boxes/pallets
        rng:              numpy Generator, so layouts are reproducible per seed
        segment_len:      length of each shelf run; gaps between runs are cross-aisles
        pickup_size:      the pickup zone is a pickup_size x pickup_size square, top-left
    """

    def __init__(self, width=15, height=15, shelves=True, clutter_density=0.0,
                 rng=None, segment_len=4, pickup_size=2):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.width, self.height = width, height
        self.grid = np.full((height, width), FREE, dtype=np.int8)
        self.pickup = set()
        self.n_shelves = 0

        if shelves:
            self._place_pickup(pickup_size)
            self._place_shelves(segment_len)
        self.n_clutter = self._place_clutter(clutter_density)
        self.delivery_cells = self._compute_delivery_cells()

    # layout
    def _place_pickup(self, size):
        for y in range(size):
            for x in range(size):
                self.pickup.add((x, y))

    def _place_shelves(self, seg_len):
        """Horizontal shelf rows every 3 cells (2-cell aisles between them),
        each row split into runs of seg_len with 1-cell gaps for cross-aisles."""
        n = 0
        y = 2
        while y <= self.height - 3:
            x = 2
            while x + seg_len - 1 <= self.width - 2:
                for dx in range(seg_len):
                    self.grid[y, x + dx] = SHELF
                n += 1
                x += seg_len + 1
            y += 3
        self.n_shelves = n

    def _place_clutter(self, density):
        """Place clutter one cell at a time, rejecting any cell that would split
        the open floor into disconnected parts. Returns how many were placed."""
        candidates = [c for c in self.free_cells() if c not in self.pickup]
        target = int(round(density * len(candidates)))
        if target == 0:
            return 0
        self.rng.shuffle(candidates)
        placed = 0
        for (x, y) in candidates:
            if placed >= target:
                break
            self.grid[y, x] = CLUTTER
            if self._floor_connected():
                placed += 1
            else:
                self.grid[y, x] = FREE
        return placed

    def _compute_delivery_cells(self):
        """Each shelf face is its own storage spot, reached from the aisle.
        A delivery cell is an open cell directly above or below a shelf cell;
        an order names exactly one delivery cell, and a drop at any other
        delivery cell is a wrong drop. Cells at the ends of a shelf (in the
        cross-aisles) are not delivery cells, which avoids any cell being
        ambiguous between two shelves.

        Returns a dict: delivery cell (x, y) -> the shelf cell (x, y) it serves.
        """
        cells = {}
        for (x, y) in self.free_cells():
            if (x, y) in self.pickup:
                continue
            spots = [(x, y + dy) for dy in (-1, 1)
                     if self.in_bounds(x, y + dy) and self.grid[y + dy, x] == SHELF]
            if len(spots) == 1:
                cells[(x, y)] = spots[0]
        return cells

    # queries
    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x, y):
        """True if the robot can stand here (inside the grid, no shelf or clutter)."""
        return self.in_bounds(x, y) and self.grid[y, x] == FREE

    def free_cells(self):
        ys, xs = np.nonzero(self.grid == FREE)
        return list(zip(xs.tolist(), ys.tolist()))

    def is_delivery_cell(self, x, y):
        return (x, y) in self.delivery_cells

    def bfs(self, start):
        """Shortest-path step counts from start to every cell (-1 = unreachable).
        Used for the shortest-path evaluation metric and goal-distance control."""
        dist = np.full((self.height, self.width), -1, dtype=np.int32)
        sx, sy = start
        dist[sy, sx] = 0
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if self.is_free(nx, ny) and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    queue.append((nx, ny))
        return dist

    def _floor_connected(self):
        free = self.free_cells()
        if not free:
            return False
        dist = self.bfs(free[0])
        return int((dist >= 0).sum()) == len(free)

    # display
    def ascii(self, robot=None, goal=None):
        """Text picture: . floor, # shelf, x clutter, P pickup, D delivery cell,
        R robot, G goal."""
        delivery = self.delivery_cells
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if robot == (x, y):
                    ch = "R"
                elif goal == (x, y):
                    ch = "G"
                elif self.grid[y, x] == SHELF:
                    ch = "#"
                elif self.grid[y, x] == CLUTTER:
                    ch = "x"
                elif (x, y) in self.pickup:
                    ch = "P"
                elif (x, y) in delivery:
                    ch = "D"
                else:
                    ch = "."
                row.append(ch)
            rows.append(" ".join(row))
        return "\n".join(rows)