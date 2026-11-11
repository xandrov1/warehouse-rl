"""Step 2 checks: prints example maps and stress-tests map generation."""
import time

import numpy as np

from env.warehouse_map import WarehouseMap

# 1. Example maps at each clutter level
examples = [
    ("Empty floor (navigation only)", dict(shelves=False, clutter_density=0.0)),
    ("Shelves, light clutter (5%)", dict(clutter_density=0.05)),
    ("Shelves, medium clutter (10%)", dict(clutter_density=0.10)),
    ("Shelves, heavy clutter (20%)", dict(clutter_density=0.20)),
]
for name, kwargs in examples:
    m = WarehouseMap(rng=np.random.default_rng(0), **kwargs)
    print(f"\n{name}: {m.n_clutter} clutter cells, {m.n_shelves} shelves")
    print(m.ascii())

# 2. Stress test: 1000 maps per level must all be fully connected
print("\nStress test (1000 maps per clutter level):")
for density in (0.0, 0.05, 0.10, 0.20):
    rng = np.random.default_rng(123)
    start = time.perf_counter()
    n_cells = []
    for _ in range(1000):
        m = WarehouseMap(clutter_density=density, rng=rng)
        assert m._floor_connected(), "open floor got split by clutter"
        # every delivery cell is open and serves exactly one shelf cell directly above/below
        for (x, y), (sx, sy) in m.delivery_cells.items():
            assert m.is_free(x, y) and sx == x and abs(sy - y) == 1
            assert not (m.is_free(x, y - 1 if sy > y else y + 1)
                        and m.grid[y - 1 if sy > y else y + 1, x] == 1), "cell touches two shelves"
        n_cells.append(len(m.delivery_cells))
    ms = (time.perf_counter() - start)  # 1000 maps -> seconds == ms per map
    print(f"  density {density:.2f}: all connected, "
          f"delivery cells per map {min(n_cells)}-{max(n_cells)}, {ms:.2f} ms per map")

# 3. Same seed -> same map (needed for reproducible seeds)
a = WarehouseMap(clutter_density=0.2, rng=np.random.default_rng(7))
b = WarehouseMap(clutter_density=0.2, rng=np.random.default_rng(7))
assert (a.grid == b.grid).all()
print("\nReproducibility check passed. Step 2 OK.")