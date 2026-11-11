# Warehouse RL: Curriculum Learning under Sparse Rewards

MSML642 · Group 12

A single robot in a 2D grid warehouse picks up an object at the pickup zone and delivers it to an assigned spot on a shelf. The robot is trained with MaskablePPO under two conditions, with a curriculum and without one, to compare learning under sparse rewards. Each condition is trained with 5 random seeds.

## Status

Project setup and the warehouse map generator are complete and tested. Training is not yet possible; the next step wraps the map in a Gymnasium environment.

| Step | What | Status |
|---|---|---|
| 1 | Project setup and shared config | Done |
| 2 | Warehouse map generator | Done |
| 3 | Gymnasium environment (`reset`, `step`, observations, rewards, action masks) | Next |
| 4 | Pygame rendering | Not started |
| 5 | Sanity checks (random agent, `check_env`) | Not started |
| 6 | Define Stages 1–4 | Not started |
| 7 | Curriculum logic (advance at >80% success over last 100 episodes) | Not started |
| 8 | Training: curriculum vs. baseline, 5 seeds each | Not started |
| 9 | Evaluation on unseen Stage 4 layouts | Not started |
| 10 | Plots, results, rollout videos | Not started |

## Setup

Requires Python 3.12.

```bash
git clone <repo-url>
cd warehouse-rl
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py
```

`check_setup.py` prints the installed package versions and runs a short MaskablePPO training run. A successful install ends with `Setup OK`. All training runs on CPU; no GPU is required.

## Project structure

```
warehouse-rl/
├── requirements.txt
├── config.py            # all shared settings in one place
├── check_setup.py       # verifies the install
├── test_map.py          # tests for the map generator
├── env/
│   ├── __init__.py
│   └── warehouse_map.py # warehouse layout generator
├── results/             # logs, saved models, eval CSVs (later)
└── figures/             # plots and videos (later)
```

## config.py

All shared settings are defined in `config.py`, so the curriculum agent and the baseline always use identical values. Settings should be changed here rather than in individual files.

- **Seeds:** `[0, 1, 2, 3, 4]` (2 agents × 5 seeds = 10 training runs)
- **Training budget:** 2M steps per run (provisional; to be tuned based on run time)
- **Curriculum:** advance to the next stage when the success rate over the last 100 episodes is above 80%
- **Rewards:** +10 correct delivery, −5 wrong drop, −1 collision, −0.01 per step
- **Evaluation:** 200 unseen Stage 4 layouts per seed

## The map (`env/warehouse_map.py`)

`python test_map.py` prints example maps and runs the generator checks.

```
P P x . . . . . . . . . . . .
P P D D D D . D D D D . . . .
. . # # # # . # # # # . . . .
. . D D D D x D D D D . . . .
. . D D D D . D D D D . . . .
. . # # # # x # # # # . . . .
x x x D D x . D D D D . . . .
. . D D D D . D D D D . . . .
. . # # # # . # # # # . . . .
. . D D D D . D D D D . . . .
. . D D x D . D D D D . . . .
. . # # # # . # # # # . . . .
. . D D D D . D D D D . . . x
. . . . . . . . . . . . . . .
. . . . . . . . . . . . . . .
```

`P` pickup zone · `#` shelf · `x` clutter · `D` possible delivery cell · `.` open floor

- **Layout:** The grid is 15×15. The pickup zone is a 2×2 square in the top-left corner. There are 4 rows of shelves with 2-cell aisles between them. Each row is split into two 4-cell shelves, and the gaps between shelves work as cross-aisles. The right side is left open as a main aisle.
- **Clutter:** Boxes and pallets cover a set fraction of the open floor: 5% (light), 10% (medium), and 20% (heavy). Final values are set in Step 6. Clutter is never placed where it would split the floor into disconnected regions, so every task is solvable and no episode is impossible by construction.
- **Shortest paths:** `bfs()` returns the shortest-path distance from a cell to every other cell. It is used for the path-length metric, for limiting goal distance in early stages, and for optional reward shaping.
- **Reproducible:** The same seed always gives the same map.
- **Speed:** About 12 ms per map at heavy clutter, fast enough to generate a new map every episode.

### Delivery rule

Each order specifies **one delivery cell**:

- Each side of a shelf cell is a separate storage spot, similar to a bin location in a real warehouse. Its delivery cell is the open cell directly in front of it (the `D` cells above and below each shelf).
- At the start of each episode, one `D` is chosen at random as the order. It is the only correct drop location.
- Dropping there gives +10. Dropping on any other `D` gives −5 and ends the episode, including other spots on the same shelf.
- DROP is masked out unless the robot is carrying something and standing on a `D`.
- The cells at the ends of shelves (in the cross-aisles) are not delivery cells, so a cell can never belong to two shelves.
- The robot never sees which cells are `D`s. It only gets the (dx, dy) direction to its goal.

A map has 64 possible delivery cells with no clutter, and 42–59 under heavy clutter, since clutter can cover some of them.

Note: under this rule, the paper's term "wrong shelf" should read "wrong delivery location."

## Next: Step 3

Gymnasium environment:

- `reset()`: build a map, place the robot, pick the goal `D`
- `step()`: move the robot, handle PICK and DROP, return rewards
- Observations: distance to the nearest obstacle in 8 directions (normalized), (dx, dy) to the current target, and a carrying flag
- `action_masks()`: block invalid PICK and DROP

## Possible extensions

- A third agent with potential-based reward shaping (using BFS distance), to compare curriculum learning against reward shaping
- Frame stacking, if the robot oscillates near dead ends (the policy has no memory of visited cells)