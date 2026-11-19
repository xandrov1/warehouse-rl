# Warehouse RL: Curriculum Learning under Sparse Rewards
 
MSML642 · Group 12
 
A single robot in a 2D grid warehouse picks up an object at the pickup zone and delivers it to an assigned spot on a shelf. The robot is trained with MaskablePPO under two conditions, with a curriculum and without one, to compare learning under sparse rewards. Each condition is trained with 5 random seeds.
 
## Status
 
The environment is complete and tested: map generator, Gymnasium environment, and Pygame rendering. Training has not started; the next steps prepare the curriculum and training runs.
 
| Step | What | Status |
|---|---|---|
| 1 | Project setup and shared config | Done |
| 2 | Warehouse map generator | Done |
| 3 | Gymnasium environment (`reset`, `step`, observations, rewards, action masks) | Done |
| 4 | Pygame rendering | Done |
| 5 | Sanity checks (random agent, `check_env`) | Next |
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
 
Tested versions:
 
```
gymnasium 1.4.0 · pygame 2.6.1 · stable-baselines3 2.9.2 · sb3-contrib 2.9.2 · torch 2.15.0 (CPU) · matplotlib 3.11.2
```
 
## Project structure
 
```
warehouse-rl/
├── requirements.txt
├── config.py            # all shared settings in one place
├── check_setup.py       # verifies the install
├── test_map.py          # tests for the map generator
├── test_env.py          # tests for the environment rules
├── watch.py             # watch the robot live or save a GIF
├── env/
│   ├── __init__.py
│   ├── warehouse_map.py # warehouse layout generator
│   ├── warehouse_env.py # Gymnasium environment
│   ├── renderer.py      # Pygame drawing
│   └── scripted.py      # scripted expert and random agents (testing and demos only)
├── results/             # logs, saved models, eval CSVs (later)
└── figures/             # plots, GIFs, and videos
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
 
## The environment (`env/warehouse_env.py`)
 
`python test_env.py` runs 9 scripted checks covering every rule below.
 
**Episode.** `reset()` builds a new map, chooses one delivery cell as the goal, and places the robot on a random open cell. Same seed, same episode.
 
**Actions.** `0` up, `1` down, `2` left, `3` right, `4` PICK, `5` DROP.
 
**Observation.** 11 values, each in [−1, 1]:
 
| Index | Meaning |
|---|---|
| 0–7 | Distance to the nearest obstacle (shelf, clutter, or wall) in 8 directions: N, NE, E, SE, S, SW, W, NW; divided by the grid size |
| 8–9 | (dx, dy) to the current target, divided by the grid size. The target is the nearest pickup cell while empty-handed, and the goal delivery cell while carrying |
| 10 | 1 if carrying an object, else 0 |
 
The robot never sees the full map.
 
**Rewards.**
 
| Event | Reward | Effect |
|---|---|---|
| Any step | −0.01 | |
| Move into shelf, clutter, or wall | −1 | Robot stays in place |
| DROP on the goal cell | +10 | Episode ends (success) |
| DROP on any other delivery cell | −5 | Episode ends (wrong drop) |
| 200 steps reached | | Episode ends (truncated) |
 
PICK has no reward; it only sets the carrying flag. No distance-based shaping is used.
 
**Action masks.** `action_masks()` is read by MaskablePPO. Moves are always allowed (collisions are penalized, not masked). PICK is allowed only when empty-handed in the pickup zone. DROP is allowed only when carrying and standing on a delivery cell.
 
**End-of-episode info.** `is_success`, `wrong_drop`, `episode_steps`, `shortest_path` (minimum number of moves for that episode, used for the path-length metric), and `collisions`.
 
**Curriculum support.**
 
- `task="navigate"`: the robot only has to reach the goal cell; PICK and DROP are always masked. Used by the early stages.
- `max_goal_dist`: limits how far the robot starts from its target, for short-distance stages.
- `set_settings(...)`: changes any setting; takes effect at the next `reset()`, so the curriculum can change stages without creating a new environment.
## Rendering (`watch.py`)
 
```bash
python watch.py                            # scripted expert, live Pygame window
python watch.py --agent random             # random (untrained) robot
python watch.py --fps 4 --clutter 0.05     # slower playback, lighter clutter
python watch.py --gif figures/expert.gif   # save a GIF instead of opening a window
```
 
Colors: blue circle = robot (orange box on it = carrying), dark green = goal cell, light green = other delivery cells, gray = shelf, orange = pickup zone, brown = clutter. The bar under the grid shows step count, episode reward, carrying status, and collisions.
 
The environment supports three render modes: `human` (live window, drawn automatically every step), `rgb_array` (image arrays for GIFs and videos), and `ansi` (text map). Rendering is off by default, so it does not slow down training.
 
The scripted agents in `env/scripted.py` are references, not learning agents. The expert uses BFS on the full map and always takes the shortest route (about +9.4 reward per episode); the random agent picks any valid action (typically runs out of steps with around −60 reward). A `--agent trained` option will be added once models are trained.
 
## Next: Step 5
 
Sanity checks before training: Stable-Baselines3's `check_env`, a random-agent baseline, and a short MaskablePPO training run to confirm the full pipeline works.
 
## Possible extensions
 
- A third agent with potential-based reward shaping (using BFS distance), to compare curriculum learning against reward shaping
- Frame stacking, if the robot oscillates near dead ends (the policy has no memory of visited cells)