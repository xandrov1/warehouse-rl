"""Step 9: evaluate every trained model on unseen layouts of each stage.

    python evaluate.py

Reads results/<agent>_s<seed>/model.zip and writes results/eval.csv.
Layouts use seeds from EVAL_SEED_OFFSET + 1000, which training never sees.
Stage 4 is the real task; Stages 2 and 3 show how far each agent got.
Actions are sampled from the policy (deterministic=False): with no memory, the
greedy policy tends to loop back and forth between two cells.
Also evaluates the checkpoints saved when a curriculum run passed a stage.
"""
import csv
import glob
import os

from sb3_contrib import MaskablePPO

import config
from train import run_episodes

N = 100  # episodes per stage (200 for the final Stage 4 numbers)

rows = []
paths = glob.glob(os.path.join(config.RESULTS_DIR, "**", "model*.zip"), recursive=True)
paths = [p for p in paths if os.path.basename(p) in ("model.zip", "model_passed_stage2.zip")
         and not (p.startswith(os.path.join(config.RESULTS_DIR, "stage2ckpt"))
                  and os.path.basename(p) == "model.zip")]
for path in sorted(paths):
    run = os.path.relpath(os.path.dirname(path), config.RESULTS_DIR)
    agent, seed = run.rsplit("_s", 1)
    checkpoint = os.path.basename(path)[:-4]
    agent = agent if checkpoint == "model" else f"{agent} ({checkpoint[6:]})"
    model = MaskablePPO.load(path)
    for stage in (2, 3, 4):
        n = config.EVAL_EPISODES if stage == 4 else N
        r = run_episodes(model, stage, n, seed_offset=config.EVAL_SEED_OFFSET + 1000, deterministic=False)
        rows.append(dict(agent=agent, seed=int(seed), stage=stage, episodes=n, **r))
        print(f"{agent:40s} seed {seed} stage {stage}: success {r['success']:6.1%}  "
              f"picked {r['picked']:6.1%}  wrong drop {r['wrong_drop']:6.1%}  reward {r['reward']:+7.2f}  collisions {r['collisions']:5.1f}", flush=True)

with open(os.path.join(config.RESULTS_DIR, "eval.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
