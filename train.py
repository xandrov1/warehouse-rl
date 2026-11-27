"""Train one MaskablePPO agent (Step 7-8).

    python train.py --agent curriculum --seed 0
    python train.py --agent baseline --seed 0 --steps 1500000

curriculum  starts at Stage 1 and moves up a stage when the success rate over the
            last 100 episodes is above 80%.
baseline    trains on Stage 4 (the real task) the whole time.

Both agents are checked on the same Stage 4 layouts every EVAL_EVERY steps, so their
learning curves can be compared directly. Writes results/<agent>_s<seed>/.
"""
import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from env.warehouse_env import WarehouseEnv


STAGES = config.STAGES  # replaced by config.STAGES_V2 with --v2


def make_env(stage):
    return lambda: WarehouseEnv(**STAGES[stage])


def run_episodes(model, stage, n, seed_offset, deterministic=True):
    """Greedy rollouts on fixed layouts. Returns success rate, mean reward,
    mean collisions and mean (steps / shortest path) over successful episodes."""
    env = WarehouseEnv(**config.STAGES[stage])
    wins, picked, wrong, rewards, collisions, ratios = 0, 0, 0, [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(),
                                      deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        wins += info["is_success"]
        picked += env.picked_at_step is not None
        wrong += info["wrong_drop"]
        rewards.append(env.episode_reward)
        collisions.append(info["collisions"])
        if info["is_success"]:
            extra = 0 if config.STAGES[stage]["task"] == "navigate" else 2  # PICK + DROP
            ratios.append(info["episode_steps"] / (info["shortest_path"] + extra))
    return dict(success=wins / n, picked=picked / n, wrong_drop=wrong / n, reward=float(np.mean(rewards)),
                collisions=float(np.mean(collisions)),
                path_ratio=float(np.mean(ratios)) if ratios else float("nan"))


class TrainingLog(BaseCallback):
    def __init__(self, agent, out_dir):
        super().__init__()
        self.agent = agent
        self.stage = 1 if agent == "curriculum" else config.FINAL_STAGE
        self.window = deque(maxlen=config.SUCCESS_WINDOW)
        self.next_eval = 0
        self.out_dir = out_dir
        self.stage_log = open(os.path.join(out_dir, "stages.csv"), "w", newline="")
        self.eval_log = open(os.path.join(out_dir, "eval_curve.csv"), "w", newline="")
        csv.writer(self.stage_log).writerow(["timesteps", "stage"])
        csv.writer(self.eval_log).writerow(["timesteps", "stage", "train_success",
                                            "stage4_success", "stage4_reward",
                                            "stage4_collisions", "stage4_path_ratio"])
        csv.writer(self.stage_log).writerow([0, self.stage])

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.window.append(bool(info.get("is_success", False)))
        if (self.agent == "curriculum" and self.stage < config.FINAL_STAGE
                and len(self.window) == config.SUCCESS_WINDOW
                and np.mean(self.window) > config.ADVANCE_THRESHOLD):
            # keep the model that just passed this stage
            self.model.save(os.path.join(self.out_dir, f"model_passed_stage{self.stage}"))
            self.stage += 1
            new = {"max_goal_dist": None, "max_order_dist": None, **STAGES[self.stage]}
            self.training_env.env_method("set_settings", **new)
            self.training_env.reset()
            self.window.clear()
            csv.writer(self.stage_log).writerow([self.num_timesteps, self.stage])
            self.stage_log.flush()
            print(f"[{self.agent}] step {self.num_timesteps}: advanced to stage {self.stage}", flush=True)
        if self.num_timesteps >= self.next_eval:
            self.next_eval += config.EVAL_EVERY
            r = run_episodes(self.model, config.FINAL_STAGE, config.EVAL_DURING_TRAINING,
                             seed_offset=config.EVAL_SEED_OFFSET)
            train_success = np.mean(self.window) if self.window else 0.0
            csv.writer(self.eval_log).writerow([self.num_timesteps, self.stage, train_success,
                                                r["success"], r["reward"], r["collisions"],
                                                r["path_ratio"]])
            self.eval_log.flush()
            print(f"[{self.agent}] step {self.num_timesteps} stage {self.stage} "
                  f"train {train_success:.2f} | stage4 success {r['success']:.2f}", flush=True)
        return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=config.AGENTS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=config.TOTAL_TIMESTEPS)
    p.add_argument("--v2", action="store_true", help="use curriculum version 2 (STAGES_V2)")
    p.add_argument("--name", default=None, help="results folder name (default: <agent>_s<seed>)")
    args = p.parse_args()
    if args.v2:
        STAGES = config.STAGES_V2

    torch.set_num_threads(1)
    name = f"{args.agent}_v2" if args.v2 else args.agent
    out_dir = os.path.join(config.RESULTS_DIR, f"{args.name or name}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    start_stage = 1 if args.agent == "curriculum" else config.FINAL_STAGE
    venv = DummyVecEnv([make_env(start_stage) for _ in range(config.N_ENVS)])
    venv.seed(args.seed)
    model = MaskablePPO("MlpPolicy", venv, seed=args.seed, verbose=0)

    t0 = time.time()
    model.learn(args.steps, callback=TrainingLog(args.agent, out_dir))
    model.save(os.path.join(out_dir, "model"))
    print(f"[{args.agent} s{args.seed}] done in {time.time() - t0:.0f}s", flush=True)
