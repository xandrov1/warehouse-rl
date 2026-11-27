"""Record the Phase 2 demo video on unseen layouts:
random robot and baseline on Stage 4, then the curriculum robot on
Stage 2 (navigation) and Stage 3 (pick and drop, where it currently gets stuck).

    python record_video.py --seed 0 --out figures/phase2_demo.mp4
"""
import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import MaskablePPO

import config
from env.scripted import random_action
from env.warehouse_env import WarehouseEnv

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0, help="which trained seed to show")
p.add_argument("--fps", type=int, default=14)
p.add_argument("--out", default="figures/phase2_demo.mp4")
args = p.parse_args()
torch.manual_seed(0)  # the trained robots sample their actions; this makes the video repeatable
RNG = np.random.default_rng(0)

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 22)
    BIG = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
except OSError:
    FONT = BIG = ImageFont.load_default()


def header(frame, text):
    img = Image.new("RGB", (frame.shape[1], frame.shape[0] + 44), (30, 34, 40))
    img.paste(Image.fromarray(frame), (0, 44))
    ImageDraw.Draw(img).text((12, 9), text, fill=(255, 255, 255), font=FONT)
    return np.array(img)


def card(size, lines, seconds=3):
    img = Image.new("RGB", size, (30, 34, 40))
    d = ImageDraw.Draw(img)
    y = size[1] // 2 - 30 * len(lines)
    for i, line in enumerate(lines):
        font = BIG if i == 0 else FONT
        w = d.textlength(line, font=font)
        d.text(((size[0] - w) / 2, y), line, fill=(255, 255, 255), font=font)
        y += 60 if i == 0 else 36
    return [np.array(img)] * (seconds * args.fps)


def episode(policy, layout_seed, label, stage):
    env = WarehouseEnv(**config.STAGES[stage], render_mode="rgb_array")
    obs, _ = env.reset(seed=layout_seed)
    frames = [header(env.render(), label)]
    done = False
    while not done:
        action = policy(env, obs)
        obs, _, term, trunc, info = env.step(action)
        frames.append(header(env.render(), label))
        done = term or trunc
    result = ("DELIVERED" if info["is_success"] else
              "WRONG DROP" if info["wrong_drop"] else "RAN OUT OF STEPS")
    end = header(env.render(), f"{result} in {info['episode_steps']} steps")
    env.close()
    return frames + [end] * (2 * args.fps), info["is_success"]


def trained(path):
    model = MaskablePPO.load(path)
    return lambda env, obs: int(model.predict(obs, action_masks=env.action_masks(),
                                              deterministic=False)[0])


random_robot = lambda env, obs: random_action(env, RNG)
baseline = trained(f"results/baseline_s{args.seed}/model.zip")
curriculum = trained(f"results/curriculum_s{args.seed}/model.zip")
navigator = trained(f"results/stage2ckpt/curriculum_s{args.seed}/model_passed_stage2.zip")

# (title card, label, policy, stage, layouts shown)
segments = [
    (["Random robot", "Stage 4: the full task, 20% clutter"], "Random robot | Stage 4", random_robot, 4, 1),
    (["Baseline after 1.5M steps", "trained on Stage 4 only"], "Baseline | Stage 4", baseline, 4, 1),
    (["Curriculum robot", "after passing Stage 2 (~350k steps)", "Stage 2: drive to the order's cell",
      "(reaches it on ~80% of unseen layouts)"], "Curriculum | Stage 2", navigator, 2, 4),
    (["Curriculum robot after 1.5M steps", "Stage 3: pick up, carry, drop", "(where training is stuck)"], "Curriculum | Stage 3", curriculum, 3, 1),
]

probe = header(np.zeros((15 * 40 + 36, 15 * 40, 3), np.uint8), "")  # 15x15 grid, 40 px cells
size = (probe.shape[1], probe.shape[0])

frames = card(size, ["Warehouse RL - Phase 2", "MSML642 - Group 12",
                     "Trained robots on unseen layouts"], seconds=3)
for title, label, policy, stage, n in segments:
    frames += card(size, title, seconds=2)
    for i in range(n):
        f, _ = episode(policy, config.EVAL_SEED_OFFSET + 1000 + i, f"{label} | layout {i + 1}", stage)
        frames += f

imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1, quality=8)
print(f"saved {len(frames)} frames ({len(frames) / args.fps:.0f}s) to {args.out}")
