"""Records rollout videos with the Pygame renderer: the scripted shortest-path robot
(shows the full pick-and-deliver task) and a random agent (the untrained starting point).
Writes MP4 if ffmpeg is installed, otherwise GIF."""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

import config
from env import WarehouseEnv
from env.scripted import scripted_action

parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=["scripted", "random", "model"], default="scripted")
parser.add_argument("--model", help="path to a saved MaskablePPO model (with --policy model)")
parser.add_argument("--clutter", type=float, default=0.10)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fps", type=int, default=8)
args = parser.parse_args()

env = WarehouseEnv(clutter_density=args.clutter, render_mode="rgb_array")
rng = np.random.default_rng(args.seed)
model = None
if args.policy == "model":
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)

frames = []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    frames.append(env.render())
    done = False
    while not done:
        mask = env.action_masks()
        if args.policy == "scripted":
            a = scripted_action(env)
        elif args.policy == "random":
            a = int(rng.choice(np.flatnonzero(mask)))
        else:
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        frames.append(env.render())
        done = term or trunc
    frames += [frames[-1]] * args.fps  # hold the last frame for a second
    outcome = info["event"] if term else "timeout"
    print(f"episode {ep + 1}: {outcome} after {info['steps']} steps "
          f"(shortest {info['shortest_path']})")
env.close()

os.makedirs(config.FIGURES_DIR, exist_ok=True)
base = os.path.join(config.FIGURES_DIR, f"demo_{args.policy}_clutter{int(args.clutter * 100)}")
if shutil.which("ffmpeg"):
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, f"{i:05d}.png"))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps),
                        "-i", os.path.join(tmp, "%05d.png"), "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", base + ".mp4"], check=True)
    print(f"Saved {base}.mp4")
else:
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(base + ".gif", save_all=True, append_images=imgs[1:], duration=1000 // args.fps, loop=0)
    print(f"Saved {base}.gif")
