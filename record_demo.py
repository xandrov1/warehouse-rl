"""Saves a GIF of the scripted robot, or of a trained model with --model."""
import argparse
import os

from PIL import Image

import config
from env.warehouse_env import WarehouseEnv, scripted_action

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="saved MaskablePPO model, e.g. results/smoke/model.zip")
parser.add_argument("--clutter", type=float, default=0.10)
args = parser.parse_args()

model = None
if args.model:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)

env = WarehouseEnv(clutter_density=args.clutter, render_mode="rgb_array")
frames = []
for ep in range(3):
    obs, _ = env.reset(seed=ep)
    frames.append(env.render())
    done = False
    while not done:
        if model:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        else:
            a = scripted_action(env)
        obs, _, term, trunc, info = env.step(a)
        frames.append(env.render())
        done = term or trunc
    frames += [frames[-1]] * 8  # pause on the last frame
    print(f"episode {ep + 1}: {info['event'] if term else 'timeout'} in {info['steps']} steps "
          f"(shortest {info['shortest_path']})")
env.close()

name = "demo_model" if model else "demo_scripted"
out = os.path.join(config.FIGURES_DIR, f"{name}_clutter{round(args.clutter * 100)}.gif")
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=125, loop=0)
print(f"Saved {out}")
