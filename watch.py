"""Watch the robot in a Pygame window, or save episodes as a GIF.

Examples:
    python watch.py                          # expert robot, 3 episodes, live window
    python watch.py --agent random           # untrained (random) robot
    python watch.py --clutter 0.05 --fps 4   # lighter clutter, slower playback
    python watch.py --gif figures/expert.gif # save to a GIF instead of a window
"""
import argparse

from PIL import Image

from env.scripted import expert_action, random_action
from env.warehouse_env import WarehouseEnv

parser = argparse.ArgumentParser()
parser.add_argument("--agent", choices=["expert", "random"], default="expert")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clutter", type=float, default=0.20)
parser.add_argument("--fps", type=int, default=8)
parser.add_argument("--gif", default=None, help="save a GIF here instead of opening a window")
args = parser.parse_args()

env = WarehouseEnv(clutter_density=args.clutter,
                   render_mode="rgb_array" if args.gif else "human")
env.metadata["render_fps"] = args.fps
policy = expert_action if args.agent == "expert" else random_action

frames = []
for ep in range(args.episodes):
    env.reset(seed=args.seed + ep)
    if args.gif:
        frames.append(env.render())
    done = False
    while not done:
        _, _, term, trunc, info = env.step(policy(env))
        if args.gif:
            frames.append(env.render())
        done = term or trunc
    result = "success" if info["is_success"] else ("wrong drop" if info["wrong_drop"] else "ran out of steps")
    print(f"episode {ep + 1}: {result} in {info['episode_steps']} steps "
          f"(shortest possible: {info['shortest_path'] + 2}), reward {env.episode_reward:+.2f}")
    if args.gif:
        frames += [frames[-1]] * args.fps  # hold the last frame for a second

env.close()

if args.gif:
    images = [Image.fromarray(f) for f in frames]
    images[0].save(args.gif, save_all=True, append_images=images[1:],
                   duration=int(1000 / args.fps), loop=0)
    print(f"saved {len(frames)} frames to {args.gif}")