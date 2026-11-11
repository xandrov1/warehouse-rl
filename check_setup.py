# check_setup.py: confirms everything installed and works together
import sys
import gymnasium as gym
import numpy as np
import pygame
import stable_baselines3 as sb3
import sb3_contrib
import torch
import matplotlib
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import config

print(f"Python            {sys.version.split()[0]}")
print(f"gymnasium         {gym.__version__}")
print(f"pygame            {pygame.version.ver}")
print(f"stable-baselines3 {sb3.__version__}")
print(f"sb3-contrib       {sb3_contrib.__version__}")
print(f"torch             {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"matplotlib        {matplotlib.__version__}")
print(f"Seeds             {config.SEEDS}")

# Tiny MaskablePPO run on a built-in env to prove training works end to end
def all_valid(env):
    return np.ones(env.action_space.n, dtype=bool)

env = ActionMasker(gym.make("CartPole-v1"), all_valid)
model = MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, seed=config.SEEDS[0], verbose=0)
model.learn(total_timesteps=1_000)
print("\nMaskablePPO smoke test passed. Setup OK.")