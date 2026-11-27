# config.py: shared settings for all experiments

SEEDS = [0, 1, 2, 3, 4]          # 5 seeds per agent -> 10 training runs total
AGENTS = ["curriculum", "baseline"]

TOTAL_TIMESTEPS = 2_000_000      # same budget for both agents (tune later)

# Curriculum advancement
SUCCESS_WINDOW = 100             # episodes used to measure success rate
ADVANCE_THRESHOLD = 0.80         # move to next stage above this

# Rewards
R_DELIVER = 10.0
R_WRONG_DROP = -5.0
R_COLLISION = -1.0
R_STEP = -0.01

# Evaluation
EVAL_EPISODES = 200              # unseen Stage 4 layouts per seed
EVAL_SEED_OFFSET = 10_000        # keeps eval layouts separate from training layouts

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

# Curriculum stages (Step 6). Each stage is a set of WarehouseEnv settings.
# Stages 1-2 only teach navigation; stages 3-4 add PICK and DROP.
# max_goal_dist limits how far the robot starts from its first target.
STAGES = {
    1: dict(task="navigate", clutter_density=0.05, max_goal_dist=6),
    2: dict(task="navigate", clutter_density=0.10, max_goal_dist=None),
    3: dict(task="full", clutter_density=0.10, max_goal_dist=6),
    4: dict(task="full", clutter_density=0.20, max_goal_dist=None),  # the real task
}
FINAL_STAGE = 4

# Version 2 of the curriculum (after the first runs stalled at Stage 3).
# Stage 3 now also keeps the order close to the pickup zone, so the first
# PICK -> carry -> DROP episodes are short enough to be found by exploration.
STAGES_V2 = dict(STAGES)
STAGES_V2[3] = dict(task="full", clutter_density=0.10, max_goal_dist=4, max_order_dist=6)

N_ENVS = 4                       # parallel environments per training run
EVAL_EVERY = 50_000              # steps between Stage 4 checks during training
EVAL_DURING_TRAINING = 50        # Stage 4 episodes per check
