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
# Episodes
MAX_EPISODE_STEPS = 200          # episode is truncated after this many steps
