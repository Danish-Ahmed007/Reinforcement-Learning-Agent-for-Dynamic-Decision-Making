"""Configuration for Lunar Lander DQN Agent."""

# Environment
ENV_NAME = "LunarLander-v3"
RENDER_MODE = None

# Training Hyperparameters
TOTAL_TIMESTEPS = 500000
LEARNING_RATE = 0.0005
BUFFER_SIZE = 50000
LEARNING_STARTS = 1000
BATCH_SIZE = 128
TAU = 1.0
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 250

# Exploration
EXPLORATION_FRACTION = 0.2
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.01

# Network Architecture
NET_ARCH = [256, 256]

# Evaluation
N_EVAL_EPISODES = 10
EVAL_FREQ = 5000

# Paths
MODEL_SAVE_PATH = "models/lunar_lander_dqn"
TENSORBOARD_LOG = "./tensorboard_logs/"
RESULTS_DIR = "results/"
PLOTS_DIR = "plots/"

# Checkpoints
CHECKPOINT_FREQ = 25000
