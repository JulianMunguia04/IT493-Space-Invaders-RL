
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Experiment reproducibility
RANDOM_SEEDS = [42, 123, 2026]

# Environment configuration
ENVIRONMENT_ID = "ALE/SpaceInvaders-v5"
SCREEN_SIZE = 84
FRAME_STACK = 4
N_ENVS = 1

# Training configuration
# Research Budget Timesteps
PILOT_TIMESTEPS = 100_000

# Extended training for final demonstration
DQN_TIMESTEPS = 500_000
PPO_TIMESTEPS = 500_000

# Evaluation settings
EVAL_FREQ = 20_000
N_EVAL_EPISODES = 3

# Checkpoint settings
CHECKPOINT_FREQ = 25_000

# Directory configuration
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DQN_MODEL_DIR = MODEL_DIR / "dqn"
PPO_MODEL_DIR = MODEL_DIR / "ppo"

GRAPH_DIR = RESULTS_DIR / "graphs"
GAMEPLAY_DIR = RESULTS_DIR / "gameplay"
SALIENCY_DIR = RESULTS_DIR / "saliency_maps"


def create_project_directories():
    """Create all required project directories."""

    directories = [
        LOG_DIR,
        MODEL_DIR,
        DQN_MODEL_DIR,
        PPO_MODEL_DIR,
        RESULTS_DIR,
        GRAPH_DIR,
        GAMEPLAY_DIR,
        SALIENCY_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print("Project directories initialized successfully!")
