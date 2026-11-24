
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

DQN_EVAL_PATH = (
    PROJECT_ROOT
    / "logs/evaluation/dqn/seed_42/extended/evaluations.npz"
)

PPO_EVAL_PATH = (
    PROJECT_ROOT
    / "logs/evaluation/ppo/seed_42/extended/evaluations.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "results/graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_evaluation_data(path, algorithm):
    """Load evaluation checkpoints saved by Stable-Baselines3."""

    if not path.exists():
        raise FileNotFoundError(f"{algorithm} evaluation file not found: {path}")

    data = np.load(path)

    timesteps = data["timesteps"]
    episode_rewards = data["results"]

    # Average reward and standard deviation at each checkpoint
    mean_rewards = np.mean(episode_rewards, axis=1)
    std_rewards = np.std(episode_rewards, axis=1)

    print(f"\n{algorithm} evaluation data")
    print(f"Checkpoints: {len(timesteps)}")
    print(f"First checkpoint: {timesteps[0]:,} timesteps")
    print(f"Final checkpoint: {timesteps[-1]:,} timesteps")
    print(f"Final mean reward: {mean_rewards[-1]:.2f}")
    print(f"Final standard deviation: {std_rewards[-1]:.2f}")

    return timesteps, mean_rewards, std_rewards


def plot_learning_curves():
    dqn_steps, dqn_mean, dqn_std = load_evaluation_data(
        DQN_EVAL_PATH, "DQN"
    )

    ppo_steps, ppo_mean, ppo_std = load_evaluation_data(
        PPO_EVAL_PATH, "PPO"
    )

    plt.figure(figsize=(12, 7))

    # DQN learning curve
    plt.plot(
        dqn_steps,
        dqn_mean,
        label="DQN",
        linewidth=2.5
    )

    plt.fill_between(
        dqn_steps,
        dqn_mean - dqn_std,
        dqn_mean + dqn_std,
        alpha=0.15
    )

    # PPO learning curve
    plt.plot(
        ppo_steps,
        ppo_mean,
        label="PPO",
        linewidth=2.5
    )

    plt.fill_between(
        ppo_steps,
        ppo_mean - ppo_std,
        ppo_mean + ppo_std,
        alpha=0.15
    )

    plt.title(
        "DQN vs PPO: Space Invaders Learning Curves",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Mean Evaluation Reward", fontsize=12)

    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()

    # Save high-resolution graph
    output_path = OUTPUT_DIR / "extended_learning_curves.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nLearning curve saved to: {output_path}")


if __name__ == "__main__":
    plot_learning_curves()
