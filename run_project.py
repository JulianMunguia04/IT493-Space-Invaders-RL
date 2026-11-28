
from pathlib import Path

import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import (
    PROJECT_ROOT,
    DQN_MODEL_DIR,
    PPO_MODEL_DIR,
    create_project_directories
)

from src.environment.env_factory import make_space_invaders_env

from src.dqn.train import train_dqn as train_dqn_model
from src.ppo.train import train_ppo as train_ppo_model

SEED = 42

EXTENDED_TIMESTEPS = 500_000

DEMONSTRATION_EPISODES = 3
EVALUATION_EPISODES = 10

def get_model_path(algorithm, prefer_extended=True):
    """
    Return the extended model if it exists.
    Otherwise, fall back to the original trained model.
    """

    if algorithm.lower() == "dqn":
        model_dir = DQN_MODEL_DIR
        model_class = DQN
    elif algorithm.lower() == "ppo":
        model_dir = PPO_MODEL_DIR
        model_class = PPO
    else:
        raise ValueError("Algorithm must be 'dqn' or 'ppo'.")

    if prefer_extended:
        extended_path = (
            model_dir / f"{algorithm.lower()}_seed_{SEED}_extended_final.zip"
        )

        if extended_path.exists():
            return extended_path

    original_path = (
        model_dir / f"{algorithm.lower()}_seed_{SEED}_final.zip"
    )

    if original_path.exists():
        return original_path

    raise FileNotFoundError(
        f"No trained {algorithm.upper()} model was found.\n"
        f"Expected: {original_path}"
    )

def train_dqn():
    """Train an extended DQN model."""

    print("\nStarting extended DQN training...")
    print(f"Timesteps: {EXTENDED_TIMESTEPS:,}")
    print(f"Seed: {SEED}")

    model = train_dqn_model(
        seed=SEED,
        total_timesteps=EXTENDED_TIMESTEPS,
        run_name="extended"
    )

    print("\nExtended DQN training completed.")
    print(
        f"Model saved to: "
        f"{DQN_MODEL_DIR / f'dqn_seed_{SEED}_extended_final.zip'}"
    )

    return model


def train_ppo():
    """Train an extended PPO model."""

    print("\nStarting extended PPO training...")
    print(f"Timesteps: {EXTENDED_TIMESTEPS:,}")
    print(f"Seed: {SEED}")

    model = train_ppo_model(
        seed=SEED,
        total_timesteps=EXTENDED_TIMESTEPS,
        run_name="extended"
    )

    print("\nExtended PPO training completed.")
    print(
        f"Model saved to: "
        f"{PPO_MODEL_DIR / f'ppo_seed_{SEED}_extended_final.zip'}"
    )

    return model

def play_model(algorithm, episodes=DEMONSTRATION_EPISODES):
    """
    Load a trained model and display it playing Space Invaders.

    The game runs in a visible window.
    """

    algorithm = algorithm.lower()

    if algorithm == "dqn":
        model_class = DQN
    elif algorithm == "ppo":
        model_class = PPO
    else:
        raise ValueError("Algorithm must be 'dqn' or 'ppo'.")

    model_path = get_model_path(algorithm)

    print(f"\nLoading {algorithm.upper()} model:")
    print(model_path)

    env = make_space_invaders_env(
        n_envs=1,
        seed=SEED,
        clip_reward=False,
        terminal_on_life_loss=False,
        render_mode="human"
    )

    try:
        env = VecTransposeImage(env)
        model = model_class.load(model_path, env=env)

        print(f"\nWatching {algorithm.upper()} play.")
        print(f"Playing {episodes} episodes.")
        print("Close the game window or wait for the episodes to finish.")

        obs = env.reset()
        completed_episodes = 0
        episode_rewards = [0.0]

        while completed_episodes < episodes:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, rewards, dones, infos = env.step(action)

            episode_rewards[-1] += float(rewards[0])

            if dones[0]:
                completed_episodes += 1

                print(
                    f"Episode {completed_episodes}: "
                    f"Reward = {episode_rewards[-1]:.2f}"
                )

                if completed_episodes < episodes:
                    episode_rewards.append(0.0)

        print(f"\n{algorithm.upper()} demonstration completed.")

    finally:
        env.close()


def play_dqn():
    """Watch the DQN agent play Space Invaders."""

    play_model("dqn")


def play_ppo():
    """Watch the PPO agent play Space Invaders."""

    play_model("ppo")

def evaluate_model(algorithm):
    """
    Evaluate a trained model over multiple episodes.

    Reward clipping and life-loss termination are disabled
    to measure the actual game performance.
    """

    algorithm = algorithm.lower()

    if algorithm == "dqn":
        model_class = DQN
    elif algorithm == "ppo":
        model_class = PPO
    else:
        raise ValueError("Algorithm must be 'dqn' or 'ppo'.")

    model_path = get_model_path(algorithm)

    env = make_space_invaders_env(
        n_envs=1,
        seed=SEED,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    try:
        env = VecTransposeImage(env)
        model = model_class.load(model_path, env=env)

        rewards, lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=EVALUATION_EPISODES,
            deterministic=True,
            return_episode_rewards=True,
            warn=False
        )

        return {
            "algorithm": algorithm.upper(),
            "rewards": rewards,
            "lengths": lengths,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths))
        }

    finally:
        env.close()


def evaluate_and_demonstrate():
    """
    Evaluate DQN and PPO, compare their performance,
    and report the results.
    """

    print("\nEvaluating DQN...")
    dqn_results = evaluate_model("dqn")

    print("\nEvaluating PPO...")
    ppo_results = evaluate_model("ppo")

    print("\n" + "=" * 60)
    print("       SPACE INVADERS: DQN vs PPO")
    print("=" * 60)

    for results in [dqn_results, ppo_results]:
        print(f"\n{results['algorithm']} RESULTS")
        print("-" * 40)

        for index, reward in enumerate(results["rewards"], start=1):
            print(f"Episode {index}: {reward:.2f}")

        print(f"\nMean reward: {results['mean_reward']:.2f}")
        print(f"Reward standard deviation: {results['std_reward']:.2f}")
        print(f"Mean episode length: {results['mean_length']:.2f}")

    print("\n" + "=" * 60)
    print("                 FINAL COMPARISON")
    print("=" * 60)

    print(
        f"DQN mean reward: {dqn_results['mean_reward']:.2f}"
    )
    print(
        f"PPO mean reward: {ppo_results['mean_reward']:.2f}"
    )

    difference = (
        ppo_results["mean_reward"] - dqn_results["mean_reward"]
    )

    print(f"\nPPO - DQN: {difference:+.2f}")

    if difference > 0:
        print("PPO achieved the higher mean reward.")
    elif difference < 0:
        print("DQN achieved the higher mean reward.")
    else:
        print("Both algorithms achieved the same mean reward.")

    print("\nEvaluation completed.")

    return {
        "dqn": dqn_results,
        "ppo": ppo_results
    }

# Terminal Menu

def main():

    create_project_directories()

    while True:

        print("\n" + "=" * 55)
        print("       SPACE INVADERS REINFORCEMENT LEARNING")
        print("=" * 55)

        print("1. Train DQN (500,000 timesteps)")
        print("2. Train PPO (500,000 timesteps)")
        print("3. Watch DQN play")
        print("4. Watch PPO play")
        print("5. Evaluate and compare DQN vs PPO")
        print("0. Exit")

        print("=" * 55)

        choice = input("Select an option: ").strip()

        try:

            if choice == "1":
                train_dqn()

            elif choice == "2":
                train_ppo()

            elif choice == "3":
                play_dqn()

            elif choice == "4":
                play_ppo()

            elif choice == "5":
                evaluate_and_demonstrate()

            elif choice == "0":
                print("\nExiting project.")
                break

            else:
                print("\nInvalid option. Please select 0-5.")

        except FileNotFoundError as error:
            print(f"\nModel not found: {error}")

        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            raise


if __name__ == "__main__":
    main()

# Run File in Powershell from root
# .\venv\Scripts\Activate.ps1
# python run_project.py