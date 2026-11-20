
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import PPO_MODEL_DIR
from src.environment.env_factory import make_space_invaders_env


def evaluate_on_fresh_seeds(training_seed, n_episodes=5):

    model_path = (
        PPO_MODEL_DIR / f"ppo_seed_{training_seed}_final.zip"
    )

    model = PPO.load(str(model_path), device="cpu")

    rewards = []
    lengths = []

    print(f"\nPPO training seed: {training_seed}")
    print("-" * 50)

    for episode in range(n_episodes):

        evaluation_seed = 50_000 + episode

        env = make_space_invaders_env(
            n_envs=1,
            seed=evaluation_seed,
            clip_reward=False,
            terminal_on_life_loss=False
        )

        env = VecTransposeImage(env)

        try:
            observation = env.reset()

            total_reward = 0
            total_length = 0
            done = False

            while not done:

                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                observation, reward, dones, infos = env.step(action)

                total_reward += float(reward[0])
                total_length += 1

                done = bool(dones[0])

            rewards.append(total_reward)
            lengths.append(total_length)

            print(
                f"Evaluation seed {evaluation_seed} | "
                f"Score: {total_reward:.2f} | "
                f"Steps: {total_length}"
            )

        finally:
            env.close()

    print("\nSummary")
    print(f"Mean score: {np.mean(rewards):.2f}")
    print(f"Score standard deviation: {np.std(rewards, ddof=1):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.2f}")

    return rewards, lengths


def main():

    for seed in [42, 123, 2026]:
        evaluate_on_fresh_seeds(seed)


if __name__ == "__main__":
    main()
