
import numpy as np

from src.environment.env_factory import make_space_invaders_env


def evaluate_agent(
    model,
    algorithm,
    seed=1000,
    n_episodes=5
):
    """
    Evaluate a trained agent using complete Space Invaders episodes.

    Evaluation uses:
    - Unclipped rewards
    - No life-loss termination
    - Identical preprocessing for both algorithms
    """

    env = make_space_invaders_env(
        n_envs=1,
        seed=seed,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    episode_rewards = []
    episode_lengths = []

    try:
        for episode in range(n_episodes):
            observation = env.reset()

            total_reward = 0
            total_length = 0
            done = False

            while not done:
                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                observation, rewards, dones, infos = env.step(action)

                total_reward += float(rewards[0])
                total_length += 1
                done = bool(dones[0])

            episode_rewards.append(total_reward)
            episode_lengths.append(total_length)

            print(
                f"{algorithm} | Episode {episode + 1}/{n_episodes} "
                f"| Score: {total_reward:.2f} "
                f"| Steps: {total_length}"
            )

    finally:
        env.close()

    results = {
        "algorithm": algorithm,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }

    print(f"\n{algorithm} Evaluation Results")
    print(f"Mean score: {results['mean_reward']:.2f}")
    print(f"Score standard deviation: {results['std_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.2f}")

    return results
