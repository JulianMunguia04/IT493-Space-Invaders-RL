import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import PPO_MODEL_DIR
from src.environment.env_factory import make_space_invaders_env


def main():

    models = {}

    for seed in [42, 123, 2026]:

        path = PPO_MODEL_DIR / f"ppo_seed_{seed}_final.zip"

        models[seed] = PPO.load(str(path), device="cpu")

    env = make_space_invaders_env(
        n_envs=1,
        seed=50000,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    env = VecTransposeImage(env)

    try:

        observation = env.reset()

        print("\nPPO action comparison")
        print("-" * 60)

        for step in range(20):

            actions = {}

            for seed, model in models.items():

                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                actions[seed] = int(action[0])

            print(
                f"Step {step + 1:02d} | "
                f"Seed 42: {actions[42]} | "
                f"Seed 123: {actions[123]} | "
                f"Seed 2026: {actions[2026]}"
            )

            observation, rewards, dones, infos = env.step(
                np.array([actions[42]])
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()