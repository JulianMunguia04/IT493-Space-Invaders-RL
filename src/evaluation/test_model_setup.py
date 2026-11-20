
from stable_baselines3 import DQN, PPO

from src.environment.env_factory import make_space_invaders_env


def main():
    print("\n========== DQN SETUP TEST ==========")

    dqn_env = make_space_invaders_env(
        n_envs=1,
        seed=42
    )

    try:
        dqn_model = DQN(
            "CnnPolicy",
            dqn_env,
            buffer_size=1_000,
            learning_starts=100,
            batch_size=32,
            device="cpu",
            verbose=1
        )

        print("DQN initialized successfully!")
        print("DQN observation shape:", dqn_model.observation_space.shape)

        del dqn_model

    finally:
        dqn_env.close()

    print("\n========== PPO SETUP TEST ==========")

    ppo_env = make_space_invaders_env(
        n_envs=1,
        seed=42
    )

    try:
        ppo_model = PPO(
            "CnnPolicy",
            ppo_env,
            n_steps=128,
            batch_size=32,
            device="cpu",
            verbose=1
        )

        print("PPO initialized successfully!")
        print("PPO observation shape:", ppo_model.observation_space.shape)

        del ppo_model

    finally:
        ppo_env.close()

    print("\nBoth model setup tests completed!")


if __name__ == "__main__":
    main()
