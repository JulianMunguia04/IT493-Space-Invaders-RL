
from src.environment.env_factory import make_space_invaders_env


def main():

    print("Creating preprocessed Space Invaders environment...")

    env = make_space_invaders_env(
        n_envs=1,
        seed=42
    )

    try:
        observation = env.reset()

        print("\nEnvironment initialized successfully!")
        print("Observation shape:", observation.shape)
        print("Observation dtype:", observation.dtype)
        print("Minimum pixel value:", observation.min())
        print("Maximum pixel value:", observation.max())

        # Verify observation dimensions
        assert observation.shape == (1, 84, 84, 4)
        assert observation.dtype.name == "uint8"

        print("\nPreprocessing verification passed!")

        # Test 10 random actions
        for step in range(10):

            action = [env.action_space.sample()]

            observation, rewards, dones, infos = env.step(action)

        print("Random action test completed!")

    finally:
        env.close()

    print("\nEnvironment closed successfully!")


if __name__ == "__main__":
    main()
