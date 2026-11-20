import gymnasium as gym
import ale_py

from src.config import ENVIRONMENT_ID


def main():

    gym.register_envs(ale_py)

    env = gym.make(
        ENVIRONMENT_ID,
        frameskip=1,
        repeat_action_probability=0.0
    )

    print("\nSpace Invaders action space")
    print("-" * 45)

    meanings = env.unwrapped.get_action_meanings()

    for index, meaning in enumerate(meanings):
        print(f"Action {index}: {meaning}")

    print("\nAction 5:")
    print(meanings[5])

    env.close()


if __name__ == "__main__":
    main()