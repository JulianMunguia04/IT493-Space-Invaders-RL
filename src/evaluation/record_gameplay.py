import cv2
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import (
    PPO_MODEL_DIR,
    DQN_MODEL_DIR,
    GAMEPLAY_DIR
)

from src.environment.env_factory import make_space_invaders_env


def record_gameplay(algorithm="ppo", seed=42):

    algorithm = algorithm.lower()

    if algorithm == "ppo":
        model_path = PPO_MODEL_DIR / f"ppo_seed_{seed}_final.zip"
        model = PPO.load(str(model_path), device="cpu")

    elif algorithm == "dqn":
        model_path = DQN_MODEL_DIR / f"dqn_seed_{seed}_final.zip"
        model = DQN.load(str(model_path), device="cpu")

    else:
        raise ValueError("Algorithm must be 'ppo' or 'dqn'.")

    env = make_space_invaders_env(
        n_envs=1,
        seed=50000,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    env = VecTransposeImage(env)

    output_path = GAMEPLAY_DIR / f"{algorithm}_seed_{seed}.mp4"

    GAMEPLAY_DIR.mkdir(parents=True, exist_ok=True)

    writer = None

    try:

        observation = env.reset()

        total_reward = 0
        total_steps = 0
        done = False

        while not done:

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, rewards, dones, infos = env.step(action)

            total_reward += float(rewards[0])
            total_steps += 1

            frame = env.envs[0].unwrapped.render()

            if frame is None:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if writer is None:

                height, width, _ = frame.shape

                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (width, height)
                )

            writer.write(frame)

            done = bool(dones[0])

    finally:

        if writer is not None:
            writer.release()

        env.close()

    print("\nGameplay recording complete!")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Training seed: {seed}")
    print(f"Score: {total_reward}")
    print(f"Agent steps: {total_steps}")
    print(f"Video: {output_path}")


if __name__ == "__main__":
    record_gameplay("ppo", 42)