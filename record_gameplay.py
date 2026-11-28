
from pathlib import Path

import gymnasium as gym
import ale_py
import numpy as np
import imageio.v2 as imageio

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from src.environment.env_factory import make_space_invaders_env

PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_DIR = PROJECT_ROOT / "results" / "gameplay"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPISODES = 3
FPS = 30

MODELS = {
    "DQN": PROJECT_ROOT / "models/dqn/dqn_seed_42_extended_final.zip",
    "PPO": PROJECT_ROOT / "models/ppo/ppo_seed_42_extended_final.zip",
}

def record_agent(algorithm, model_path):

    print(f"\nLoading {algorithm} model...")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if algorithm == "DQN":
        model = DQN.load(model_path, device="cpu")
    else:
        model = PPO.load(model_path, device="cpu")

    print(f"Loaded {algorithm} successfully.")

    env = make_space_invaders_env(
        n_envs=1,
        seed=SEED,
        clip_reward=False,
        terminal_on_life_loss=False,
        render_mode="rgb_array",
    )

    env = VecTransposeImage(env)

    video_path = VIDEO_DIR / f"{algorithm.lower()}_500k.mp4"

    all_frames = []
    episode_rewards = []

    try:
        for episode in range(EPISODES):

            obs = env.reset()
            done = False
            total_reward = 0

            print(f"\n{algorithm}: Recording episode {episode + 1}/{EPISODES}")

            while not done:

                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                obs, rewards, dones, infos = env.step(action)

                total_reward += float(rewards[0])
                done = bool(dones[0])

                frame = env.venv.envs[0].render()

                if frame is not None:
                    all_frames.append(
                        np.asarray(frame, dtype=np.uint8)
                    )

            episode_rewards.append(total_reward)

            print(
                f"{algorithm} Episode {episode + 1}: "
                f"Reward = {total_reward:.2f}"
            )

    finally:
        env.close()

    if not all_frames:
        raise RuntimeError(
            f"No frames were captured for {algorithm}. "
            "Check the environment render configuration."
        )

    print(f"\nSaving {algorithm} video...")

    imageio.mimsave(
        video_path,
        all_frames,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )

    print(f"Video saved: {video_path}")

    print(f"\n{algorithm} recording summary")
    print(f"Episodes: {EPISODES}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Frames recorded: {len(all_frames)}")
    print(f"Video duration: {len(all_frames) / FPS:.1f} seconds")

    return video_path

if __name__ == "__main__":

    gym.register_envs(ale_py)

    print("=" * 50)
    print("SPACE INVADERS - 500K GAMEPLAY RECORDINGS")
    print("=" * 50)

    for algorithm, model_path in MODELS.items():
        record_agent(algorithm, model_path)

    print("\nBoth recordings completed!")
    print(f"Videos are located in: {VIDEO_DIR}")
