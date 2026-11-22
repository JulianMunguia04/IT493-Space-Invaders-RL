
import gymnasium as gym
import ale_py

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def make_space_invaders_env(
    n_envs=1,
    seed=42,
    monitor_dir=None,
    clip_reward=True,
    terminal_on_life_loss=True,
    render_mode=None
):
    """
    Create a preprocessed Space Invaders environment.

    Preprocessing:
    - Frame skipping (4)
    - Grayscale conversion
    - Resize to 84x84
    - Stack 4 consecutive frames

    Training defaults:
    - Reward clipping enabled
    - Life-loss termination enabled

    Evaluation:
    - Reward clipping disabled
    - Life-loss termination disabled

    Rendering:
    - None: run without a visible game window
    - "human": display the game window
    """

    gym.register_envs(ale_py)

    env_kwargs = {
        "frameskip": 1,
        "repeat_action_probability": 0.0
    }

    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = make_atari_env(
        "ALE/SpaceInvaders-v5",
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        env_kwargs=env_kwargs,
        wrapper_kwargs={
            "screen_size": 84,
            "frame_skip": 4,
            "clip_reward": clip_reward,
            "terminal_on_life_loss": terminal_on_life_loss
        }
    )

    env = VecFrameStack(
        env,
        n_stack=4,
        channels_order="last"
    )

    return env
