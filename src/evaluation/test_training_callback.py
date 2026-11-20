
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.config import LOG_DIR
from src.evaluation.logger import ExperimentLogger
from src.evaluation.training_callback import TrainingLoggerCallback


def main():
    print("Creating test environment...")

    env = Monitor(gym.make("CartPole-v1"))

    logger = ExperimentLogger(
        algorithm="callback_test",
        seed=42,
        log_dir=LOG_DIR
    )

    callback = TrainingLoggerCallback(
        experiment_logger=logger
    )

    print("Starting short PPO training test...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        tensorboard_log=str(LOG_DIR / "tensorboard")
    )

    model.learn(
        total_timesteps=2_000,
        callback=callback,
        tb_log_name="callback_test"
    )

    env.close()

    print("\nCallback test completed!")


if __name__ == "__main__":
    main()
