
from stable_baselines3 import DQN, PPO

from src.environment.env_factory import make_space_invaders_env
from src.evaluation.logger import ExperimentLogger
from src.evaluation.training_callback import TrainingLoggerCallback
from src.config import LOG_DIR

from src.dqn.config import DQN_CONFIG
from src.ppo.config import PPO_CONFIG


def test_dqn():
    print("\n========== DQN PILOT TEST ==========")

    env = make_space_invaders_env(seed=42)

    logger = ExperimentLogger(
        algorithm="dqn_pilot",
        seed=42,
        log_dir=LOG_DIR
    )

    callback = TrainingLoggerCallback(logger)

    # Temporary settings for the pilot test.
    # These allow DQN to start learning within 1,000 timesteps.
    dqn_test_config = {
        key: value
        for key, value in DQN_CONFIG.items()
        if key not in ("policy", "learning_starts", "buffer_size")
    }

    dqn_test_config.update({
        "learning_starts": 100,
        "buffer_size": 2_000
    })

    try:
        model = DQN(
            DQN_CONFIG["policy"],
            env,
            seed=42,
            **dqn_test_config
        )

        print("\nTraining DQN for 1,000 timesteps...")

        model.learn(
            total_timesteps=1_000,
            callback=callback
        )

        print("DQN pilot test passed!")

    finally:
        env.close()


def test_ppo():
    print("\n========== PPO PILOT TEST ==========")

    env = make_space_invaders_env(seed=42)

    logger = ExperimentLogger(
        algorithm="ppo_pilot",
        seed=42,
        log_dir=LOG_DIR
    )

    callback = TrainingLoggerCallback(logger)

    try:
        model = PPO(
            PPO_CONFIG["policy"],
            env,
            seed=42,
            **{
                key: value
                for key, value in PPO_CONFIG.items()
                if key != "policy"
            }
        )

        print("\nTraining PPO for 512 timesteps...")

        model.learn(
            total_timesteps=512,
            callback=callback
        )

        print("PPO pilot test passed!")

    finally:
        env.close()


def main():
    test_dqn()
    test_ppo()

    print("\nBoth Space Invaders pilot tests completed!")


if __name__ == "__main__":
    main()
