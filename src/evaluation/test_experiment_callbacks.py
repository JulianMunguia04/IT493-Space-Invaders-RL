
from stable_baselines3 import DQN, PPO

from src.config import LOG_DIR
from src.dqn.config import DQN_CONFIG
from src.ppo.config import PPO_CONFIG

from src.environment.env_factory import make_space_invaders_env
from src.evaluation.logger import ExperimentLogger
from src.evaluation.experiment_callbacks import make_experiment_callbacks


def test_dqn_callbacks():
    print("\n========== DQN CALLBACK INTEGRATION TEST ==========")

    seed = 42
    env = make_space_invaders_env(seed=seed)
    eval_env = None

    logger = ExperimentLogger(
        algorithm="dqn_integration_test",
        seed=seed,
        log_dir=LOG_DIR
    )

    try:
        callbacks, eval_env = make_experiment_callbacks(
            algorithm="dqn",
            seed=seed,
            experiment_logger=logger,
            eval_freq=512,
            checkpoint_freq=512,
            n_eval_episodes=1
        )

        # Temporary settings to allow actual DQN learning
        dqn_test_config = {
            key: value
            for key, value in DQN_CONFIG.items()
            if key not in ("policy", "learning_starts", "buffer_size")
        }

        dqn_test_config.update({
            "learning_starts": 100,
            "buffer_size": 2_000
        })

        model = DQN(
            DQN_CONFIG["policy"],
            env,
            seed=seed,
            **dqn_test_config
        )

        print("Running DQN callback test...")

        model.learn(
            total_timesteps=512,
            callback=callbacks
        )

        print("DQN callback integration test passed!")

    finally:
        env.close()

        if eval_env is not None:
            eval_env.close()


def test_ppo_callbacks():
    print("\n========== PPO CALLBACK INTEGRATION TEST ==========")

    seed = 42
    env = make_space_invaders_env(seed=seed)
    eval_env = None

    logger = ExperimentLogger(
        algorithm="ppo_integration_test",
        seed=seed,
        log_dir=LOG_DIR
    )

    try:
        callbacks, eval_env = make_experiment_callbacks(
            algorithm="ppo",
            seed=seed,
            experiment_logger=logger,
            eval_freq=512,
            checkpoint_freq=512,
            n_eval_episodes=1
        )

        model = PPO(
            PPO_CONFIG["policy"],
            env,
            seed=seed,
            **{
                key: value
                for key, value in PPO_CONFIG.items()
                if key != "policy"
            }
        )

        print("Running PPO callback test...")

        model.learn(
            total_timesteps=512,
            callback=callbacks
        )

        print("PPO callback integration test passed!")

    finally:
        env.close()

        if eval_env is not None:
            eval_env.close()


def main():
    test_dqn_callbacks()
    test_ppo_callbacks()

    print("\nAll experiment callback integration tests passed!")


if __name__ == "__main__":
    main()
