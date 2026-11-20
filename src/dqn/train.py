
from stable_baselines3 import DQN

from src.config import (
    RANDOM_SEEDS,
    DQN_MODEL_DIR,
    LOG_DIR,
    create_project_directories
)

from src.environment.env_factory import make_space_invaders_env
from src.evaluation.logger import ExperimentLogger
from src.evaluation.experiment_callbacks import make_experiment_callbacks

from src.dqn.config import DQN_CONFIG, DQN_TIMESTEPS


def train_dqn(seed):
    print("\n" + "=" * 50)
    print(f"STARTING DQN TRAINING | SEED {seed}")
    print("=" * 50)

    env = make_space_invaders_env(
        n_envs=1,
        seed=seed
    )

    logger = ExperimentLogger(
        algorithm="dqn",
        seed=seed,
        log_dir=LOG_DIR
    )

    eval_env = None

    try:
        # Create evaluation and checkpoint callbacks
        callbacks, eval_env = make_experiment_callbacks(
            algorithm="dqn",
            seed=seed,
            experiment_logger=logger
        )

        # Initialize DQN
        model = DQN(
            policy=DQN_CONFIG["policy"],
            env=env,
            seed=seed,
            **{
                key: value
                for key, value in DQN_CONFIG.items()
                if key != "policy"
            },
            tensorboard_log=str(LOG_DIR / "tensorboard")
        )

        print(f"\nTraining DQN for {DQN_TIMESTEPS:,} timesteps...")

        model.learn(
            total_timesteps=DQN_TIMESTEPS,
            callback=callbacks,
            tb_log_name=f"dqn_seed_{seed}"
        )

        # Save final model
        model_path = DQN_MODEL_DIR / f"dqn_seed_{seed}_final"

        model.save(str(model_path))

        print(f"\nFinal DQN model saved to: {model_path}.zip")

    finally:
        env.close()

        if eval_env is not None:
            eval_env.close()


def main():
    create_project_directories()

    print("DQN EXPERIMENT")
    print(f"Training budget: {DQN_TIMESTEPS:,} timesteps")
    print(f"Random seeds: {RANDOM_SEEDS}")

    for seed in RANDOM_SEEDS:
        train_dqn(seed)

    print("\nAll DQN experiments completed!")


if __name__ == "__main__":
    main()
