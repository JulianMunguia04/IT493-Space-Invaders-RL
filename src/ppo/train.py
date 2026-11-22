
from stable_baselines3 import PPO

from src.config import (
    RANDOM_SEEDS,
    PPO_MODEL_DIR,
    LOG_DIR,
    create_project_directories
)

from src.environment.env_factory import make_space_invaders_env
from src.evaluation.logger import ExperimentLogger
from src.evaluation.experiment_callbacks import make_experiment_callbacks

from src.ppo.config import PPO_CONFIG, PPO_TIMESTEPS




def train_ppo(seed, total_timesteps=None, run_name="original"):

    if total_timesteps is None:
        total_timesteps = PPO_TIMESTEPS

    create_project_directories()

    print("\n" + "=" * 50)
    print(f"STARTING PPO TRAINING | SEED {seed}")
    print(f"TRAINING BUDGET: {total_timesteps:,}")
    print("=" * 50)

    env = make_space_invaders_env(
        n_envs=1,
        seed=seed
    )

    logger = ExperimentLogger(
        algorithm="ppo",
        seed=seed,
        log_dir=LOG_DIR,
        run_name=run_name
    )

    eval_env = None

    try:
        callbacks, eval_env = make_experiment_callbacks(
            algorithm="ppo",
            seed=seed,
            experiment_logger=logger,
            run_name=run_name
        )

        model = PPO(
            policy=PPO_CONFIG["policy"],
            env=env,
            seed=seed,
            **{
                key: value
                for key, value in PPO_CONFIG.items()
                if key != "policy"
            },
            tensorboard_log=str(LOG_DIR / "tensorboard")
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"ppo_seed_{seed}_{run_name}"
        )

        filename = (
            f"ppo_seed_{seed}_final"
            if run_name == "original"
            else f"ppo_seed_{seed}_{run_name}_final"
        )

        model_path = PPO_MODEL_DIR / filename
        model.save(str(model_path))

        print(f"\nFinal PPO model saved to: {model_path}.zip")

        return model

    finally:
        env.close()

        if eval_env is not None:
            eval_env.close()

        logger.close()

def main():
    create_project_directories()

    print("PPO EXPERIMENT")
    print(f"Training budget: {PPO_TIMESTEPS:,} timesteps")
    print(f"Random seeds: {RANDOM_SEEDS}")

    for seed in RANDOM_SEEDS:
        train_ppo(seed)

    print("\nAll PPO experiments completed!")


if __name__ == "__main__":
    main()
