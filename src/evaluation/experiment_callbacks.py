
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback
)

from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import (
    EVAL_FREQ,
    N_EVAL_EPISODES,
    CHECKPOINT_FREQ,
    N_ENVS,
    LOG_DIR,
    MODEL_DIR
)

from src.environment.env_factory import make_space_invaders_env
from src.evaluation.training_callback import TrainingLoggerCallback


def make_experiment_callbacks(
    algorithm,
    seed,
    experiment_logger,
    eval_freq=None,
    checkpoint_freq=None,
    n_eval_episodes=None
):
    """
    Build the callbacks used during a full experiment.

    Includes:
    1. Training episode logging
    2. Periodic full-game evaluation
    3. Periodic model checkpoints

    Optional frequency arguments allow shorter integration tests.
    """

    algorithm = algorithm.lower()

    if algorithm not in ("dqn", "ppo"):
        raise ValueError("Algorithm must be 'dqn' or 'ppo'.")

    # Use production settings unless test overrides are supplied
    eval_freq = EVAL_FREQ if eval_freq is None else eval_freq

    checkpoint_freq = (
        CHECKPOINT_FREQ
        if checkpoint_freq is None
        else checkpoint_freq
    )

    n_eval_episodes = (
        N_EVAL_EPISODES
        if n_eval_episodes is None
        else n_eval_episodes
    )

    if eval_freq <= 0 or checkpoint_freq <= 0 or n_eval_episodes <= 0:
        raise ValueError("Evaluation and checkpoint settings must be positive.")

    # Create directories for this experiment
    run_model_dir = MODEL_DIR / algorithm / f"seed_{seed}"

    best_model_dir = run_model_dir / "best"
    checkpoint_dir = run_model_dir / "checkpoints"

    evaluation_log_dir = (
        LOG_DIR / "evaluation" / algorithm / f"seed_{seed}"
    )

    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_log_dir.mkdir(parents=True, exist_ok=True)

    # Separate evaluation environment:
    # - Actual, unclipped rewards
    # - Complete games rather than life-loss termination
    eval_env = make_space_invaders_env(
        n_envs=1,
        seed=seed + 10_000,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    # Match the image observation format used by the training model
    eval_env = VecTransposeImage(eval_env)

    # 1. Training episode logger
    training_callback = TrainingLoggerCallback(
        experiment_logger=experiment_logger
    )

    # 2. Periodic evaluation callback
    evaluation_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(evaluation_log_dir),
        eval_freq=max(eval_freq // N_ENVS, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 3. Periodic checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // N_ENVS, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{algorithm}_seed_{seed}",
        save_replay_buffer=(algorithm == "dqn"),
        verbose=1
    )

    # Combine all callbacks
    callbacks = CallbackList([
        training_callback,
        evaluation_callback,
        checkpoint_callback
    ])

    return callbacks, eval_env
