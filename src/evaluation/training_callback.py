
import time
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, experiment_logger, verbose=0):
        super().__init__(verbose)
        self.experiment_logger = experiment_logger
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        infos = self.locals.get("infos", [])

        for info in infos:
            episode = info.get("episode")

            if episode is not None:
                self.experiment_logger.log_episode(
                    timestep=self.num_timesteps,
                    reward=episode["r"],
                    length=episode["l"]
                )

        return True

    def _on_training_end(self):
        self.experiment_logger.close()

        elapsed = time.time() - self.start_time

        print(
            f"Training completed in {elapsed:.2f} seconds."
        )
