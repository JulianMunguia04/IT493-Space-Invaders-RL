
import csv
import time
from pathlib import Path


class ExperimentLogger:

    def __init__(self, algorithm, seed, log_dir, run_name="original"):

        self.algorithm = algorithm
        self.seed = seed
        self.run_name = run_name

        self.log_dir = (
            Path(log_dir)
            / algorithm
            / f"seed_{seed}"
            / run_name
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "training_metrics.csv"

        self.start_time = time.time()

        self.file = open(
            self.csv_path,
            mode="w",
            newline=""
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestep",
            "episode_reward",
            "episode_length",
            "elapsed_seconds"
        ])

    def log_episode(self, timestep, reward, length):

        elapsed = time.time() - self.start_time

        self.writer.writerow([
            timestep,
            reward,
            length,
            elapsed
        ])

        self.file.flush()

    def close(self):

        if not self.file.closed:
            self.file.close()

        print(f"Experiment logs saved to: {self.csv_path}")
