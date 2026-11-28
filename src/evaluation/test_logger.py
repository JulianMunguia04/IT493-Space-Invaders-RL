
from src.config import LOG_DIR
from src.evaluation.logger import ExperimentLogger


def main():

    logger = ExperimentLogger(
        algorithm="test",
        seed=42,
        log_dir=LOG_DIR
    )

    logger.log_episode(1000, 120.0, 450)
    logger.log_episode(2000, 150.0, 520)
    logger.log_episode(3000, 180.0, 600)

    logger.close()

    print("Logger test completed successfully!")


if __name__ == "__main__":
    main()
