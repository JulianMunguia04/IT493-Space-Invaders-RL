
import csv
import numpy as np

from pathlib import Path

from src.config import (
    RANDOM_SEEDS,
    LOG_DIR,
    MODEL_DIR,
    DQN_MODEL_DIR,
    PPO_MODEL_DIR,
    RESULTS_DIR,
)


ALGORITHMS = ["dqn", "ppo"]

EXPECTED_TIMESTEPS = np.array([
    20_000,
    40_000,
    60_000,
    80_000,
    100_000
])

EXPECTED_EPISODES = 3


def audit_experiments():
    print("\n" + "=" * 65)
    print("SPACE INVADERS EXPERIMENT AUDIT")
    print("=" * 65)

    all_valid = True
    evaluation_details = []
    per_seed_results = {}

    for algorithm in ALGORITHMS:

        print(f"\n{'-' * 65}")
        print(f"ALGORITHM: {algorithm.upper()}")
        print("-" * 65)

        per_seed_results[algorithm] = []

        model_dir = (
            DQN_MODEL_DIR if algorithm == "dqn"
            else PPO_MODEL_DIR
        )

        for seed in RANDOM_SEEDS:

            print(f"\nChecking {algorithm.upper()} | Seed {seed}")

            evaluation_path = (
                LOG_DIR / "evaluation" / algorithm
                / f"seed_{seed}" / "evaluations.npz"
            )

            training_log_path = (
                LOG_DIR / algorithm
                / f"seed_{seed}" / "training_metrics.csv"
            )

            final_model_path = (
                model_dir / f"{algorithm}_seed_{seed}_final.zip"
            )

            best_model_path = (
                MODEL_DIR / algorithm / f"seed_{seed}"
                / "best" / "best_model.zip"
            )

            required_files = {
                "Evaluation data": evaluation_path,
                "Training log": training_log_path,
                "Final model": final_model_path,
                "Best model": best_model_path,
            }

            for name, path in required_files.items():
                exists = path.exists()

                print(
                    f"  {'PASS' if exists else 'FAIL'}: "
                    f"{name}"
                )

                if not exists:
                    all_valid = False

            if not evaluation_path.exists():
                continue

            # Load evaluation data
            data = np.load(evaluation_path)

            timesteps = data["timesteps"]
            rewards = data["results"]
            episode_lengths = data["ep_lengths"]

            # Validate array dimensions
            if not np.array_equal(
                timesteps, EXPECTED_TIMESTEPS
            ):
                print("  FAIL: Unexpected evaluation timesteps")
                all_valid = False

            if rewards.shape != (5, EXPECTED_EPISODES):
                print("  FAIL: Unexpected reward array shape")
                all_valid = False
                continue

            if episode_lengths.shape != (
                5, EXPECTED_EPISODES
            ):
                print("  FAIL: Unexpected episode length shape")
                all_valid = False
                continue

            # Validate numerical values
            if not np.all(np.isfinite(rewards)):
                print("  FAIL: Non-finite reward values")
                all_valid = False

            if not np.all(np.isfinite(episode_lengths)):
                print("  FAIL: Non-finite episode lengths")
                all_valid = False

            if np.any(episode_lengths <= 0):
                print("  FAIL: Invalid episode lengths")
                all_valid = False

            if np.any(rewards < 0):
                print("  WARNING: Negative evaluation rewards found")

            print("  PASS: Evaluation arrays loaded")
            print(f"  Evaluation shape: {rewards.shape}")
            print(f"  Episode length shape: {episode_lengths.shape}")

            # Calculate mean reward for each checkpoint
            mean_scores = rewards.mean(axis=1)

            for i, timestep in enumerate(timesteps):

                for episode in range(EXPECTED_EPISODES):

                    evaluation_details.append([
                        algorithm,
                        seed,
                        int(timestep),
                        episode + 1,
                        float(rewards[i, episode]),
                        int(episode_lengths[i, episode])
                    ])

            final_mean = float(mean_scores[-1])
            final_episode_std = float(
                rewards[-1].std(ddof=1)
            )

            per_seed_results[algorithm].append({
                "seed": seed,
                "mean_scores": mean_scores,
                "final_mean": final_mean,
                "final_episode_std": final_episode_std
            })

            print(f"  Final evaluation mean: {final_mean:.2f}")
            print(
                "  Final episode standard deviation: "
                f"{final_episode_std:.2f}"
            )

    # Export episode-level data
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = RESULTS_DIR / "evaluation_details.csv"

    with open(details_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "algorithm",
            "seed",
            "timestep",
            "episode",
            "reward",
            "episode_length"
        ])

        writer.writerows(evaluation_details)

    # Export seed-level summary and learning curves
    seed_summary_path = RESULTS_DIR / "seed_summary.csv"
    curve_summary_path = RESULTS_DIR / "learning_curve_summary.csv"

    with open(seed_summary_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "algorithm",
            "seed",
            "final_mean_reward",
            "final_episode_std"
        ])

        for algorithm in ALGORITHMS:
            for result in per_seed_results[algorithm]:
                writer.writerow([
                    algorithm,
                    result["seed"],
                    result["final_mean"],
                    result["final_episode_std"]
                ])

    with open(curve_summary_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "algorithm",
            "timestep",
            "mean_reward_across_seeds",
            "std_across_seeds"
        ])

        for algorithm in ALGORITHMS:

            results = per_seed_results[algorithm]

            if len(results) != len(RANDOM_SEEDS):
                all_valid = False
                continue

            scores = np.array([
                result["mean_scores"] for result in results
            ])

            for i, timestep in enumerate(EXPECTED_TIMESTEPS):

                writer.writerow([
                    algorithm,
                    int(timestep),
                    float(scores[:, i].mean()),
                    float(scores[:, i].std(ddof=1))
                ])

    # Final comparison
    print("\n" + "=" * 65)
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 65)

    for algorithm in ALGORITHMS:

        results = per_seed_results[algorithm]

        if len(results) != len(RANDOM_SEEDS):
            print(f"{algorithm.upper()}: INCOMPLETE")
            all_valid = False
            continue

        final_scores = np.array([
            result["final_mean"] for result in results
        ])

        print(f"\n{algorithm.upper()}")
        print(f"  Seed scores: {final_scores}")
        print(f"  Overall mean: {final_scores.mean():.2f}")
        print(
            "  Standard deviation across seeds: "
            f"{final_scores.std(ddof=1):.2f}"
        )

    print("\n" + "=" * 65)

    print(
        "AUDIT STATUS:",
        "PASSED" if all_valid else "ISSUES FOUND"
    )

    print("\nGenerated files:")
    print(details_path)
    print(seed_summary_path)
    print(curve_summary_path)


if __name__ == "__main__":
    audit_experiments()
