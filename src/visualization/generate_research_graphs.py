from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"

LEARNING_CURVE_FILE = RESULTS_DIR / "learning_curve_summary.csv"
SEED_SUMMARY_FILE = RESULTS_DIR / "seed_summary.csv"
EVALUATION_FILE = RESULTS_DIR / "evaluation_details.csv"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

learning_curves = pd.read_csv(LEARNING_CURVE_FILE)
seed_summary = pd.read_csv(SEED_SUMMARY_FILE)
evaluation = pd.read_csv(EVALUATION_FILE)

algorithms = ["dqn", "ppo"]
algorithm_labels = {
    "dqn": "DQN",
    "ppo": "PPO"
}

print("Research datasets loaded successfully.")

def generate_learning_curves():

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in algorithms:

        data = (
            learning_curves[
                learning_curves["algorithm"] == algorithm
            ]
            .sort_values("timestep")
        )

        timesteps = data["timestep"].to_numpy()
        means = data["mean_reward_across_seeds"].to_numpy()
        stds = data["std_across_seeds"].to_numpy()

        ax.plot(
            timesteps,
            means,
            marker="o",
            linewidth=2,
            markersize=6,
            label=algorithm_labels[algorithm]
        )

        ax.fill_between(
            timesteps,
            means - stds,
            means + stds,
            alpha=0.15
        )

    ax.set_title(
        "DQN vs PPO: Learning Curves",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)

    ax.set_xlim(0, 100000)
    ax.set_xticks(np.arange(0, 100001, 20000))

    ax.grid(True, alpha=0.3)
    ax.legend(title="Algorithm")

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "learning_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Generated: learning_curves.png")

def generate_final_performance():

    fig, ax = plt.subplots(figsize=(8, 6))

    means = []
    stds = []

    for algorithm in algorithms:

        scores = seed_summary.loc[
            seed_summary["algorithm"] == algorithm,
            "final_mean_reward"
        ].to_numpy()

        means.append(np.mean(scores))
        stds.append(np.std(scores, ddof=1))

    positions = np.arange(len(algorithms))

    bars = ax.bar(
        positions,
        means,
        yerr=stds,
        capsize=8,
        width=0.55
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(["DQN", "PPO"])

    ax.set_title(
        "Final Performance Comparison",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylabel("Mean Final Evaluation Reward")
    ax.set_xlabel("Algorithm")

    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{mean:.2f}",
            ha="center",
            fontweight="bold"
        )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "final_performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Generated: final_performance.png")

def generate_seed_performance():

    final_timestep = evaluation["timestep"].max()

    final_data = evaluation[
        evaluation["timestep"] == final_timestep
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    seed_markers = {
        42: "o",
        123: "s",
        2026: "^"
    }

    positions = {
        "dqn": 0,
        "ppo": 1
    }

    offsets = {
        42: -0.16,
        123: 0,
        2026: 0.16
    }

    for algorithm in algorithms:

        algorithm_data = final_data[
            final_data["algorithm"] == algorithm
        ]

        for seed in [42, 123, 2026]:

            seed_data = algorithm_data[
                algorithm_data["seed"] == seed
            ]

            x = positions[algorithm] + offsets[seed]

            ax.scatter(
                np.full(len(seed_data), x),
                seed_data["reward"],
                marker=seed_markers[seed],
                s=65,
                alpha=0.8,
                label=f"Seed {seed}" if algorithm == "dqn" else None
            )

        overall_mean = algorithm_data["reward"].mean()

        ax.hlines(
            overall_mean,
            positions[algorithm] - 0.28,
            positions[algorithm] + 0.28,
            linewidth=2
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DQN", "PPO"])

    ax.set_title(
        "Final Evaluation Scores Across Training Seeds",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("Evaluation Episode Reward")
    ax.set_xlabel("Algorithm")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Training Seed")

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "individual_seed_performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Generated: individual_seed_performance.png")

def generate_evaluation_stability():

    # Calculate the standard deviation of the three
    # evaluation episodes for each training seed
    # at every evaluation checkpoint.

    per_seed_variability = (
        evaluation
        .groupby(["algorithm", "seed", "timestep"])["reward"]
        .std()
        .reset_index(name="episode_reward_std")
    )

    # Calculate the average within-seed variability
    # and its variation across the three training seeds.

    stability_summary = (
        per_seed_variability
        .groupby(["algorithm", "timestep"])["episode_reward_std"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in algorithms:

        data = (
            stability_summary[
                stability_summary["algorithm"] == algorithm
            ]
            .sort_values("timestep")
        )

        timesteps = data["timestep"].to_numpy()
        means = data["mean"].to_numpy()
        stds = data["std"].fillna(0).to_numpy()

        ax.plot(
            timesteps,
            means,
            marker="o",
            linewidth=2,
            label=algorithm_labels[algorithm]
        )

        ax.fill_between(
            timesteps,
            np.maximum(means - stds, 0),
            means + stds,
            alpha=0.15
        )

    ax.set_title(
        "Evaluation Episode Variability During Training",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Within-Seed Reward Standard Deviation")

    ax.set_xlim(0, 100000)
    ax.set_xticks(np.arange(0, 100001, 20000))

    ax.grid(True, alpha=0.3)
    ax.legend(title="Algorithm")

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "evaluation_stability.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Generated: evaluation_stability.png")

def main():

    print("\nGenerating research graphs...")
    print("-" * 45)

    generate_learning_curves()
    generate_final_performance()
    generate_seed_performance()
    generate_evaluation_stability()

    print("-" * 45)
    print("All research graphs generated successfully!")

    print(f"\nOutput directory: {GRAPH_DIR}")


if __name__ == "__main__":
    main()