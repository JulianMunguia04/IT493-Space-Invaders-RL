
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import PROJECT_ROOT
from src.environment.env_factory import make_space_invaders_env


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = "cpu"

DQN_PATH = (
    PROJECT_ROOT
    / "models/dqn/seed_42/checkpoints/dqn_seed_42_100000_steps.zip"
)

PPO_PATH = (
    PROJECT_ROOT
    / "models/ppo/seed_42/checkpoints/ppo_seed_42_100000_steps.zip"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "saliency_maps"

# Capture observations at these points in the PPO gameplay sequence.
CAPTURE_STEPS = [0, 100, 300, 500]


# --------------------------------------------------
# Collect common game observations
# --------------------------------------------------

def collect_observations(ppo_model, env):
    """
    Collect observations from one deterministic PPO gameplay
    sequence. Both models will be analyzed using these same states.
    """

    observation = env.reset()
    captured = {}

    total_steps = max(CAPTURE_STEPS)

    for step in range(total_steps + 1):

        if step in CAPTURE_STEPS:
            captured[step] = observation.copy()
            print(f"Captured observation at step {step}")

        action, _ = ppo_model.predict(
            observation,
            deterministic=True
        )

        observation, rewards, dones, infos = env.step(action)

    return captured


# --------------------------------------------------
# Calculate gradient-based saliency
# --------------------------------------------------

def calculate_saliency(model, observation, algorithm):
    """
    Calculate the absolute input gradients for the selected action.

    DQN target: selected action's Q-value.
    PPO target: selected action's policy logit.
    """

    input_tensor, _ = model.policy.obs_to_tensor(observation)

    # Convert image observations to floating point so gradients
    # can be calculated with respect to the input pixels.
    input_tensor = (
        input_tensor.float()
        .detach()
        .requires_grad_(True)
    )

    model.policy.zero_grad(set_to_none=True)

    if algorithm == "dqn":

        q_values = model.policy.q_net(input_tensor)

        selected_action = q_values.argmax(dim=1).item()

        target_score = q_values[0, selected_action]

    elif algorithm == "ppo":

        distribution = model.policy.get_distribution(input_tensor)

        logits = distribution.distribution.logits

        selected_action = logits.argmax(dim=1).item()

        target_score = logits[0, selected_action]

    else:
        raise ValueError("Algorithm must be 'dqn' or 'ppo'.")

    gradients = torch.autograd.grad(
        outputs=target_score,
        inputs=input_tensor
    )[0]

    # Combine the four stacked grayscale frames into one
    # spatial saliency map.
    saliency = gradients.abs().sum(dim=1)[0]

    saliency = saliency.detach().cpu().numpy()

    # Normalize using a high percentile to reduce the influence
    # of extreme gradient values.
    upper_limit = np.percentile(saliency, 99.5)

    if upper_limit > 0:
        saliency = np.clip(saliency / upper_limit, 0, 1)

    return saliency, selected_action


# --------------------------------------------------
# Generate visualization
# --------------------------------------------------

def generate_maps(dqn_model, ppo_model, observations):

    algorithms = [
        ("DQN", dqn_model, "dqn"),
        ("PPO", ppo_model, "ppo")
    ]

    all_maps = {}

    for label, model, algorithm in algorithms:

        all_maps[label] = []

        for step, observation in observations.items():

            saliency, action = calculate_saliency(
                model,
                observation,
                algorithm
            )

            # VecTransposeImage gives shape:
            # (environment, stacked_frames, height, width)
            # The last frame is the newest frame.
            game_frame = observation[0, -1]

            all_maps[label].append({
                "step": step,
                "frame": game_frame,
                "saliency": saliency,
                "action": action
            })

            print(
                f"{label} | Step {step} | "
                f"Selected action: {action}"
            )

    return all_maps


# --------------------------------------------------
# Save individual algorithm visualizations
# --------------------------------------------------

def save_algorithm_grid(all_maps, algorithm):

    maps = all_maps[algorithm]

    fig, axes = plt.subplots(
        1,
        len(maps),
        figsize=(16, 4)
    )

    for ax, item in zip(axes, maps):

        ax.imshow(item["frame"], cmap="gray")

        heatmap = ax.imshow(
            item["saliency"],
            cmap="inferno",
            alpha=0.65,
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"Step {item['step']}\n"
            f"Action {item['action']}"
        )

        ax.axis("off")

    fig.colorbar(
        heatmap,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="Normalized gradient magnitude"
    )

    fig.suptitle(
        f"{algorithm} Gradient-Based Saliency",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    filename = OUTPUT_DIR / f"{algorithm.lower()}_saliency.png"

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# --------------------------------------------------
# Save direct DQN vs PPO comparison
# --------------------------------------------------

def save_comparison(all_maps):

    fig, axes = plt.subplots(
        2,
        len(CAPTURE_STEPS),
        figsize=(16, 8)
    )

    for row, algorithm in enumerate(["DQN", "PPO"]):

        for col, item in enumerate(all_maps[algorithm]):

            ax = axes[row, col]

            ax.imshow(item["frame"], cmap="gray")

            heatmap = ax.imshow(
                item["saliency"],
                cmap="inferno",
                alpha=0.65,
                vmin=0,
                vmax=1
            )

            ax.set_title(
                f"Step {item['step']} | Action {item['action']}"
            )

            ax.axis("off")

            if col == 0:
                ax.set_ylabel(
                    algorithm,
                    fontsize=14,
                    fontweight="bold"
                )

    fig.suptitle(
        "DQN vs PPO: Visual Saliency Comparison",
        fontsize=17,
        fontweight="bold"
    )

    fig.colorbar(
        heatmap,
        ax=axes,
        fraction=0.02,
        pad=0.02,
        label="Normalized gradient magnitude"
    )

    fig.tight_layout()

    filename = OUTPUT_DIR / "dqn_vs_ppo_saliency.png"

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained models...")

    dqn_model = DQN.load(DQN_PATH, device=DEVICE)
    ppo_model = PPO.load(PPO_PATH, device=DEVICE)

    print("\nCreating evaluation environment...")

    env = make_space_invaders_env(
        n_envs=1,
        seed=42,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    env = VecTransposeImage(env)

    try:

        print("\nCollecting shared gameplay observations...")

        observations = collect_observations(
            ppo_model,
            env
        )

        print("\nCalculating saliency maps...")

        all_maps = generate_maps(
            dqn_model,
            ppo_model,
            observations
        )

        print("\nSaving visualizations...")

        save_algorithm_grid(all_maps, "DQN")
        save_algorithm_grid(all_maps, "PPO")
        save_comparison(all_maps)

        print("\n" + "=" * 50)
        print("SALiency MAP GENERATION COMPLETED")
        print("=" * 50)

        print(f"Output directory: {OUTPUT_DIR}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
