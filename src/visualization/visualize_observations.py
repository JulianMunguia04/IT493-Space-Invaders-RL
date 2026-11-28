
import matplotlib.pyplot as plt

from src.environment.env_factory import make_space_invaders_env


def main():

    env = make_space_invaders_env(seed=42)

    try:
        observation = env.reset()

        for step in range(100):
            action = [env.action_space.sample()]

            observation, rewards, dones, infos = env.step(action)

            if dones[0]:
                observation = env.reset()

        frames = observation[0]

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))

        for i in range(4):
            axes[i].imshow(frames[:, :, i], cmap="gray", vmin=0, vmax=255)
            axes[i].set_title(f"Frame {i + 1}")
            axes[i].axis("off")

        fig.suptitle("Space Invaders - Gameplay Frame Stack")

        plt.tight_layout()

        plt.savefig(
            "results/graphs/frame_stack.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()

        print("Saved visualization to results/graphs/frame_stack.png")

    finally:
        env.close()


if __name__ == "__main__":
    main()
