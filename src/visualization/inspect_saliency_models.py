
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from src.config import PROJECT_ROOT
from src.environment.env_factory import make_space_invaders_env


def main():
    device = "cpu"

    dqn_path = (
        PROJECT_ROOT
        / "models/dqn/seed_42/checkpoints/dqn_seed_42_100000_steps.zip"
    )

    ppo_path = (
        PROJECT_ROOT
        / "models/ppo/seed_42/checkpoints/ppo_seed_42_100000_steps.zip"
    )

    print("Loading final DQN model...")
    dqn = DQN.load(dqn_path, device=device)

    print("Loading final PPO model...")
    ppo = PPO.load(ppo_path, device=device)

    print("\nCreating evaluation environment...")

    env = make_space_invaders_env(
        n_envs=1,
        seed=42,
        clip_reward=False,
        terminal_on_life_loss=False
    )

    env = VecTransposeImage(env)

    observation = env.reset()

    print("\nObservation shape:", observation.shape)
    print("Observation dtype:", observation.dtype)

    dqn_tensor, _ = dqn.policy.obs_to_tensor(observation)
    ppo_tensor, _ = ppo.policy.obs_to_tensor(observation)

    print("\nTesting DQN network...")

    with torch.no_grad():
        dqn_q_values = dqn.policy.q_net(dqn_tensor)

    print("DQN Q-value shape:", dqn_q_values.shape)
    print("DQN selected action:", dqn_q_values.argmax(dim=1).item())

    print("\nTesting PPO network...")

    with torch.no_grad():
        ppo_distribution = ppo.policy.get_distribution(ppo_tensor)
        ppo_logits = ppo_distribution.distribution.logits

    print("PPO logits shape:", ppo_logits.shape)
    print("PPO selected action:", ppo_logits.argmax(dim=1).item())

    print("\nBoth models successfully processed the same observation.")

    env.close()


if __name__ == "__main__":
    main()
