import gymnasium as gym
import ale_py

# Register Atari environments
gym.register_envs(ale_py)

# Create Space Invaders
env = gym.make(
    "ALE/SpaceInvaders-v5",
    render_mode="human"
)

# Start the game
observation, info = env.reset(seed=42)

print("Environment successfully initialized!")
print("Observation shape:", observation.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Let a random agent play for 1,000 steps
for step in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

print("Space Invaders test completed!")

# Run python src/environment/test_space_invaders.py