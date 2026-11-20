
# DQN experiment configuration

DQN_CONFIG = {
    # Neural network
    "policy": "CnnPolicy",

    # Training
    "learning_rate": 0.0001,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "batch_size": 32,
    "gamma": 0.99,

    # Learning frequency
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1_000,

    # Exploration
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,

    # Hardware
    "device": "cpu",
    "verbose": 1,
}

DQN_TIMESTEPS = 100_000
