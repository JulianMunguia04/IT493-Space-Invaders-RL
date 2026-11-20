
# PPO experiment configuration

PPO_CONFIG = {
    # Neural network
    "policy": "CnnPolicy",

    # Training
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "gae_lambda": 0.95,

    # Policy updates
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,

    # Policy optimization
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Hardware
    "device": "cpu",
    "verbose": 1,
}

PPO_TIMESTEPS = 100_000
