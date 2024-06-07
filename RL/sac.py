import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import MlpPolicy

from callbacks import CustomLoggingCallback
from environment import SQLiEnv

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)

# Create the environment
env = make_vec_env(lambda: SQLiEnv(), n_envs=1)
env.seed(seed)

# Instantiate the model with a learning rate scheduler
model = SAC(
    MlpPolicy,
    env,
    verbose=1,
    seed=seed,
    device="cpu"
)

# Create the callback
callback = CustomLoggingCallback()

# Train the model for a specified number of time steps
total_timesteps = 1000  # Total number of interactions with the server

model.learn(
    total_timesteps=total_timesteps, callback=callback, log_interval=1000
)

# Save the final model
final_model_folder = "final_model"
os.makedirs(final_model_folder, exist_ok=True)
model.save(os.path.join(final_model_folder, "model"))

print("Training finished.")
