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

# Instantiate the model
model = SAC(MlpPolicy, env, verbose=1, seed=seed, device="cpu")

# Create the callback
callback = CustomLoggingCallback()

# Train the model for a specified number of time steps
total_timesteps = 3000000  # Total number of interactions with the server

model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)

# Save the final model
model.save("sac_sqli_agent_final")

