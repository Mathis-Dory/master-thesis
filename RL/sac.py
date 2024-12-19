import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.sac.policies import MlpPolicy

from callbacks import CustomLoggingCallback
from environment import SQLiEnv

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)

# Create the environment
env = make_vec_env(lambda: SQLiEnv(), n_envs=1)
env.seed(seed)

# Define action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions)
)

# Instantiate the SAC model with a higher entropy coefficient and action noise
model = SAC(
    policy=MlpPolicy,
    env=env,
    verbose=1,
    seed=seed,
    ent_coef=0.2,  # Encourages exploration
    action_noise=action_noise,
    device="cpu",
)

# Custom callback for logging
callback = CustomLoggingCallback()

# Train the model
total_timesteps = 1000000
model.learn(
    total_timesteps=total_timesteps, callback=callback, log_interval=100
)

# Save the final model
model_folder = "final_model"
os.makedirs(model_folder, exist_ok=True)
model.save(os.path.join(model_folder, "sac_sql_injection_model"))

print("Training finished.")
