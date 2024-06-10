import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import MlpPolicy
from callbacks import CustomLoggingCallback
from environment import SQLiEnv
from stable_baselines3.common.noise import NormalActionNoise

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)

# Create the environment
env = make_vec_env(lambda: SQLiEnv(), n_envs=1)
env.seed(seed)

# Define the noise to add to actions
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

# Instantiate the model with a higher entropy coefficient and action noise
model = SAC(
    MlpPolicy,
    env,
    verbose=1,
    seed=seed,
    ent_coef=0.2,  # Increase entropy coefficient to encourage exploration
    action_noise=action_noise,
    device="cpu"
)

# Create the callback
callback = CustomLoggingCallback()

# Train the model for a specified number of time steps
total_timesteps = 400000  # Total number of interactions with the server

model.learn(
    total_timesteps=total_timesteps, callback=callback, log_interval=1000
)

# Save the final model
final_model_folder = "final_model"
os.makedirs(final_model_folder, exist_ok=True)
model.save(os.path.join(final_model_folder, "model"))

print("Training finished.")
