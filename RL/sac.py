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

# Define the action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

# Instantiate the model with modified parameters for increased exploration
model = SAC(
    MlpPolicy,
    env,
    verbose=1,
    seed=seed,
    device="cpu",
    learning_rate=0.0003,  # Adjust the learning rate as needed
    buffer_size=1000000,  # Increase the replay buffer size
    action_noise=action_noise,  # Add action noise for exploration
    ent_coef="auto_1.0",  # Adjust the temperature parameter for entropy
)

# Create the callback
callback = CustomLoggingCallback()

# Train the model for a specified number of time steps
total_timesteps = 20000  # Total number of interactions with the server

model.learn(
    total_timesteps=total_timesteps, callback=callback, log_interval=2500
)

# Save the final model
model.save("sac_sqli_agent_final")
