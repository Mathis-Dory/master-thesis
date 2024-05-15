from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import MlpPolicy

from callbacks import CustomLoggingCallback
from environment import SQLiEnv

# Create the environment
env = make_vec_env(lambda: SQLiEnv(), n_envs=1)

# Instantiate the model
model = SAC(MlpPolicy, env, verbose=1)

# Create the callback
callback = CustomLoggingCallback()

# Train the model
model.learn(total_timesteps=2000000, callback=callback)
model.save("sac_sqli_agent_final")
