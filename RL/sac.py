from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import MlpPolicy

from callbacks import CustomLoggingCallback
from environement import SQLiEnv

# Create the environment
env = SQLiEnv()
env = make_vec_env(lambda: env, n_envs=1)

# Instantiate the model
model = SAC(MlpPolicy, env, verbose=1)

# Create the custom callback
callback = CustomLoggingCallback(env, verbose=1)

# Train the model with the custom callback
model.learn(total_timesteps=10000, callback=callback)
model.save("sac_sqli_agent_final")
