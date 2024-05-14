import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomLoggingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.env = env
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log information at each step
        if self.n_calls % 100 == 0:
            action = self.locals["actions"]
            index = int(action[0] * (len(self.env.payloads) - 1))
            static_vector = self.env.payloads.iloc[index]["static_vector"]
            dynamic_vector = self.env.payloads.iloc[index]["dynamic_vector"]

            print(f"Step {self.n_calls}")
            print(f"Current Reward: {self.locals['rewards']}")
            print(f"Current Action: {action}")
            print(f"Current Observation: {self.locals['new_obs']}")
            print(f"Static Vector: {static_vector}")
            print(f"Dynamic Vector: {dynamic_vector}")

        return True

    def _on_rollout_end(self) -> None:
        # Log information at the end of each rollout
        rollout_rewards = np.sum(self.locals["rewards"])
        rollout_length = len(self.locals["rewards"])
        self.episode_rewards.append(rollout_rewards)
        self.episode_lengths.append(rollout_length)
        print(
            f"Rollout ended. Total Reward: {rollout_rewards}, Length: {rollout_length}"
        )

    def _on_training_end(self) -> None:
        # Log information at the end of training
        print("Training finished.")
        print(f"Average Episode Reward: {np.mean(self.episode_rewards)}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths)}")
