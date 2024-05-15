import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            action = self.locals["actions"]
            env = self.training_env.envs[0].env

            action = action[0] if isinstance(action, np.ndarray) else action
            payload_index = int(action[0] * (len(env.flat_tokens) - 1))
            static_vector = env.flat_tokens[payload_index]

            print(f"Step {self.n_calls}")
            print(f"Current Reward: {self.locals['rewards']}")
            print(f"Current Action: {action}")
            print(f"Current Observation: {self.locals['new_obs']}")
            print(f"Static Vector: {static_vector}")

        return True

    def _on_rollout_end(self) -> None:
        rollout_rewards = np.sum(self.locals["rewards"])
        rollout_length = len(self.locals["rewards"])
        self.episode_rewards.append(rollout_rewards)
        self.episode_lengths.append(rollout_length)

    def _on_training_end(self) -> None:
        print("Training finished.")
        print(f"Average Episode Reward: {np.mean(self.episode_rewards)}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths)}")
