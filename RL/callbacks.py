import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = 0
        self.episodes = 0
        self.success_rates = []
        self.image_dir = "images"
        os.makedirs(self.image_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            env = self.training_env.envs[0].env
            payload = (
                env.last_payload
            )  # Access the last payload generated in the environment

            print(f"Step {self.n_calls}")
            print(f"Current Reward: {self.locals['rewards']}")
            print(f"Last Payload: {payload}")
            print(f"Current Observation: {self.locals['new_obs']}")

        if self.locals["dones"][0]:
            self.episodes += 1
            if (
                self.locals["rewards"][0] > 0
            ):  # Assuming a positive reward indicates success
                self.successes += 1
            success_rate = (
                self.successes / self.episodes if self.episodes > 0 else 0
            )
            self.success_rates.append(success_rate)

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

        # Plot success rate over episodes
        episodes = range(1, self.episodes + 1)
        plt.figure()
        plt.plot(episodes, self.success_rates, label="Success Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.title("Success Rate over Episodes")
        plt.legend()

        # Save the plot to a file
        plt.savefig(os.path.join(self.image_dir, "success_rate.png"))
        plt.close()
