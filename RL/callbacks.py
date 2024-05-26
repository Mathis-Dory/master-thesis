import os
import time

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, save_path="models"):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = 0
        self.episodes = 0
        self.success_rates = []
        self.payloads = []
        self.observations = []
        self.rewards = []
        self.step_rewards = []
        self.current_episode_length = 0
        self.image_dir = "images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.start_time = time.time()  # Record the start time
        self.last_save_time = self.start_time  # Record the last save time

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            env = self.training_env.envs[0].env
            payload = env.last_payload

            print(f"Step {self.n_calls}")
            print(f"Current Reward: {self.locals['rewards']}")
            print(f"Last Payload: {payload}")
            print(f"Current Observation: {self.locals['new_obs']}")

            self.payloads.append(payload)
            self.observations.append(self.locals["new_obs"])
            self.rewards.append(self.locals["rewards"])

        self.step_rewards.append(self.locals["rewards"][0])
        self.current_episode_length += 1

        if (
            self.locals.get("dones", [False])[0]
            or self.locals.get("truncated", [False])[0]
        ):
            self.episodes += 1
            episode_reward = np.sum(
                self.step_rewards[-self.current_episode_length :]
            )
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            if (
                self.locals["rewards"][0] > 0
            ):  # Assuming a positive reward indicates success
                self.successes += 1
            success_rate = (
                self.successes / self.episodes if self.episodes > 0 else 0
            )
            self.success_rates.append(success_rate)
            self.current_episode_length = (
                0  # Reset current episode length for the next episode
            )

        if self.n_calls % 100000 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.last_save_time
            self.last_save_time = current_time
            print(
                f"Time taken for the last 100,000 steps: {elapsed_time:.2f} seconds"
            )
            self.model.save(
                os.path.join(self.save_path, f"model_step_{self.n_calls}")
            )

        return True

    def _on_training_end(self) -> None:
        total_training_time = (
            time.time() - self.start_time
        )  # Calculate total training time
        print("Training finished.")
        print(f"Total Training Time: {total_training_time:.2f} seconds")

        if len(self.episode_rewards) > 0:
            avg_episode_reward = np.mean(self.episode_rewards)
        else:
            avg_episode_reward = 0.0

        if len(self.episode_lengths) > 0:
            avg_episode_length = np.mean(self.episode_lengths)
        else:
            avg_episode_length = 0.0

        print(f"Average Episode Reward: {avg_episode_reward}")
        print(f"Average Episode Length: {avg_episode_length}")

        # Plot success rate over episodes
        episodes = range(1, self.episodes + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, self.success_rates, label="Success Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.title("Success Rate over Episodes")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "success_rate.png"))
        plt.close()

        # Plot episode rewards over time
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, self.episode_rewards, label="Episode Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Episode Rewards over Episodes")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "episode_rewards.png"))
        plt.close()

        # Plot episode lengths over time
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, self.episode_lengths, label="Episode Lengths")
        plt.xlabel("Episodes")
        plt.ylabel("Length")
        plt.title("Episode Lengths over Episodes")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "episode_lengths.png"))
        plt.close()

        # Plot rewards over time
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(self.rewards)), self.rewards, label="Rewards")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Rewards over Time")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "rewards_over_time.png"))
        plt.close()

        # Save payloads and observations to a text file
        with open(
            os.path.join(self.save_path, "payloads_and_observations.txt"), "w"
        ) as f:
            for step, (payload, obs, reward) in enumerate(
                zip(self.payloads, self.observations, self.rewards)
            ):
                f.write(f"Step {step}:\n")
                f.write(f"Payload: {payload}\n")
                f.write(f"Observation: {obs}\n")
                f.write(f"Reward: {reward}\n")
                f.write("\n")

        # Save rewards to a CSV file
        np.savetxt(
            os.path.join(self.save_path, "rewards.csv"),
            self.rewards,
            delimiter=",",
        )
