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
        self.success_rates_per_observation = []
        self.payloads = []
        self.observations = []
        self.rewards = []
        self.image_dir = "images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.start_time = time.time()  # Record the start time
        self.last_save_time = self.start_time  # Record the last save time
        self.observation_labels = [
            "exploit_char_used",
            "exploit_char_beginning",
            "no_weird_pattern",
            "odd_escape_char_count",
            "query_valid",
            "data_found",
            "flag_found",
        ]

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

            self.payloads.append(payload)
            self.observations.append(self.locals["new_obs"])
            self.rewards.append(self.locals["rewards"])

        if (
            self.locals.get("dones", [False])[0]
            or self.locals.get("truncated", [False])[0]
        ):
            self.episodes += 1
            if (
                self.locals["rewards"][0] > 0
            ):  # Assuming a positive reward indicates success
                self.successes += 1
            success_rate = (
                self.successes / self.episodes if self.episodes > 0 else 0
            )
            self.success_rates.append(success_rate)
            self.success_rates_per_observation.append(
                self.locals["new_obs"][0]
            )  # Record success rate per observation

        # Every 100,000 steps, measure the time taken and save the model
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

    def _on_rollout_end(self) -> None:
        rollout_rewards = np.sum(self.locals["rewards"])
        rollout_length = len(self.locals["rewards"])
        self.episode_rewards.append(rollout_rewards)
        self.episode_lengths.append(rollout_length)

    def _on_training_end(self) -> None:
        total_training_time = (
            time.time() - self.start_time
        )  # Calculate total training time
        print("Training finished.")
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        print(f"Average Episode Reward: {np.mean(self.episode_rewards)}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths)}")

        # Plot success rate over episodes
        episodes = range(1, self.episodes + 1)
        plt.figure(figsize=(12, 8))  # Increase the figure size
        plt.plot(episodes, self.success_rates, label="Success Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.title("Success Rate over Episodes")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "success_rate.png"))
        plt.close()

        # Plot success rates for every observation
        plt.figure(figsize=(12, 8))  # Increase the figure size
        for i, label in enumerate(self.observation_labels):
            plt.plot(
                range(len(self.success_rates_per_observation)),
                [obs[i] for obs in self.success_rates_per_observation],
                label=label,
            )
        plt.xlabel("Observation Steps")
        plt.ylabel("Success Rate")
        plt.title("Success Rate per Observation")
        plt.legend(fontsize="small")  # Reduce the font size of the legend

        # Ensure there's enough data to plot
        if len(self.success_rates_per_observation) > 0:
            # Set the x-axis scale to represent every 50,000 steps
            plt.xticks(
                ticks=np.arange(
                    0, len(self.success_rates_per_observation), 50000
                ),
                labels=np.arange(
                    0, len(self.success_rates_per_observation), 50000
                ),
            )

        # Save the plot to a file
        plt.savefig(
            os.path.join(self.image_dir, "success_rate_per_observation.png")
        )
        plt.close()

        # Plot the last payloads and observations
        if self.payloads:
            plt.figure(figsize=(12, 8))
            plt.plot(range(len(self.payloads)), self.rewards, label="Rewards")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.title("Rewards over Time")
            plt.legend()
            plt.savefig(os.path.join(self.image_dir, "rewards_over_time.png"))
            plt.close()

            for i, label in enumerate(self.observation_labels):
                plt.figure(figsize=(12, 8))
                plt.plot(
                    range(len(self.observations)),
                    [obs[0][i] for obs in self.observations],
                    label=label,
                )
                plt.xlabel("Steps")
                plt.ylabel(label)
                plt.title(f"{label} over Time")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        self.image_dir,
                        f"{label.lower().replace(' ', '_')}_over_time.png",
                    )
                )
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
