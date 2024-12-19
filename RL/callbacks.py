import os
import time

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, save_path="models"):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.failures = []
        self.payloads = []
        self.observations = []
        self.rewards = []
        self.entropies = []
        self.action_distribution = []
        self.cumulative_reward = []
        self.best_mean_reward = -np.inf
        self.best_model_steps = []
        self.best_model_episodes = []
        self.episode_end_steps = []
        self.current_episode_length = 0
        self.episodes = 0

        # Paths for saving
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.image_dir = os.path.join(self.save_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.save_path, "best_model")
        self.start_time = time.time()

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].env
        payload = getattr(env, "last_payload", "Unknown")
        reward = self.locals["rewards"][0]
        actions = self.locals["actions"]
        new_obs = self.locals["new_obs"]
        done = self.locals.get("dones", [False])[0]

        # Log current step information
        if self.n_calls % 100 == 0:
            print(
                f"Step {self.n_calls}: Reward: {reward}, Payload: {payload}, Observation: {new_obs}"
            )

        self.payloads.append(payload)
        self.observations.append(new_obs)
        self.rewards.append(reward)
        self.current_episode_length += 1

        # Cumulative reward
        self.cumulative_reward.append(
            self.cumulative_reward[-1] + reward
            if self.cumulative_reward
            else reward
        )

        # Action entropy and distribution
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().detach().numpy()
        entropy = -np.sum(actions * np.log(actions + 1e-10), axis=-1).mean()
        self.entropies.append(entropy)
        self.action_distribution.extend(actions.flatten())

        # Handle episode end
        if done:
            self._log_episode()

        return True

    def _log_episode(self):
        # Log metrics for completed episode
        episode_reward = np.sum(self.rewards[-self.current_episode_length :])
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        self.episode_end_steps.append(self.n_calls)
        self.episodes += 1

        # Log success and failure
        success = (
            1 if self.rewards[-1] == -1 else 0
        )  # Adjust success logic as needed
        self.successes.append(success)
        self.failures.append(1 - success)  # Opposite of success

        mean_reward = np.mean(
            self.episode_rewards[-25:]
        )  # mean of 25 last episodes
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.best_model_path)
            self.best_model_steps.append(self.n_calls)
            self.best_model_episodes.append(self.episodes)
            print(
                f"New best model saved at step {self.n_calls}, episode {self.episodes}, with mean reward {mean_reward:.2f}"
            )

        self.current_episode_length = 0

    def _on_training_end(self):
        # Final logging and plotting
        self._finalize_training()

    def _finalize_training(self):
        total_time = time.time() - self.start_time
        print(f"Training finished in {total_time:.2f} seconds.")
        print(f"Total Episodes: {self.episodes}")
        print(f"Successful Episodes: {np.sum(self.successes)}")
        print(f"Failed Episodes: {np.sum(self.failures)}")
        print(f"Average Episode Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths):.2f}")

        # Save payloads and observations
        self._save_payloads_and_observations()

        # Generate plots
        self._generate_plots()

    def _save_payloads_and_observations(self):
        save_file = os.path.join(
            self.save_path, "payloads_and_observations.txt"
        )
        with open(save_file, "w") as f:
            for step, (payload, obs, reward) in enumerate(
                zip(self.payloads, self.observations, self.rewards)
            ):
                f.write(
                    f"Step {step}:\nPayload: {payload}\nObservation: {obs}\nReward: {reward}\n\n"
                )

    def _generate_plots(self):
        episodes = range(1, self.episodes + 1)
        steps = range(len(self.cumulative_reward))

        sns.set(style="whitegrid", palette="muted", font_scale=1.2)

        # Success and failure rate
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=self.successes,
            label="Success Indicator",
            marker="o",
            color="blue",
            linewidth=2,
        )
        sns.lineplot(
            x=episodes,
            y=self.failures,
            label="Failure Indicator",
            marker="x",
            color="red",
            linewidth=2,
        )
        plt.title("Success and Failure Rates Over Episodes", fontsize=16)
        plt.xlabel("Episodes", fontsize=14)
        plt.ylabel("Indicator (0 or 1)", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "success_and_failure.png"))
        plt.close()

        # Episode rewards and success rate combined
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=self.successes,
            label="Success Indicator",
            marker="o",
            color="blue",
            linewidth=2,
        )
        sns.lineplot(
            x=episodes,
            y=self.episode_rewards,
            label="Episode Rewards",
            color="green",
            linewidth=2,
        )
        plt.title("Success Indicator and Episode Rewards", fontsize=16)
        plt.xlabel("Episodes", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "success_and_rewards.png"))
        plt.close()

        # Episode lengths
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=episodes,
            y=self.episode_lengths,
            label="Episode Lengths",
            color="orange",
            linewidth=2,
        )
        plt.title("Episode Lengths Over Episodes", fontsize=16)
        plt.xlabel("Episodes", fontsize=14)
        plt.ylabel("Length (Steps)", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "episode_lengths.png"))
        plt.close()

        # Entropy
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=steps[: len(self.entropies)],
            y=self.entropies,
            label="Entropy",
            color="purple",
            linewidth=2,
        )
        plt.title("Entropy Over Steps", fontsize=16)
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel("Entropy", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "entropy.png"))
        plt.close()

        # Cumulative rewards
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=steps,
            y=self.cumulative_reward,
            label="Cumulative Reward",
            color="red",
            linewidth=2,
        )
        plt.title("Cumulative Reward Over Steps", fontsize=16)
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel("Cumulative Reward", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "cumulative_rewards.png"))
        plt.close()

        # Action distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(
            self.action_distribution,
            bins=50,
            kde=True,
            color="green",
            label="Action Distribution",
            linewidth=0,
        )
        plt.title("Action Distribution", fontsize=16)
        plt.xlabel("Action Values", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "action_distribution.png"))
        plt.close()

        # Reward distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(
            self.episode_rewards,
            bins=30,
            kde=True,
            color="blue",
            label="Reward Distribution",
            linewidth=0,
        )
        plt.title("Reward Distribution Across Episodes", fontsize=16)
        plt.xlabel("Reward", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "reward_distribution.png"))
        plt.close()


callback = CustomLoggingCallback()
