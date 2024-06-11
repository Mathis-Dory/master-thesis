import os
import time
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import torch


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, save_path="models"):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.episodes = 0
        self.payloads = []
        self.observations = []
        self.rewards = []
        self.current_episode_length = 0
        self.best_mean_reward = -np.inf
        self.best_model_path = os.path.join(save_path, "best_model")
        self.image_dir = "images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.start_time = time.time()  # Record the start time
        self.last_save_time = self.start_time  # Record the last save time
        self.episode_end_steps = []  # Track steps where episodes end
        self.cumulative_reward = []
        self.entropies = []  # Track entropy of actions
        self.action_distribution = []
        self.best_model_steps = []  # Track steps when best model was saved
        self.best_model_episodes = []  # Track episodes when best model was saved

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].env
        payload = env.last_payload
        if self.n_calls % 100 == 0:
            print(f"Step {self.n_calls}")
            print(f"Current Reward: {self.locals['rewards']}")
            print(f"Last Payload: {payload}")
            print(f"Current Observation: {self.locals['new_obs']}")

        self.payloads.append(payload)
        self.observations.append(self.locals["new_obs"])
        self.rewards.append(self.locals["rewards"][0])
        self.current_episode_length += 1

        # Track cumulative reward
        if len(self.cumulative_reward) > 0:
            self.cumulative_reward.append(self.cumulative_reward[-1] + self.locals["rewards"][0])
        else:
            self.cumulative_reward.append(self.locals["rewards"][0])

        # Track entropy of the actions
        actions = self.locals["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().detach().numpy()
        entropy = -np.sum(actions * np.log(actions + 1e-10), axis=-1).mean()
        self.entropies.append(entropy)

        # Track action distribution
        self.action_distribution.extend(actions)

        if (
                self.locals.get("dones", [False])[0]
                or self.locals.get("truncated", [False])[0]
        ):
            self.episodes += 1
            episode_reward = np.mean(self.rewards[-self.current_episode_length:])
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_end_steps.append(self.n_calls - 1)  # Record the step number

            # Update the success rate if the episode is successful
            success = 1 if self.locals["rewards"][0] == -1 else 0
            self.successes.append(success)

            self.current_episode_length = (
                0  # Reset current episode length for the next episode
            )

            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                self.best_model_steps.append(self.n_calls)
                self.best_model_episodes.append(self.episodes)
                print(f"New best model saved with mean reward: {self.best_mean_reward}")

        return True

    def _on_training_end(self) -> None:
        # Ensure the last episode end step is captured
        if self.current_episode_length > 0:
            self.episodes += 1
            episode_reward = np.mean(self.rewards[-self.current_episode_length:])
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_end_steps.append(self.n_calls - 1)
            success = 1 if self.rewards[-1] == -1 else 0
            self.successes.append(success)
            self.current_episode_length = 0

        total_training_time = (
                time.time() - self.start_time
        )  # Calculate total training time
        print("Training finished.")
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        print(f"Total Episodes: {self.episodes}")

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

        # Debugging: Print all collected data
        print("Episode Rewards:", self.episode_rewards)
        print("Episode Lengths:", self.episode_lengths)
        print("Success Rates:", self.successes)

        # Print when the best model was saved
        print("Best model was saved at the following steps and episodes:")
        for step, episode in zip(self.best_model_steps, self.best_model_episodes):
            print(f"Step: {step}, Episode: {episode}")

        # Plot success rate over episodes
        episodes = range(1, self.episodes + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, self.successes, label="Success Indicator", marker='o')
        plt.xlabel("Episodes")
        plt.ylabel("Success Indicator")
        plt.title("Success Indicator over Episodes")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "success_rate.png"))
        plt.close()

        # Plot average episode rewards over episodes
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, self.episode_rewards, label="Episode Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Average Episode Rewards over Episodes")
        plt.legend()
        plt.ylim([min(self.episode_rewards), max(self.episode_rewards) * 1.1])  # Adjust the scale dynamically
        plt.savefig(os.path.join(self.image_dir, "average_episode_rewards.png"))
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

        # Plot cumulative reward over time
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(self.cumulative_reward)), self.cumulative_reward, label="Cumulative Reward")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward over Steps")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "cumulative_reward.png"))
        plt.close()

        # Plot action entropy over steps
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(self.entropies)), self.entropies, label="Action Entropy")
        plt.xlabel("Steps")
        plt.ylabel("Entropy")
        plt.title("Action Entropy over Steps")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "action_entropy.png"))
        plt.close()

        # Plot action distribution
        actions = np.array(self.action_distribution)
        plt.figure(figsize=(12, 8))
        plt.hist(actions.flatten(), bins=50, label="Action Distribution")
        plt.xlabel("Action Values")
        plt.ylabel("Frequency")
        plt.title("Action Distribution")
        plt.legend()
        plt.savefig(os.path.join(self.image_dir, "action_distribution.png"))
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


# Use the custom callback
callback = CustomLoggingCallback()
