import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import subprocess
import requests
import time

from environment import SQLiEnv

# Load the best saved model
best_model_path = "final_model/model.zip"
model = SAC.load(best_model_path)


# Function to test the model against the server
def test_model(model, env, total_challenges=25):
    start_time = time.time()
    successes = []
    episode_lengths = []
    step_counts = []  # Track the number of steps for each challenge

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    for challenge_id in range(1, total_challenges + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_length = 0
        success = 0
        phase = 1

        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            action = action + action_noise()
            obs, reward, done, truncated, info = env.step(action)
            episode_length += 1

            # Transition between phases based on the observation space flags
            if phase == 1 and obs[0] == 1:
                phase = 2
            elif phase == 2 and obs[1] == 1:
                phase = 3

            if done:
                if reward == -1:
                    success = 1
                break

        successes.append(success)
        episode_lengths.append(episode_length)
        step_counts.append(episode_length)  # Number of steps is the episode length

    total_time = time.time() - start_time
    print(f"Total time taken to test with the SAC agent: {total_time:.2f} seconds")
    return successes, episode_lengths, step_counts


# Function to check if the server is up
def check_server(url, retries=5, delay=2):
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is up.")
                return True
        except requests.ConnectionError:
            pass
        print(f"Server not available, retrying... ({i + 1}/{retries})")
        time.sleep(delay)
    print("Server is not available after retries.")
    return False


# Function to use SQLMap to exploit the server
def run_sqlmap(url):
    sqlmap_command = [
        "python", "C:\\sqlmap\\sqlmap.py",  # Adjust the path to your sqlmap.py location
        "-u", url,
        "--data", "username_payload=test&password_payload=test",
        "--batch",
        "--level=5",
        "--risk=3",
        "--dump",
        "-T", "auth_bypass",
        "-p", "username_payload",
        "--technique=B",
        "--ignore-code", "500",
        "--flush-session"
    ]

    process = subprocess.Popen(sqlmap_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output = ""
    payload_count = 0
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            output += line
            payload_count += 1

    return output, payload_count


# Test SQLMap against all challenges
def test_sqlmap(total_challenges=25):
    start_time = time.time()
    all_sqlmap_outputs = []
    payload_counts = []
    successes = []

    for challenge_id in range(1, total_challenges + 1):
        response = requests.get("http://localhost:5959/reset")
        if response.status_code == 200:
            print("Environment reset successfully.")
        else:
            print("Failed to reset the environment.")

        print(f"Testing SQLMap on challenge {challenge_id}/{total_challenges}")
        url = f"http://localhost:5959/challenge/1"
        # Check if server is up
        if not check_server(url):
            print(f"Skipping challenge {challenge_id} due to server issues.")
            continue

        sqlmap_output, payload_count = run_sqlmap(url)
        print(f"SQLMap output for challenge {challenge_id}: {sqlmap_output}")
        all_sqlmap_outputs.append(sqlmap_output)
        payload_counts.append(payload_count)

        # Determine success based on presence of "flag_challenge"
        if "flag_challenge" in sqlmap_output:
            successes.append(1)
        else:
            successes.append(0)

        # Reset the environment
        response = requests.get("http://localhost:5959/reset")
        if response.status_code == 200:
            print("Environment reset successfully.")
        else:
            print("Failed to reset the environment.")

        # Add a delay to ensure the server has time to reset
        time.sleep(2)

    total_time = time.time() - start_time
    print(f"Total time taken to test with SQLMap: {total_time:.2f} seconds")

    return all_sqlmap_outputs, payload_counts, successes


# Function to plot the results
def plot_results(successes, payload_or_steps_counts, title_prefix, ylabel, image_name):
    episodes = range(1, len(successes) + 1)

    os.makedirs("evaluate_results", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.plot(episodes, successes, label="Success Indicator", marker='o')
    plt.xlabel("Challenges")
    plt.ylabel("Success Indicator")
    plt.title(f"{title_prefix} Success Indicator over Challenges")
    plt.legend()
    plt.savefig(os.path.join("evaluate_results", f"{image_name}_success_rate.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(episodes, payload_or_steps_counts, label=ylabel)
    plt.xlabel("Challenges")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} {ylabel} over Challenges")
    plt.legend()
    plt.savefig(os.path.join("evaluate_results", f"{image_name}_{ylabel.replace(' ', '_').lower()}.png"))
    plt.close()

    print(f"{title_prefix} testing completed. Results saved.")


# Function to create the environment
def create_env():
    return Monitor(SQLiEnv(), "logs", allow_early_resets=True)


# Main function to run the desired test
def main(test_type="model", image_name="result"):
    env = create_env()

    if test_type == "model":
        successes, episode_lengths, step_counts = test_model(model, env)
        plot_results(successes, step_counts, "Model", "Steps", image_name)
    elif test_type == "sqlmap":
        all_sqlmap_outputs, payload_counts, successes = test_sqlmap()
        plot_results(successes, payload_counts, "SQLMap", "Payloads", image_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Model or SQLMap against SQLi challenges.")
    parser.add_argument("--test_type", type=str, choices=["model", "sqlmap"], required=True,
                        help="Specify which test to run: 'model' or 'sqlmap'.")
    parser.add_argument("--image_name", type=str, default="result", help="Specify the custom name for the images.")

    args = parser.parse_args()
    main(test_type=args.test_type, image_name=args.image_name)
