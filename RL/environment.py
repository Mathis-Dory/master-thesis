import re
import time

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

tokens = {
    "escape_chars": ["'", '"', '`'],
    "comments": ["--", "#", "//", "/*", "*/"],
    "functions": ["OFFSET", "LIMIT"],
    "spacial_chars": [")", "(", ",", " "],
    "ints": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "operators": ["AND", "OR", "NOT", "LIKE"],
    "math_operators": ["+", "-", "*", "/", "%", "="],
}


class SQLiEnv(gym.Env):
    def __init__(self):
        self.base_url = "http://localhost:5959/challenge/"
        self.current_challenge_id = 1
        self.max_attempts = 5000  # Maximum attempts per challenge
        self.attempts = 0  # Attempt counter
        self.tokens = tokens
        self.exploit_char_found = False  # Flag for finding exploit character
        # Flatten the token list for easy indexing
        self.flat_tokens = [token for category in self.tokens.values() for token in category]
        self.token_count = len(self.flat_tokens)

        # Define the action space (40 tokens to choose from) + 1 for the length of the payload
        self.action_space = spaces.Box(low=0, high=1, shape=(41,), dtype=np.float32)

        # Define the observation space (adjust size based on the combined length of static and dynamic vectors)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Example size

    def step(self, action):
        # If the exploit character is found, generate the rest of the payload
        if self.exploit_char_found:
            payload_length = int(action[0] * 39) + 1  # Scale to get a value between 1 and 40
            payload_tokens = [self.flat_tokens[int(a * (self.token_count - 1))] for a in action[1:payload_length + 1]]
        else:
            # generate a payload of length 1 to find the exploit character
            payload_length = 1
            # Force to choose in the escape character category
            payload_tokens = [self.flat_tokens[int(a * (len(self.tokens["escape_chars"]) - 1))] for a in
                              action[1:payload_length + 1]]

        # Generate the payload using the next 'payload_length' actions
        payload = ' '.join(payload_tokens)

        response = requests.get(f"{self.base_url}{self.current_challenge_id}")
        if "Login" in response.text:
            data = {"username_payload": payload, "password_payload": ""}
        elif "Filter" in response.text:
            data = {"payload": payload}

        start_time = time.time()
        response = requests.post(f"{self.base_url}{self.current_challenge_id}", data=data)
        end_time = time.time()

        state, reward, done = self.analyze_response(response.text, payload, response.status_code)

        self.attempts += 1

        if done or self.attempts >= self.max_attempts:
            self.current_challenge_id += 1
            self.attempts = 0
            self.exploit_char_found = False  # Reset exploit character found flag
            self.exploit_char = ""  # Reset exploit character
            if self.current_challenge_id > 10:
                self.current_challenge_id = 1  # Reset to the first challenge
                requests.get("http://localhost:5959/reset")  # Call the reset endpoint
            done = True

        truncated = False  # Reset on a specific condition, if needed
        info = {"response_time": end_time - start_time}

        return np.array(state), reward, done, truncated, info

    def reset(self, **kwargs):
        self.current_challenge_id = 1
        self.attempts = 0
        self.exploit_char_found = False  # Reset exploit character found flag
        self.exploit_char = ""  # Reset exploit character
        requests.get("http://localhost:5959/reset")
        return np.zeros(self.observation_space.shape), {}

    def analyze_response(self, response_text, payload, response_status):
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"

        if re.search(flag_pattern, response_text):
            print(f"Flag found! with payload: {payload}")
            return [1, 1, 1], -1, True  # High reward for finding the flag

        # If the agent found the escape character
        if self.exploit_char_found:
            if self.exploit_char not in payload:
                # Worst penalty if the agent did not use the exploit character when it is available
                print("The agent did not use the exploit character!")
                return [0, 0, 0], -100, False
            if response_status == 200:
                # Agent used the exploit character and the query is valid
                if "flag_" in response_text:
                    # Agent found wrong flag
                    print("The agent found the wrong flag!")
                    return [1, 1, 0], -10, False
                else:
                    # Agent did a valid query but nothing is found
                    print("The agent did a valid query but nothing is found!")
                    return [1, 1, 0], -25, False
            if response_status == 500:
                # The query is still invalid
                print("The agent did invalid query!")
                return [1, 0, 0], -75, False
        else:
            if response_status == 200:
                # If the response status is 200, it means the query is valid without escaping
                print("The agent did not find the exploit character!")
                return [0, 1, 0], -100, False
            if response_status == 500:
                self.exploit_char_found = True
                self.exploit_char = payload
                print(f"Exploit character found: {self.exploit_char}")
                return [1, 0, 0], -25, False

        return [0, 0, 0], -100, False  # Default state define as the worst state


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
