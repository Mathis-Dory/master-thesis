import re
import time

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

# Define tokens with more specific SQL elements for better grammar adherence
tokens = {
    "escape_chars": ["'", '"', ""],
    "comments": ["--", "#", "//", "/*", "*/"],
    "functions": ["OFFSET", "LIMIT"],
    "special_chars": [")", "(", " "],
    "tautologies": ["1=1"],
    "ints": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "operators": ["AND", "OR"],
}


class SQLiEnv(gym.Env):
    def __init__(self):
        self.base_url = "http://localhost:5959/challenge/"
        self.current_challenge_id = 1
        self.max_attempts = 100000  # Maximum attempts per challenge
        self.attempts = 0  # Attempt counter
        self.tokens = tokens
        self.last_payload = ""  # Store the last payload
        self.exploit_char_found = False  # Flag for finding exploit character
        self.exploit_char = ""  # Store the found exploit character
        # Flatten the token list for easy indexing
        self.flat_tokens = [token for category in self.tokens.values() for token in category]
        self.token_count = len(self.flat_tokens)

        # Shape 22, first value is the payload length, the rest are 20 tokens
        self.action_space = spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        # Observation space: [exploit_char_used, query_valid, comments_at_the_end, data_found, flag]

    def step(self, action):
        if self.exploit_char_found:
            payload_length = int(action[0] * 20) + 1  # Scale to get a value between 1 and 20
            payload_tokens = [self.flat_tokens[int(a * (self.token_count - 1))] for a in action[1:payload_length + 1]]
        else:
            payload_length = 1
            payload_tokens = [self.flat_tokens[int(a * (len(self.tokens["escape_chars"]) - 1))] for a in
                              action[1:payload_length + 1]]

        # Ensure the payload always starts with the exploit character and ends with a comment
        if not self.exploit_char_found:
            escape_char = self.flat_tokens[int(action[1] * (len(self.tokens["escape_chars"]) - 1))]
            payload_tokens[0] = escape_char
        else:
            payload_tokens[0] = self.exploit_char

        if len(payload_tokens) > 1:
            payload_tokens[-1] = self.flat_tokens[int(action[payload_length] * (len(self.tokens["comments"]) - 1))]

        payload = ' '.join(payload_tokens)
        self.last_payload = payload

        response = requests.get(f"{self.base_url}{self.current_challenge_id}", timeout=10)
        if "Login" in response.text:
            data = {"username_payload": payload, "password_payload": ""}
        elif "Filter" in response.text:
            data = {"payload": payload}

        start_time = time.time()
        response = requests.post(f"{self.base_url}{self.current_challenge_id}", data=data, timeout=10)
        end_time = time.time()

        state, reward, done = self.analyze_response(response.text, payload, response.status_code)
        self.attempts += 1
        truncated = self.attempts >= self.max_attempts

        if done or truncated:
            self.current_challenge_id += 1
            self.attempts = 0
            self.exploit_char_found = False
            self.exploit_char = ""
            if self.current_challenge_id > 10:
                self.current_challenge_id = 1
                requests.get("http://localhost:5959/reset")

        info = {"response_time": end_time - start_time}

        return np.array(state), reward, done, truncated, info

    def reset(self, **kwargs):
        self.attempts = 0
        self.exploit_char_found = False
        self.exploit_char = ""
        if self.current_challenge_id == 1:
            requests.get("http://localhost:5959/reset")
        return np.zeros(self.observation_space.shape), {}

    def analyze_response(self, response_text, payload, response_status):
        state, reward = self.set_flags(payload, response_text, response_status)
        done = state == [1, 1, 1, 1, 1]
        return state, reward, done

    def set_flags(self, payload, response_text, response_status):
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0, 0]

        if response_status == 200:
            space[1] = 1

        if re.search(flag_pattern, response_text):
            space[4] = 1
            space[3] = 1
        elif "flag_" in response_text:
            space[3] = 1

        # Check the payload ends with a comment and only one comment is used in the payload
        comments_in_payload = [comment for comment in self.tokens["comments"] if comment in payload]
        if any([payload.endswith(comment) for comment in self.tokens["comments"]]) and len(comments_in_payload) == 1:
            space[2] = 1

        if payload.startswith(self.exploit_char + ' ') and self.exploit_char_found:
            space[0] = 1

        reward = self.set_reward(space, payload)
        return space, reward

    def set_reward(self, space, payload):
        reward = 0
        if space[0] == 0:
            reward -= 100  # Penalty for not using the exploit character at the beginning
        if space[1] == 0:
            if not self.exploit_char_found:
                reward -= 10  # Reward for finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload.split(' ')[0]  # Set the first token as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 50  # Penalty for crashing the server if we already found the exploit character

        if space[2] == 0:
            reward -= 35  # Reward for using comments at the end and only once
        if space[3] == 0:
            reward -= 15  # Reward for dumping data
        if space == [1, 1, 1, 1, 1]:
            reward = -1  # High reward for finding the flag

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
