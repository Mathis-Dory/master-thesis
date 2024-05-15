import re
import time

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

tokens = {
    "escape_chars": ["'", '"', " "],
    "comments": ["--", "#", "//", "/*", "*/"],
    "functions": ["OFFSET", "LIMIT"],
    "spacial_chars": [")", "(", ",", " "],
    "tautologies": ["1=1", "1=0"],
    "ints": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "operators": ["AND", "OR", "LIKE"],
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
        # Flatten the token list for easy indexing
        self.flat_tokens = [token for category in self.tokens.values() for token in category]
        self.token_count = len(self.flat_tokens)

        # Define the action space (40 tokens to choose from) + 1 for the length of the payload
        self.action_space = spaces.Box(low=0, high=1, shape=(21,), dtype=np.float32)

        # Define the observation space (adjust size based on the combined length of static and dynamic vectors)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        # Observation space: [exploit_char_beginning, query_valid, comments_at_the_end,
        # data_found, flag]

    def step(self, action):
        # If the exploit character is found, generate the rest of the payload
        if self.exploit_char_found:
            payload_length = int(action[0] * 19) + 1  # Scale to get a value between 1 and 40
            payload_tokens = [self.flat_tokens[int(a * (self.token_count - 1))] for a in action[1:payload_length + 1]]
        else:
            # generate a payload of length 1 to find the exploit character
            payload_length = 1
            # Force to choose in the escape character category
            payload_tokens = [self.flat_tokens[int(a * (len(self.tokens["escape_chars"]) - 1))] for a in
                              action[1:payload_length + 1]]

        # Generate the payload using the next 'payload_length' actions
        payload = ' '.join(payload_tokens)
        self.last_payload = payload

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

        # If it fails to find the flag after the maximum attempts, the episode is done
        truncated = self.attempts >= self.max_attempts

        if done or truncated:
            self.current_challenge_id += 1
            self.attempts = 0
            self.exploit_char_found = False  # Reset exploit character found flag
            self.exploit_char = ""  # Reset exploit character
            if self.current_challenge_id > 10:
                self.current_challenge_id = 1  # Reset to the first challenge
                requests.get("http://localhost:5959/reset")  # Call the reset endpoint

        info = {"response_time": end_time - start_time}

        return np.array(state), reward, done, truncated, info

    def reset(self, **kwargs):
        self.attempts = 0
        self.exploit_char_found = False  # Reset exploit character found flag
        self.exploit_char = ""  # Reset exploit character
        # Only call the reset endpoint if we are at the first challenge
        if self.current_challenge_id == 1:
            requests.get("http://localhost:5959/reset")
        return np.zeros(self.observation_space.shape), {}

    def analyze_response(self, response_text, payload, response_status):
        state, reward = self.set_flags(payload, response_text, response_status)
        done = False
        if state == [1, 1, 1, 1, 1]:
            done = True
        return state, reward, done

    def set_flags(self, payload, response_text, response_status):
        """
        This function set the observation space
        by putting flags on or off depending on the conditions
        Observation space: [exploit_char_used exploit_char_beginning, query_valid, comments_at_the_end,
        data_found, flag]
        """
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0, 0]
        if response_status == 200:
            # if the query is valid
            space[1] = 1

        if re.search(flag_pattern, response_text):
            # Flag found
            space[4] = 1
            # Also set the data found flag
            space[3] = 1
        else:
            # another flag found
            if "flag_" in response_text:
                space[3] = 1

        if any([payload.endswith(comment) for comment in self.tokens["comments"]]):
            # Comments at the end
            space[2] = 1

        if payload.startswith(self.exploit_char + ' ') and self.exploit_char_found:
            # Exploit character at the beginning
            space[0] = 1

        reward = self.set_reward(space, payload)
        return space, reward

    def set_reward(self, space, payload):
        reward = 0
        if space[0] == 0:
            # Penalty for not using the exploit character at the beginning
            reward += -100
        if space[1] == 0:
            if not self.exploit_char_found:
                # Small penalty for invalid query if the exploit character was not found before
                reward += -10
                self.exploit_char_found = True
                self.exploit_char = payload
                print(f"Exploit character found: {self.exploit_char}")
            else:
                # penalty for invalid query if the exploit character was found before
                reward += -50

        if space[2] == 0:
            # Penalty for not having comments at the end
            reward += -35
        if space[3] == 0:
            # Penalty for not finding the data
            reward += -15
        if space[4] == 0:
            # Penalty for not finding the flag
            reward += -10
        if space == [1, 1, 1, 1, 1]:
            # High reward for finding the flag
            reward = -1

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
