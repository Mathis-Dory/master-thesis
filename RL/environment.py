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
    "comments": ["--", "#", "/*"],
    "functions": ["OFFSET", "LIMIT"],
    "special_chars": [")", "(", " "],
    "tautologies": ["1=1", "1=0"],
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        # Observation space: [exploit_char_used, exploit_char_beginning, comments_used_once, comments_at_the_end,
        # no_multiples_op/func/tautologies_in_row, no_multiples_int_in_row_wth_space, query_valid, data_found,
        # flag_found]

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
        done = state == [1, 1, 1, 1, 1, 1, 1, 1, 1]
        return state, reward, done

    def set_flags(self, payload, response_text, response_status):
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Observation space: [exploit_char_used, exploit_char_beginning, comments_used_once, comments_at_the_end,
        # no_multiples_op/func/tautologies_in_row, no_multiples_int_in_row_wth_space, query_valid, data_found,
        # flag_found]
        if self.exploit_char in payload:
            space[0] = 1  # Exploit character used somewhere in the payload

        if payload.startswith(f"{self.exploit_char} "):
            space[1] = 1  # Exploit character used at the beginning

        # Check only one comment is used in the payload
        comments_in_payload = [comment for comment in self.tokens["comments"] if comment in payload]
        if len(comments_in_payload) == 1:
            space[2] = 1

        # Check if the comment is at the end of the payload
        if any([payload.endswith(comment) for comment in self.tokens["comments"]]):
            space[3] = 1

        # Reward for not using two operators, functions, or tautologies in a row
        no_consecutive_invalid_tokens = True
        payload_tokens = payload.split()

        for i in range(len(payload_tokens) - 1):
            if (payload_tokens[i] in self.tokens["operators"] and payload_tokens[i + 1] in self.tokens["operators"]) or \
                    (payload_tokens[i] in self.tokens["functions"] and payload_tokens[i + 1] in self.tokens["functions"]) or \
                    (payload_tokens[i] in self.tokens["tautologies"] and payload_tokens[i + 1] in self.tokens["tautologies"]):
                no_consecutive_invalid_tokens = False
                break

        if no_consecutive_invalid_tokens:
            space[4] = 1

        # Reward for not using two integers in a row separated by a space
        no_consecutive_integers = True
        for i in range(len(payload_tokens) - 1):
            if payload_tokens[i] in self.tokens["ints"] and payload_tokens[i + 1] in self.tokens["ints"]:
                no_consecutive_integers = False
                break

        if no_consecutive_integers:
            space[5] = 1

        if response_status == 200:
            space[6] = 1  # Query is valid

        elif "flag_" in response_text:
            space[7] = 1  # Data found

        if re.search(flag_pattern, response_text):
            space[7] = 1  # Data found
            space[8] = 1  # Flag found

        reward = self.set_reward(space, payload)
        return space, reward

    def set_reward(self, space, payload):
        reward = 0

        # Observation space: [exploit_char_used, exploit_char_beginning, comments_used_once, comments_at_the_end,
        # no_multiples_op/func/tautologies_in_row, no_multiples_int_in_row_wth_space, query_valid, data_found,
        # flag_found]

        if space[0] == 0:
            reward -= 1000  # Penalty for not using the exploit character (very bad)

        if space[1] == 0:
            reward -= 75  # Penalty for not using the exploit character at the beginning (bad but still ok)

        if space[2] == 0:
            reward -= 150  # Penalty for not using comments once and only once (very bad)

        if space[3] == 0:
            reward -= 100  # Penalty for not using comments at the end (bad but still ok)

        if space[4] == 0:
            reward -= 200  # Penalty for using multiple operators, functions, or tautologies in a row (very bad)

        if space[5] == 0:
            reward -= 175  # Penalty for using multiple integers in a row with a space (very bad)

        if space[6] == 0:
            if not self.exploit_char_found:
                reward -= 20  # Reward for finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload.split(' ')[0]  # Set the first token as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 50  # Penalty for crashing the server if we already found the exploit character (bad but ok
                # because of the exploration)

        if space[7] == 0:
            reward -= 40  # Penalty for not finding any data (bad)

        if space[8] == 1 and space != [1, 1, 1, 1, 1, 1, 1, 1, 1]:
            reward = -5  # Small penalty for finding the flag but not in optimal conditions

        if space == [1, 1, 1, 1, 1, 1, 1, 1, 1]:
            reward = -1  # High reward for finding the flag

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
