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
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        # Observation space: [exploit_char_used, exploit_char_beginning,
        # no_weird_pattern, odd_escape_char_count, query_valid, data_found, flag_found]

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
        start_time = 0
        end_time = 0
        data = {}
        try:
            response_get = self.session.get(f"{self.base_url}{self.current_challenge_id}", timeout=30)
            if "Login" in response_get.text:
                data = {"username_payload": payload, "password_payload": ""}
            elif "Filter" in response_get.text:
                data = {"payload": payload}

            start_time = time.time()
            response = self.session.post(f"{self.base_url}{self.current_challenge_id}", data=data, timeout=30)
            end_time = time.time()

            state, reward, done, truncated = self.analyze_response(response.text, payload, response.status_code,
                                                                   response_get)
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

    def analyze_response(self, response_text, payload, response_status, response_get):
        state, reward = self.set_flags(payload, response_text, response_status, response_get)
        done = state == [1, 1, 1, 1, 1, 1, 1]
        truncated = False
        return state, reward, done, truncated

    def set_flags(self, payload, response_text, response_status, response_get):
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0, 0, 0, 0]
        # Observation space: [exploit_char_used, exploit_char_beginning,
        # no_weird_pattern, odd_escape_char_count, query_valid, data_found, flag_found]

        if self.exploit_char in payload:
            space[0] = 1  # Exploit character used somewhere in the payload

        if payload.startswith(f"{self.exploit_char} "):
            space[1] = 1  # Exploit character used at the beginning

        payload_tokens = payload.split()
        used_tokens = set()  # Track used tokens for diversity reward

        no_consecutive_invalid_tokens = True
        error_penalty = 0  # Used to proportionally increase the penalty depending on the number of errors
        for i in range(len(payload_tokens) - 1):
            if payload_tokens[i] in self.tokens["operators"] and payload_tokens[i + 1] in self.tokens["operators"]:
                no_consecutive_invalid_tokens = False  # Penalize for consecutive operators
                error_penalty += 1
            if payload_tokens[i] in self.tokens["functions"] and payload_tokens[i + 1] in self.tokens["functions"]:
                no_consecutive_invalid_tokens = False  # Penalize for consecutive functions
                error_penalty += 1
            if payload_tokens[i] in self.tokens["tautologies"] and payload_tokens[i + 1] in self.tokens["tautologies"]:
                no_consecutive_invalid_tokens = False  # Penalize for consecutive tautologies
                error_penalty += 1
            if payload_tokens[i] in self.tokens["escape_chars"] and payload_tokens[i + 1] in self.tokens[
                "escape_chars"]:
                no_consecutive_invalid_tokens = False  # Penalize for consecutive escape characters
                error_penalty += 1

            if ((payload_tokens[i] == "(" and payload_tokens[i + 1] == ")")
                    or
                    (payload_tokens[i] == ")" and payload_tokens[i + 1] == "(")):
                no_consecutive_invalid_tokens = False  # Penalize empty parentheses () or )(
                error_penalty += 1

            if ((payload_tokens[i] in self.tokens["functions"] and payload_tokens[i + 1] in self.tokens[
                "tautologies"])
                    or
                    (payload_tokens[i] in self.tokens["tautologies"] and payload_tokens[i + 1] in self.tokens["ints"])
                    or
                    (payload_tokens[i] in self.tokens["ints"] and payload_tokens[i + 1] in self.tokens["tautologies"])
                    or (payload_tokens[i] in self.tokens["escape_chars"]
                        and payload_tokens[i + 1] in self.tokens["tautologies"])
                    or (
                            payload_tokens[i] in self.tokens["escape_chars"] and payload_tokens[i + 1] in self.tokens[
                        "ints"])):
                no_consecutive_invalid_tokens = False  # Penalize invalid sequences of functions, tautologies,
                # and integers
                # 1 1=1
                # LIMIT 1=1
                # 1=1 1
                # ' 1
                # ' 1=1
                error_penalty += 1
            if payload_tokens[i] in self.tokens["ints"] and payload_tokens[i + 1] in self.tokens["ints"]:
                no_consecutive_invalid_tokens = False  # Penalize consecutive integers
                error_penalty += 1

            if (payload_tokens[i] in self.tokens["operators"] and payload_tokens[i + 1] in self.tokens["ints"]
                    or payload_tokens[i] in self.tokens["operators"] and payload_tokens[i + 1] in self.tokens[
                        "tautologies"]):
                no_consecutive_invalid_tokens = False  # Penalize using int or tautologies after operator
                error_penalty += 1

            if payload_tokens[i] in self.tokens["operators"] and payload_tokens[i + 1] in self.tokens["comments"]:
                no_consecutive_invalid_tokens = False  # Penalize if operators are used before comments
                error_penalty += 1

            used_tokens.add(payload_tokens[i])
        used_tokens.add(payload_tokens[-1])

        if no_consecutive_invalid_tokens:
            space[2] = 1

        # Check for even number of occurrences of the found exploit character
        exploit_char_count = payload.count(self.exploit_char)
        space[3] = 1 if exploit_char_count % 2 == 1 else 0

        if response_status == 200:
            space[4] = 1  # Query is valid

        # Successful bypass of the password check
        elif (response_status == 200 and ("wrong" or "fail") not in response_text.lower()
              and len(response_text) != len(response_get.text)):
            space[5] = 1  # Data found

        if re.search(flag_pattern, response_text):
            space[5] = 1  # Data found
            space[6] = 1  # Flag found

        reward = self.set_reward(space, payload, used_tokens, error_penalty)
        return space, reward

    def set_reward(self, space, payload, used_tokens, error_penalty):
        reward = 0

        # Observation space: [exploit_char_used, exploit_char_beginning,
        # no_weird_pattern, odd_escape_char_count, query_valid, data_found, flag_found]

        if space[0] == 0:
            reward -= 1000  # Penalty for not using the exploit character (very bad)

        if space[1] == 0:
            reward -= 75  # Penalty for not using the exploit character at the beginning
            # (should be forced due to code logic)

        if space[2] == 0:
            reward -= 200 + ((0.25 * error_penalty) * 10)  # Penalty for using consecutive invalid tokens

        # Penalty for using an odd number of occurrences of the exploit character
        if space[3] == 0:
            reward -= 200  # Adjust the penalty value as needed

        if space[4] == 0:
            if not self.exploit_char_found:
                reward -= 20  # Penalty for not finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload.split(' ')[0]  # Set the first token as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 60  # Penalty for crashing the server if we already found the exploit character
                # (bad but ok because of the exploration)

        if space[5] == 0:
            reward -= 40  # Penalty for not finding any data (bad)

        if space[6] == 1 and space != [1, 1, 1, 1, 1, 1, 1]:
            reward -= 5  # Small penalty for bypassing password check

        # Penalty for overly simplistic payload
        if len(payload.split()) <= 4:  # if payload is too short
            reward -= 500

        # Penalty for lack of diversity in token usage
        if space == [1, 1, 1, 1, 0, 0, 0] or space == [1, 1, 1, 1, 1, 0, 0]:
            reward -= (20 - len(used_tokens)) * 4
            # Increase penalty for lower diversity when syntax correct but no data or nearly correct

        if space == [1, 1, 1, 1, 1, 1, 1]:
            reward = -1  # High reward for finding the flag

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
