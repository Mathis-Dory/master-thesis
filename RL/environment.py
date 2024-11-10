import logging
import os
import re

import gymnasium as gym
import numpy as np
import requests
from dotenv import load_dotenv
from gymnasium import spaces
from nltk.parse.generate import generate
from numpy._typing import ArrayLike
from stable_baselines3.common.monitor import Monitor

from cfg import cfg_phase1, cfg_phase2, cfg_phase3
from utils import (
    setup_session, extract_tokens_from_grammar, get_valid_insertion_points,
)

# Load environment variables from .env file
load_dotenv()

NUM_CHALLENGES = int(os.getenv("NUM_CHALLENGES", 1))  # Default to 1 if not set

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SQLiEnv(gym.Env):
    """
    Custom environment for SQL injection attacks using reinforcement learning.
    """

    def __init__(self):
        super().__init__()
        self.base_url = "http://localhost:5959/challenge/"
        self.session = setup_session()
        self._initialize_flags()

        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )
        # Observation space: [query_valid, data_found, flag_found]

        # Pre-generate sequences
        self.phase1_sequences = list(generate(cfg_phase1, n=10))
        self.phase2_sequences = list(generate(cfg_phase2, n=10))
        self.phase3_sequences = list(generate(cfg_phase3, n=10))

    def _initialize_flags(self):
        self.current_challenge_id = 1
        self.last_payload = ""
        self.exploit_char_found = False
        self.found_parenthesis_structure = False
        self.exploit_char = ""
        self.parentheses_structure = ""
        self.step_count = 0
        self.max_steps_per_episode = 5000

    def _generate_payload(self, phase: int, action: ArrayLike) -> str:
        """
        Generate the SQL injection payload based on the current phase and action.

        :param phase: The current phase of the environment.
        :param action: The action taken by the agent.
        :return: The generated SQL injection payload.
        """
        action = np.clip(action, 0, 1)  # Clip the action values to [0, 1]
        grammar = {
            1: self.phase1_sequences,
            2: self.phase2_sequences,
            3: self.phase3_sequences,
        }[phase]
        action_index = int(action[0] * (len(grammar) - 1))

        payload = self._build_payload(phase, grammar[action_index], action)
        logging.debug(f"Generated payload for phase {phase}: {payload}")
        return payload

    def _build_payload(self, phase: int, selected_clause, action):
        # Simplifies payload construction based on phase and chosen grammar clause
        if phase == 1:
            return " ".join(selected_clause)
        elif phase == 2:
            return f"{self.exploit_char} {' '.join(selected_clause)}"
        else:
            clause = " ".join(selected_clause)
            return self._build_complex_payload(clause, action)

    def _build_complex_payload(self, clause: str, action: ArrayLike) -> str:
        """
        Builds a complex SQL injection payload by incorporating the identified exploit character,
        valid parentheses structure, and a comment to bypass filters.
        """
        match = re.search(r"(\)+)\s*(--|#|/\*)", self.parentheses_structure)
        if match:
            parentheses = match.group(1)
            comment = match.group(2).strip() + " "  # Ensure a space after the comment character
            num_parentheses = len(parentheses)
            parts = clause.split()

            # Identify CFG tokens by parsing the grammar
            cfg_tokens = extract_tokens_from_grammar(cfg_phase3)

            # Determine valid insertion points for parentheses, avoiding CFG tokens
            valid_insertion_points = get_valid_insertion_points(parts, cfg_tokens)

            if not valid_insertion_points:
                valid_insertion_points = [0]

            group_parentheses = action[1] > 0.5
            if group_parentheses:
                split_index = int(action[2] * len(valid_insertion_points))  # Choose a split index based on action
                if split_index < len(valid_insertion_points):
                    parts.insert(valid_insertion_points[split_index],
                                 parentheses)  # Insert all parentheses at the chosen index
            else:
                # Distribute parentheses based on action indices
                split_indices = sorted(
                    [int(action[i + 2] * len(valid_insertion_points)) for i in range(num_parentheses)]
                )
                for i, index in enumerate(split_indices):
                    if index < len(valid_insertion_points):
                        parts.insert(valid_insertion_points[index] + i, ')')

            clause_with_parentheses = ' '.join(parts)
            payload = f"{self.exploit_char} {clause_with_parentheses}".strip()
        else:
            # No valid parentheses structure found, generate payload without escape character before comment
            payload = f"{self.exploit_char} {clause}".strip()
            # remove the exploit char in the parentheses structure
            comment = self.parentheses_structure.replace(self.exploit_char, "")

        payload = f"{payload} {comment}"

        return payload



    def step(self, action):
        self.step_count += 1
        # Ensure the agent goes through phases sequentially
        phase = (
            1 if not self.exploit_char_found else (2 if not self.found_parenthesis_structure else 3)
        )
        payload = self.generate_payload(phase, action)
        response = self._send_request(payload)
        self.last_payload = payload
        return self._process_response(response, payload)

    def _send_request(self, payload):
        try:
            response_get = self.session.get(f"{self.base_url}{self.current_challenge_id}", timeout=30)
            data = {"username_payload": payload, "password_payload": ""} if "Login" in response_get.text else {
                "payload": payload}
            response = self.session.post(f"{self.base_url}{self.current_challenge_id}", data=data, timeout=30)
            return response, response_get
        except requests.exceptions.RequestException as e:
            logging.error(f"Connection failed: {e}")
            return None, None

    def _process_response(self, response, payload):
        if response:
            state, reward, done = self._analyze_response(response[0], payload, response[1])
            truncated = self.step_count >= self.max_steps_per_episode
            return np.array(state), reward, done, truncated, {"response_time": response[0].elapsed.total_seconds()}
        return np.zeros(self.observation_space.shape), -1000, False, False, {}

    def _analyze_response(self, response, payload, response_get):
        state, reward = self.set_flags(payload, response.text, response.status_code, response_get)
        done = all(state)
        return state, reward, done

    def _set_flags(self, payload, response_text, response_status, response_get):
        state = [0, 0, 0]
        if response_status == 200:
            state[0] = 1
            if self.exploit_char_found and not self.found_parenthesis_structure:
                self.found_parenthesis_structure = True
                self.parentheses_structure = payload
                logging.info(f"Phase 2 success - Parentheses structure found: {self.parentheses_structure}")
        if response_status == 200 and ("fail" or "wrong") not in response_text.lower():
            state[1] = 1
        if re.search(rf"flag_challenge_{self.current_challenge_id}\{{[^}}]+}}", response_text):
            state[1], state[2] = 1, 1
            logging.info(f"Flag found in payload: {payload}")
        reward = self.calculate_reward(state, payload)
        return state, reward

    def _calculate_reward(self, space: list[int], payload: str) -> int:
        """
        Set the reward for the current state based on the response from the server.

        :param space: The current observation space.
        :param payload: The payload sent to the server.
        :return: The reward for the current state.
        """
        reward = 0

        # Observation space: [query_valid, data_found, flag_found]

        if space[0] == 0:
            if not self.exploit_char_found:
                reward -= 20  # Penalty for finding the exploit character by crashing the server
                self.exploit_char_found = True
                self.exploit_char = payload[0]  # Set the first character as the escape character
                print(f"Exploit character found: {self.exploit_char}")
            else:
                reward -= 500  # Penalty for crashing the server if we already found the exploit character
                # (bad but ok because of the exploration)

        if space[1] == 0:
            reward -= 200  # Penalty for not finding any data (bad)

        if space[1] == 1 and space != [1, 1, 1]:
            reward -= 100  # Small penalty for bypassing the password check but not the good one

        if space == [1, 1, 1]:
            reward = -1  # High reward for finding the flag

        return reward

    def reset(self, **kwargs):
        self._initialize_flags()
        if NUM_CHALLENGES > 1:
            self.current_challenge_id = (self.current_challenge_id % NUM_CHALLENGES) + 1
        self.session.get("http://localhost:5959/reset")
        logging.info("Environment reset")
        return np.zeros(self.observation_space.shape), {}


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
