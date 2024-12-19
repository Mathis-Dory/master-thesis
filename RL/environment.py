import logging
import os
import re

import gymnasium as gym
import numpy as np
import requests
from dotenv import load_dotenv
from gymnasium import spaces
from nltk.parse.generate import generate
from numpy.typing import ArrayLike
from stable_baselines3.common.monitor import Monitor

from cfg import cfg_phase1, cfg_phase2, cfg_phase3
from utils import (
    setup_session, generate_atomic_clause)

# Load environment variables from .env file
load_dotenv()

NUM_CHALLENGES = int(os.getenv("NUM_CHALLENGES", 1))  # Default to 1 if not set

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cfg_phase3_rules = [
    (rule.lhs, rule.rhs)
    for rule in cfg_phase3.productions()  # Extract rules from the grammar object
]


class SQLiEnv(gym.Env):
    """
   Custom environment for SQL injection attacks using reinforcement learning.

   SQLiEnv class simulates a SQL injection attack, where an RL agent generates payloads
   in phases. It iteratively learns by interacting with server responses, aiming to find
   a flag through progressive injection stages.
   """

    def __init__(self):
        """Initialize the SQLiEnv environment, action, and observation spaces."""
        super().__init__()
        self.current_challenge_id = None
        self.last_payload = None
        self.base_url = "http://localhost:5959/challenge/"
        self.session = setup_session()
        self._initialize_flags()

        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
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
        """Initialize tracking flags for the environment's state."""
        self.current_challenge_id = 1
        self.last_payload = ""
        self.exploit_char_found = False
        self.found_parenthesis_structure = False
        self.exploit_char = ""
        self.parentheses_structure = ""
        self.comment_char = ""
        self.step_count = 0
        self.max_steps_per_episode = 5000

    def _generate_payload(self, phase: int, action: ArrayLike) -> str:
        """
         Generate the SQL injection payload based on the current phase and action.

         :param phase: The current phase of the environment.
         :param action: ArrayLike, the action taken by the agent.
         :return: The generated SQL injection payload as a string.
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

    def _build_payload(self, phase: int, selected_clause: str, action: ArrayLike) -> str:
        """
        Simplify payload construction based on phase and chosen grammar clause.

        :param phase: Integer representing the current phase of the environment.
        :param selected_clause: The clause generated from the CFG grammar.
        :param action: ArrayLike, the action vector influencing payload structure.
        :return: A string representing the generated payload.
        """
        if phase == 1:
            return " ".join(selected_clause)
        elif phase == 2:
            return f"{self.exploit_char} {' '.join(selected_clause)}"
        else:
            # Build complex payload for phase 3
            return self._build_complex_payload(action)

    def _build_complex_payload(self, action: ArrayLike) -> str:
        """
        Build the payload for phase 3, ensuring parentheses do not break atomic CFG units.

        :param action: ArrayLike, action vector influencing payload structure.
        :return: Fully constructed SQL injection payload.
        """
        # Generate atomic clauses from CFG
        generated_clauses = generate_atomic_clause(cfg_phase3, n=100)  # Generate array of 100 atomic clauses

        # Select a generated clause based on the action
        action_index = int(action[0] * (len(generated_clauses) - 1))

        selected_clause_units = generated_clauses[action_index]  # Select 1 atomic clause

        # Map actions to parentheses placement
        parentheses_actions = action[1:]  # Use remaining actions for parentheses placement
        num_parentheses = len(self.parentheses_structure)
        insertion_indices = sorted(
            [round(a * (len(selected_clause_units))) for a in parentheses_actions[:num_parentheses]]
        )

        # Insert parentheses into the clause
        clause_with_parentheses = []
        last_index = 0
        for idx in insertion_indices:
            clause_with_parentheses.extend(selected_clause_units[last_index:idx])
            clause_with_parentheses.append(")")  # Insert parenthesis at the specified index
            last_index = idx

        clause_with_parentheses.extend(selected_clause_units[last_index:])

        # Combine the clause with the escape character and comment
        selected_clause = " ".join(clause_with_parentheses)
        payload = f"{self.exploit_char} {selected_clause}" if self.exploit_char else selected_clause
        payload_with_comment = f"{payload} {self.comment_char}"
        return payload_with_comment

    def step(self, action: ArrayLike) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
       Execute a step in the environment using the given action and process the response.

       :param action: ArrayLike, the action taken by the agent.
       :return: Tuple containing the current state (np.ndarray), reward (float),
                done (bool), truncated (bool), and additional info (dict).
       """
        self.step_count += 1
        # Ensure the agent goes through phases sequentially
        phase = (
            1 if not self.exploit_char_found else (2 if not self.found_parenthesis_structure else 3)
        )
        payload = self._generate_payload(phase, action)
        response = self._send_request(payload)
        self.last_payload = payload
        return self._process_response(response, payload)

    def _send_request(self, payload: str) -> tuple[requests.Response | None, requests.Response | None]:
        """
        Send the generated payload to the server and handle any connection errors.

        :param payload: The SQL injection payload string to be sent to the server.
        :return: Tuple containing the POST and GET requests' responses (or None if request failed).
        """
        try:
            response_get = self.session.get(f"{self.base_url}{self.current_challenge_id}", timeout=30)
            data = {"username_payload": payload, "password_payload": ""} if "Login" in response_get.text else {
                "payload": payload}
            response = self.session.post(f"{self.base_url}{self.current_challenge_id}", data=data, timeout=30)
            return response, response_get
        except requests.exceptions.RequestException as e:
            logging.error(f"Connection failed: {e}")
            return None, None

    def _process_response(self, response: tuple[requests.Response | None, requests.Response | None], payload: str) \
            -> tuple[np.ndarray, float, bool, bool, dict]:
        """
         Process the server's response to determine the current state, reward, and completion status.

         :param response: Tuple containing the server's GET and POST responses.
         :param payload: The SQL injection payload string that was sent.
         :return: Tuple containing the state (np.ndarray), reward (float), done (bool),
                  truncated (bool), and additional info (dict).
         """
        if response:
            state, reward, done = self._analyze_response(response[0], payload)
            truncated = self.step_count >= self.max_steps_per_episode
            return np.array(state), reward, done, truncated, {"response_time": response[0].elapsed.total_seconds()}
        return np.zeros(self.observation_space.shape), -1000, False, False, {}

    def _analyze_response(
            self, response: requests.Response, payload: str
    ) -> tuple[list[int], int, bool]:
        """
        Analyze the server's response to update environment state and determine reward.

        :param response: The server's response object.
        :param payload: The payload sent to the server.
        :return: A tuple containing the state (list of observation flags), reward, and completion status.
        """
        state, reward = self._set_flags(payload, response.text, response.status_code)
        done = all(state)
        return state, reward, done

    def _set_flags(self, payload: str, response_text: str, response_status: int) -> tuple[list[int], int]:
        """
        Update state flags based on server response status and content.

        :param payload: The payload sent to the server.
        :param response_text: The text content of the server's response.
        :param response_status: The HTTP status code of the response.
        :return: Tuple containing the state as a list of integers and reward as an integer.
        """
        state = [0, 0, 0]
        if response_status == 200:
            state[0] = 1
            if self.exploit_char_found and not self.found_parenthesis_structure:
                self.found_parenthesis_structure = True
                self.parentheses_structure = ''.join(re.findall(r'[)]+', payload))
                self.comment_char = f"{payload.split()[-1]} "  # Ensure adding whitespace after comment character
                logging.info(f"Phase 2 success - Parentheses structure found: {self.parentheses_structure or 
                                                                               'Without parentheses'}")
        if response_status == 200 and ("fail" or "wrong") not in response_text.lower():
            state[1] = 1
        if re.search(rf"flag_challenge_{self.current_challenge_id}\{{[^}}]+}}", response_text):
            state[1], state[2] = 1, 1
            logging.info(f"Flag found in payload: {payload}")
        reward = self._calculate_reward(state, payload)
        return state, reward

    def _calculate_reward(self, space: list[int], payload: str) -> int:
        """
        Set the reward for the current state based on the response from the server.

        :param space: The current observation space as a list of integers.
        :param payload: The payload sent to the server.
        :return: The calculated reward as an integer.
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

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state, advancing to the next challenge if available.

        :param kwargs: Additional keyword arguments for compatibility with gym reset().
        :return: Tuple containing the reset observation state (np.ndarray) and additional info (dict).
        """
        self._initialize_flags()
        if NUM_CHALLENGES > 1:
            self.current_challenge_id = (self.current_challenge_id % NUM_CHALLENGES) + 1
        self.session.get("http://localhost:5959/reset")
        logging.info("Environment reset")
        return np.zeros(self.observation_space.shape), {}


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
