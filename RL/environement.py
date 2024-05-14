import re
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import requests
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


class SQLiEnv(gym.Env):
    def __init__(self):
        self.base_url = "http://localhost:5959/challenge/"
        self.payloads = pd.read_pickle("dataset/preprocessed_data.pkl")
        self.current_challenge_id = 1
        self.max_attempts = 1000  # Maximum attempts per challenge
        self.attempts = 0  # Attempt counter

        self.exploit_char_found = False  # Flag for finding exploit character
        self.exploit_char = ""  # Store the found exploit character

        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Continuous action space where
        # it chooses action with index 0 to 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Continuous observation space
        # Receive three continuous values
        # 1st value: return 1 if the agent found the exploit character, 0 otherwise
        # 2nd value: return 1 is the query is valid, 0 otherwise
        # 3rd value: return 1 if the agent found the flag, 0 otherwise

    def step(self, action):
        index = int(action[0] * (len(self.payloads) - 1))
        selected_payload = self.payloads.iloc[index]['full_payload']

        response = requests.get(f"{self.base_url}{self.current_challenge_id}")
        if "Login" in response.text:
            data = {"username_payload": selected_payload, "password_payload": ""}
        elif "Filter" in response.text:
            data = {"payload": selected_payload}

        start_time = time.time()
        response = requests.post(f"{self.base_url}{self.current_challenge_id}", data=data)
        end_time = time.time()

        state, reward, done = self.analyze_response(response.text, selected_payload, response.status_code)

        self.attempts += 1

        if done or self.attempts >= self.max_attempts:
            self.current_challenge_id += 1
            self.attempts = 0
            self.exploit_char_found = False  # Reset exploit character found flag
            self.exploit_char = ""  # Reset exploit character
            if self.current_challenge_id > 10:
                self.current_challenge_id = 1  # Reset to the first challenge
                done = True

        truncated = False
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
            print("Flag found!")
            return [1, 1, 1], -1, True

        # If the agent found the escape character
        if self.exploit_char_found:
            if payload[0] != self.exploit_char:
                # Worst penalty if the agent did not use the exploit character when it is available
                return [0, 0, 0], -100, False
            if response_status == 200:
                # Agent used the exploit character and the query is valid
                if "flag_" in response_text:
                    # Agent found wrong flag
                    return [1, 1, 0], -10, False
                else:
                    # Agent did a valid query but nothing is found
                    return [1, 1, 0], -25, False
            if response_status == 500:
                # The query is still invalid
                return [1, 0, 0], -75, False
        else:
            if response_status == 200:
                # If the response status is 200, it means the query is valid without escaping
                return [0, 1, 0], -100, False
            if response_status == 500:
                self.exploit_char_found = True
                self.exploit_char = payload[0]
                print(f"Exploit character found: {self.exploit_char}")
                return [1, 0, 0], -25, False

        return [0, 0, 0], -100, False  # Default state define as the worst state


env = Monitor(SQLiEnv(), "logs", allow_early_resets=True)
