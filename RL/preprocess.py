import re
import urllib

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    return pd.DataFrame(lines, columns=["payload"])


def load_usernames(file_path):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    return lines


def preprocess_payload(payload):
    payload = urllib.parse.unquote(payload)
    payload = re.sub(r"--.*|\#.*", "", payload)
    payload = re.sub(
        r"\b\d+\b", "0", payload
    )  # Normalize numeric literals to '0'
    payload = re.sub(r"'[^']*'", "''", payload)
    payload = re.sub(r"\s+", " ", payload).strip()
    return payload


def create_static_vector(payload):
    patterns = {
        "tautology": r"(\bor\b|\band\b)\s*\(?\s*('1'='1'|1=1|\bTRUE\b)\s*\)?",
        "admin_bypass": r"\badmin'\s*\bor\b\s*('1'='1'|1=1|\bTRUE\b)",
        "extended_condition": r"(\bor\b|\band\b)\s+.+=\s+\1",
    }
    return [
        1 if re.search(pattern, payload, re.IGNORECASE) else 0
        for pattern in patterns.values()
    ]


def create_dynamic_vector(payload, all_payloads):
    encoder = OneHotEncoder()
    encoded_payloads = encoder.fit_transform(
        all_payloads[["payload"]]
    ).toarray()
    payload_idx = all_payloads.index[
        all_payloads["payload"] == payload
    ].tolist()[0]
    return encoded_payloads[payload_idx]


def process_auth():
    payload_file_path = "dataset/auth/auth.txt"

    data = load_data(payload_file_path)

    data["payload"] = data["payload"].apply(preprocess_payload)
    data["static_vector"] = data["payload"].apply(create_static_vector)
    data["dynamic_vector"] = data["payload"].apply(
        lambda x: create_dynamic_vector(x, data)
    )

    data["full_payload"] = data.apply(lambda x: f"{x['payload']}", axis=1)

    return data


if __name__ == "__main__":
    data = process_auth()
    data.to_pickle("dataset/preprocessed_data.pkl")
