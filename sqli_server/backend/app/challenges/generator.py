import logging
import random
import secrets
from typing import List, Tuple

# Choices of SQLi vulnerabilities
SQLI_ARCHETYPES = [
    "Error-based SQL Injection",
    "Union-based SQL Injection",
    "Stacked Queries",
    "Time-based Blind SQL Injection",
    "Boolean-based Blind SQL Injection",
    "No vulnerabilities",
]

TEMPLATES = [
    "auth",
    "filter",
    "filter",
    "filter",
]  # Boost the probability of filter challenges to 75%


def generate_challenges(nbr: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Generates a list of SQLi vulnerabilities and associated flags.
    :param nbr: Number of challenges to generate
    :return: Tuple of challenges types and flags
    """
    challenges = random.choices(SQLI_ARCHETYPES, k=nbr)
    templates = random.choices(TEMPLATES, k=nbr)
    flags = [
        f"flag_challenge_{idx + 1}{{{secrets.token_hex(16)}}}"
        if v != "No vulnerabilities"
        else None
        for idx, v in enumerate(challenges)
    ]
    logging.debug(f"Generated vulnerabilities: {challenges}")
    logging.debug(f"Flags are {flags}")
    return challenges, flags, templates


def generate_auth_queries(payload_1, payload_2):
    return f"SELECT * FROM user WHERE username = '{payload_1}' AND password = '{payload_2}'"


def generate_filter_queries():
    return
