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

TEMPLATES = ["auth", "filter"]


def generate_challenges(nbr: int) -> Tuple[List[str], List[str]]:
    """
    Generates a list of SQLi vulnerabilities and associated flags.
    :param nbr: Number of challenges to generate
    :return: Tuple of challenges types and flags
    """
    vulns = random.choices(SQLI_ARCHETYPES, k=nbr)
    flags = [
        f"flag{{{secrets.token_hex(16)}}}" if v != "No vulnerabilities" else None for v in vulns
    ]
    logging.debug(f"Generated vulnerabilities: {vulns}")
    logging.debug(f"Flags are {flags}")
    return vulns, flags
