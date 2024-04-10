import logging
import random
import secrets
import string
from typing import List

# Choice of SQLi vulnerabilities
SQLI_ARCHETYPES = [
    "Error-based SQL Injection",
    "Union-based SQL Injection",
    "Stacked Queries",
    "Time-based Blind SQL Injection",
    "Boolean-based Blind SQL Injection",
    "No vulnerabilities",
]


def generate_challenges(nbr: int = 10) -> List[str]:
    """Generates a list of SQLi challenges."""
    challenges = random.choices(SQLI_ARCHETYPES, k=nbr)
    flags = []
    for challenge in challenges:
        if challenge != "No vulnerabilities":
            random_string = "".join(secrets.choice(string.ascii_letters) for _ in range(32))
            flag = "flag{" + random_string + "}"
            flags.append(flag)
        else:
            flags.append(None)

    logging.debug(f"Flags are {flags}")
    return challenges
