import requests
from nltk.parse.generate import generate
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def generate_atomic_clause(cfg, n=1):
    """
    Generate atomic clauses from CFG, ensuring no splitting of complex RHS units.

    :param cfg: The context-free grammar.
    :param n: Number of clauses to generate.
    :return: List of atomic clauses (preserving CFG unit structure).
    """
    clauses = list(generate(cfg, n=n))
    atomic_clauses = []

    for clause in clauses:
        atomic_clause = []
        i = 0
        while i < len(clause):
            unit = clause[i]
            if (
                unit == "LIMIT 1 OFFSET"
                and i + 1 < len(clause)
                and clause[i + 1].isdigit()
            ):
                # Combine "LIMIT 1 OFFSET" and NUMBER into a single atomic unit
                atomic_clause.append(f"{unit} {clause[i + 1]}")
                i += 1  # Skip the next unit as it has been merged
            elif (
                unit == "ORDER BY"
                and i + 1 < len(clause)
                and clause[i + 1].isdigit()
            ):
                # Combine "ORDER BY" and COLUMN into a single atomic unit
                atomic_clause.append(f"{unit} {clause[i + 1]}")
                i += 1  # Skip the next unit as it has been merged
            else:
                atomic_clause.append(unit)
            i += 1

        atomic_clauses.append(atomic_clause)
    return atomic_clauses
