import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def get_valid_insertion_points(parts, tokens):
    """
    Get valid insertion points for parentheses, avoiding splitting CFG tokens.

    :param parts: The list of parts of the clause.
    :param tokens: The set of CFG tokens to avoid splitting.
    :return: A list of valid insertion points.
    """
    valid_insertion_points = []
    for i in range(len(parts) + 1):
        if i < len(parts):
            if parts[i] in tokens:
                continue
        if i > 0:
            if parts[i - 1] in tokens:
                continue
        valid_insertion_points.append(i)
    return valid_insertion_points


def extract_tokens_from_grammar(grammar):
    """
    Extract all tokens from the given CFG grammar.

    :param grammar: The CFG grammar to extract tokens from.
    :return: A set of CFG tokens.
    """
    tokens = set()
    for production in grammar.productions():
        rhs = production.rhs()
        for symbol in rhs:
            if isinstance(symbol, str):
                tokens.add(symbol)
    return tokens
