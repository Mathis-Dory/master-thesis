import requests
from nltk import CFG
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def get_valid_insertion_points(parts, tokens):
    """
    Get valid insertion points for parentheses, ensuring no token is split.

    :param parts: List of parts of the clause.
    :param tokens: Set of CFG tokens to avoid splitting.
    :return: List of valid insertion points.
    """
    valid_insertion_points = []
    for i in range(len(parts) + 1):
        if i < len(parts) and (parts[i] in tokens):
            continue  # Avoid insertion before or after CFG tokens
        if i > 0 and (parts[i - 1] in tokens):
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


def update_cfg_with_comment(cfg, comment_char):
    """
    Dynamically modify the CFG to incorporate both the parentheses structure and the comment character.

    :param cfg: Original CFG object.
    :param comment_char: The comment character (e.g., '--') to be included at the end of the payload.
    :return: A modified CFG with the comment character embedded.
    """

    # Define the comment structure rule (e.g., '--' or any other form of comment)
    comment_rule = f'Comment -> "{comment_char}"'

    # Extract the original CFG rules as a string
    original_rules = "\n".join(str(rule) for rule in cfg.productions())

    # Append the comment structure to the original rules
    updated_cfg_rules = f"{original_rules}\n{comment_rule}"

    # Create a new CFG with the updated rules
    return CFG.fromstring(updated_cfg_rules)


def update_cfg_with_parenthesis(cfg, parentheses_count):
    """
    Dynamically modify the CFG to incorporate the correct number of parentheses for phase 3.

    :param cfg: Original CFG object for phase 3.
    :param parentheses_count: The number of parentheses to include.
    :return: A modified CFG with the correct number of parentheses.
    """
    # Generate the parentheses rule based on the parentheses count
    parentheses_rule = f'PARENTHESIS -> {"".join([")"] * parentheses_count)}'

    # Extract the original CFG rules as a string
    original_rules = "\n".join(str(rule) for rule in cfg.productions())

    # Append the dynamic parentheses rule to the original rules
    updated_cfg_rules = f"{original_rules}\n{parentheses_rule}"

    # Create a new CFG with the updated rules
    return CFG.fromstring(updated_cfg_rules)
