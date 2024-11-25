import requests
from nltk import CFG
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def update_cfg_with_parenthesis(cfg, parentheses_count):
    """
    Dynamically modify the CFG to incorporate the correct number of parentheses for phase 3.

    :param cfg: Original CFG object for phase 3.
    :param parentheses_count: The number of parentheses to include.
    :return: A modified CFG with the correct number of parentheses.
    """
    # Generate the parentheses rule based on the parentheses count
    parentheses_terminals = " ".join(['")"'] * parentheses_count)
    parentheses_rule = f"PARENTHESIS -> {parentheses_terminals}"

    # Extract the original CFG rules as a string
    original_rules = "\n".join(str(rule) for rule in cfg.productions())

    # Append the dynamic parentheses rule to the original rules
    updated_cfg_rules = f"{original_rules}\n{parentheses_rule}"

    # Parse the updated CFG
    return CFG.fromstring(updated_cfg_rules)


def extract_valid_parenthesis_positions(flat_tokens):
    """
    Determine valid positions for parentheses insertion between flattened tokens.

    :param flat_tokens: List of flattened tokens.
    :return: A list of valid positions for parentheses insertion.
    """
    valid_positions = [0]  # Start of the payload
    for i in range(1, len(flat_tokens)):
        valid_positions.append(i)
    valid_positions.append(len(flat_tokens))  # End of the payload
    return valid_positions


def distribute_parentheses(flat_tokens, parentheses_count):
    """
    Distribute closing parentheses across valid positions within the payload.

    :param flat_tokens: List of flattened tokens.
    :param parentheses_count: Number of closing parentheses to insert.
    :return: Tokens with parentheses distributed appropriately.
    """
    valid_positions = extract_valid_parenthesis_positions(flat_tokens)

    # Ensure parentheses are distributed
    for _ in range(parentheses_count):
        if not valid_positions:
            break
        # Pick a random valid position to distribute parentheses
        insert_pos = valid_positions.pop(0)  # Pop from the front for simplicity
        flat_tokens.insert(insert_pos, ")")

    return flat_tokens


def extract_flat_tokens(cfg, clause):
    """
    Flatten tokens from a clause based on the CFG rules, preserving compound tokens.

    :param cfg: The CFG object.
    :param clause: The generated clause (list of strings).
    :return: A list of flattened tokens, respecting CFG compound structures.
    """
    rules = {
        str(prod.lhs()): [str(rhs) for rhs in prod.rhs()]
        for prod in cfg.productions()
    }
    tokens = []
    i = 0

    while i < len(clause):
        matched = False
        # Try to match multi-token rules
        for lhs, rhs_list in rules.items():
            for rhs in rhs_list:
                rhs_tokens = rhs.split()
                if clause[i : i + len(rhs_tokens)] == rhs_tokens:
                    tokens.append(" ".join(rhs_tokens))  # Treat as one token
                    i += len(rhs_tokens)
                    matched = True
                    break
            if matched:
                break

        # If no match, treat the current word as a single token
        if not matched:
            tokens.append(clause[i])
            i += 1

    return tokens
