import logging
import random
import secrets
from typing import List, Tuple

from flask import current_app
from sqlalchemy import inspect, String

from app.database import db

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
        (
            f"flag_challenge_{idx + 1}{{{secrets.token_hex(16)}}}"
            if v != "No vulnerabilities"
            else None
        )
        for idx, v in enumerate(challenges)
    ]
    logging.debug(f"Generated vulnerabilities: {challenges}")
    logging.debug(f"Flags are {flags}")
    logging.debug(f"Templates are {templates}")
    return challenges, flags, templates


def generate_auth_queries(payload_1, payload_2):
    return


def generate_filter_queries(payload):
    return


def generate_filter_default_queries(
    available_tables: List[db.Model],
) -> List[str]:
    """
    Generate GET queries dynamically based on available tables and columns in order
    to display default data in the template (should not contain any flags).
    :param available_tables: List of available tables
    :return: return a list of queries
    """
    queries = []
    templates = current_app.config["INIT_DATA"]["templates"]
    filter_templates = [t for t in templates if t == "filter"]

    # Exclude 'Users' and 'AuthBypass' tables by their table names
    available_tables = [
        t for t in available_tables if t.__tablename__ not in ["user", "auth_bypass"]
    ]

    for _ in filter_templates:
        if available_tables:
            table = random.choice(available_tables)
            available_columns = get_columns(table)
            if available_columns:
                selected_columns = random.sample(
                    available_columns, random.randint(1, len(available_columns))
                )
                flag_exclusion = [
                    f"{col.name} NOT LIKE 'flag_challenge_%'"
                    for col in table.__table__.columns
                    if col.name in selected_columns and isinstance(col.type, String)
                ]
                if flag_exclusion:
                    where_clause = " AND ".join(flag_exclusion)
                    base_query = f"SELECT {', '.join(selected_columns)} FROM {table.__tablename__} WHERE {where_clause}"
                else:
                    base_query = f"SELECT {', '.join(selected_columns)} FROM {table.__tablename__}"

                if random.choice([True, False]):
                    order_by = random.choice(
                        [
                            col.name
                            for col in table.__table__.columns
                            if col.name in selected_columns
                        ]
                    )
                    base_query += f" ORDER BY {order_by} DESC"
                if random.choice([True, False]):
                    limit = random.randint(1, 5)
                    base_query += f" LIMIT {limit}"

                queries.append(base_query)
            else:
                logging.debug(
                    f"No available columns found for table {table.__tablename__}"
                )
        else:
            logging.debug("No available tables found for generating queries.")

    logging.debug(f"Generated following filter queries: {queries}")
    return queries


def get_columns(table: db.Model) -> List[str]:
    """
    Get the column names of a table except the id one.
    :param table: Table object
    :return: List of column names
    """
    return [
        column.key
        for column in inspect(table).mapper.column_attrs
        if column.key.lower() not in ["id"]
    ]
