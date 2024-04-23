import logging
import random
import secrets
from typing import List, Tuple, Any

from flask import current_app
from sqlalchemy import inspect, String, Text, Integer, Float

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
    for idx, _ in enumerate(challenges):
        logging.debug(
            f"Generated vulnerabilities: {challenges[idx]} with flag: {flags[idx]} and template: {templates[idx]}"
        )
    return challenges, flags, templates


def generate_filter_default_queries(
    available_tables: List[db.Model],
) -> List[str] or None:
    """
    Generate GET queries dynamically based on available tables and columns in order
    to display default data in the template, ensuring that all generated queries are valid
    and adhere to SQL syntax and logic, particularly in relation to GROUP BY, ORDER BY
    clauses, and the use of logical operators in WHERE clauses.
    :param available_tables: List of available tables excluding 'user' and 'auth_bypass'
    :return: List of complex queries
    """
    queries = []
    templates = current_app.config["INIT_DATA"]["templates"]
    filter_templates = [t for t in templates if t == "filter"]

    # Define SQL functions and operators
    numeric_functions = ["MAX", "SUM", "AVG", "MIN"]
    text_functions = ["CONCAT", "SUBSTRING", "REPLACE"]
    other_functions = ["COUNT"]
    compare_operators = [
        ">",
        "<",
        "=",
        "LIKE",
        "NOT LIKE",
    ]
    conditional_operators = ["AND", "OR"]
    end_operators = ["HAVING", "ORDER BY", "GROUP BY", "LIMIT", "DISTINCT"]

    # Exclude 'Users' and 'AuthBypass' tables
    available_tables = [
        t for t in available_tables if t.__tablename__ not in ["user", "auth_bypass"]
    ]

    for _ in filter_templates:
        if available_tables:
            if random.choice([True, False]):
                # We join tables
                tables = random.sample(available_tables, len(available_tables))
                # TODO
            else:
                # We do not join tables
                table = random.choice(available_tables)
                columns = get_columns(table)
                if random.choice([True, True, False]):
                    # Select all columns
                    if random.choice([True, True, True, False]):
                        # Include a function, but first we check which columns are numeric or text
                        text_columns, numeric_columns = get_column_type(table, columns)
                        if random.choice([True, False]):
                            # Apply the function to a text column
                            generate_text_function(text_functions, text_columns)
                        else:
                            # Apply the function to a numeric column
                            generate_numeric_function(
                                numeric_functions, numeric_columns
                            )
                    else:
                        # Do not include a function
                        query = f"SELECT * FROM {table.__tablename__}"

                else:
                    # Select random columns
                    columns = random.sample(columns, random.randint(1, len(columns)))
                    if random.choice([True, False]):
                        # Include a function, but first we check which columns are numeric or text
                        text_columns, numeric_columns = get_column_type(table, columns)
                        if random.choice([True, False]):
                            # Apply the function to a text column
                            generate_text_function(text_functions, text_columns)
                        else:
                            # Apply the function to a numeric column
                            generate_numeric_function(
                                numeric_functions, numeric_columns
                            )
                    else:
                        # Do not include a function
                        query = (
                            f"SELECT {', '.join(columns)} FROM {table.__tablename__}"
                        )
                if random.choice([True, False]):
                    # Add a general function
                    function = random.choice(other_functions)
                    if function == "COUNT":
                        # TODO
                        pass
                    else:
                        logging.error("Unknown function !")
                else:
                    # TODO
                    pass
            if random.choice([True, False]):
                # We do a UNION clause
                pass
            else:
                # We do not do a UNION clause
                pass
        else:
            logging.error("No tables available for filter challenges !")
            return
    queries.append(query)

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


def get_column_type(table: Any, columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Helper function to determine the type of column
    :param table: Table object
    :param columns: List of column names
    :return: Tuple of text and numeric columns
    """
    text_columns = [
        c
        for c in columns
        if (isinstance(table.__table__.columns[c].type, (String, Text)))
    ]
    numeric_columns = [
        c
        for c in columns
        if (isinstance(table.__table__.columns[c].type, (Integer, Float)))
    ]
    return text_columns, numeric_columns


def generate_text_function(
    text_functions: List[str], text_columns: List[str]
) -> str or None:
    text_function = random.choice(text_functions)
    if text_function == "CONCAT":
        if len(text_columns) > 1:
            # Select two random text columns and concatenate them
            columns = random.sample(text_columns, 2)
            query = f"SELECT {text_function}({columns[0]}, '_', {columns[1]})"
        else:
            # Select one random text column and concatenate it with a string
            column = random.sample(text_columns, 1)
            query = f"SELECT {text_function}({column[0]}, '_I_AM_CONCATENATE')"
    elif text_function == "SUBSTRING":
        # Select a random text column and apply the SUBSTRING function
        column = random.sample(text_columns, 1)
        query = f"SELECT {text_function}({column[0]}, 1, {len(column[0]) // 2})"
    elif text_function == "REPLACE":
        # Select a random text column and apply the REPLACE function
        column = random.sample(text_columns, 1)
        query = f"SELECT {text_function}({column[0]}, 'A', 'B')"
    else:
        logging.error("Unknown text function !")
        return
    return query


def generate_numeric_function(
    numeric_functions: List[str], numeric_columns: List[str]
) -> str or None:
    numeric_function = random.choice(numeric_functions)
    if numeric_function == "SUM":
        if len(numeric_columns) > 1:
            # Select two random numeric columns and sum them
            columns = random.sample(numeric_columns, 2)
            query = f"SELECT {numeric_function}({columns[0]}) + {numeric_function}({columns[1]})"
        else:
            # Select one random numeric column and sum it
            column = random.sample(numeric_columns, 1)
            query = f"SELECT {numeric_function}({column[0]})"
    elif numeric_function == "AVG":
        column = random.sample(numeric_columns, 1)
        query = f"SELECT {numeric_function}({column[0]})"
    elif numeric_function == "MAX":
        column = random.sample(numeric_columns, 1)
        query = f"SELECT {numeric_function}({column[0]})"
    elif numeric_function == "MIN":
        column = random.sample(numeric_columns, 1)
        query = f"SELECT {numeric_function}({column[0]})"
    else:
        logging.error("Unknown numeric function !")
        return
    return query


def transform_get_to_post(query: str, payload: str) -> None or str:
    return ""
