import logging
import random
import secrets
from typing import List, Tuple, Any

from faker import Faker
from flask import current_app

from app.database import db
from challenges.utils import (
    get_columns,
    get_column_type,
    find_group_functions,
    find_columns_in_query,
    LOGICAL_OPERATORS,
    END_OPERATORS,
    COMPARE_OPERATORS,
    exclude_tables,
    filter_flags,
    NUMERIC_FUNCTIONS,
    TEXT_FUNCTIONS,
    display_table,
)

# Choices of SQLi vulnerabilities
SQLI_ARCHETYPES = [
    "Error-based SQL Injection",
    "Union-based SQL Injection",
    "Stacked Queries",
    "Time-based Blind SQL Injection",
    "Boolean-based Blind SQL Injection",
    "No vulnerabilities",
]

TEMPLATES = ["filter"]  # Remove auth challenges for the moment


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


def generate_default_queries_filter(
    available_tables: List[db.Model],
) -> List[str] or None:
    """
    Generate a list of default filter queries.
    :param available_tables: List of available tables excluding 'customer' and 'auth_bypass'
    :return: List of complex queries
    """
    queries = []
    templates = current_app.config["INIT_DATA"]["templates"]
    filter_templates = [t for t in templates if t == "filter"]

    # Exclude 'Customer' and 'AuthBypass' tables
    available_tables = exclude_tables(
        available_tables, excluded_tables=["customer", "auth_bypass"]
    )

    for _ in filter_templates:
        if available_tables:
            # Create a basic query like "SELECT column1, column2 FROM table"
            query, text_columns, numeric_columns, columns = basic_query(
                available_tables
            )
            # Add a WHERE clause to remove any flag pattern in the query
            if text_columns:
                query = filter_flags(query, text_columns)
            else:
                # The WHERE is initially added in the filter_flags function
                query += " WHERE"

            # Add extra WHERE clause for future filtering
            query += "( "
            query = generate_conditional_query(
                columns=columns,
                text_columns=text_columns,
                numeric_columns=numeric_columns,
                query=query,
            )
            # Add extra logical operators to the query
            query = add_logical_operator(
                columns=columns,
                text_columns=text_columns,
                numeric_columns=numeric_columns,
                query=query,
            )
            query += " )"
            # Inspect the query to check if we applied a numerical function and if yes, add a GROUP BY clause
            if find_group_functions(query):
                query += f" GROUP BY {', '.join(columns)}"

            query = generate_end_operator(
                query=query,
                columns=columns,
                available_tables=available_tables,
                text_columns=text_columns,
                numeric_columns=numeric_columns,
            )

            queries.append(query)
        else:
            logging.error("No tables available for filter challenges !")
            return

    logging.debug(f"Generated following filter queries: {queries}")
    return queries


def basic_query(
    available_tables: List[db.Model],
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Generate a basic query with a random table and columns.
    :param available_tables: List of available tables
    :return: Query, text columns, numeric columns, columns
    """
    table = random.choice(available_tables)
    columns = get_columns(table)
    if random.choice([True, True, False]):
        # Select all columns
        text_columns, numeric_columns = get_column_type(table, columns)
        if random.choice([True, True, False]):
            query, used_columns = apply_sql_function(
                columns=columns,
                text_columns=text_columns,
                numeric_columns=numeric_columns,
            )
        else:
            # Do not include a function
            query = f"SELECT {', '.join(columns)}"

    else:
        # Select random columns
        columns = random.sample(columns, random.randint(1, len(columns)))
        text_columns, numeric_columns = get_column_type(table, columns)
        if random.choice([True, True, False]):
            query, used_columns = apply_sql_function(
                columns=columns,
                text_columns=text_columns,
                numeric_columns=numeric_columns,
            )
        else:
            # Do not include a function
            if len(columns) > 1:
                query = f"SELECT {', '.join(columns)}"
            else:
                query = f"SELECT {columns[0]}"

    table_format = display_table(table)
    query += f" FROM {table_format}"

    return query, text_columns, numeric_columns, columns


def apply_sql_function(
    columns: List[str],
    text_columns: List[str] or None = None,
    numeric_columns: List[str] or None = None,
) -> Tuple[str, List[str]]:
    """
    Apply a SQL function to a column.
    :param columns: List of columns
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :return: Query with a SQL function
    """
    if random.choice([True, False]):
        # Apply the function to a text column
        query, used_columns = generate_text_function(text_columns)
    else:
        # Apply the function to a numeric column
        query, used_columns = generate_numeric_function(numeric_columns)
    # Concat the rest of the columns to the query without the column used in the function
    if 1 < len(columns) != len(used_columns) and "(" in query:
        query += f", {', '.join([c for c in columns if c not in used_columns])}"
    elif 1 < len(columns) == len(used_columns) and "(" in query:
        return query, used_columns

    elif len(columns) > 1 and "(" not in query:
        query += f"{', '.join([c for c in columns if c not in used_columns])}"
    else:
        query += f"{columns[0]}"
    return query, used_columns


def generate_text_function(text_columns: List[str]) -> Tuple[str, List[str]]:
    """
    Generate a text function query.
    :param text_columns: List of text columns
    :return: Query with a text function
    """
    if not text_columns:
        logging.debug("No text columns available to apply a function.")
        return "SELECT ", []
    text_function = random.choice(TEXT_FUNCTIONS)
    if text_function == "CONCAT":
        if len(text_columns) > 1:
            # Select two random text columns and concatenate them
            columns = random.sample(text_columns, 2)
            query = f"SELECT {text_function}({columns[0]}, '_', {columns[1]})"
            return query, columns
        else:
            # Select one random text column and concatenate it with a string
            column = random.choice(text_columns)
            query = f"SELECT {text_function}({column}, '_I_AM_CONCATENATE')"
            return query, column
    elif text_function == "SUBSTRING":
        # Select a random text column and apply the SUBSTRING function
        column = random.choice(text_columns)
        query = f"SELECT {text_function}({column}, 1, {len(column[0]) // 2})"
        return query, column
    elif text_function == "REPLACE":
        # Select a random text column and apply the REPLACE function
        column = random.choice(text_columns)
        query = f"SELECT {text_function}({column}, 'a', 'b')"
        return query, column

    logging.error("Unknown text function !")
    return "SELECT ", []


def generate_numeric_function(
    numeric_columns: List[str],
) -> Tuple[str, List[str]]:
    """
    Generate a numeric function query.
    :param numeric_columns: List of numeric columns
    :return: Query with a numeric function
    """
    if not numeric_columns:
        logging.debug("No numeric columns available to apply a function.")
        return "SELECT ", []
    numeric_function = random.choice(NUMERIC_FUNCTIONS)
    if numeric_function == "SUM":
        if len(numeric_columns) > 1:
            # Select two random numeric columns and sum them
            columns = random.sample(numeric_columns, 2)
            query = f"SELECT {numeric_function}({columns[0]}) + {numeric_function}({columns[1]})"
            return query, columns
        else:
            # Select one random numeric column and sum it
            column = random.choice(numeric_columns)
            query = f"SELECT {numeric_function}({column})"
            return query, column
    elif numeric_function == "AVG":
        column = random.choice(numeric_columns)
        query = f"SELECT {numeric_function}({column})"
        return query, column
    elif numeric_function == "MAX":
        column = random.choice(numeric_columns)
        query = f"SELECT {numeric_function}({column})"
        return query, column
    elif numeric_function == "MIN":
        column = random.choice(numeric_columns)
        query = f"SELECT {numeric_function}({column})"
        return query, column
    elif numeric_function == "COUNT":
        column = random.choice(numeric_columns)
        query = f"SELECT {numeric_function}({column})"
        return query, column
    logging.error("Unknown numeric function !")
    return "SELECT ", []


def generate_conditional_query(
    columns: List[str],
    text_columns: List[str],
    numeric_columns: List[str],
    query: str,
) -> str:
    """
    Generate a conditional query based on the column type.
    :param columns: List of columns
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :param query: Original query
    :return: The query with a conditional operator
    """
    column = random.choice(columns)
    if column in text_columns:
        if random.choice([True, True, True, False]):
            # Add a text comparison
            query += (
                f"{column} {random.choice(COMPARE_OPERATORS['text'])}"
                f" '{random.choice(['%', ''])}{Faker().random_letter()}{random.choice(['%', ''])}'"
            )
        else:
            # Add a NULL comparison
            query += f"{column} {random.choice(COMPARE_OPERATORS['null'])}"

    elif column in numeric_columns:
        if random.choice([True, True, True, False]):
            # Add a numeric comparison
            query += f"{column} {random.choice(COMPARE_OPERATORS['numeric'])} {random.randint(0, 100)}"
        else:
            # Add a NULL comparison
            query += f"{column} {random.choice(COMPARE_OPERATORS['null'])}"
    else:
        query += f"{column} {random.choice(COMPARE_OPERATORS['null'])}"
    return query


def add_logical_operator(
    columns: List[str],
    text_columns: List[str],
    numeric_columns: List[str],
    query: str,
) -> str:
    """
    Add a logical operator to the query.
    :param columns: List of columns
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :param query: Original query
    :return: The query with a logical operator
    """
    if random.choice([True, False]):
        # Add a logical operator
        query += f" {random.choice(LOGICAL_OPERATORS)} "
        query = generate_conditional_query(
            columns, text_columns, numeric_columns, query
        )
    return query


def generate_end_operator(
    query: str,
    columns: List[str],
    available_tables: List[Any],
    text_columns: List[str],
    numeric_columns: List[str],
) -> str:
    """
    Add an end operator to the query.
    :param query: Original query
    :param columns: List of columns
    :param available_tables: List of available tables
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :return: Query with end operator
    """
    if random.choice([True, False]):
        operator = random.choice(END_OPERATORS)
        if operator == "ORDER BY":
            query += f" {operator} {random.choice(columns)}"
        elif operator == "LIMIT":
            query += f" {operator} {random.randint(1, 20)}"
        elif operator == "DISTINCT":
            query = query.replace("SELECT", "SELECT DISTINCT")
        elif operator == "UNION":
            query += generate_union_query(
                query, available_tables, text_columns, numeric_columns
            )
        else:
            logging.error("Unknown end operator !")
    return query


def generate_union_query(
    query: str,
    available_tables: List[Any],
    text_columns: List[str],
    numeric_columns: List[str],
) -> str:
    """
    Generate a UNION query with type casting for matching column types.
    :param query: Original query
    :param available_tables: List of available tables
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :return: Union query
    """
    # Choosing another random table
    union_table = random.choice(available_tables)
    columns_union = get_columns(union_table)
    text_columns_union, numeric_columns_union = get_column_type(
        union_table, columns_union
    )

    # Ensuring that the columns match the first query's columns
    query_union = " UNION SELECT "
    original_columns = find_columns_in_query(query)

    query_parts = []
    for col1 in original_columns:
        if col1 in text_columns:
            # If the column in the first table is a text type, find a text type column in the second table or cast
            col2 = next(
                (col for col in columns_union if col in text_columns_union),
                None,
            )
            if col2:
                query_parts.append(f"{col2}")
            else:
                col2 = next(
                    (col for col in columns_union if col in numeric_columns_union),
                    None,
                )
                query_parts.append(f"CAST({col2} AS VARCHAR)")
        elif col1 in numeric_columns:
            # If the column in the first table is a numeric type, find a numeric type column in the second table or cast
            col2 = next(
                (col for col in columns_union if col in numeric_columns_union),
                None,
            )
            if col2:
                query_parts.append(f"{col2}")
            else:
                col2 = next(
                    (col for col in columns_union if col in text_columns_union),
                    None,
                )
                query_parts.append(f"CAST({col2} AS NUMERIC)")

    table_format = display_table(union_table)
    query_union += f"{', '.join(query_parts)} FROM {table_format}"
    query_union = filter_flags(query_union, text_columns_union)
    query_union += " ( "
    query_union = generate_conditional_query(
        columns=columns_union,
        text_columns=text_columns_union,
        numeric_columns=numeric_columns_union,
        query=query_union,
    )
    query_union += " ) "

    return query_union


def transform_get_to_post(query: str, payload: str) -> None or str:
    return ""
