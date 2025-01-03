import logging
import random
import re
import secrets
from typing import List, Tuple, Any

from faker import Faker
from flask import current_app

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
    SQLI_ARCHETYPES,
    TEMPLATES,
    add_parenthesis,
    add_quotes,
)
from database import Base


def generate_challenges(nbr: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Generates a list of SQLi vulnerabilities and associated flags.
    :param nbr: Number of archetypes to generate
    :return: Tuple of archetypes types and flags
    """
    archetypes = random.choices(SQLI_ARCHETYPES, k=nbr)
    templates = random.choices(TEMPLATES, k=nbr)
    flags = [
        (
            f"flag_challenge_{idx + 1}{{{secrets.token_hex(16)}}}"
            if v != "No vulnerabilities"
            else None
        )
        for idx, v in enumerate(archetypes)
    ]
    return archetypes, flags, templates


def generate_default_queries(
    created_tables: List[Base],
) -> Tuple[List[str], List[str]] or None:
    """
    Generate a list of default queries for filter and auth challenges.
    :param created_tables: List of available tables in the db
    :return: List of complex queries
    """
    queries = []
    decomposed_queries = []
    templates = current_app.config["INIT_DATA"]["TEMPLATES"]
    archetypes = current_app.config["INIT_DATA"]["ARCHETYPES"]

    # Exclude and 'AuthBypass' tables
    created_tables = exclude_tables(
        created_tables, excluded_tables=["auth_bypass"]
    )

    for idx, t in enumerate(templates):
        if t == "filter":
            if created_tables:
                query = generate_filter_query(created_tables)
                queries.append(query)
                if "No vulnerabilities" in archetypes[idx]:
                    decomposed_query = extract_random_condition(query)
                    decomposed_query[3] = re.sub(
                        r"['\"]?:payload['\"]?",
                        ":payload",
                        decomposed_query[3],
                    )
                    decomposed_queries.append(decomposed_query)
                else:
                    decomposed_queries.append(extract_random_condition(query))
            else:
                logging.error("No tables available for filter challenges !")
                return
        elif t == "auth":
            if "No vulnerabilities" in archetypes[idx]:
                queries.append(
                    f"SELECT username, password FROM"
                    f" auth_bypass WHERE (username= :username AND password= :password) LIMIT 1"
                )
            else:
                queries.append(generate_default_queries_auth())
            # We do not need to decompose the auth queries
            decomposed_queries.append(None)

        else:
            logging.error("Unknown template !")
            return

    return queries, decomposed_queries


def generate_filter_query(available_tables: List[Base]) -> str:
    """
    Generate a complex filter query.
    :param available_tables: List of available tables
    :return: Query
    """
    (
        query,
        text_columns,
        numeric_columns,
        date_time_columns,
        columns,
    ) = basic_query(available_tables)
    # Add a WHERE clause to remove any flag pattern in the query
    if text_columns:
        query = filter_flags(query, text_columns)
    else:
        # The WHERE is initially added in the filter_flags function
        query += " WHERE"

    # Add extra WHERE clause for future filtering
    query += " ( "
    query = generate_conditional_query(
        columns=columns,
        text_columns=text_columns,
        numeric_columns=numeric_columns,
        date_time_columns=date_time_columns,
        query=query,
    )
    # Add extra logical operators to the query
    query = add_logical_operator(
        columns=columns,
        text_columns=text_columns,
        numeric_columns=numeric_columns,
        date_time_columns=date_time_columns,
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
    return query


def basic_query(
    available_tables: List[Base],
) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """
    Generate a basic query with a random table and columns.
    :param available_tables: List of available tables
    :return: Query, text columns, numeric columns, columns
    """
    table = random.choice(available_tables)
    columns = get_columns(table)
    if random.choice([True, True, False]):
        # Select all columns
        (
            text_columns,
            numeric_columns,
            date_time_columns,
        ) = get_column_type(table, columns)
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
        text_columns, numeric_columns, date_time_columns = get_column_type(
            table, columns
        )
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

    return query, text_columns, numeric_columns, date_time_columns, columns


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
    # If the amount of available columns is greater than the amount of used columns and the query contains a function
    if 1 < len(columns) != len(used_columns) and "(" in query:
        query += f", {', '.join([c for c in columns if c not in used_columns])}"
    # If more than one column is used, and it was the amount of available columns
    elif 1 < len(columns) == len(used_columns) and "(" in query:
        return query, used_columns
    # If we did not apply a function, and we have more than one column
    elif len(columns) > 1 and "(" not in query:
        query += f"{', '.join([c for c in columns])}"
    # If we have only one column without a function
    elif len(columns) == 1 and "(" not in query:
        query += f"{columns[0]}"
    # If we have only one column, but we applied a function on it and then select it again
    else:
        query += f", {columns[0]}"
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
    date_time_columns: List[str],
    query: str,
) -> str:
    """
    Generate a conditional query based on the column type.
    :param columns: List of columns
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :param date_time_columns: List of date time columns
    :param query: Original query
    :return: The query with a conditional operator
    """
    column = random.choice(columns)
    if column in text_columns:
        # Add a text comparison
        payload = add_parenthesis(
            f"'{random.choice(['%', '']) + Faker().random_letter() + random.choice(['%', ''])}'"
        )
        query += (
            f"{column} {random.choice(COMPARE_OPERATORS['text'])} {payload}"
        )

    elif column in numeric_columns:
        # Add a numeric comparison
        payload = add_quotes(random.randint(0, 1000))
        payload = add_parenthesis(payload)
        query += (
            f"{column} {random.choice(COMPARE_OPERATORS['numeric'])} {payload}"
        )

    elif column in date_time_columns:
        # Add a date comparison
        payload = add_quotes(Faker().date_time_this_year())
        payload = add_parenthesis(payload)
        query += (
            f"{column} {random.choice(COMPARE_OPERATORS['numeric'])} {payload}"
        )

    else:
        query += f"{column} {random.choice(COMPARE_OPERATORS['null'])}"

    return query


def add_logical_operator(
    columns: List[str],
    text_columns: List[str],
    numeric_columns: List[str],
    date_time_columns: List[str],
    query: str,
) -> str:
    """
    Add a logical operator to the query.
    :param columns: List of columns
    :param text_columns: List of text columns
    :param numeric_columns: List of numeric columns
    :param date_time_columns: List of date time columns
    :param query: Original query
    :return: The query with a logical operator
    """
    if random.choice([True, False]):
        # Add a logical operator
        query += f" {random.choice(LOGICAL_OPERATORS)} "
        query = generate_conditional_query(
            columns=columns,
            text_columns=text_columns,
            numeric_columns=numeric_columns,
            date_time_columns=date_time_columns,
            query=query,
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
    (
        text_columns_union,
        numeric_columns_union,
        date_time_columns,
    ) = get_column_type(union_table, columns_union)

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
                    (
                        col
                        for col in columns_union
                        if col in numeric_columns_union
                    ),
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
                if current_app.config["INIT_DATA"]["DBMS"] == "postgres:latest":
                    query_parts.append(
                        f"CASE WHEN {col2} ~ '^[0-9]+(\.[0-9]+)?$' THEN CAST({col2} AS NUMERIC) ELSE NULL END"
                    )
                elif current_app.config["INIT_DATA"]["DBMS"] == "mysql:latest":
                    query_parts.append(
                        f"IF({col2} REGEXP '^-?[0-9]+(\\.[0-9]+)?$',  CAST({col2} AS DECIMAL), NULL)"
                    )
                else:
                    logging.error("Unknown DBMS !")

    table_format = display_table(union_table)
    query_union += f"{', '.join(query_parts)} FROM {table_format}"
    query_union = filter_flags(query_union, text_columns_union)
    query_union += " ( "
    query_union = generate_conditional_query(
        columns=columns_union,
        text_columns=text_columns_union,
        numeric_columns=numeric_columns_union,
        date_time_columns=date_time_columns,
        query=query_union,
    )
    query_union += " ) "

    return query_union


def extract_random_condition(
    query: str,
) -> List[str] | None:
    """
    Extract a random condition from the query.
    :param query: SQL query
    :return: List including column name, comparison value, condition and the new POST query
    """
    conditions = []
    parts = re.split(r"\bUNION\b", query, flags=re.IGNORECASE)

    for part in parts:
        where_parts = re.split(r"\bWHERE\b", part, flags=re.IGNORECASE)
        if len(where_parts) < 2:
            continue

        where_clause = where_parts[-1]
        # Remove GROUP BY and ORDER BY for now, but keep LIMIT for separate processing
        limit_match = re.search(
            r"\bLIMIT\s+\d+", where_clause, flags=re.IGNORECASE
        )
        if limit_match:
            conditions.append(limit_match.group(0))

        # Clean the where clause from GROUP BY, ORDER BY and LIMIT
        clean_where_clause = re.sub(
            r"GROUP BY.*|ORDER BY.*|LIMIT.*",
            "",
            where_clause,
            flags=re.IGNORECASE,
        )

        # Split the remaining part of the where clause to get conditions
        potential_conditions = re.split(
            r"\sAND\s|\sOR\s", clean_where_clause, flags=re.IGNORECASE
        )

        for condition in potential_conditions:
            if "%flag_challenge%" not in condition:
                # Further strip and clean each condition to remove leading/trailing spaces
                clean_condition = condition.strip()
                # Both following conditions are to handle parentheses in the condition and keep only the ones around the
                # comparison value
                # Remove the parentheses in front of the condition if present
                if clean_condition.startswith("("):
                    clean_condition = clean_condition[1:].strip()
                # Count the amount of opening and closing parentheses and remove the last one if it's not equal
                if clean_condition.count("(") != clean_condition.count(")"):
                    clean_condition = clean_condition[:-1]
                if (
                    "IS NULL" in clean_condition.upper()
                    or "IS NOT NULL" in clean_condition.upper()
                ):
                    continue
                if clean_condition:
                    conditions.append(clean_condition)

    if conditions:
        chosen_condition = random.choice(conditions)
        logging.debug(
            f"Randomly chosen condition: {chosen_condition} in the query: {query}"
        )
        if "LIMIT" in chosen_condition:
            # Handle LIMIT specially, there is no column
            limit_value = re.search(
                r"\bLIMIT\s+(\d+)", chosen_condition, flags=re.IGNORECASE
            ).group(1)
            post_query = query.replace(f"LIMIT {limit_value}", "LIMIT :payload")
            return ["LIMIT", limit_value, chosen_condition, post_query]

        # Extract column name and comparison value using regex
        match = re.match(
            r"\s*([\w.]+)\s*((?:NOT\s+)?(?:LIKE|IN)|[=!<>]{1,2})\s*(.*)",
            chosen_condition,
            flags=re.IGNORECASE,
        )
        if match:
            column_name = match.group(1)
            comparison_value = match.group(3).strip()
            comparison_value_clean = (
                match.group(3)
                .strip()
                .replace("(", "")
                .replace(")", "")
                .replace("'", "")
                .replace('"', "")
            )
            new_compare = comparison_value.replace(
                comparison_value_clean, ":payload"
            )
            new_condition = f"{column_name} {match.group(2)} {new_compare}"
            post_query = query.replace(chosen_condition, new_condition)
            return [column_name, comparison_value, chosen_condition, post_query]
        else:
            logging.error(
                "Error when extracting column and value from the condition."
            )
            return
    else:
        logging.error("No suitable condition found in the query.")
        return


def generate_default_queries_auth() -> List[str] or None:
    """
    Generate query SELECT username, password FROM customers WHERE username = payload1 AND password = payload2 LIMIT 1
    :return: List of queries
    """
    # Enter the username and password
    username_payload = add_quotes("payload1")
    password_payload = add_quotes("payload2")
    username_payload = add_parenthesis(username_payload)
    password_payload = add_parenthesis(password_payload)
    if random.choice([True, False]):
        # Add parentheses around the where
        query = (
            f"SELECT username, password FROM auth_bypass WHERE (username={username_payload}"
            f" AND password={password_payload}) LIMIT 1"
        )
    else:
        query = (
            f"SELECT username, password FROM auth_bypass WHERE username={username_payload}"
            f" AND password={password_payload} LIMIT 1"
        )
    return query


def generate_random_settings():
    """
    Generates random settings for the blacklist check.
    """
    blacklist = [
        "like",
        "select",
        "+",
        "union",
        "-",
        "%",
        " ",
        "/*",
        "*/",
        "=",
        ">",
        "<",
        "/",
        "*",
        "()",
        "(",
        ")",
        ",",
        ";",
        "#",
        "sleep",
        "\\",
        "between",
    ]
    check_count = random.randint(1, len(blacklist))
    items_to_check = random.sample(blacklist, check_count)
    checks = []
    for item in items_to_check:
        case_check = random.choice([1, 2, 3])
        checks.append((item, case_check))
    return checks


def add_limitations(payload: str, settings: List[Tuple[str, int]]) -> None:
    """
    Add limitations to the payload based on pre-determined session settings.
    :param payload: Payload to check
    :param settings: List of settings
    """
    if current_app.config["ENABLE_BLACKLIST"] == "True":
        logging.debug(f"Blacklisted item: {settings}")
        for item, case_check in settings:
            if case_check == 1 and item.lower() in payload.lower():
                raise ValueError("Payload contains a blacklisted element")
            elif case_check == 2 and item.upper() in payload.upper():
                raise ValueError("Payload contains a blacklisted element")
            elif case_check == 3 and (
                item in payload or item.lower() in payload.lower()
            ):
                raise ValueError("Payload contains a blacklisted element")
