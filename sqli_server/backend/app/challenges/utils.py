import logging
import re
from typing import Tuple, List, Any

from flask import current_app
from sqlalchemy import String, Text, inspect, Integer, Float

from app.database import db

# Define SQL functions and operators
NUMERIC_FUNCTIONS = ["MAX", "SUM", "AVG", "MIN", "COUNT"]
TEXT_FUNCTIONS = ["CONCAT", "SUBSTRING", "REPLACE"]
COMPARE_OPERATORS = {
    "numeric": ["=", "!=", ">", "<", ">=", "<="],
    "text": ["=", "!=", "LIKE", "NOT LIKE"],
    "null": ["IS NULL", "IS NOT NULL"],
}

LOGICAL_OPERATORS = ["AND", "OR"]
END_OPERATORS = ["ORDER BY", "LIMIT", "DISTINCT", "UNION"]


def find_columns_in_query(query: str) -> List[str]:
    """
    Extracts the column names or the first argument of SQL functions from the SELECT clause of a SQL query,
    handling nested functions and multiple layers of parentheses.
    :param query: SQL query string to parse.
    :return: List of column or expression names.
    """
    pattern = r"SELECT\s+(.*?)\s+FROM"
    match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)
    if match:
        columns_part = match.group(1)
        return parse_columns(columns_part)
    return []


def parse_columns(columns_part: str) -> List[str]:
    """
    Parses the SELECT clause to extract columns or the first argument from functions,
    accounting for nested parentheses and aliases.
    :param columns_part: The part of the query string that lists the columns.
    :return: A list of base column names or the first argument of functions.
    """
    columns = []
    current_col = []
    bracket_level = 0
    i = 0
    while i < len(columns_part):
        char = columns_part[i]
        if char == "," and bracket_level == 0:
            columns.append(extract_first_column("".join(current_col).strip()))
            current_col = []
        elif char == "(":
            bracket_level += 1
            current_col.append(char)
        elif char == ")":
            bracket_level -= 1
            current_col.append(char)
        else:
            current_col.append(char)
        i += 1

    if current_col:  # Add the last column if there's any residue
        columns.append(extract_first_column("".join(current_col).strip()))

    return columns


def extract_first_column(column: str) -> str:
    """
    Extracts the first column or argument from a given SQL expression, especially useful for SQL functions.
    :param column: The column expression or function from the query.
    :return: The base column name or the first argument inside the function.
    """
    # Remove nested functions and subqueries by focusing on the first complete segment inside any outermost parentheses
    if "(" in column:
        start = column.find("(") + 1
        end = start
        depth = 1
        while end < len(column) and depth > 0:
            if column[end] == "(":
                depth += 1
            elif column[end] == ")":
                depth -= 1
            end += 1
        return column[start : end - 1].split(",")[0].strip().split()[0]
    return column.split(",")[0].strip().split()[0]


def find_group_functions(query: str) -> bool:
    """
    Try to find if we need to add a GROUP BY clause in the query
    :param query: the actual query to test
    :return: boolean
    """
    functions_to_find = NUMERIC_FUNCTIONS + ["CONCAT"]
    pattern = r"(\b" + r"\b|\b".join(functions_to_find) + r"\b)\s*\(\s*([^)]+)\s*\)"

    # Find all occurrences of numerical functions in the query
    matches = re.findall(pattern, query)
    return bool(matches)


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


def exclude_tables(
    available_tables: List[Any], excluded_tables: List[str]
) -> List[Any]:
    """
    Exclude tables from the available tables
    :param available_tables: List of tables
    :param excluded_tables: List of table names to exclude
    :return: List of tables without the excluded ones
    """
    return [t for t in available_tables if t.__tablename__ not in excluded_tables]


def filter_flags(query, columns):
    """
    Appends WHERE clauses to the provided SQL query to filter out rows based on the specified columns containing
    the '%flag_challenge%' pattern. Adds additional logical connectors as needed.

    :param query: The base SQL query string to which conditions will be added.
    :param columns: A list of column names on which to apply the filtering conditions.
    :return: The modified query string with the appropriate WHERE or AND conditions added.
    """
    if not columns:
        return query  # No columns to filter, return the original query

    # Start building the condition string
    conditions = " AND ".join(f"{col} NOT LIKE '%flag_challenge%'" for col in columns)

    # Check if the query already contains a WHERE clause
    if " WHERE " in query:
        # Already has a WHERE clause, append with AND
        query += f" AND {conditions}"
    else:
        # No WHERE clause yet, start one
        query += f" WHERE {conditions}"

    return query + " AND "


def display_table(table: Any) -> str:
    """
    Display the table name with the correct syntax for the current DBMS
    :param table: Table object
    :return: Table name string
    """
    if current_app.config["INIT_DATA"]["dbms"] == "mysql:latest":
        return f"`{table.__tablename__}`"
    elif current_app.config["INIT_DATA"]["dbms"] == "postgres:latest":
        return f'"{table.__tablename__}"'
    else:
        logging.error("Unknown DBMS !")
