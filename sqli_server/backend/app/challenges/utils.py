import re
from typing import Tuple, List, Any

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
END_OPERATORS = ["HAVING", "ORDER BY", "LIMIT", "DISTINCT", "UNION"]


def find_columns_in_query(query: str) -> List[str]:
    """
    Find all columns in a query
    :param query:  to parse
    :return: List of columns
    """
    pattern = r"(?<=SELECT\s)(.*?)(?=\sFROM)"
    matches = re.search(pattern, query)
    if matches:
        return [col.strip() for col in matches.group(0).split(",")]
    return []


def find_numerical_functions(query: str) -> bool:
    """
    Try to find if the query contains a numerical function
    :param query: the actual query to test
    :return: boolean
    """
    pattern = r"(\b" + r"\b|\b".join(NUMERIC_FUNCTIONS) + r"\b)\s*\(\s*([^)]+)\s*\)"

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
