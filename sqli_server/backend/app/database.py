import logging

import psycopg2
import pymysql
import time
from flask import current_app
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

Base = declarative_base()


def get_engine() -> Engine:
    """
    Get the database engine.
    :return: Database engine
    """
    db_uri = current_app.config["SQLALCHEMY_DATABASE_URI"]
    engine = create_engine(
        db_uri,
        pool_size=200,
        max_overflow=0,
        pool_recycle=280,
        pool_pre_ping=True,
    )
    return engine


def get_session() -> Session:
    """
    Get the database session.
    :return: Database session
    """
    engine = get_engine()
    session = sessionmaker(bind=engine)
    return session()


def init_db() -> None:
    """
    Initialize the database.
    :return: None
    """
    engine = get_engine()
    Base.metadata.create_all(engine)


def configure_database_uri(dbms: str) -> str or None:
    """
    Configure the database URI based on the selected DBMS.
    :param dbms: DBMS image name
    :return: Database URI or None
    """
    db_password = current_app.config["DB_PASSWORD"]
    if dbms == "mysql:latest":
        return f"mysql+pymysql://root:{db_password}@sqli-challenge-db:3306/sqli_challenge"
    elif dbms == "postgres:latest":
        return f"postgresql://postgres:{db_password}@sqli-challenge-db:5432/sqli_challenge"
    else:
        logging.error(f"Error when choosing the DBMS URI for: {dbms}")
        return None


def wait_for_db(host: str, port: int) -> None:
    """
    Wait for database to become available to prevent URI connection failing.
    :param host: Database name
    :param port: Database port
    :return: None
    """
    if port == 3306:
        while True:
            try:
                conn = pymysql.connect(
                    host=host,
                    port=port,
                    user="root",
                    password=current_app.config["DB_PASSWORD"],
                )
                logging.info("Connected to MySQL DB!")
                conn.close()
                break
            except pymysql.err.OperationalError:
                logging.info("MySQL Database not ready, waiting ...")
                time.sleep(2)
    elif port == 5432:
        while True:
            try:
                conn = psycopg2.connect(
                    dbname="sqli_challenge",
                    user="postgres",
                    password=current_app.config["DB_PASSWORD"],
                    host=host,
                    port=port,
                )
                logging.info("Connected to PostgreSQL DB!")
                conn.close()
                break
            except psycopg2.OperationalError:
                logging.info("PostgreSQL Database not ready, waiting ...")
                time.sleep(2)
    else:
        logging.error(f"Can not connect to DBMS on port: {port}")
        return None
