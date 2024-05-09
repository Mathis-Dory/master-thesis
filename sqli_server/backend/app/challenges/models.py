import logging
import random
from typing import List

from faker import Faker
from flask import current_app
from sqlalchemy import Column, String, Integer, Float, MetaData

from challenges.generator import (
    generate_default_queries,
)
from challenges.utils import log_challenges
from database import Base, get_session


class AuthBypass(Base):
    __tablename__ = "auth_bypass"
    id = Column(Integer, primary_key=True)
    username = Column(String(80), nullable=False)
    password = Column(String(80), nullable=False)


class Customers(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    username = Column(String(80), nullable=False)
    password = Column(String(80), nullable=False)
    age = Column(Integer, nullable=True)
    email = Column(String(80), nullable=True)


class Product(Base):
    __tablename__ = "product"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    price = Column(Float, nullable=True)
    description = Column(String(120), nullable=True)


class Order(Base):
    __tablename__ = "order"
    id = Column(Integer, primary_key=True)
    url = Column(String(120), nullable=False)
    product = Column(String(80), nullable=False)
    destination = Column(String(80), nullable=True)
    amount = Column(Integer, nullable=True)
    price = Column(Float, nullable=True)


class Comment(Base):
    __tablename__ = "comment"
    id = Column(Integer, primary_key=True)
    opinion = Column(String(120), nullable=True)
    username = Column(String(80), nullable=False)


class Grade(Base):
    __tablename__ = "grade"
    id = Column(Integer, primary_key=True)
    grade = Column(Float, nullable=False)
    student = Column(String(80), nullable=False)
    classroom = Column(Integer, nullable=True)


class Menu(Base):
    __tablename__ = "menu"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    price = Column(Float, nullable=False)
    description = Column(String(120), nullable=True)


class Car(Base):
    __tablename__ = "car"
    id = Column(Integer, primary_key=True)
    model = Column(String(80), nullable=True)
    brand = Column(String(80), nullable=False)
    year = Column(Integer, nullable=True)
    price = Column(Float, nullable=False)


class Movie(Base):
    __tablename__ = "movie"
    id = Column(Integer, primary_key=True)
    title = Column(String(80), nullable=False)
    year = Column(Integer, nullable=False)
    genre = Column(String(80), nullable=True)
    rating = Column(Float, nullable=False)


class Book(Base):
    __tablename__ = "book"
    id = Column(Integer, primary_key=True)
    title = Column(String(80), nullable=False)
    author = Column(String(80), nullable=False)
    year = Column(Integer, nullable=True)
    price = Column(Float, nullable=True)


class Music(Base):
    __tablename__ = "music"
    id = Column(Integer, primary_key=True)
    title = Column(String(80), nullable=False)
    artist = Column(String(80), nullable=False)
    year = Column(Integer, nullable=True)
    genre = Column(String(80), nullable=True)


class Equipment(Base):
    __tablename__ = "equipment"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    brand = Column(String(80), nullable=True)
    price = Column(Float, nullable=False)
    description = Column(String(120), nullable=True)


def create_selected_tables(selected_models):
    session = get_session()  # Get a new session
    meta = MetaData()
    for model in selected_models:
        model.__table__.tometadata(meta)
    meta.create_all(bind=session.bind)  # Use session's bind for engine


def populate_model(model: Base, num_entries: int, faker: Faker) -> None:
    null_frequency = current_app.config["NULL_FREQUENCY"]
    session = get_session()  # Get a new session
    for _ in range(num_entries):
        if model == Customers:
            entry = Customers(
                username=faker.user_name(),
                password=faker.password(),
                age=(
                    random.randint(1, 99)
                    if random.random() > null_frequency
                    else None
                ),
                email=(
                    faker.email() if random.random() > null_frequency else None
                ),
            )
        elif model == Product:
            entry = Product(
                name=faker.word(),
                price=(
                    random.uniform(1, 1000)
                    if random.random() > null_frequency
                    else None
                ),
                description=(
                    faker.sentence()
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == Order:
            entry = Order(
                url=faker.url(),
                product=faker.word(),
                destination=(
                    faker.city() if random.random() > null_frequency else None
                ),
                amount=(
                    random.randint(1, 100)
                    if random.random() > null_frequency
                    else None
                ),
                price=(
                    random.uniform(1, 1000)
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == Comment:
            entry = Comment(
                opinion=(
                    faker.sentence()
                    if random.random() > null_frequency
                    else None
                ),
                username=faker.user_name(),
            )
        elif model == Grade:
            entry = Grade(
                grade=random.uniform(0, 10),
                student=faker.user_name(),
                classroom=(
                    random.randint(1, 15)
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == Menu:
            entry = Menu(
                name=faker.word(),
                price=random.uniform(1, 100),
                description=(
                    faker.sentence()
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == Car:
            entry = Car(
                model=(
                    faker.word() if random.random() > null_frequency else None
                ),
                brand=faker.word(),
                year=(
                    random.randint(1, 2024)
                    if random.random() > null_frequency
                    else None
                ),
                price=random.uniform(1, 100000),
            )
        elif model == Movie:
            entry = Movie(
                title=faker.word(),
                year=random.randint(1, 2024),
                genre=(
                    faker.word() if random.random() > null_frequency else None
                ),
                rating=random.uniform(0, 10),
            )
        elif model == Book:
            entry = Book(
                title=faker.word(),
                author=faker.name(),
                year=(
                    random.randint(1, 2024)
                    if random.random() > null_frequency
                    else None
                ),
                price=(
                    random.uniform(1, 1000)
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == Music:
            entry = Music(
                title=faker.word(),
                artist=faker.name(),
                year=(
                    random.randint(1, 2024)
                    if random.random() > null_frequency
                    else None
                ),
                genre=(
                    faker.word() if random.random() > null_frequency else None
                ),
            )
        elif model == Equipment:
            entry = Equipment(
                name=faker.word(),
                brand=(
                    faker.word() if random.random() > null_frequency else None
                ),
                price=random.uniform(1, 1000),
                description=(
                    faker.sentence()
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == AuthBypass:
            continue
        else:
            logging.error(
                f"Error when faking data because of unknown model: {model}"
            )
            return
        session.add(entry)
    session.commit()


def populate_db(templates: List[str], flags: List[str]) -> None:
    """
    Populate the database with random data and flags.
    :param templates: List of templates
    :param flags: List of flags
    """
    faker = Faker(["en_US"])
    Faker.seed(current_app.config["SEED"])
    tables = [
        AuthBypass,
        Customers,
        Product,
        Order,
        Comment,
        Grade,
        Menu,
        Car,
        Movie,
        Book,
        Music,
        Equipment,
    ]
    indices_tables = list(
        range(2, len(tables))
    )  # Exclude the User table initially and the AuthBypass table
    n = random.randint(1, len(indices_tables))
    selected_indices = random.sample(indices_tables, n)
    # Add the Customers table and the AuthBypass table to the selected tables by default
    selected_tables = [tables[i] for i in selected_indices] + [
        Customers,
        AuthBypass,
    ]

    create_selected_tables(selected_tables)
    logging.debug(f"Tables created for {len(selected_tables)} models.")

    for model in selected_tables:
        num_entries = random.randint(500, 1000)
        populate_model(model, num_entries, faker)

    insert_flags(selected_tables, templates, flags)
    logging.info("Random data and flags inserted.")
    logging.info("Generating default queries ...")

    queries, decomposed_queries = generate_default_queries(
        created_tables=selected_tables
    )

    current_app.config["QUERIES"] = queries

    current_app.config["DECOMPOSED_FILTER_QUERIES"] = decomposed_queries

    log_challenges(templates=templates, queries=queries)


def insert_flags(
    selected_tables: List[Base], templates: List[str], flags: List[str]
) -> None:
    """
    Randomly insert flags into the database but if it is an auth challenge, insert the
    relevant flag to AuthBypass table.
    :param selected_tables: List of all tables
    :param templates: List of templates
    :param flags: List of flags to insert
    """
    logging.info(f"Inserting flags into the database...")
    session = get_session()
    for idx, flag in enumerate(flags):
        if flag is not None:
            if templates[idx] == "auth":
                entry = AuthBypass(username=Faker().name(), password=flag)
                session.add(entry)
            else:
                # Randomly select one of the selected table, then randomly take one of the string
                # column and update it by the flag but exclude the AuthBypass table
                tables_for_random_insertion = [
                    table
                    for table in selected_tables
                    if table.__name__ != "AuthBypass"
                ]
                table = random.choice(tables_for_random_insertion)
                columns = [
                    column
                    for column in table.__table__.columns
                    if isinstance(column.type, String)
                ]
                flag_inserted = False
                while not flag_inserted:
                    column = random.choice(columns)
                    rows = session.query(table).all()
                    if rows:
                        row = random.choice(rows)
                        # Check if the selected cell already contains a flag
                        if getattr(row, column.name) in flags:
                            logging.warning(
                                "Selected cell already contains a flag, retrying..."
                            )
                        else:
                            setattr(row, column.name, flag)
                            session.add(row)
                            flag_inserted = True
                    else:
                        logging.error("No rows in the table to put the flag")
                        break
    session.commit()

    logging.info("Flags inserted into the database.")
