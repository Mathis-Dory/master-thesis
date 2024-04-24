import logging
import random
from typing import List

from faker import Faker
from flask import current_app
from sqlalchemy import MetaData

from challenges.generator import generate_filter_default_queries
from database import db


class AuthBypass(db.Model):
    __tablename__ = "auth_bypass"
    id = db.Column(db.Integer, primary_key=True)
    flag = db.Column(db.String(80), nullable=False)


class Customers(db.Model):
    __tablename__ = "customers"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    email = db.Column(db.String(80), nullable=True)


class Product(db.Model):
    __tablename__ = "product"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    price = db.Column(db.Float, nullable=True)
    description = db.Column(db.String(120), nullable=True)


class Order(db.Model):
    __tablename__ = "order"
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(120), nullable=False)
    product = db.Column(db.String(80), nullable=False)
    destination = db.Column(db.String(80), nullable=True)
    amount = db.Column(db.Integer, nullable=True)
    price = db.Column(db.Float, nullable=True)


class Comment(db.Model):
    __tablename__ = "comment"
    id = db.Column(db.Integer, primary_key=True)
    opinion = db.Column(db.String(120), nullable=True)
    username = db.Column(db.String(80), nullable=False)


class Grade(db.Model):
    __tablename__ = "grade"
    id = db.Column(db.Integer, primary_key=True)
    grade = db.Column(db.Float, nullable=False)
    student = db.Column(db.String(80), nullable=False)
    classroom = db.Column(db.Integer, nullable=True)


class Menu(db.Model):
    __tablename__ = "menu"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=True)


class Car(db.Model):
    __tablename__ = "car"
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(80), nullable=True)
    brand = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=True)
    price = db.Column(db.Float, nullable=False)


class Movie(db.Model):
    __tablename__ = "movie"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    genre = db.Column(db.String(80), nullable=True)
    rating = db.Column(db.Float, nullable=False)


class Book(db.Model):
    __tablename__ = "book"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    author = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=True)
    price = db.Column(db.Float, nullable=True)


class Music(db.Model):
    __tablename__ = "music"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    artist = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=True)
    genre = db.Column(db.String(80), nullable=True)


class Equipment(db.Model):
    __tablename__ = "equipment"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    brand = db.Column(db.String(80), nullable=True)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=True)


def create_selected_tables(selected_models: List[db.Model]) -> None:
    """Create tables for the selected models only."""
    # Create a new metadata instance
    meta = MetaData()
    # Bind models to the new metadata instance
    for model in selected_models:
        # Attach each table to the new metadata object
        model.__table__.tometadata(meta)
    # Create tables
    meta.create_all(bind=db.engine)


def populate_model(model: db.Model, num_entries: int, faker: Faker) -> None:
    null_frequency = current_app.config["NULL_FREQUENCY"]
    for _ in range(num_entries):
        if model == Customers:
            entry = Customers(
                username=faker.user_name(),
                password=faker.password(),
                age=(
                    random.randint(1, 99) if random.random() > null_frequency else None
                ),
                email=faker.email() if random.random() > null_frequency else None,
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
                    faker.sentence() if random.random() > null_frequency else None
                ),
            )
        elif model == Order:
            entry = Order(
                url=faker.url(),
                product=faker.word(),
                destination=faker.city() if random.random() > null_frequency else None,
                amount=(
                    random.randint(1, 100) if random.random() > null_frequency else None
                ),
                price=(
                    random.uniform(1, 1000)
                    if random.random() > null_frequency
                    else None
                ),
            )
        elif model == Comment:
            entry = Comment(
                opinion=faker.sentence() if random.random() > null_frequency else None,
                username=faker.user_name(),
            )
        elif model == Grade:
            entry = Grade(
                grade=random.uniform(0, 10),
                student=faker.user_name(),
                classroom=(
                    random.randint(1, 15) if random.random() > null_frequency else None
                ),
            )
        elif model == Menu:
            entry = Menu(
                name=faker.word(),
                price=random.uniform(1, 100),
                description=(
                    faker.sentence() if random.random() > null_frequency else None
                ),
            )
        elif model == Car:
            entry = Car(
                model=faker.word() if random.random() > null_frequency else None,
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
                genre=faker.word() if random.random() > null_frequency else None,
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
                genre=faker.word() if random.random() > null_frequency else None,
            )
        elif model == Equipment:
            entry = Equipment(
                name=faker.word(),
                brand=faker.word() if random.random() > null_frequency else None,
                price=random.uniform(1, 1000),
                description=(
                    faker.sentence() if random.random() > null_frequency else None
                ),
            )
        elif model == AuthBypass:
            continue
        else:
            logging.error(f"Error when faking data because of unknown model: {model}")
            return
        db.session.add(entry)
    db.session.commit()


def populate_db(templates: List[str], flags: List[str]) -> None:
    """Populate the database with dummy data for selected models."""
    faker = Faker(["en_US"])
    Faker.seed(current_app.config["SEED"])
    tables = current_app.config["DEFINED_TABLES"]
    indices_tables = list(
        range(2, len(tables))
    )  # Exclude the User table initially and the AuthBypass table
    n = random.randint(1, len(indices_tables))
    selected_indices = random.sample(indices_tables, n)
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
    filter_queries = generate_filter_default_queries(available_tables=selected_tables)
    current_app.config["FILTER_QUERIES"] = dict(
        zip(
            [i + 1 for i, template in enumerate(templates) if template == "filter"],
            filter_queries,
        )
    )


def insert_flags(
    selected_tables: List[db.Model], templates: List[str], flags: List[str]
) -> None:
    """
    Randomly insert flags into the database but if it is an auth challenge, insert the
    relevant flag to AuthBypass table.
    :param selected_tables: List of all tables
    :param templates: List of templates
    :param flags: List of flags to insert
    """
    logging.info(f"Inserting flags into the database...")
    for idx, flag in enumerate(flags):
        if flag is not None:
            if templates[idx] == "auth":
                entry = AuthBypass(flag=flag)
                db.session.add(entry)
            else:
                # Randomly select one of the selected table, then randomly take one of the string
                # column and update it by the flag but exclude the AuthBypass table
                tables_for_random_insertion = [
                    table for table in selected_tables if table.__name__ != "AuthBypass"
                ]
                table = random.choice(tables_for_random_insertion)
                columns = [
                    column
                    for column in table.__table__.columns
                    if isinstance(column.type, db.String)
                ]
                flag_inserted = False
                while not flag_inserted:
                    column = random.choice(columns)
                    rows = table.query.all()
                    if rows:
                        row = random.choice(rows)
                        # Check if the selected cell already contains a flag
                        if getattr(row, column.name) in flags:
                            logging.debug(
                                "Selected cell already contains a flag, retrying..."
                            )
                        else:
                            setattr(row, column.name, flag)
                            db.session.add(row)
                            flag_inserted = True
                    else:
                        logging.error("No rows in the table to put the flag")
                        break
    db.session.commit()

    logging.info("Flags inserted into the database.")
