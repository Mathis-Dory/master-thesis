import logging
import random
from typing import List, Type

from faker import Faker
from flask import current_app
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

from database import db

Base: Type[declarative_base()] = db.Model


class AuthBypass(db.Model):
    __tablename__ = "auth_bypass"
    id = db.Column(db.Integer, primary_key=True)
    flag = db.Column(db.String(80), nullable=False)


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    email = db.Column(db.String(80), unique=True, nullable=False)


class Product(db.Model):
    __tablename__ = "product"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=False)


class Order(db.Model):
    __tablename__ = "order"
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(120), nullable=False)
    product = db.Column(db.String(80), nullable=False)
    destination = db.Column(db.String(80), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)


class Comment(db.Model):
    __tablename__ = "comment"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(120), nullable=False)
    user = db.Column(db.String(80), nullable=False)


class Grade(db.Model):
    __tablename__ = "grade"
    id = db.Column(db.Integer, primary_key=True)
    grade = db.Column(db.Float, nullable=False)
    student = db.Column(db.String(80), nullable=False)
    classroom = db.Column(db.Integer, nullable=False)


class Menu(db.Model):
    __tablename__ = "menu"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=False)


class Car(db.Model):
    __tablename__ = "car"
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(80), nullable=False)
    brand = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)


class Movie(db.Model):
    __tablename__ = "movie"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    genre = db.Column(db.String(80), nullable=False)
    rating = db.Column(db.Float, nullable=False)


class Book(db.Model):
    __tablename__ = "book"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    author = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)


class Music(db.Model):
    __tablename__ = "music"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    artist = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    genre = db.Column(db.String(80), nullable=False)


class Equipment(db.Model):
    __tablename__ = "equipment"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    brand = db.Column(db.String(80), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=False)


def create_selected_tables(selected_models: list[Type[Base]]) -> None:
    """Create tables for the selected models only."""
    # Create a new metadata instance
    meta = MetaData()
    # Bind models to the new metadata instance
    for model in selected_models:
        # Attach each table to the new metadata object
        model.__table__.tometadata(meta)
    # Create tables
    meta.create_all(bind=db.engine)


def populate_model(model: Type[Base], num_entries: int, faker: Faker) -> None:
    for _ in range(num_entries):
        if model == User:
            entry = User(
                username=faker.user_name(),
                password=faker.password(),
                age=random.randint(12, 99),
                email=faker.email(),
            )
        elif model == Product:
            entry = Product(
                name=faker.word(), price=random.uniform(1, 1000), description=faker.sentence()
            )
        elif model == Order:
            entry = Order(
                url=faker.url(),
                product=faker.word(),
                destination=faker.city(),
                amount=random.randint(1, 100),
                price=random.uniform(1, 1000),
            )
        elif model == Comment:
            entry = Comment(text=faker.sentence(), user=faker.user_name())
        elif model == Grade:
            entry = Grade(
                grade=random.uniform(0, 10),
                student=faker.user_name(),
                classroom=random.randint(1, 15),
            )
        elif model == Menu:
            entry = Menu(
                name=faker.word(), price=random.uniform(1, 100), description=faker.sentence()
            )
        elif model == Car:
            entry = Car(
                model=faker.word(),
                brand=faker.word(),
                year=random.randint(1990, 2024),
                price=random.uniform(1000, 100000),
            )
        elif model == Movie:
            entry = Movie(
                title=faker.word(),
                year=random.randint(1900, 2024),
                genre=faker.word(),
                rating=random.uniform(0, 10),
            )
        elif model == Book:
            entry = Book(
                title=faker.word(),
                author=faker.name(),
                year=random.randint(1900, 2024),
                price=random.uniform(1, 1000),
            )
        elif model == Music:
            entry = Music(
                title=faker.word(),
                artist=faker.name(),
                year=random.randint(1900, 2024),
                genre=faker.word(),
            )
        elif model == Equipment:
            entry = Equipment(
                name=faker.word(),
                brand=faker.word(),
                price=random.uniform(1, 1000),
                description=faker.sentence(),
            )
        else:
            logging.error(f"Error when faking data because of unknown model: {model}")
            return
        db.session.add(entry)


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
    selected_models = [tables[i] for i in selected_indices] + [User, AuthBypass]

    create_selected_tables(selected_models)
    logging.debug(f"Tables created for {len(selected_models)} models.")

    for model in selected_models:
        num_entries = random.randint(30, 100)
        populate_model(model, num_entries, faker)
        logging.debug(f"Populated {model.__name__} with {num_entries} entries.")

    db.session.commit()
    insert_flags(selected_models, templates, flags)
    logging.info("Database population complete and flags inserted.")


def insert_flags(selected_tables: List, templates: List[str], flags: List[str]) -> None:
    """
    Randomly insert flags into the database but if it is an auth challenge, insert the
    relevant flag to AuthBypasss table.
    :param selected_tables: List of all tables
    :param templates: List of templates
    :param flags: List of flags to insert
    """
    logging.info(f"Inserting flags into the database... {flags}")
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
                            logging.debug("Selected cell already contains a flag, retrying...")
                        else:
                            setattr(row, column.name, flag)
                            db.session.add(row)
                            flag_inserted = True
                    else:
                        logging.error("No rows in the table to put the flag")
                        break
    db.session.commit()

    logging.info("Flags inserted into the database.")
