import random

from faker import Faker
from flask import current_app

from database import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False)
    password = db.Column(db.String(20), nullable=False)
    Age = db.Column(db.Integer, nullable=True)
    email = db.Column(db.String(50), unique=True, nullable=False)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=False)


class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(120), nullable=False)
    product = db.Column(db.String(50), nullable=False)
    destination = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)


class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(120), nullable=False)
    user = db.Column(db.String(30), nullable=False)


class Grade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    grade = db.Column(db.Float, nullable=False)
    student = db.Column(db.String(30), nullable=False)
    classroom = db.Column(db.Integer, nullable=False)


class Menu(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=False)


class Car(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(50), nullable=False)
    brand = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    genre = db.Column(db.String(50), nullable=False)
    rating = db.Column(db.Float, nullable=False)


class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), nullable=False)
    author = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)


class Music(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), nullable=False)
    artist = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    genre = db.Column(db.String(50), nullable=False)


class Equipment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    brand = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(120), nullable=False)


def populate_db() -> None:
    faker = Faker(["en_US"])
    Faker.seed(current_app.config["SEED"])
    tables = current_app.config["DEFINED_TABLES"]
    for table in tables:
        db.session.query(table).delete()

    # Create array with index from 1 to len(tables) excluding 0 because we select it by default later
    indices_tables = list(range(1, len(tables) - 1))
    # Choose number of tables to populate from 1 to len(tables) - 1 (excluding 0)
    n = random.randint(1, len(tables) - 1)
    # Select n tables randomly but excluding 0
    selected_tables = random.sample(indices_tables, n)
    # Add index 0 (User) to the selected tables
    selected_tables.append(0)

    for table_idx in selected_tables:
        data_amount = random.randint(50, 100)
        # Populate the selected tables
        if table_idx == 0:
            for _ in range(data_amount):
                user = User(
                    username=faker.user_name(),
                    password=faker.password(),
                    Age=random.randint(12, 99),
                    email=faker.email(),
                )
                db.session.add(user)
        elif table_idx == 1:
            for _ in range(data_amount):
                product = Product(
                    name=faker.word(),
                    price=random.uniform(1, 1000),
                    description=faker.sentence(),
                )
                db.session.add(product)
        elif table_idx == 2:
            for _ in range(data_amount):
                order = Order(
                    url=faker.url(),
                    product=faker.word(),
                    destination=faker.city(),
                    amount=random.randint(1, 100),
                    price=random.uniform(1, 1000),
                )
                db.session.add(order)
        elif table_idx == 3:
            for _ in range(data_amount):
                comment = Comment(text=faker.sentence(), user=faker.user_name())
                db.session.add(comment)
        elif table_idx == 4:
            for _ in range(data_amount):
                grade = Grade(
                    grade=random.uniform(0, 10),
                    student=faker.user_name(),
                    classroom=random.randint(1, 15),
                )
                db.session.add(grade)
        elif table_idx == 5:
            for _ in range(data_amount):
                menu = Menu(
                    name=faker.word(),
                    price=random.uniform(1, 100),
                    description=faker.sentence(),
                )
                db.session.add(menu)
        elif table_idx == 6:
            for _ in range(data_amount):
                car = Car(
                    model=faker.word(),
                    brand=faker.word(),
                    year=random.randint(1990, 2024),
                    price=random.uniform(1000, 100000),
                )
                db.session.add(car)
        elif table_idx == 7:
            for _ in range(data_amount):
                movie = Movie(
                    title=faker.word(),
                    year=random.randint(1900, 2024),
                    genre=faker.word(),
                    rating=random.uniform(0, 10),
                )
                db.session.add(movie)
        elif table_idx == 8:
            for _ in range(data_amount):
                book = Book(
                    title=faker.word(),
                    author=faker.name(),
                    year=random.randint(1900, 2024),
                    price=random.uniform(1, 1000),
                )
                db.session.add(book)
        elif table_idx == 9:
            for _ in range(data_amount):
                music = Music(
                    title=faker.word(),
                    artist=faker.name(),
                    year=random.randint(1900, 2024),
                    genre=faker.word(),
                )
                db.session.add(music)
        elif table_idx == 10:
            for _ in range(data_amount):
                equipment = Equipment(
                    name=faker.word(),
                    brand=faker.word(),
                    price=random.uniform(1, 1000),
                    description=faker.sentence(),
                )
                db.session.add(equipment)
    db.session.commit()
