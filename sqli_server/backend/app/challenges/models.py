import random

from faker import Faker

from database import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    price = db.Column(db.Float, nullable=False)


def populate_db():
    faker = Faker()
    db.session.query(User).delete()
    db.session.query(Product).delete()

    # Generate random users and products
    for _ in range(50):
        user = User(username=faker.user_name(), email=faker.email())
        product = Product(name=faker.word(), price=random.uniform(1.0, 1000.0))
        db.session.add(user)
        db.session.add(product)

    db.session.commit()
