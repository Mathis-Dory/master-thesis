import os

from challenges.models import (
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
    AuthBypass,
)


class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DBMS_IMAGES = ["mysql:latest", "postgres:latest"]
    CHALLENGES_EPISODES = int(os.getenv("CHALLENGES_EPISODES", 10))
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    # Defined tables in the models.py file
    DEFINED_TABLES = [
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
    SEED = os.getenv("SEED", 42)
    NULL_FREQUENCY = float(os.getenv("NULL_FREQUENCY", 0.1))
