import os

from challenges.models import (
    User,
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
    CHALLENGES_EPISODES = os.getenv("CHALLENGES_EPISODES", 10)
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    # Defined tables in the models.py file
    DEFINED_TABLES = [
        AuthBypass,
        User,
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
