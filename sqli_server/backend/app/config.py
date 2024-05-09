import os


class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DBMS_IMAGES = ["mysql:latest"]
    CHALLENGES_EPISODES = int(os.getenv("CHALLENGES_EPISODES", 10))
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    SEED = os.getenv("SEED")
    NULL_FREQUENCY = float(os.getenv("NULL_FREQUENCY", 0.1))
