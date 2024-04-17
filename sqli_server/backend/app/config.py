import os


class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DBMS_IMAGES = ["postgres:latest"]
    CHALLENGES_EPISODES = os.getenv("CHALLENGES_EPISODES", 10)
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
