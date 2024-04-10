import logging
import os
import random

import docker
from dotenv import load_dotenv
from flask import Flask, jsonify, Response

from challenges.generator import generate_challenges

app = Flask(__name__)
load_dotenv("../env")

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

client = docker.from_env()

# DBMS choice to generate the environment
dbms_images = ["mysql:latest", "postgres:latest"]


def start_db_instance(db_image: str) -> docker.models.containers.Container:
    container_name = "sqli-challenge-db"
    """Starts a new DB instance in a Docker container."""
    # Stops existing container named 'sqli-challenge-db' if it exists
    try:
        container = client.containers.get(container_name)
        container.stop()
        container.remove()
    except docker.errors.NotFound:
        pass

    if db_image == "mysql:latest":
        environment = [
            "MYSQL_ROOT_PASSWORD=password",
        ]
        ports = {"3306/tcp": 3306}
    elif db_image == "postgres:latest":
        environment = (
            [
                "POSTGRES_PASSWORD=password",
            ],
        )
        ports = {"5432/tcp": 5432}

    else:
        raise ValueError(f"Invalid DBMS image: {db_image}")
    logging.info(f"Starting database : {db_image} ...")
    container = client.containers.run(
        image=db_image,
        name=container_name,
        environment=environment,
        ports=ports,
        detach=True,
    )
    logging.info(f"Database started")
    return container


@app.route("/start", methods=["GET"])
def start() -> Response:
    """Endpoint to start the challenge environment."""
    selected_dbms = random.choice(dbms_images)
    # Start DB instance
    container = start_db_instance(selected_dbms)
    logging.info(f"Starting challenges generation ...")
    challenges = generate_challenges(nbr=10)
    logging.info(f"{len(challenges)} challenges generated")
    return jsonify(
        {
            "dbms": selected_dbms,
            "challenges": challenges,
            "container_id": container.id,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5959)
