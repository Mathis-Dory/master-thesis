import logging
import random

import docker
from docker import DockerClient
from flask import Flask, jsonify, current_app

from app.challenges.generator import generate_challenges
from app.config import Config
from challenges.models import populate_db, db
from .database import configure_database_uri, wait_for_db


def create_app() -> Flask:
    """
    Create the Flask application.
    :return: Flask application
    """
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    random.seed(app.config["SEED"])

    # Setup logging based on configuration
    logging.basicConfig(
        level=app.config["LOG_LEVEL"],
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sqlalchemy_logger = logging.getLogger("sqlalchemy")
    sqlalchemy_logger.setLevel(logging.ERROR)
    faker_logger = logging.getLogger("faker")
    faker_logger.setLevel(logging.ERROR)

    with app.app_context():
        init_data = app.config["INIT_DATA"] = initialize_environment(app)

        if not init_data:
            logging.error("Failed to initialize the challenge environment")
            return app

        app.config.update(init_data)
        dbms = init_data["dbms"]
        db_uri = configure_database_uri(dbms)
        if not db_uri:
            logging.error("Failed to configure database URI")
            return app
        app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
        db.init_app(app)
        populate_db(templates=init_data["templates"], flags=init_data["flags"])

    @app.route("/", methods=["GET"])
    def root() -> jsonify:
        """
        Display the container information
        :return: JSON response
        """
        container_info = app.config["INIT_DATA"]
        return jsonify(container_info)

    from app.challenges.routes import challenges_bp

    app.register_blueprint(challenges_bp, url_prefix="/challenge")

    return app


def initialize_environment(app: Flask) -> dict or None:
    """
    Initialize the challenge environment.
    :param app: Flask application
    :return: Dictionary with the DBMS, challenges types, and container ID
    """
    logging.info("Initializing the challenge environment...")
    client = docker.from_env()
    selected_dbms = random.choice(app.config["DBMS_IMAGES"])
    container = start_db_instance(client, selected_dbms)

    if selected_dbms == "mysql:latest":
        wait_for_db("sqli-challenge-db", 3306)
    elif selected_dbms == "postgres:latest":
        wait_for_db("sqli-challenge-db", 5432)
    else:
        logging.error(f"Unable to start DBMS: {selected_dbms}")
        return None
    challenges, flags, templates = generate_challenges(
        nbr=app.config["CHALLENGES_EPISODES"]
    )
    logging.info(
        f"{len(challenges)} challenges generated and database started with ID: {container.id}"
    )

    return {
        "dbms": selected_dbms,
        "challenges": challenges,
        "flags": flags,
        "templates": templates,
        "container_id": container.id,
    }


def start_db_instance(
    client: DockerClient, db_image: str
) -> docker.models.containers.Container or None:
    """
    Starts a new DB instance in a Docker container.
    :param client: Docker client
    :param db_image: Docker image to use
    :return Container object or None
    """
    container_name = "sqli-challenge-db"
    try:
        container = client.containers.get(container_name)
        container.stop()
        container.remove()
    except docker.errors.NotFound:
        pass

    if db_image == "mysql:latest":
        environment = {
            "MYSQL_ROOT_PASSWORD": current_app.config["DB_PASSWORD"],
            "MYSQL_DATABASE": "sqli_challenge",
        }
        ports = {"3306/tcp": 3306}
    elif db_image == "postgres:latest":
        environment = {
            "POSTGRES_PASSWORD": current_app.config["DB_PASSWORD"],
            "POSTGRES_DB": "sqli_challenge",
        }
        ports = {"5432/tcp": 5432}

    else:
        logging.error(f"Can not assign environment variables for DBMS: {db_image}")
        return None

    logging.info(f"Starting database: {db_image} ...")
    container = client.containers.run(
        image=db_image,
        name=container_name,
        environment=environment,
        ports=ports,
        detach=True,
        network="sqli_server_internal",
    )
    container.reload()
    logging.info("Database started")
    return container
