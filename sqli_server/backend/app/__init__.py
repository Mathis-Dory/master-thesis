import logging
import random

import docker
from docker import DockerClient
from flask import Flask, current_app, jsonify

from app.challenges.generator import (
    generate_challenges,
    generate_random_settings,
)
from app.challenges.routes import challenges_bp
from app.config import Config
from challenges.models import populate_db
from .database import (
    configure_database_uri,
    wait_for_db,
    init_db,
    get_engine,
)


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

    logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
    logging.getLogger("faker").setLevel(logging.ERROR)
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    with app.app_context():
        init_data = app.config["INIT_DATA"] = initialize_environment()

        if not init_data:
            logging.error("Failed to initialize the challenge environment")
            return app

        dbms = init_data["DBMS"]
        db_uri = configure_database_uri(dbms)
        if not db_uri:
            logging.error("Failed to configure database URI")
            return app
        app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
        init_db()
        populate_db(templates=init_data["TEMPLATES"], flags=init_data["FLAGS"])
        app.config["INIT_DATA"]["LIMITATIONS"] = generate_random_settings()

    app.register_blueprint(challenges_bp, url_prefix="/challenge")

    @app.route("/", methods=["GET"])
    def root() -> jsonify:
        """
        Display the container information
        :return: JSON response
        """
        container_info = app.config["INIT_DATA"]
        return jsonify(container_info)

    @app.route("/reset", methods=["GET"])
    def reset():
        """
        Reset the challenge environment and restart a new db container.
        :return: JSON response
        """
        old_engine = get_engine()
        old_engine.dispose()
        init_data = app.config["INIT_DATA"]
        client = docker.from_env()
        container = client.containers.get(init_data["CONTAINER_ID"])
        container.stop()
        container.remove()
        logging.info("Database container stopped and removed")

        new_data = app.config["INIT_DATA"] = initialize_environment()
        if not new_data:
            logging.error("Failed to initialize the challenge environment")
            return jsonify({"message": "Failed to reset the environment"})

        dbms = new_data["DBMS"]
        db_uri = configure_database_uri(dbms)
        if not db_uri:
            logging.error("Failed to configure database URI")
            return jsonify({"message": "Failed to reset the environment"})

        app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
        init_db()
        populate_db(templates=new_data["TEMPLATES"], flags=new_data["FLAGS"])
        app.config["INIT_DATA"] = new_data
        app.config["INIT_DATA"]["LIMITATIONS"] = generate_random_settings()

        return jsonify({"message": "Environment reset successfully"})

    return app


def initialize_environment() -> dict or None:
    """
    Initialize the challenge environment.
    :param app: Flask application
    :return: Dictionary with the DBMS, archetypes types, and container ID
    """
    logging.info("Initializing the challenge environment...")
    client = docker.from_env()
    selected_dbms = random.choice(current_app.config["DBMS_IMAGES"])
    container = start_db_instance(client, selected_dbms)

    if selected_dbms == "mysql:latest":
        wait_for_db("sqli-challenge-db", 3306)
    elif selected_dbms == "postgres:latest":
        wait_for_db("sqli-challenge-db", 5432)
    else:
        logging.error(f"Unable to start DBMS: {selected_dbms}")
        return None
    archetypes, flags, templates = generate_challenges(
        nbr=current_app.config["CHALLENGES_EPISODES"]
    )
    logging.info(
        f"{len(archetypes)} challenges generated and database started with ID: {container.id}"
    )

    return {
        "DBMS": selected_dbms,
        "ARCHETYPES": archetypes,
        "FLAGS": flags,
        "TEMPLATES": templates,
        "CONTAINER_ID": container.id,
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
    if db_image not in current_app.config["DBMS_IMAGES"]:
        logging.error(f"Unsupported DBMS image: {db_image}")
        return None

    container_name = "sqli-challenge-db"
    try:
        container = client.containers.get(container_name)
        logging.info(
            f"Stopping and removing existing container: {container.id}"
        )
        container.stop()
        container.remove()
        logging.info("Removing unused images and volumes")
        client.images.prune()
        client.volumes.prune()
    except docker.errors.NotFound:
        logging.info("No existing container to remove.")

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
        logging.error(
            f"Can not assign environment variables for DBMS: {db_image}"
        )
        return None

    try:
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
    except Exception as e:
        logging.error(f"Failed to start database container: {e}")
        return None
