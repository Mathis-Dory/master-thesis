import logging
import random

import docker
from app.challenges.generator import (
    generate_challenges,
    generate_random_settings,
)
from app.challenges.routes import challenges_bp
from app.config import Config
from docker import DockerClient
from flask import Flask, current_app, jsonify

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

        # Stop and remove the existing container
        try:
            container = client.containers.get(init_data["CONTAINER_ID"])
            container.stop()
            container.remove()
            logging.info("Database container stopped and removed")
        except docker.errors.NotFound:
            logging.warning("No database container found to stop or remove.")
        except Exception as e:
            logging.error(f"Error stopping/removing container: {e}")

        # List and remove associated volumes
        try:
            volumes = client.volumes.list(
                filters={"label": "app=sqli_challenge"}
            )
            if volumes:
                for volume in volumes:
                    logging.info(
                        f"Found volume: {volume.name} with labels: {volume.attrs['Labels']}"
                    )
                    try:
                        volume.remove(force=True)
                        logging.info(f"Removed volume: {volume.name}")
                    except Exception as e:
                        logging.error(
                            f"Failed to remove volume {volume.name}: {e}"
                        )
            else:
                logging.info(
                    "No volumes found with label 'app=sqli_challenge'."
                )
        except Exception as e:
            logging.error(f"Error listing/removing volumes: {e}")

        # Remove dangling images based on label
        try:
            images = client.images.list(
                filters={"dangling": True, "label": "app=sqli_challenge"}
            )
            if images:
                for image in images:
                    logging.info(
                        f"Found dangling image: {image.id} with labels: {image.attrs['Labels']}"
                    )
                    try:
                        client.images.remove(image.id, force=True)
                        logging.info(f"Removed dangling image: {image.id}")
                    except Exception as e:
                        logging.error(
                            f"Failed to remove dangling image {image.id}: {e}"
                        )
            else:
                logging.info(
                    "No dangling images found with label 'app=sqli_challenge'."
                )
        except Exception as e:
            logging.error(f"Error listing/removing dangling images: {e}")

        # Restart the environment
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
    :return: Container object or None
    """
    if db_image not in current_app.config["DBMS_IMAGES"]:
        logging.error(f"Unsupported DBMS image: {db_image}")
        return None

    container_name = "sqli-challenge-db"
    volume_name = "sqli_challenge_volume"

    # Stop and remove the existing container, if any
    try:
        container = client.containers.get(container_name)
        logging.info(
            f"Stopping and removing existing container: {container.id}"
        )
        container.stop()
        container.remove()
    except docker.errors.NotFound:
        logging.info("No existing container to remove.")
    except Exception as e:
        logging.error(f"Error stopping/removing container: {e}")

    # Remove any old volumes with the specific label
    try:
        volumes = client.volumes.list(filters={"label": "app=sqli_challenge"})
        if volumes:
            for volume in volumes:
                logging.info(f"Removing old volume: {volume.name}")
                try:
                    volume.remove(force=True)
                    logging.info(f"Volume {volume.name} removed.")
                except Exception as e:
                    logging.error(f"Failed to remove volume {volume.name}: {e}")
        else:
            logging.info("No old volumes found to remove.")
    except Exception as e:
        logging.error(f"Error listing/removing volumes: {e}")

    # Create a new volume
    try:
        volume = client.volumes.create(
            name=volume_name,
            labels={"app": "sqli_challenge"},
        )
        logging.info(f"Created new volume: {volume.name}")
    except Exception as e:
        logging.error(f"Failed to create volume: {e}")
        return None

    if db_image == "mysql:latest":
        environment = {
            "MYSQL_ROOT_PASSWORD": current_app.config["DB_PASSWORD"],
            "MYSQL_DATABASE": "sqli_challenge",
        }
        ports = {"3306/tcp": 3306}
        volume_binding = {volume_name: {"bind": "/var/lib/mysql", "mode": "rw"}}
    elif db_image == "postgres:latest":
        environment = {
            "POSTGRES_PASSWORD": current_app.config["DB_PASSWORD"],
            "POSTGRES_DB": "sqli_challenge",
        }
        ports = {"5432/tcp": 5432}
        volume_binding = {
            volume_name: {"bind": "/var/lib/postgresql/data", "mode": "rw"}
        }
    else:
        logging.error(
            f"Cannot assign environment variables for DBMS: {db_image}"
        )
        return None

    try:
        logging.info(f"Starting database container with image: {db_image}")
        container = client.containers.run(
            image=db_image,
            name=container_name,
            environment=environment,
            ports=ports,
            detach=True,
            network="sqli_server_internal",
            volumes=volume_binding,
            labels={"app": "sqli_challenge"},
        )
        logging.info(f"Database container started with volume: {volume_name}")
        return container
    except Exception as e:
        logging.error(f"Failed to start database container: {e}")
        return None
