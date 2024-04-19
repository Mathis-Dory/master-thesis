import logging

from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import db
from challenges.generator import generate_auth_queries, generate_filter_queries

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/<int:challenge_number>", methods=["POST", "GET"])
def challenge(challenge_number: int) -> str:
    episodes = int(current_app.config["CHALLENGES_EPISODES"])
    templates = current_app.config["INIT_DATA"]["templates"]

    if challenge_number < 1 or challenge_number > episodes:
        abort(404)
    selected_template = templates[challenge_number - 1]
    error = None
    result = None

    if request.method == "POST":
        username_payload = request.form.get("username_payload", "")
        password_payload = request.form.get("password_payload", "")
        try:
            if selected_template == "auth":
                query = generate_auth_queries(username_payload, password_payload)
                # send the payload using a randomly vulnerable query logic when POST
                result = db.session.execute(text(query))
                logging.debug(result)
            elif selected_template == "filter":
                query = generate_filter_queries()
                # send the payload using a randomly vulnerable query logic when POST
            else:
                logging.error(f"Unknown template: {selected_template}")
        except SQLAlchemyError as e:
            logging.error(str(e.__dict__["orig"]))

    return render_template(f"{selected_template}.html")
