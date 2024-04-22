import logging

from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import db
from challenges.generator import (
    generate_auth_queries,
)

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/<int:challenge_number>", methods=["POST", "GET"])
def challenge(challenge_number: int) -> str:
    episodes = int(current_app.config["CHALLENGES_EPISODES"])
    templates = current_app.config["INIT_DATA"]["templates"]

    if challenge_number < 1 or challenge_number > episodes:
        abort(404)
    selected_template = templates[challenge_number - 1]
    logging.debug(f"Selected template: {selected_template}")
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
                pass
                # send the payload using a randomly vulnerable query logic when POST
            else:
                logging.error(f"Unknown template: {selected_template}")
        except SQLAlchemyError as e:
            logging.error(str(e.__dict__["orig"]))

    elif request.method == "GET":
        try:
            if selected_template == "filter":
                filter_queries = current_app.config.get("FILTER_QUERIES", {})
                logging.debug(f"Filter queries: {filter_queries}")
                query = filter_queries[challenge_number]
                result_proxy = db.session.execute(text(query))
                result = result_proxy.fetchall()
                # Get column names from the result set
                columns = result_proxy.keys()
                # Create a list of dictionaries for each row in the result set
                items = [{col: val for col, val in zip(columns, row)} for row in result]

        except SQLAlchemyError as e:
            logging.error(str(e.__dict__["orig"]))

    return render_template(f"{selected_template}.html", items=items)
