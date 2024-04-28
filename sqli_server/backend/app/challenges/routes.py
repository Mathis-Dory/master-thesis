import logging

from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import text

from app import db
from challenges.generator import (
    regenerate_filter_with_payload,
)

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/<int:challenge_number>", methods=["POST", "GET"])
def challenge(challenge_number: int) -> str:
    """
    Display the challenge page based on the challenge number.
    :param challenge_number: Challenge number
    :return: Rendered template
    """

    episodes = int(current_app.config["CHALLENGES_EPISODES"])
    templates = current_app.config["INIT_DATA"]["TEMPLATES"]
    challenges = current_app.config["INIT_DATA"]["ARCHETYPES"]

    if challenge_number < 1 or challenge_number > episodes:
        abort(404)

    selected_template = templates[challenge_number - 1]
    sqli_archetype = challenges[challenge_number - 1]

    if selected_template == "filter":
        queries = current_app.config.get("QUERIES")
        extracted_filter_queries = current_app.config.get(
            "EXTRACTED_FILTER_QUERIES", {}
        )
        query = queries[challenge_number - 1]
        [column, value, condition] = extracted_filter_queries[challenge_number - 1]
        if ("<" or ">" or "<=" or ">=") in condition:
            numerical_condition = True
        else:
            numerical_condition = False

        if request.method == "POST":
            payload = request.form.get("payload", "")
            query = regenerate_filter_with_payload(
                query=query,
                value=value,
                condition=condition,
                payload=payload,
            )
            result_proxy = db.session.execute(text(query))
            response = result_proxy.fetchall()
            columns = result_proxy.keys()
            items = [{col: val for col, val in zip(columns, row)} for row in response]
            return render_template(
                f"{selected_template}.html",
                items=items,
                column=column,
                value=value,
                numerical_condition=numerical_condition,
            )

        elif request.method == "GET":
            result_proxy = db.session.execute(text(query))
            response = result_proxy.fetchall()
            columns = result_proxy.keys()
            items = [{col: val for col, val in zip(columns, row)} for row in response]
            return render_template(
                f"{selected_template}.html",
                items=items,
                column=column,
                value=value,
                numerical_condition=numerical_condition,
            )
        else:
            logging.error(f"Unknown method: {request.method}")
            abort(404)

    elif selected_template == "auth":
        if request.method == "GET":
            return render_template(f"{selected_template}.html")
        elif request.method == "POST":
            pass
        else:
            logging.error(f"Unknown method: {request.method}")
            abort(404)

    else:
        logging.error(f"Unknown template: {selected_template}")
        abort(404)
