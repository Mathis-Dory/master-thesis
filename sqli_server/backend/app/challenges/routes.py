import logging

from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import text

from app import db
from challenges.generator import (
    generate_vulnerable_request,
)

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/<int:challenge_number>", methods=["POST", "GET"])
def challenge(challenge_number: int) -> str:
    episodes = int(current_app.config["CHALLENGES_EPISODES"])
    templates = current_app.config["INIT_DATA"]["templates"]
    challenges = current_app.config["INIT_DATA"]["challenges"]
    filter_queries = current_app.config.get("FILTER_QUERIES", {})
    extracted_queries = current_app.config.get("EXTRACTED_QUERIES", {})

    if challenge_number < 1 or challenge_number > episodes:
        abort(404)

    selected_template = templates[challenge_number - 1]
    logging.debug(f"Selected template: {selected_template}")

    sqli_archetype = challenges[challenge_number - 1]
    query = filter_queries[challenge_number]
    [column, value, condition] = extracted_queries[challenge_number - 1]
    if ("<" or ">" or "<=" or ">=") in condition:
        numerical_condition = True
    else:
        numerical_condition = False
    logging.debug(f"COLUMN: {column}, VALUE: {value}, CONDITION: {condition}")
    if request.method == "POST":
        payload = request.form.get("payload", "")
        if selected_template == "filter":
            query = generate_vulnerable_request(query, value, condition, payload)
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
            logging.error(f"Unknown template: {selected_template}")
            abort(404)

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
        abort(404)
