import logging

from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import text

from app import db
from challenges.generator import (
    transform_get_to_post,
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
    filter_queries = current_app.config.get("FILTER_QUERIES", {})
    query = filter_queries[challenge_number]
    if request.method == "GET":
        if selected_template == "filter":
            result_proxy = db.session.execute(text(query))
            response = result_proxy.fetchall()
            # Get column names from the response set
            columns = result_proxy.keys()
            # Create a list of dictionaries for each row in the response set
            items = [{col: val for col, val in zip(columns, row)} for row in response]

            # Generate the input fields for the filter form
            input_fields = []
            for col in columns:
                if "price" in col:
                    input_fields.append(
                        {
                            "name": col,
                            "type": "number",
                            "placeholder": "Enter price range",
                            "attributes": {"min": 0, "step": 0.01},
                        }
                    )
        return render_template(f"{selected_template}.html", items=items)

    elif request.method == "POST":
        username_payload = request.form.get("username_payload", "")
        password_payload = request.form.get("password_payload", "")
        payload = request.form.get("payload", "")

        if selected_template == "auth":
            # send the payload using a randomly vulnerable query logic when POST
            pass
        elif selected_template == "filter":
            query = transform_get_to_post(query, payload)
            result_proxy = db.session.execute(text(query))
            response = result_proxy.fetchall()
            logging.debug(response)

        else:
            logging.error(f"Unknown template: {selected_template}")

    else:
        abort(404)
