import logging
import random
from time import sleep
from typing import Any

from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import text, Result
from sqlalchemy.exc import SQLAlchemyError

from app import db
from challenges.generator import (
    add_limitations,
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
        decomposed_filter_queries = current_app.config.get(
            "DECOMPOSED_FILTER_QUERIES", []
        )
        get_query = queries[challenge_number - 1]
        [column, value, condition, post_query] = decomposed_filter_queries[
            challenge_number - 1
        ]
        # Used to determine if the condition is numerical one in order to display the correct label in template
        if ("<" or ">" or "<=" or ">=") in condition:
            numerical_condition = True
        else:
            numerical_condition = False

        if request.method == "POST":
            payload = request.form.get("payload", "")
            if "No vulnerabilities" not in sqli_archetype:
                post_query = post_query.replace(":payload", payload)
            try:
                add_limitations(
                    payload, current_app.config["INIT_DATA"]["LIMITATIONS"]
                )
                if "No vulnerabilities" in sqli_archetype:
                    result_proxy = db.session.execute(
                        text(post_query), {"payload": payload}
                    )
                    logging.info(f"Generated query: {post_query}")
                else:
                    result_proxy = db.session.execute(text(post_query))
                    logging.info(f"Generated query: {post_query}")
                return process_filer_response(
                    result_proxy=result_proxy,
                    selected_template=selected_template,
                    sqli_archetype=sqli_archetype,
                    column=column,
                    value=value,
                    numerical_condition=numerical_condition,
                )

            except SQLAlchemyError as e:
                error = str(e)
                logging.info(f"SQLAlchemyError: {error}")
                return select_correct_template(
                    template=selected_template,
                    sqli_archetype=sqli_archetype,
                    parameters={
                        "column": column,
                        "value": value,
                        "numerical_condition": numerical_condition,
                        "sql_error": (
                            error if "Errors" in sqli_archetype else None
                        ),
                    },
                )
            except ValueError as e:
                return select_correct_template(
                    template=selected_template,
                    sqli_archetype=sqli_archetype,
                    parameters={
                        "column": column,
                        "value": value,
                        "numerical_condition": numerical_condition,
                        "sql_error": e if "Errors" in sqli_archetype else None,
                    },
                )

        elif request.method == "GET":
            result_proxy = db.session.execute(text(get_query))
            return process_filer_response(
                result_proxy=result_proxy,
                selected_template=selected_template,
                sqli_archetype=sqli_archetype,
                column=column,
                value=value,
                numerical_condition=numerical_condition,
            )
        else:
            logging.error(f"Unknown method: {request.method}")
            abort(404)

    elif selected_template == "auth":
        if request.method == "GET":
            return select_correct_template(
                template=selected_template,
                sqli_archetype=sqli_archetype,
            )
        elif request.method == "POST":
            payload1 = request.form.get("username_payload")
            payload2 = request.form.get("password_payload")
            get_query = current_app.config.get("QUERIES")[challenge_number - 1]

            # If the challenge has vulnerabilities the payload is used to inject SQL else,
            # the bind parameters are used directly when calling the execute method
            if "No vulnerabilities" not in sqli_archetype:
                get_query = get_query.replace("payload1", payload1)
                get_query = get_query.replace("payload2", payload2)
            try:
                add_limitations(
                    payload1, current_app.config["INIT_DATA"]["LIMITATIONS"]
                )
                add_limitations(
                    payload2, current_app.config["INIT_DATA"]["LIMITATIONS"]
                )
                logging.info(f"Generated query: {get_query}")
                # If the challenge has no vulnerabilities, the bind parameters are used to avoid SQL injection
                if "No vulnerabilities" in sqli_archetype:
                    result_proxy = db.session.execute(
                        text(get_query),
                        {"username": payload1, "password": payload2},
                    )
                else:
                    result_proxy = db.session.execute(text(get_query))

                response = result_proxy.fetchone()
                if response:
                    user_dict = {
                        "auth": True,
                        "username": response[0],
                        "password": response[1],
                    }
                    value = user_dict
                else:
                    value = {"auth": False}

                return select_correct_template(
                    template=selected_template,
                    sqli_archetype=sqli_archetype,
                    parameters={"value": value},
                )
            except SQLAlchemyError as e:
                error = str(e)
                return select_correct_template(
                    template=selected_template,
                    sqli_archetype=sqli_archetype,
                    parameters={
                        "sql_error": (
                            error if "Errors" in sqli_archetype else None
                        ),
                    },
                )
            except ValueError as e:
                return select_correct_template(
                    template=selected_template,
                    sqli_archetype=sqli_archetype,
                    parameters={
                        "sql_error": e if "Errors" in sqli_archetype else None,
                    },
                )

        else:
            logging.error(f"Unknown method: {request.method}")
            abort(404)

    else:
        logging.error(f"Unknown template: {selected_template}")
        abort(404)


def select_correct_template(
    template: str,
    sqli_archetype: str,
    parameters: dict[str, bool | list[dict] | Any] = None,
) -> str:
    """
    Select the correct template based on the SQLi archetype and template
    :param template name
    :param sqli_archetype: SQLi archetype
    :param parameters: Parameters to pass to the template
    :return: Rendered template
    """
    if template == "filter":
        if "In-band" in sqli_archetype:
            template_name = "filter_inband"
        elif "Boolean-based" in sqli_archetype:
            template_name = "filter_boolean"

        elif "Time-based" in sqli_archetype:
            template_name = "filter_time"
            if parameters.get("items"):
                if len(parameters["items"]) > 0:
                    sleep(5)  # Add some delay to simulate a time-based attack

        elif "No vulnerabilities" in sqli_archetype:
            template_name = random.choice(
                ["filter_inband", "filter_boolean", "filter_time"]
            )
            if template_name == "filter_time":
                sleep(5)

        else:
            logging.error(
                f"Impossible to render a template for : {sqli_archetype}"
            )
            abort(404)

        if parameters.get("sql_error"):
            return render_template(
                f"filters/{template_name}.html",
                column=parameters["column"],
                value=parameters["value"],
                numerical_condition=parameters["numerical_condition"],
                sql_error=parameters.get("sql_error"),
            )
        else:
            return render_template(
                f"filters/{template_name}.html",
                items=parameters["items"],
                column=parameters["column"],
                value=parameters["value"],
                numerical_condition=parameters["numerical_condition"],
            )
    elif template == "auth":
        if "In-band" in sqli_archetype:
            template_name = "auth_inband"

        elif "Boolean-based" in sqli_archetype:
            template_name = "auth_boolean"

        elif "Time-based" in sqli_archetype:
            template_name = "auth_time"

        elif "No vulnerabilities" in sqli_archetype:
            template_name = random.choice(
                ["auth_inband", "auth_boolean", "auth_time"]
            )
        else:
            logging.error(
                f"Impossible to render a template for : {sqli_archetype}"
            )
            abort(404)

        if parameters is None:
            return render_template(f"auth/{template_name}.html")

        else:
            if parameters.get("sql_error"):
                return render_template(
                    f"auth/{template_name}.html",
                    sql_error=parameters.get("sql_error"),
                )
            else:
                return render_template(
                    f"auth/{template_name}.html", value=parameters["value"]
                )


def process_filer_response(
    result_proxy: Result,
    selected_template: str,
    sqli_archetype: str,
    column: str,
    value: str,
    numerical_condition: bool,
) -> str:
    """
    Process the filter response and return the correct template
    :param result_proxy: ResultProxy object
    :param selected_template: Selected template
    :param sqli_archetype: SQLi archetype
    :param column: Column name
    :param value: Value to filter
    :param numerical_condition: Numerical condition
    """
    response = result_proxy.fetchall()
    columns = result_proxy.keys()
    items = [{col: val for col, val in zip(columns, row)} for row in response]

    return select_correct_template(
        template=selected_template,
        sqli_archetype=sqli_archetype,
        parameters={
            "items": items,
            "column": column,
            "value": value,
            "numerical_condition": numerical_condition,
        },
    )
