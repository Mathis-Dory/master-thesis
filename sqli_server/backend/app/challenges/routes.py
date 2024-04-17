import random

from flask import Blueprint, request, render_template, abort, current_app

from .generator import TEMPLATES, generate_challenges

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/<int:challenge_number>", methods=["POST", "GET"])
def challenge(challenge_number: int) -> str:
    episodes = int(current_app.config["CHALLENGES_EPISODES"])
    if challenge_number < 1 or challenge_number > episodes:
        abort(404)

    index = challenge_number - 1
    vulns, flags = generate_challenges(nbr=episodes)
    template = random.choice(TEMPLATES)
    challenge_type = vulns[index]
    flag = flags[index]
    error = None

    if request.method == "POST":
        user_input = request.form["user_input"]
        result = "Simulated result based on query execution"
    else:
        query = None
        result = None

    return render_template(f"{template}.html", query=query, result=result, flag=flag, error=error)
