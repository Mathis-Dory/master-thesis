from flask import Blueprint, abort, current_app, render_template

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/<int:challenge_number>", methods=["POST", "GET"])
def challenge(challenge_number: int) -> str:
    episodes = int(current_app.config["CHALLENGES_EPISODES"])
    templates = current_app.config["INIT_DATA"]["templates"]

    if challenge_number < 1 or challenge_number > episodes:
        abort(404)

    selected_template = templates[challenge_number - 1]
    return render_template(f"{selected_template}.html")
