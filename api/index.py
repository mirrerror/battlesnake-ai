import typing
from .battlesnake_utils import GameState
from .minimax import find_best_move
import logging
from flask import Flask, request

app = Flask("Battlesnake")


def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "pascommeilfaut",
        "color": "#7F00FF",
        "head": "safe",
        "tail": "round-bum",
    }


def start(game_state: typing.Dict):
    print(f"GAME START\n")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    state = GameState(game_state)
    next_move = find_best_move(state)
    print(f"MOVE {state.turn}: My name is {state.my_name}. I will move {next_move}.")
    return {"move": next_move}


@app.route("/")
def on_info():
    return info()


@app.route("/start", methods=["POST"])
def on_start():
    game_state = request.get_json()
    start(game_state)
    return "ok"


@app.route("/move", methods=["POST"])
def on_move():
    game_state = request.get_json()
    return move(game_state)


@app.route("/end", methods=["POST"])
def on_end():
    game_state = request.get_json()
    end(game_state)
    return "ok"


@app.after_request
def identify_server(response):
    response.headers.set(
        "server", "battlesnake/github/starter-snake-python"
    )
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Battlesnake Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the Battlesnake server on."
    )
    args = parser.parse_args()

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    port = args.port
    print(f"\nRunning Battlesnake server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)