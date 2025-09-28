"""
Provides a high-level API to interact with the Battlesnake game state.

This module contains classes that wrap the JSON game state object provided by
the Battlesnake engine.
"""

from __future__ import annotations
import typing


class Coord:
    """
    Represents a single coordinate on the game board.

    You can access its x and y values via properties.
    It supports equality checks with other Coord objects and with dictionaries.
    """
    def __init__(self, data: typing.Dict[str, int]):
        self._x = data['x']
        self._y = data['y']

    @property
    def x(self) -> int:
        """The x-coordinate."""
        return self._x

    @property
    def y(self) -> int:
        """The y-coordinate."""
        return self._y

    def __repr__(self) -> str:
        return f"Coord(x={self.x}, y={self.y})"

    def __eq__(self, other) -> bool:
        """Checks for equality with another Coord or a dict."""
        if isinstance(other, Coord):
            return self.x == other.x and self.y == other.y
        if isinstance(other, dict):
            return self.x == other.get('x') and self.y == other.get('y')
        return False

    def __hash__(self) -> int:
        """Allows Coord objects to be stored in sets and dictionary keys."""
        return hash((self.x, self.y))

    def to_dict(self) -> typing.Dict[str, int]:
        """Converts the Coord object back to a dictionary."""
        return {'x': self.x, 'y': self.y}


def manhattan_distance(p1: Coord, p2: Coord) -> int:
    """Calculates the Manhattan distance (number of moves on a grid) between two points."""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


class Snake:
    """Represents a snake in the game."""
    def __init__(self, snake_data: dict):
        self._data = snake_data
        self._body_coords = [Coord(part) for part in self._data['body']]

    @property
    def id(self) -> str:
        """The snake's unique ID."""
        return self._data['id']

    @property
    def name(self) -> str:
        """The snake's name."""
        return self._data['name']

    @property
    def health(self) -> int:
        """The snake's current health."""
        return self._data['health']

    @property
    def body(self) -> list[Coord]:
        """A list of the snake's body coordinates, from head to tail."""
        return self._body_coords

    @property
    def head(self) -> Coord:
        """The snake's head coordinate."""
        return self.body[0]

    @property
    def tail(self) -> Coord:
        """The snake's tail coordinate."""
        return self.body[-1]

    @property
    def length(self) -> int:
        """The snake's current length."""
        return self._data['length']

    def __repr__(self) -> str:
        return f"Snake(id={self.id}, name='{self.name}', length={self.length})"


class GameState:
    """A wrapper for the Battlesnake game state JSON object."""

    def __init__(self, game_state: dict):
        """Initializes the GameState object."""
        self._state = game_state
        self._board = self._state['board']

        self.food: list[Coord] = [Coord(f) for f in self._board['food']]
        self.hazards: list[Coord] = [Coord(h) for h in self._board['hazards']]
        self.snakes: list[Snake] = [Snake(s) for s in self._board['snakes']]

        # find your snake object from the list of all snakes
        my_id = self._state['you']['id']
        self.you: Snake = next((s for s in self.snakes if s.id == my_id), None)

        # pre-calculate all snake body positions for quick lookups
        self._all_snake_bodies = set()
        for snake in self.snakes:
            self._all_snake_bodies.update(snake.body)

    @property
    def game_id(self) -> str:
        """The unique ID for the current game."""
        return self._state['game']['id']

    @property
    def game_timeout(self) -> int:
        """The timeout for the game in milliseconds."""
        return self._state['game']['timeout']

    @property
    def turn(self) -> int:
        """The current turn number."""
        return self._state['turn']

    @property
    def board_width(self) -> int:
        """The width of the game board."""
        return self._board['width']

    @property
    def board_height(self) -> int:
        """The height of the game board."""
        return self._board['height']

    @property
    def my_id(self) -> str:
        """Your snake's unique ID."""
        return self.you.id if self.you else ""

    @property
    def my_name(self) -> str:
        """Your snake's name."""
        return self.you.name if self.you else ""

    @property
    def my_health(self) -> int:
        """Your snake's current health."""
        return self.you.health if self.you else 0

    @property
    def my_body(self) -> list[Coord]:
        """A list of your snake's body coordinates, from head to tail."""
        return self.you.body if self.you else []

    @property
    def my_head(self) -> Coord:
        """Your snake's head coordinate."""
        return self.you.head if self.you else None

    @property
    def my_tail(self) -> Coord:
        """Your snake's tail coordinate."""
        return self.you.tail if self.you else None

    @property
    def my_length(self) -> int:
        """Your snake's current length."""
        return self.you.length if self.you else 0

    @property
    def opponents(self) -> list[Snake]:
        """A list of all opponent snake objects on the board."""
        return [snake for snake in self.snakes if snake.id != self.my_id]

    def is_on_board(self, coord: Coord) -> bool:
        """Checks if a given coordinate is within the board boundaries."""
        return 0 <= coord.x < self.board_width and 0 <= coord.y < self.board_height

    def is_safe(self, coord: Coord) -> bool:
        """
        Checks if a given coordinate is safe to move to.
        A move is safe if it is on the board and not occupied by any snake's body.
        """
        if not self.is_on_board(coord):
            return False
        if coord in self._all_snake_bodies:
            return False
        return True

    def find_closest_food(self) -> typing.Optional[Coord]:
        """Finds the food closest to your snake's head."""
        if not self.food or not self.you:
            return None

        closest = min(
            self.food,
            key=lambda f: manhattan_distance(self.my_head, f)
        )
        return closest

    def _get_valid_moves(self, snake: Snake) -> typing.Dict[str, Coord]:
        """Returns a dictionary of valid moves for a given snake."""
        if not snake: return {}
        moves = {}
        head = snake.head
        potential_moves = {
            "up": Coord({'x': head.x, 'y': head.y + 1}),
            "down": Coord({'x': head.x, 'y': head.y - 1}),
            "left": Coord({'x': head.x - 1, 'y': head.y}),
            "right": Coord({'x': head.x + 1, 'y': head.y})
        }

        # don't move backwards
        if snake.length > 1:
            neck = snake.body[1]
            if potential_moves.get("up") == neck: del potential_moves["up"]
            if potential_moves.get("down") == neck: del potential_moves["down"]
            if potential_moves.get("left") == neck: del potential_moves["left"]
            if potential_moves.get("right") == neck: del potential_moves["right"]

        # only add moves that are safe
        for move, coord in potential_moves.items():
            if self.is_safe(coord):
                moves[move] = coord

        return moves