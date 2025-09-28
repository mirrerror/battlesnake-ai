import collections
import copy
from .battlesnake_utils import GameState, Coord, manhattan_distance

"""
Flood-Fill Caching
Cache for flood-fill results to avoid re-computation within the same turn.
The key is a frozenset of all obstacle coordinates (snake bodies).
The value is another dictionary mapping a start_coord to its calculated space.
This cache is cleared at the start of each 'find_best_move' call.
"""
_flood_fill_cache = {}


def _available_space(state: GameState, start_coord: Coord) -> int:
    """Calculates reachable squares using a flood-fill algorithm (BFS) with caching."""
    if not start_coord:
        return 0

    # create a hashable representation of the board's obstacles (all snake bodies) for caching
    # a frozenset is used because it's hashable and order-independent
    obstacles = frozenset(part for s in state.snakes for part in s.body)

    # check if this board state is already in the cache
    if obstacles in _flood_fill_cache:
        # if so, check if the calculation for this specific start_coord has been done
        if start_coord in _flood_fill_cache[obstacles]:
            return _flood_fill_cache[obstacles][start_coord]

    queue = collections.deque([start_coord])
    visited = {start_coord}
    count = 0
    while queue:
        coord = queue.popleft()
        count += 1
        for move in ["up", "down", "left", "right"]:
            next_coord_data = {}
            if move == "up":
                next_coord_data = {'x': coord.x, 'y': coord.y + 1}
            elif move == "down":
                next_coord_data = {'x': coord.x, 'y': coord.y - 1}
            elif move == "left":
                next_coord_data = {'x': coord.x - 1, 'y': coord.y}
            elif move == "right":
                next_coord_data = {'x': coord.x + 1, 'y': coord.y}

            next_coord = Coord(next_coord_data)

            # check safety against the current state's obstacles
            if state.is_safe(next_coord) and next_coord not in visited:
                visited.add(next_coord)
                queue.append(next_coord)

    # store the result in the cache before returning
    if obstacles not in _flood_fill_cache:
        _flood_fill_cache[obstacles] = {}
    _flood_fill_cache[obstacles][start_coord] = count

    return count

def _find_path_to_tail(state: GameState) -> bool:
    """
    Finds if a path exists from the snake's head to its tail using BFS.
    The tail square itself is considered a valid destination.
    """
    if not state.my_head or not state.my_tail:
        return False

    start_coord = state.my_head
    end_coord = state.my_tail

    queue = collections.deque([start_coord])
    visited = {start_coord}

    while queue:
        coord = queue.popleft()

        if coord == end_coord:
            return True  # path found

        for move in ["up", "down", "left", "right"]:
            next_coord_data = {}
            if move == "up":
                next_coord_data = {'x': coord.x, 'y': coord.y + 1}
            elif move == "down":
                next_coord_data = {'x': coord.x, 'y': coord.y - 1}
            elif move == "left":
                next_coord_data = {'x': coord.x - 1, 'y': coord.y}
            elif move == "right":
                next_coord_data = {'x': coord.x + 1, 'y': coord.y}

            next_coord = Coord(next_coord_data)

            # the key is to check if the next square is safe or if it's the tail
            # the tail is a valid target because it will be empty on the next turn
            if (state.is_safe(next_coord) or next_coord == end_coord) and next_coord not in visited:
                visited.add(next_coord)
                queue.append(next_coord)

    return False  # no path found

def _get_score(state: GameState) -> float:
    """
    Calculates the score for the current game state using multiple weighted heuristics.
    """
    # heuristic 1: survival
    if not state.you:
        return -float('inf')  # lost the game, this is the worst possible outcome

    # heuristic 2: space control (flood fill)
    available_space = _available_space(state, state.my_head)
    # a low available space count is a strong indicator of being trapped
    if available_space <= state.my_length:
        return -10000  # high penalty for being trapped

    # heuristic 3: length advantage
    opp_lengths = [opp.length for opp in state.opponents]
    longest_opponent = max(opp_lengths) if opp_lengths else 0
    length_advantage = state.my_length - longest_opponent

    # heuristic 4: food proximity
    food_score = 0.0
    closest_food = state.find_closest_food()
    if closest_food:
        dist_to_food = manhattan_distance(state.my_head, closest_food)
        # score is higher for closer food. normalized by board size
        food_proximity = (state.board_width + state.board_height - dist_to_food)
        # make food much more important at lower health
        if state.my_health < 40:
            food_score = food_proximity * 2.5
        else:
            food_score = food_proximity

    # heuristic 5: center control
    center_x = state.board_width // 2
    center_y = state.board_height // 2
    center_coord = Coord({'x': center_x, 'y': center_y})
    dist_to_center = manhattan_distance(state.my_head, center_coord)
    # score is higher for being closer to the center
    center_control_score = (state.board_width - dist_to_center)

    # heuristic 6: opponent head-to-head threat
    threat_penalty = 0
    for opponent in state.opponents:
        if opponent.length >= state.my_length:
            dist_to_opp_head = manhattan_distance(state.my_head, opponent.head)
            if dist_to_opp_head <= 2:
                 threat_penalty += (3 - dist_to_opp_head)

    # heuristic 7: path to tail
    path_to_tail_penalty = 0
    # this is crucial when not hungry to avoid self-trapping
    if state.my_health > 50:
        if not _find_path_to_tail(state):
            # apply a massive penalty for any move that cuts off the path to the tail
            path_to_tail_penalty = -20000

    # weights for each heuristic
    W_SPACE = 2.0
    W_LENGTH = 5.0
    W_FOOD = 1.5
    W_CENTER = 0.5
    W_THREAT = -10.0

    final_score = (
        available_space * W_SPACE +
        length_advantage * W_LENGTH +
        food_score * W_FOOD +
        center_control_score * W_CENTER +
        threat_penalty * W_THREAT +
        path_to_tail_penalty
    )

    return float(final_score)


def _is_game_over(state: GameState) -> bool:
    """Checks if the game is over for our snake."""
    return not state.you

def _minimax(state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
    """The recursive MiniMax function with Alpha-Beta Pruning."""
    if depth == 0 or _is_game_over(state):
        return _get_score(state)

    if is_maximizing:
        max_score = -float('inf')
        my_moves = state._get_valid_moves(state.you)
        if not my_moves: return _get_score(state)

        for move_name, move_coord in my_moves.items():
            hypothetical_state_data = copy.deepcopy(state._state)
            me_data = next(s for s in hypothetical_state_data['board']['snakes'] if s['id'] == state.my_id)
            me_data['body'].insert(0, move_coord.to_dict())
            me_data['head'] = move_coord.to_dict()
            me_data['body'].pop()

            child_state = GameState(hypothetical_state_data)
            score = _minimax(child_state, depth - 1, alpha, beta, False)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha: # pruning
                break
        return max_score
    else:  # minimizing player
        min_score = float('inf')
        my_moves = state._get_valid_moves(state.you)
        if not my_moves: return _get_score(state)

        for move_name, move_coord in my_moves.items():
            hypothetical_state_data = copy.deepcopy(state._state)
            me_data = next(s for s in hypothetical_state_data['board']['snakes'] if s['id'] == state.my_id)
            me_data['body'].insert(0, move_coord.to_dict())
            me_data['head'] = move_coord.to_dict()
            me_data['body'].pop()

            child_state = GameState(hypothetical_state_data)
            score = _minimax(child_state, depth - 1, alpha, beta, True)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha: # pruning
                break
        return min_score if min_score != float('inf') else _get_score(state)


def find_best_move(state: GameState, depth: int = 3) -> str:
    """
    Finds the best move using the MiniMax algorithm with alpha-beta pruning
    and move ordering. Returns the best move ("up", "down", "left", or "right").
    """
    # the cache is only valid for the duration of a single turn's calculation
    global _flood_fill_cache
    _flood_fill_cache.clear()

    best_score = -float('inf')
    best_move = "down"  # default move

    my_moves = state._get_valid_moves(state.you)
    if not my_moves:
        return best_move

    if len(my_moves) == 1:
        return list(my_moves.keys())[0]

    # move ordering
    # create a list of moves to be sorted based on available space
    sorted_moves = []
    for move_name, move_coord in my_moves.items():
        # create a quick hypothetical state to evaluate the move
        temp_state_data = copy.deepcopy(state._state)
        me_data = next(s for s in temp_state_data['board']['snakes'] if s['id'] == state.my_id)
        me_data['head'] = move_coord.to_dict()
        me_data['body'].insert(0, move_coord.to_dict())
        me_data['body'].pop()

        temp_state = GameState(temp_state_data)
        move_score = _available_space(temp_state, move_coord)
        sorted_moves.append({'name': move_name, 'coord': move_coord, 'score': move_score})

    # sort moves in descending order of their score (more space is better)
    sorted_moves.sort(key=lambda m: m['score'], reverse=True)

    # perform MiniMax search using the sorted move list
    for move in sorted_moves:
        move_name = move['name']
        move_coord = move['coord']

        # simulate our move
        hypothetical_state_data = copy.deepcopy(state._state)
        me_data = next(s for s in hypothetical_state_data['board']['snakes'] if s['id'] == state.my_id)
        me_data['body'].insert(0, move_coord.to_dict())
        me_data['head'] = move_coord.to_dict()

        # check for food
        ate_food = False
        for food_coord in state.food:
            if move_coord == food_coord:
                ate_food = True
                me_data['health'] = 100
                me_data['length'] += 1
                hypothetical_state_data['board']['food'].remove(food_coord.to_dict())
                break

        if not ate_food:
            me_data['body'].pop()

        hypothetical_state = GameState(hypothetical_state_data)

        score = _minimax(hypothetical_state, depth - 1, -float('inf'), float('inf'), False)

        if score > best_score:
            best_score = score
            best_move = move_name

    return best_move
