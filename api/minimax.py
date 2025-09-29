import collections
import copy
from .battlesnake_utils import GameState, Coord, Snake, manhattan_distance

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
    # if so, check if the calculation for this specific start_coord has been done
    if obstacles in _flood_fill_cache and start_coord in _flood_fill_cache[obstacles]:
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

    start_coord, end_coord = state.my_head, state.my_tail

    queue = collections.deque([start_coord])
    visited = {start_coord}

    while queue:
        coord = queue.popleft()

        if coord == end_coord:
            return True # path found

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

    return False # no path found


def _get_score(state: GameState) -> float:
    """Calculates the score for the current game state using multiple weighted heuristics."""
    # heuristic 1: survival
    if not state.you:
        return -float('inf') # lost the game, this is the worst possible outcome

    # heuristic 2: space control (flood fill)
    available_space = _available_space(state, state.my_head)
    # a low available space count is a strong indicator of being trapped
    if available_space <= state.my_length:
        return -10000 # high penalty for being trapped

    # heuristic 3: length advantage
    opp_lengths = [opp.length for opp in state.opponents]
    longest_opponent = max(opp_lengths) if opp_lengths else 0
    length_advantage = state.my_length - longest_opponent

    # heuristic 4: food proximity
    food_score = 0.0
    closest_food = state.find_closest_food()
    if closest_food:
        dist_to_food = manhattan_distance(state.my_head, closest_food)
        food_proximity = (state.board_width + state.board_height - dist_to_food)
        # score is higher for closer food. normalized by board size
        # food is much more important at lower health
        food_score = food_proximity * 2.5 if state.my_health < 40 else food_proximity

    # heuristic 5: center control
    center_x, center_y = state.board_width // 2, state.board_height // 2
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
    if state.my_health > 50 and not _find_path_to_tail(state):
        # apply a massive penalty for any move that cuts off the path to the tail
        path_to_tail_penalty = -20000

    # weights for each heuristic
    W_SPACE, W_LENGTH, W_FOOD, W_CENTER, W_THREAT = 2.0, 5.0, 1.5, 0.5, -10.0
    final_score = (
            available_space * W_SPACE + length_advantage * W_LENGTH +
            food_score * W_FOOD + center_control_score * W_CENTER +
            threat_penalty * W_THREAT + path_to_tail_penalty
    )
    return float(final_score)


def _is_game_over(state: GameState) -> bool:
    """Checks if the game is over for our snake."""
    return not state.you


def _simulate_move(state: GameState, snake: Snake, move_coord: Coord) -> GameState:
    """Creates a new GameState object representing the board after a snake makes a move."""
    # simulate the move
    new_state_data = copy.deepcopy(state._state)
    snake_to_move = next((s for s in new_state_data['board']['snakes'] if s['id'] == snake.id), None)
    if not snake_to_move:
        return GameState(new_state_data)

    snake_to_move['head'] = move_coord.to_dict()
    snake_to_move['body'].insert(0, move_coord.to_dict())

    # check for food
    ate_food = False
    current_food = list(new_state_data['board']['food'])
    for i, food_pos in enumerate(current_food):
        if food_pos['x'] == move_coord.x and food_pos['y'] == move_coord.y:
            ate_food = True
            snake_to_move['health'] = 100
            snake_to_move['length'] += 1
            new_state_data['board']['food'].pop(i)
            break

    if not ate_food:
        snake_to_move['body'].pop()
        snake_to_move['health'] -= 1

    return GameState(new_state_data)


def _solo_minimax(state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
    """The recursive MiniMax function for a single player (solo mode)."""
    if depth == 0 or _is_game_over(state):
        return _get_score(state)

    my_moves = state._get_valid_moves(state.you)
    if not my_moves:
        return _get_score(state)

    if is_maximizing: # maximizing
        max_score = -float('inf')
        for move_coord in my_moves.values():
            child_state = _simulate_move(state, state.you, move_coord)
            score = _solo_minimax(child_state, depth - 1, alpha, beta, False)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha: break # pruning
        return max_score
    else: # minimizing
        min_score = float('inf')
        for move_coord in my_moves.values():
            child_state = _simulate_move(state, state.you, move_coord)
            score = _solo_minimax(child_state, depth - 1, alpha, beta, True)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha: break # pruning
        return min_score


def _multiplayer_minimax(state: GameState, depth: int, agent_index: int, alpha: float, beta: float) -> float:
    """The recursive MiniMax function for multiple players."""
    if _is_game_over(state):
        return _get_score(state)

    agents = [state.you] + state.opponents
    if agent_index >= len(agents): # end of a turn cycle
        return _multiplayer_minimax(state, depth - 1, 0, alpha, beta)

    if depth == 0:
        return _get_score(state)

    current_agent = agents[agent_index]
    possible_moves = state._get_valid_moves(current_agent)
    if not possible_moves: # if a snake has no moves, its turn is skipped
        return _multiplayer_minimax(state, depth, agent_index + 1, alpha, beta)

    if agent_index == 0: # maximizing player (me)
        max_score = -float('inf')
        for move_coord in possible_moves.values():
            child_state = _simulate_move(state, current_agent, move_coord)
            score = _multiplayer_minimax(child_state, depth, agent_index + 1, alpha, beta)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha: break # pruning
        return max_score
    else:  # minimizing player (opponent)
        min_score = float('inf')
        for move_coord in possible_moves.values():
            child_state = _simulate_move(state, current_agent, move_coord)
            score = _multiplayer_minimax(child_state, depth, agent_index + 1, alpha, beta)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha: break # pruning
        return min_score


def find_best_move(state: GameState, depth: int = 7) -> str:
    """Finds the best move using the appropriate MiniMax algorithm based on game mode."""
    # the cache is only valid for the duration of a single turn's calculation
    global _flood_fill_cache
    _flood_fill_cache.clear()

    my_moves = state._get_valid_moves(state.you)
    if not my_moves: return "down" # default move
    if len(my_moves) == 1: return list(my_moves.keys())[0]

    # move ordering: evaluate moves based on available space to improve pruning
    sorted_moves = []
    for move_name, move_coord in my_moves.items():
        temp_state = _simulate_move(state, state.you, move_coord)
        move_score = _available_space(temp_state, move_coord)
        sorted_moves.append({'name': move_name, 'coord': move_coord, 'score': move_score})
    # sort moves in descending order of their score (more space is better)
    sorted_moves.sort(key=lambda m: m['score'], reverse=True)

    best_score = -float('inf')
    best_move = sorted_moves[0]['name'] # default to the move with the most space

    # decide which MiniMax algorithm to use
    is_multiplayer = len(state.opponents) > 0

    # perform MiniMax search using the sorted move list
    for move in sorted_moves:
        hypothetical_state = _simulate_move(state, state.you, move['coord'])

        if is_multiplayer:
            score = _multiplayer_minimax(hypothetical_state, depth - 1, 1, -float('inf'), float('inf'))
        else:
            score = _solo_minimax(hypothetical_state, depth - 1, -float('inf'), float('inf'), False)

        if score > best_score:
            best_score = score
            best_move = move['name']

    return best_move
