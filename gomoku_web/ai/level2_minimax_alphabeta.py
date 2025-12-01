from ai.core import (
    EMPTY, BLACK, WHITE, N,
    normalize_board, check_winner_board,
    generate_moves, find_immediate_win
)
from ai.level1_greedy import greedy_move


def evaluate_for(board, ai_stone):
    """
    Evaluate the entire board state: 
    Positive score favors ai_stone, negative score favors opponent
    
    Args:
        board: Current board state
        ai_stone: AI's stone color
    
    Returns:
        int: Evaluation score (positive favors AI, negative favors opponent)
    """
    score_ai = 0
    score_opp = 0
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    opp = -ai_stone

    for x in range(N):
        for y in range(N):
            p = board[x][y]
            if p == EMPTY:
                continue

            for dx, dy in dirs:
               # Ensure we only count each line once (starting from the "beginning")
                px, py = x - dx, y - dy
                if 0 <= px < N and 0 <= py < N and board[px][py] == p:
                    continue

                count = 0
                i, j = x, y
                while 0 <= i < N and 0 <= j < N and board[i][j] == p:
                    count += 1
                    i += dx
                    j += dy

                # Count consecutive stones in this direction
                open_ends = 0
                if 0 <= i < N and 0 <= j < N and board[i][j] == EMPTY:
                    open_ends += 1
                if 0 <= px < N and 0 <= py < N and board[px][py] == EMPTY:
                    open_ends += 1

                # Assign value based on line length and number of open ends
                if count >= 5:
                    val = 1_000_000
                elif count == 4:
                    val = 50000 if open_ends == 2 else 8000
                elif count == 3:
                    val = 3000 if open_ends == 2 else 400
                elif count == 2:
                    val = 200 if open_ends == 2 else 40
                else:
                    val = 5

                if p == ai_stone:
                    score_ai += val
                elif p == opp:
                    score_opp += val

    return score_ai - score_opp


def minimax(board, depth, alpha, beta, current_player, ai_stone, last_move):
    """
    Minimax search with alpha-beta pruning
    
    Args:
        board: Current board state
        depth: Search depth remaining
        alpha: Alpha value for alpha-beta pruning (best value for maximizing player)
        beta: Beta value for alpha-beta pruning (best value for minimizing player)
        current_player: Current player to move (ai_stone or -ai_stone)
        ai_stone: AI's stone color
        last_move: Last move made (x, y, player) for quick win check
    
    Returns:
        tuple: (evaluation_score, best_move)
    """
    # 1) Terminal node check: check if last move resulted in a win
    if last_move is not None:
        lx, ly, lp = last_move
        if check_winner_board(board, lx, ly, lp):
            if lp == ai_stone:
                return 1_000_000 - depth, None
            else:
                return -1_000_000 + depth, None

    # 2) Depth limit reached → return static evaluation
    if depth == 0:
        return evaluate_for(board, ai_stone), None

    # 3)Generate candidate moves
    moves = generate_moves(board)
    if not moves:
        return 0, None

    best_move = None

    # 4)  AI's turn (maximizing player)
    if current_player == ai_stone:
        value = -10**18
        for x, y in moves:
            board[x][y] = ai_stone
            child_val, _ = minimax(
                board, depth - 1, alpha, beta,
                -ai_stone, ai_stone, (x, y, ai_stone)
            )
            board[x][y] = EMPTY

            if child_val > value:
                value = child_val
                best_move = (x, y)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value, best_move

    # 5)Opponent's turn(minimizing player)
    else:
        opp = -ai_stone
        value = 10**18
        for x, y in moves:
            board[x][y] = opp
            child_val, _ = minimax(
                board, depth - 1, alpha, beta,
                ai_stone, ai_stone, (x, y, opp)
            )
            board[x][y] = EMPTY

            if child_val < value:
                value = child_val
                best_move = (x, y)

            beta = min(beta, value)
            if alpha >= beta:
                break

        return value, best_move


def alphabeta_move(board, ai_stone):
    """
     Level2 interface: Called by backend
    
    Strategy:
    - First check for immediate winning move
    - Otherwise use Minimax with alpha-beta pruning
    - Fallback to level1 greedy_move if no valid move found
    
    Args:
        board: Current board state
        ai_stone: AI's stone color
    
    Returns:
        tuple: Best move coordinates (x, y)
    """
    board = normalize_board(board)

    # 1) Priority: immediate winning move
    win = find_immediate_win(board, ai_stone)
    if win:
        return win

    # 2)  Minimax search with alpha-beta pruning
    _, mv = minimax(
        board,
        depth=2,                 # Currently set to 2, can be increased for higher levels
        alpha=-10**18,
        beta=10**18,
        current_player=ai_stone,
        ai_stone=ai_stone,
        last_move=None
    )

    # 3) Fallback: if no move found, use greedy algorithm
    if mv is None:
        return greedy_move(board, ai_stone)

    return mv
