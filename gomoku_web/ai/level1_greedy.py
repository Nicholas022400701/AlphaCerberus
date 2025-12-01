import random
from ai.core import (
    EMPTY, BLACK, WHITE, N, K,
    normalize_board, check_winner_board,
    find_immediate_win
)


def count_dir(board, x, y, dx, dy, p):
    """
    Count consecutive stones of color p in a given direction from position (x, y)
    
    Args:
        board: Current board state
        x, y: Starting position coordinates
        dx, dy: Direction vector to search
        p: Stone color to count
    
    Returns:
        int: Number of consecutive stones in the specified direction
    """
    c = 0
    i, j = x + dx, y + dy
    # Count consecutive stones in the specified direction
    while 0 <= i < N and 0 <= j < N and board[i][j] == p:
        c += 1
        i += dx
        j += dy
    return c


def score_point(board, x, y, p):
    """
    Calculate a simple heuristic score for placing stone p at position (x, y)
    (Only considers the consecutive stone patterns at this point)
    
    Args:
        board: Current board state
        x, y: Position to evaluate
        p: Stone color to evaluate for
    
    Returns:
        int: Heuristic score for the move
    """
    # If position is not empty, return very low score
    if board[x][y] != EMPTY:
        return -10**9

    score = 0
    # Four directions: horizontal, vertical, two diagonals
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in dirs:
        # Count consecutive stones in both directions
        left = count_dir(board, x, y, -dx, -dy, p)
        right = count_dir(board, x, y, dx, dy, p)
        cnt = left + right + 1  # Total consecutive stones including current move

        # Score based on the length of consecutive stones
        if cnt >= K:
            score += 1_000_000  # Winning move (K or more in a row)
        elif cnt == 4:
            score += 50000     # Four in a row (very strong)
        elif cnt == 3:
            score += 5000      # Three in a row (strong)
        elif cnt == 2:
            score += 300       # Two in a row (moderate)
        elif cnt == 1:
            score += 20        # Single stone (weak)

    return score


def greedy_move(board, ai_stone):
    """
    Level 1: Greedy AI
    - First priority: play immediate winning move if available
    - Otherwise: use heuristic scoring for each empty position
      - Own score multiplied by 1.1 (slight preference for offense)
      - Plus opponent's score (to consider defensive blocking)
    
    Args:
        board: Current board state
        ai_stone: AI's stone color (BLACK or WHITE)
    
    Returns:
        tuple: Best move coordinates (x, y)
    """
    # Ensure board is in proper 2D list format
    board = normalize_board(board)

    # 1) First check for immediate winning move
    win_move = find_immediate_win(board, ai_stone)
    if win_move is not None:
        return win_move

    # 2) If no immediate win, perform simple heuristic search
    best_score = -10**18  # Initialize with very low score
    best_move = None
    opp = -ai_stone  # Opponent's stone color

    # Evaluate all empty positions on the board
    for x in range(N):
        for y in range(N):
            if board[x][y] != EMPTY:
                continue

            # Calculate score for AI's move and opponent's potential response
            my_score = score_point(board, x, y, ai_stone)    # Offensive score
            opp_score = score_point(board, x, y, opp)        # Defensive score (blocking)
            total_score = my_score * 1.1 + opp_score         # Combined score with offensive bias

            # Update best move if current score is better
            if total_score > best_score:
                best_score = total_score
                best_move = (x, y)

    # 3) Fallback: if no move found (shouldn't happen), choose random empty position
    if best_move is None:
        empties = [(i, j) for i in range(N) for j in range(N) if board[i][j] == EMPTY]
        return random.choice(empties) if empties else (0, 0)

    return best_move