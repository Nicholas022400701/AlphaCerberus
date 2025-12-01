import random
from ai.core import (
    EMPTY, BLACK, WHITE, N, K,
    normalize_board, check_winner_board,
    find_immediate_win
)


def count_dir(board, x, y, dx, dy, p):
    """
    Count consecutive stones of color p in a given direction
    starting from (x, y) and moving along direction (dx, dy).
    """
    c = 0
    i, j = x + dx, y + dy
    while 0 <= i < N and 0 <= j < N and board[i][j] == p:
        c += 1
        i += dx
        j += dy
    return c


def score_point(board, x, y, p):
    """
    Evaluate the heuristic score of placing stone p at (x, y).
    The scoring considers only the number of connected stones
    that this move would form.
    """
    if board[x][y] != EMPTY:
        return -10**9  # Illegal move position

    score = 0
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dx, dy in dirs:
        left = count_dir(board, x, y, -dx, -dy, p)
        right = count_dir(board, x, y, dx, dy, p)
        cnt = left + right + 1  # Total consecutive stones

        # Simple pattern scoring
        if cnt >= K:
            score += 1_000_000   # Winning move
        elif cnt == 4:
            score += 50000
        elif cnt == 3:
            score += 5000
        elif cnt == 2:
            score += 300
        elif cnt == 1:
            score += 20

    return score


def greedy_move(board, ai_stone):
    """
    Level 1 Greedy AI.
    - First priority: make the immediate winning move if available.
    - Otherwise: evaluate every empty position with a heuristic.
    - Special rule: Only the center point (Tengen) receives a huge
      positional bonus. Points around the center do not receive bonuses.
    """
    board = normalize_board(board)

    # 1) Check for immediate winning move
    win_move = find_immediate_win(board, ai_stone)
    if win_move is not None:
        return win_move

    best_score = -10**18
    best_move = None
    opp = -ai_stone

    CENTER = (N // 2, N // 2)
    CENTER_BONUS = 2_000_000  # Huge score only for the center

    for x in range(N):
        for y in range(N):
            if board[x][y] != EMPTY:
                continue

            # Offensive and defensive scores
            my_score = score_point(board, x, y, ai_stone)
            opp_score = score_point(board, x, y, opp)

            # Positional bonus: ONLY the center point is rewarded
            pos = CENTER_BONUS if (x, y) == CENTER else 0

            total_score = my_score * 1.1 + opp_score + pos

            if total_score > best_score:
                best_score = total_score
                best_move = (x, y)

    # Fallback in case no move is chosen (rare)
    if best_move is None:
        empties = [(i, j) for i in range(N) for j in range(N) if board[i][j] == EMPTY]
        return random.choice(empties) if empties else (0, 0)

    return best_move
