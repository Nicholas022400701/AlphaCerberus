# Define board state constants
EMPTY = 0    # Empty position
BLACK = 1    # Black stone
WHITE = -1   # White stone

# Board parameters
N = 15  # Board size 15x15
K = 5   # Number of consecutive stones needed to win


def normalize_board(board):
    """Ensure the board is in list of lists format (2D list)"""
    return [list(row) for row in board]


def check_winner_board(board, x, y, p):
    """
    Check if placing stone p at position (x, y) forms K consecutive stones (win condition)
    
    Args:
        board: Current board state
        x, y: Coordinates of the last move
        p: Stone color (BLACK or WHITE)
    
    Returns:
        bool: Whether K consecutive stones are formed
    """
    # Four directions to check: horizontal, vertical, main diagonal, anti-diagonal
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in dirs:
        c = 1  # Consecutive stone count, starts with 1 (current move)

        # Forward direction check
        i, j = x + dx, y + dy
        # Count consecutive stones in the positive direction
        while 0 <= i < N and 0 <= j < N and board[i][j] == p:
            c += 1
            i += dx
            j += dy

        # Backward direction check
        i, j = x - dx, y - dy
        # Count consecutive stones in the negative direction
        while 0 <= i < N and 0 <= j < N and board[i][j] == p:
            c += 1
            i -= dx
            j -= dy

        # If any direction has K or more consecutive stones, return win
        if c >= K:
            return True

    return False


def generate_moves(board):
    """
    Generate candidate moves based on current board state
    
    Strategy:
    - If board is empty, play at center
    - Otherwise, search for empty positions within 2 steps of existing stones
    
    Returns:
        list: List of candidate move coordinates
    """
    # Get all existing stone positions on the board
    stones = [(i, j) for i in range(N) for j in range(N) if board[i][j] != EMPTY]
    
    # If board is empty, return center position
    if not stones:
        c = N // 2
        return [(c, c)]

    # Use set to avoid duplicate candidate moves
    moves = set()
    # Search around each existing stone
    for x, y in stones:
        # Search within 2 steps radius
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                # Check if position is within board and empty
                if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == EMPTY:
                    moves.add((nx, ny))

    return list(moves)


def find_immediate_win(board, stone):
    """
    Check if there's an immediate winning move (checkmate move)
    
    Args:
        board: Current board state
        stone: Current player's stone color
    
    Returns:
        tuple: Winning move coordinates (x, y), or None if no immediate win exists
    """
    # Iterate through all board positions
    for x in range(N):
        for y in range(N):
            # Skip non-empty positions
            if board[x][y] != EMPTY:
                continue
                
            # Try placing stone at this position
            board[x][y] = stone
            # Check if this move wins the game
            if check_winner_board(board, x, y, stone):
                # Restore board state
                board[x][y] = EMPTY
                return x, y  # Return winning position
            # Restore board state
            board[x][y] = EMPTY
            
    return None  # No immediate winning move found