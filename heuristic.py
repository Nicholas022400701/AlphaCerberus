# heuristic.py
import numpy as np
import random

# Pattern score table (heuristic scoring based on game theory)
# The logic here: Never miss any killing move, never overlook any defense
SCORE_MAP = {
    "WIN": 100000,      # Five in a row
    "BLOCK_WIN": 50000, # Block five in a row
    "OPEN_4": 10000,    # Open four
    "BLOCK_OPEN_4": 5000,
    "OPEN_3": 1000,     # Open three
    "BLOCK_OPEN_3": 500,
    "OPEN_2": 100,
    "OTHER": 0
}

class GreedyGomokuAgent:
    def __init__(self, board_size=15):
        self.board_size = board_size

    def select_move(self, board, current_player):
        """
        Input board: (15, 15) numpy array, 1=P1, -1=P2, 0=Empty
        Input current_player: 1 or -1
        """
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] == 0:
                    legal_moves.append((r, c))
        
        if not legal_moves: return None

        # 1. Calculate kill moves (find critical attack/defense points)
        # This is a simplified detection, only looks at current move
        best_score = -1
        best_moves = []

        for r, c in legal_moves:
            score = self._evaluate_move(board, r, c, current_player)
            if score > best_score:
                best_score = score
                best_moves = [(r, c)]
            elif score == best_score:
                best_moves.append((r, c))
        
        # Randomly select from highest scoring moves (to avoid complete determinism)
        if best_moves:
            move = random.choice(best_moves)
            return move[0] * self.board_size + move[1]
        
        # Fallback: random
        move = random.choice(legal_moves)
        return move[0] * self.board_size + move[1]

    def _evaluate_move(self, board, r, c, player):
        """ Evaluate the value of placing a stone at (r,c) """
        # Try placing stone at this point
        board[r, c] = player
        my_score = self._check_patterns(board, r, c, player)
        board[r, c] = 0 # Backtrack

        # Try simulating opponent's move at this point (defense score)
        opponent = -player
        board[r, c] = opponent
        opp_score = self._check_patterns(board, r, c, opponent)
        board[r, c] = 0 # Backtrack

        # Defense score slightly discounted, prioritize attack unless must defend
        if opp_score >= SCORE_MAP["WIN"]:
             # If opponent can win with this move, it's a must defend, extremely high score
            return SCORE_MAP["BLOCK_WIN"]
        
        # Combined score: attack + defense * 0.8
        return my_score + opp_score * 0.8

    def _check_patterns(self, board, r, c, player):
        """ Scan four directions, detect highest level pattern """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_score = 0

        for dr, dc in directions:
            line = []
            # Take line segment centered at (r,c) with radius 4
            for k in range(-4, 5):
                nr, nc = r + k*dr, c + k*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    val = board[nr, nc]
                    if val == player: line.append(1)
                    elif val == 0: line.append(0)
                    else: line.append(2) # Opponent
                else:
                    line.append(3) # Border (Wall)
            
            # Simple pattern matching
            s = "".join(map(str, line))
            
            # Five in a row
            if "11111" in s: return SCORE_MAP["WIN"]
            # Open four (011110)
            if "011110" in s: score = SCORE_MAP["OPEN_4"]
            # Blocked four (011112, 211110, 10111, etc) - simplified handling
            elif "1111" in s: score = SCORE_MAP["BLOCK_OPEN_4"] # Close to winning
            # Open three (01110, 010110)
            elif "01110" in s or "010110" in s: score = SCORE_MAP["OPEN_3"]
            # Open two
            elif "0110" in s: score = SCORE_MAP["OPEN_2"]
            else: score = 0
            
            max_score = max(max_score, score)
        
        return max_score