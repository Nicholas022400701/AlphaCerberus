# game.py
import numpy as np
from numba import int8, boolean, int32, void, njit
from numba.experimental import jitclass
import logging
from config import CONF

P1 = 1; P2 = -1; EMPTY = 0

# =============================================================================
#  STATIC HELPERS (Numba Optimized)
# =============================================================================

@njit
def _check_exact_win_static(board, r, c, p, board_size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for i in range(4):
        dx, dy = directions[i]
        count = 1
        for k in range(1, 6):
            nr, nc = r + k*dx, c + k*dy
            if 0<=nr<board_size and 0<=nc<board_size and board[nr, nc]==p: count+=1
            else: break
        for k in range(1, 6):
            nr, nc = r - k*dx, c - k*dy
            if 0<=nr<board_size and 0<=nc<board_size and board[nr, nc]==p: count+=1
            else: break
        if count >= 5: return True
    return False

@njit
def _check_four_or_five_static(board, r, c, p, board_size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for i in range(4):
        dx, dy = directions[i]
        count = 1
        for k in range(1, 5):
            nr, nc = r + k*dx, c + k*dy
            if 0<=nr<board_size and 0<=nc<board_size and board[nr, nc]==p: count+=1
            else: break
        for k in range(1, 5):
            nr, nc = r - k*dx, c - k*dy
            if 0<=nr<board_size and 0<=nc<board_size and board[nr, nc]==p: count+=1
            else: break
        if count >= 4: return True
    return False

@njit
def _check_open_three_static(board, r, c, p, board_size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for i in range(4):
        dx, dy = directions[i]
        count = 1; open_ends = 0
        for k in range(1, 5):
            nr, nc = r+k*dx, c+k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        for k in range(1, 5):
            nr, nc = r-k*dx, c-k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        if count == 3 and open_ends == 2: return True
    return False

@njit
def _check_open_four_static(board, r, c, p, board_size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for i in range(4):
        dx, dy = directions[i]
        count = 1; open_ends = 0
        for k in range(1, 5):
            nr, nc = r+k*dx, c+k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        for k in range(1, 5):
            nr, nc = r-k*dx, c-k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        if count == 4 and open_ends == 2: return True
    return False

@njit
def _check_double_threat_static(board, r, c, p, board_size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    threat_count = 0
    for i in range(4):
        dx, dy = directions[i]
        count = 1; open_ends = 0
        for k in range(1, 5):
            nr, nc = r+k*dx, c+k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        for k in range(1, 5):
            nr, nc = r-k*dx, c-k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        if count == 4 and open_ends >= 1: threat_count += 1
        elif count == 3 and open_ends == 2: threat_count += 1
        if threat_count >= 2: return True
    return False

# --- Solver Helpers ---
@njit
def _find_forced_block_static(board, attacker, board_size):
    threats_count = 0; block_move = -1
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] == EMPTY:
                if _check_exact_win_static(board, r, c, attacker, board_size):
                    threats_count += 1; block_move = r * board_size + c
                    if threats_count > 1: return -1
    if threats_count == 0: return -2
    return block_move

@njit
def _find_open_four_threats_static(board, p, board_size):
    threats_r = []; threats_c = []
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] == EMPTY:
                if _check_exact_win_static(board, r, c, p, board_size):
                    threats_r.append(r); threats_c.append(c)
                    if len(threats_r) >= 2: return threats_r, threats_c
    return threats_r, threats_c

@njit
def _get_vcf_candidates_static(board, p, board_size):
    cands = []
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] == EMPTY:
                if _check_four_or_five_static(board, r, c, p, board_size): cands.append(r*board_size+c)
    return cands

@njit
def _get_vct_candidates_static(board, p, board_size):
    cands = []
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] == EMPTY:
                if _check_open_three_static(board, r, c, p, board_size): cands.append(r*board_size+c)
    return cands

@njit
def _vcf_recursive_static(board, p, depth, board_size):
    if depth <= 0: return -1
    cands = _get_vcf_candidates_static(board, p, board_size)
    for move in cands:
        r, c = divmod(move, board_size)
        if _check_exact_win_static(board, r, c, p, board_size): return move
        board[r, c] = p
        block_move = _find_forced_block_static(board, p, board_size)
        if block_move == -1: board[r, c] = EMPTY; return move
        if block_move == -2: board[r, c] = EMPTY; continue
        br, bc = divmod(block_move, board_size)
        board[br, bc] = -p
        res = _vcf_recursive_static(board, p, depth - 1, board_size)
        board[br, bc] = EMPTY; board[r, c] = EMPTY
        if res != -1: return move
    return -1

@njit
def _vct_recursive_static(board, p, depth, width, board_size):
    if depth <= 0: return -1
    vcf_win = _vcf_recursive_static(board, p, 6, board_size)
    if vcf_win != -1: return vcf_win
    cands = _get_vct_candidates_static(board, p, board_size)
    count = 0
    for move in cands:
        if count >= width: break
        count += 1
        r, c = divmod(move, board_size)
        board[r, c] = p
        tr, tc = _find_open_four_threats_static(board, p, board_size)
        if len(tr) >= 2: board[r, c] = EMPTY; return move
        if len(tr) == 0: board[r, c] = EMPTY; continue
        br, bc = tr[0], tc[0]
        board[br, bc] = -p
        res = _vct_recursive_static(board, p, depth - 1, width, board_size)
        board[br, bc] = EMPTY; board[r, c] = EMPTY
        if res != -1: return move
    return -1

@njit
def _check_pattern_score_static(board, r, c, p, board_size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for i in range(4):
        dx, dy = directions[i]
        count = 1; open_ends = 0
        for k in range(1, 5):
            nr, nc = r + k*dx, c + k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        for k in range(1, 5):
            nr, nc = r - k*dx, c - k*dy
            if 0<=nr<board_size and 0<=nc<board_size:
                if board[nr, nc]==p: count+=1
                elif board[nr, nc]==EMPTY: open_ends+=1; break
                else: break
            else: break
        if count >= 5: return 2
        if count == 4 and open_ends > 0: return 2
        if count == 3:
            if open_ends == 2: return 2
            if open_ends == 1: return 1
        if count == 2 and open_ends == 2: return 1
    return 0

# --- Main Class ---
spec = [
    ('board', int8[:, :]), ('current_player', int8), ('last_move', int32),
    ('move_count', int32), ('done', boolean), ('winner', int8),
    ('BOARD_SIZE', int32), ('N_IN_ROW', int32), ('INPUT_CHANNELS', int32),
]

@jitclass(spec)
class GomokuEnv:
    def __init__(self, board_size, n_in_row, input_channels):
        self.BOARD_SIZE = board_size
        self.N_IN_ROW = n_in_row
        self.INPUT_CHANNELS = input_channels
        self.reset()

    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = P1
        self.last_move = -1; self.move_count = 0; self.done = False; self.winner = EMPTY

    def get_legal_moves_mask_flat(self):
        return self.board.flatten() == EMPTY

    def get_state(self):
        state = np.zeros((self.INPUT_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        player = self.current_player; opponent = -player
        state[0] = (self.board == player)
        state[1] = (self.board == opponent)
        if self.last_move != -1:
            x, y = divmod(self.last_move, self.BOARD_SIZE)
            state[2, x, y] = 1
        if player == P1: state[3, :, :] = 1 
        
        if self.INPUT_CHANNELS >= 8:
            for r in range(self.BOARD_SIZE):
                for c in range(self.BOARD_SIZE):
                    if self.board[r, c] == EMPTY:
                        score_self = _check_pattern_score_static(self.board, r, c, player, self.BOARD_SIZE)
                        if score_self >= 2: state[4, r, c] = 1
                        elif score_self == 1: state[6, r, c] = 1
                        score_opp = _check_pattern_score_static(self.board, r, c, opponent, self.BOARD_SIZE)
                        if score_opp >= 2: state[5, r, c] = 1
                        elif score_opp == 1: state[7, r, c] = 1
        return state

    def get_urgent_move(self):
        player = self.current_player; opponent = -player
        
        block_win = -1; block_open4 = -1; make_open4 = -1
        make_double_threat = -1; block_double_threat = -1
        
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == EMPTY:
                    action = r * self.BOARD_SIZE + c
                    
                    if _check_exact_win_static(self.board, r, c, player, self.BOARD_SIZE): return action, True 
                    if _check_exact_win_static(self.board, r, c, opponent, self.BOARD_SIZE): block_win = action; continue
                    
                    if _check_open_four_static(self.board, r, c, player, self.BOARD_SIZE): make_open4 = action
                    if _check_open_four_static(self.board, r, c, opponent, self.BOARD_SIZE): block_open4 = action

                    if _check_double_threat_static(self.board, r, c, player, self.BOARD_SIZE): make_double_threat = action
                    if _check_double_threat_static(self.board, r, c, opponent, self.BOARD_SIZE): block_double_threat = action
        
        if block_win != -1: return block_win, False
        if make_open4 != -1: return make_open4, True
        if block_open4 != -1: return block_open4, False
        if make_double_threat != -1: return make_double_threat, True 
        if block_double_threat != -1: return block_double_threat, False
        
        return -1, False

    def solve_vcf(self, max_depth):
        return _vcf_recursive_static(self.board, self.current_player, max_depth, self.BOARD_SIZE)

    def solve_vct(self, max_depth, max_width):
        return _vct_recursive_static(self.board, self.current_player, max_depth, max_width, self.BOARD_SIZE)
    
    # [FIX] Exposed API for external callers
    def _check_exact_win(self, r, c, p):
        return _check_exact_win_static(self.board, r, c, p, self.BOARD_SIZE)
    
    def _check_open_four(self, r, c, p):
        return _check_open_four_static(self.board, r, c, p, self.BOARD_SIZE)

    def step(self, action):
        if self.done: return
        x, y = divmod(action, self.BOARD_SIZE)
        if self.board[x, y] != EMPTY: self.done = True; self.winner = -self.current_player; return
        self.board[x, y] = self.current_player
        self.last_move = action; self.move_count += 1
        if _check_exact_win_static(self.board, x, y, self.current_player, self.BOARD_SIZE):
            self.done = True; self.winner = self.current_player; return
        if self.move_count == self.BOARD_SIZE * self.BOARD_SIZE: self.done = True; self.winner = EMPTY; return
        self.current_player = -self.current_player

    def copy_from(self, other):
        self.board = other.board.copy()
        self.current_player = other.current_player
        self.last_move = other.last_move; self.move_count = other.move_count
        self.done = other.done; self.winner = other.winner

def initialize_game_engine():
    logging.info("Initializing Numba Game Engine...")
    env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
    env.step(0)
    env.get_state() 
    env.solve_vcf(2)
    env.get_urgent_move()
    logging.info("Game Engine ready.")