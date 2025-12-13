import os
import sys
import ctypes
import threading

# =============================================================================
# Intel MKL Signal Handling Patch
# Prevents Fortran runtime from intercepting Ctrl+C and crashing the process.
# =============================================================================

# Disable Intel Fortran default console handler
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

try:
    kernel32 = ctypes.windll.kernel32
    HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

    def _kernel_ctrl_handler(ctrl_type):
        # 0: CTRL_C_EVENT, 1: CTRL_BREAK_EVENT
        if ctrl_type in (0, 1, 2):
            print(f"\n[System] Termination signal received ({ctrl_type}). Exiting...")
            # Force exit to bypass Python cleanup and avoid MKL errors
            os._exit(1)
            return True
        return False

    _handler_ref = HandlerRoutine(_kernel_ctrl_handler)
    if not kernel32.SetConsoleCtrlHandler(_handler_ref, True):
        print("[System] Warning: Failed to register kernel control handler.")

except Exception as e:
    print(f"[System] Signal handler setup failed: {e}")

# =============================================================================
# Server Application
# =============================================================================

import numpy as np
import torch
import time
import logging
from flask import Flask, request, jsonify, render_template

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Server")

# Add parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from ai.registry import AI_LEVELS
    from config import CONF
    from model import AlphaCerberusNet
    from mcts import MCTS
    from game import GomokuEnv, initialize_game_engine
except ImportError as e:
    logger.critical(f"Module import failed: {e}")
    sys.exit(1)

# Warmup: Pre-compile Numba JIT functions at startup to avoid delay at game start
logger.info("Warming up Numba JIT functions...")
initialize_game_engine()
logger.info("Numba warmup complete.")

app = Flask(__name__)

class InferenceEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.device = CONF.DEVICE
        self.model = AlphaCerberusNet().to(self.device)
        self.ready = False
        self.env = None
        self.mcts = None
        
        # Locate model file
        model_path = os.path.join(parent_dir, CONF.SL_MODEL_PATH)
        if not os.path.exists(model_path):
            model_path = os.path.join(parent_dir, CONF.MODEL_PATH)
            
        if os.path.exists(model_path):
            logger.info(f"Loading model: {model_path} [{self.device}]")
            try:
                # Fix FutureWarning locally: Explicitly set weights_only=True
                # This fixes the warning without modifying model.py
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Handle both full checkpoint dicts and direct state dicts
                if 'model_state_dict' in state: 
                    state = state['model_state_dict']
                
                self.model.load_state_dict(state)
                self.model.eval()
                self.ready = True
                self._reset_state()
                self._warmup_inference()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.error("Model not found. Neural network features disabled.")

    def force_reset(self):
        with self.lock:
            self._reset_state()

    def _reset_state(self):
        self.env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
        self.mcts = MCTS(self)
        logger.info("Game state reset.")

    def _warmup_inference(self):
        """Warmup neural network inference to avoid delay at first game move."""
        logger.info("Warming up neural network inference...")
        dummy_state = np.zeros((CONF.INPUT_CHANNELS, CONF.BOARD_SIZE, CONF.BOARD_SIZE), dtype=np.float32)
        self.infer(dummy_state)
        logger.info("Neural network warmup complete.")

    def infer(self, state_np, stop_signal=None):
        # Test-Time Augmentation (TTA)
        # Generates 8 symmetries of the board state to improve prediction stability
        if not state_np.flags['C_CONTIGUOUS']: 
            state_np = np.ascontiguousarray(state_np)
        
        variations = []
        for i in range(4):
            rot = np.rot90(state_np, i, axes=(1, 2)).copy()
            variations.append(rot)
            variations.append(np.flip(rot, axis=2).copy())

        batch = torch.from_numpy(np.stack(variations)).float().to(self.device)
        with torch.inference_mode():
            p_logits, v_out = self.model(batch)
            p_probs = torch.softmax(p_logits, dim=1).cpu().numpy()
            v_out = v_out.cpu().numpy()

        # Aggregate results
        p_accum = np.zeros((CONF.BOARD_SIZE, CONF.BOARD_SIZE), dtype=np.float32)
        v_accum = 0.0
        for idx in range(8):
            p_2d = p_probs[idx].reshape(CONF.BOARD_SIZE, CONF.BOARD_SIZE)
            v = v_out[idx][0]
            rot_k = idx // 2
            is_flip = (idx % 2 == 1)
            
            # Inverse transform
            if is_flip: p_2d = np.flip(p_2d, axis=1)
            if rot_k > 0: p_2d = np.rot90(p_2d, -rot_k)
            
            p_accum += p_2d
            v_accum += v
            
        return p_accum.flatten() / 8.0, v_accum / 8.0

    def sync_and_move(self, board_matrix, last_move_idx, role_name):
        # Sync internal game state with frontend board
        current_stones = np.count_nonzero(board_matrix)
        internal_stones = self.env.move_count
        
        if current_stones < internal_stones or current_stones == 0:
            logger.info(f"[{role_name}] State mismatch or new game. Resetting.")
            self._reset_state()

        # Apply opponent's last move
        if last_move_idx is not None:
            if self.env.board.flatten()[last_move_idx] == 0:
                self.env.step(last_move_idx)
                self.mcts.reuse_tree(last_move_idx)

        return self._search_move(role_name)

    def _search_move(self, role_name):
        if self.env.done: 
            return None

        t0 = time.time()
        # Set simulation count for MCTS
        CONF.MCTS_SIMULATION_SCHEDULE = [(0, 800)]
        
        logger.info(f"[{role_name}] MCTS Start...")
        mcts_res = self.mcts.run_search(self.env)
        if mcts_res is None: return None
        
        policy = mcts_res[0] if isinstance(mcts_res, tuple) else mcts_res
        action = np.argmax(policy)
        win_rate = self.mcts.root.Q
        
        dt = time.time() - t0
        r, c = divmod(action, CONF.BOARD_SIZE)
        
        logger.info(f"[{role_name}] Move: ({r}, {c}) | Value: {win_rate:.3f} | Time: {dt:.2f}s")
        
        self.env.step(action)
        self.mcts.reuse_tree(action)
        
        return divmod(action, CONF.BOARD_SIZE)

engine = InferenceEngine()

@app.route("/reset", methods=["POST"])
def reset_route():
    engine.force_reset()
    return jsonify({"status": "reset_complete"})

@app.route("/")
def index(): 
    return render_template("index.html")

@app.route("/ai_move", methods=["POST"])
def ai_move():
    data = request.get_json()
    board = np.array(data["board"])
    level = int(data["level"])
    role_name = data.get("role", "AI")
    
    last_move_idx = None
    if data.get("last_move"):
        last_move_idx = data["last_move"][0] * CONF.BOARD_SIZE + data["last_move"][1]

    # Level 3: Neural MCTS
    if level == 3:
        if not engine.ready: return jsonify({"error": "Engine not ready"})
        try:
            with engine.lock:
                move = engine.sync_and_move(board, last_move_idx, role_name)
            if move: return jsonify({"x": int(move[0]), "y": int(move[1])})
            else: return jsonify({"error": "Game Over"})
        except Exception as e:
            logger.exception("Engine Error")
            return jsonify({"error": str(e)})

    # Legacy Levels (Greedy / Minimax)
    ai_stone = int(data["ai_stone"])
    ai_func = AI_LEVELS.get(level)
    
    if ai_func:
        try:
            x, y = ai_func(board, ai_stone)
            return jsonify({"x": int(x), "y": int(y)})
        except Exception as e:
             logger.error(f"Legacy AI Error: {e}")

    # Fallback: Random
    import random
    from ai.core import N, EMPTY
    empties = [(i, j) for i in range(N) for j in range(N) if board[i][j] == EMPTY]
    x, y = random.choice(empties) if empties else (0, 0)
    return jsonify({"x": int(x), "y": int(y)})

if __name__ == "__main__":
    print("\nAlpha-Cerberus Server Running On 127.0.0.1:5050 ...\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
