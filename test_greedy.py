# test_greedy.py
import torch
import numpy as np
import logging
import sys
import os
import multiprocessing as mp
import time

# Import core components
try:
    from config4greedy import CONF
    from game import GomokuEnv, P1, P2
    from model import AlphaGomokuNet, load_model
    from mcts import MCTS
    from heuristic import GreedyGomokuAgent
except ImportError as e:
    print(f"Error: Missing dependency files. Ensure heuristic.py, config.py, etc., are in the project root.\nDetails: {e}")
    sys.exit(1)

# Set logging to show only critical information
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# --- 1. Replicate Battle.py's inference pipeline ---
class LocalInferencePipeline:
    """
    [Battle equivalent] Local synchronous inference pipeline.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def infer(self, state_np, stop_signal=None):
        # 1. Preprocess (C, H, W) -> (1, C, H, W)
        if state_np.ndim == 3:
            state_np = state_np[np.newaxis, ...]
        
        # Ensure memory is contiguous
        if not state_np.flags['C_CONTIGUOUS']:
            state_np = np.ascontiguousarray(state_np)

        state_tensor = torch.from_numpy(state_np).float().to(self.device)
        
        # 2. Inference
        try:
            with torch.inference_mode():
                policy_logits, value = self.model(state_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1)
                
                # Convert back to numpy
                policy = policy_probs.cpu().numpy().flatten()
                value = value.item()
                
                return policy, value
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return None

# --- 2. Replicate Battle.py's board printing ---
def print_board(board, last_move=-1):
    size = CONF.BOARD_SIZE
    print("\n" + "   " + " ".join([f"{i:2}" for i in range(size)]))
    print("  " + "---" * size)
    
    for r in range(size):
        row_str = f"{r:2}|"
        for c in range(size):
            idx = r * size + c
            char = ". "
            if board[r, c] == P1:  char = "● " # Black
            if board[r, c] == P2:  char = "○ " # White
            
            # Highlight the last move
            if idx == last_move:
                char = "[" + char[0] + "]"
            else:
                char = " " + char
            
            row_str += char
        print(row_str)
    print("")

# --- 3. Core duel logic ---
def run_duel(sl_model_path, ai_plays_black=True):
    device = CONF.DEVICE
    print("="*60)
    print(f" 🔥 Duel Start: SL AI ({'Black' if ai_plays_black else 'White'}) vs Greedy ({'White' if ai_plays_black else 'Black'})")
    print(f"    Model: {sl_model_path}")
    print("="*60)

    # A. Initialize model
    model = AlphaGomokuNet().to(device)
    if not load_model(model, sl_model_path, device):
        print("❌ Model loading failed, skipping.")
        return None

    # B. Initialize environment and components
    env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
    pipeline = LocalInferencePipeline(model, device)
    mcts = MCTS(pipeline)
    greedy_agent = GreedyGomokuAgent(CONF.BOARD_SIZE)
    
    dummy_stop_signal = mp.Event()

    # C. Assign roles
    if ai_plays_black:
        ai_player = P1
        greedy_player = P2
    else:
        greedy_player = P1
        ai_player = P2

    print_board(env.board)

    # D. Game loop
    while not env.done:
        # --- AI turn ---
        if env.current_player == ai_player:
            print(f"🤖 AI ({'●' if ai_player==P1 else '○'}) thinking...", end="", flush=True)
            t0 = time.time()
            
            # [Battle rule] Run MCTS
            original_schedule = CONF.MCTS_SIMULATION_SCHEDULE
            CONF.MCTS_SIMULATION_SCHEDULE = [(0, 800)] 
            
            result = mcts.run_search(env, stop_signal=dummy_stop_signal)
            if isinstance(result, tuple):
                policy, _ = result  
            else:
                policy = result     
            
            CONF.MCTS_SIMULATION_SCHEDULE = original_schedule # Restore
            
            if policy is None:
                print(" AI resigns (error).")
                break
                
            action = np.argmax(policy)
            print(f" Move: {divmod(action, CONF.BOARD_SIZE)} (Time {time.time()-t0:.2f}s)")
            
            env.step(action)
            mcts.reuse_tree(action) # [Battle rule] Update tree, keep state synchronized

        # --- Greedy turn ---
        else:
            print(f"👾 Greedy ({'●' if greedy_player==P1 else '○'}) thinking...", end="", flush=True)
            action = greedy_agent.select_move(env.board, env.current_player)
            print(f" Move: {divmod(action, CONF.BOARD_SIZE)}")
            
            env.step(action)
            mcts.reuse_tree(action) # [Battle rule] Critical! Tell AI where the opponent moved

        print_board(env.board, env.last_move)

    # E. Settlement
    winner_name = "Draw"
    if env.winner == ai_player: winner_name = "SL AI"
    elif env.winner == greedy_player: winner_name = "Greedy"
    
    print(f"🏁 Game Over! Winner: {winner_name}\n")
    return winner_name

def main():
    # Check if SL model exists
    sl_path = CONF.SL_MODEL_PATH # Usually 'sl_best.ckpt'
    if not os.path.exists(sl_path):
        # Try to find pth
        if os.path.exists("model_latest.pth"):
            print(f"⚠️ {sl_path} not found, using model_latest.pth instead.")
            sl_path = "model_latest.pth"
        else:
            print(f"❌ Model file {sl_path} not found. Run SLTrain.py or ensure the file exists.")
            return

    # Round 1: AI as Black (first move)
    res1 = run_duel(sl_path, ai_plays_black=True)
    
    # Round 2: AI as White (second move)
    res2 = run_duel(sl_path, ai_plays_black=False)

    print("="*40)
    print("Final Report:")
    print(f"Round 1 (AI Black): {res1}")
    print(f"Round 2 (AI White): {res2}")
    
    if res1 == "SL AI" and res2 == "SL AI":
        print("✅ Test Passed: SL model defeated Greedy algorithm in both rounds! Solid foundation.")
    elif res1 == "SL AI" or res2 == "SL AI":
        print("⚠️ Barely Passed: Mixed results. SL model may have weaknesses.")
    else:
        print("❌ Test Failed: SL model was crushed by the Greedy algorithm. Stop RL immediately and optimize SL data!")
    print("="*40)

if __name__ == "__main__":
    main()