# MassArena.py
import torch
import numpy as np
import logging
import sys
import os
import time
import multiprocessing as mp
from tqdm import tqdm
from torch.amp import autocast

# 1. Mount team member code
TEAM_DIR = os.path.join(os.getcwd(), 'gomoku_web')
if TEAM_DIR not in sys.path: sys.path.append(TEAM_DIR)

try:
    from ai.level2_minimax_alphabeta import alphabeta_move
except ImportError:
    print("❌ Missing gomoku_web/ai/level2_minimax_alphabeta.py")
    sys.exit(1)

try:
    from config import CONF
    from game import GomokuEnv, P1, P2
    from model import AlphaCerberusNet
    from mcts import MCTS
except ImportError:
    print("❌ Missing core files.")
    sys.exit(1)

logging.basicConfig(level=logging.ERROR)

# =============================================================================
# Dynamic Resource Calculator
# =============================================================================
def calc_optimal_workers():
    """
    Calculate optimal GPU worker count based on VRAM size.
    """
    if not torch.cuda.is_available():
        print("⚠️ No GPU found. Falling back to CPU Strategy.")
        return max(1, mp.cpu_count() - 2), "cpu"

    # Get VRAM info (Bytes)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    
    # Empirical formula: PyTorch process overhead is huge on Windows
    # Base overhead ~800MB, model+cache ~200MB, safety margin ~200MB -> 1.2 GB / Worker
    VRAM_PER_WORKER = 1.2 * (1024 ** 3) 
    
    # Reserve 1GB for system display and other applications
    SAFE_MARGIN = 1.0 * (1024 ** 3)
    
    available_mem = total_mem - SAFE_MARGIN
    optimal_count = int(available_mem / VRAM_PER_WORKER)
    
    # Limit range: at least 1, at most not exceeding CPU core count (otherwise scheduling overhead is too large)
    optimal_count = max(1, min(optimal_count, mp.cpu_count()))
    
    print(f"🧠 GPU Analysis:")
    print(f"   Total VRAM: {total_mem / 1e9:.2f} GB")
    print(f"   Est. Cost/Worker: {VRAM_PER_WORKER / 1e9:.2f} GB")
    print(f"   Safe Limit: {optimal_count} Concurrent Workers")
    
    return optimal_count, "cuda:0"

# =============================================================================
# Battle Process (GPU Accelerated)
# =============================================================================
def worker_play_game_gpu(args):
    game_id, model_path, my_role_black, device_str = args
    
    # Initialize device independently
    device = torch.device(device_str)
    
    # Load model
    model = AlphaCerberusNet().to(device)
    try:
        # Load safely using weights_only=True
        state = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state: state = state['model_state_dict']
        model.load_state_dict(state)
        model.eval()
    except:
        return "Error"

    # Local inference pipeline (with AMP acceleration)
    class LocalPipe:
        def __init__(self, m, d): self.m = m; self.d = d
        def infer(self, s, sig=None):
            if not s.flags['C_CONTIGUOUS']: s = np.ascontiguousarray(s)
            
            # Copy may incur GPU overhead, but necessary for process safety
            t = torch.from_numpy(s).unsqueeze(0).float().to(self.d)
            
            # [Key optimization] Enable half-precision inference, 2x speed, half VRAM
            with torch.inference_mode(), autocast(device_type=self.d.type):
                p, v = self.m(t)
                return torch.softmax(p, dim=1).float().cpu().numpy().flatten(), v.item()
    
    mcts = MCTS(LocalPipe(model, device))
    env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
    
    # Tactical config: 800 simulations
    CONF.MCTS_SIMULATION_SCHEDULE = [(0, 800)] 
    
    # Warm up
    env.step(0); env.solve_vcf(2); env.reset()
    
    my_role = P1 if my_role_black else P2
    
    while not env.done:
        if env.current_player == my_role:
            # Our turn (GPU MCTS)
            res = mcts.run_search(env)
            if res is None or (isinstance(res, tuple) and res[0] is None):
                 return "Loss" # Surrender
            
            policy = res[0] if isinstance(res, tuple) else res
            action = np.argmax(policy) # Competition mode: Argmax
            
            env.step(action)
            mcts.reuse_tree(action)
        else:
            # Opponent turn (CPU Minimax)
            try:
                b_list = env.board.astype(int).tolist()
                x, y = alphabeta_move(b_list, env.current_player)
                action = x * CONF.BOARD_SIZE + y
                env.step(action)
                mcts.reuse_tree(action)
            except:
                return "Win" # Opponent crashed

    if env.winner == my_role: return "Win"
    elif env.winner == 0: return "Draw"
    else: return "Loss"

# =============================================================================
# Main Control
# =============================================================================
if __name__ == "__main__":
    # Required for Windows
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    MODEL_PATH = "sl_best.ckpt"
    if not os.path.exists(MODEL_PATH):
        print("Model not found."); sys.exit()
        
    # 1. Dynamically calculate concurrency
    NUM_PROCESSES, DEVICE_STR = calc_optimal_workers()
    
    TOTAL_GAMES = 50 # Sample size
    
    print(f"🔥 MASS ARENA (GPU EDITION) STARTED")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Workers: {NUM_PROCESSES} (Device: {DEVICE_STR})")
    print("-" * 50)

    # 2. Prepare tasks
    tasks = [(i, MODEL_PATH, i % 2 == 0, DEVICE_STR) for i in range(TOTAL_GAMES)]
    
    wins, losses, draws = 0, 0, 0
    errors = 0
    
    # 3. Execute in parallel
    # Note: Each process startup will be slow (due to CUDA initialization), please wait patiently for the progress bar to start moving
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        for res in tqdm(pool.imap_unordered(worker_play_game_gpu, tasks), total=TOTAL_GAMES):
            if res == "Win": wins += 1
            elif res == "Loss": losses += 1
            elif res == "Draw": draws += 1
            else: errors += 1
            
    print("\n" + "="*40)
    print(f"📊 RESULTS ({TOTAL_GAMES} Games)")
    if errors > 0: print(f"⚠️ Errors: {errors}")
    print(f"🏆 Wins:   {wins} ({wins/TOTAL_GAMES*100:.1f}%)")
    print(f"💀 Losses: {losses} ({losses/TOTAL_GAMES*100:.1f}%)")
    print(f"🤝 Draws:  {draws} ({draws/TOTAL_GAMES*100:.1f}%)")
    print("="*40)
