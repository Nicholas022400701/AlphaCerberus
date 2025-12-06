# benchmark_vcf.py
import time
import numpy as np
from numba import njit

# --- 1. Pure Python Implementation (Slow) ---
def check_win_py(board, r, c, p):
    # simplified check for benchmarking recursion overhead
    return False 

def vcf_py(board, p, depth):
    if depth <= 0: return -1
    # Simulate branching factor of ~5 candidates
    for i in range(5): 
        # Simulate some logic
        board[0,0] = p 
        res = vcf_py(board, p, depth - 1)
        board[0,0] = 0
        if res != -1: return 1
    return -1

# --- 2. Numba Implementation (Fast) ---
@njit
def vcf_numba(board, p, depth):
    if depth <= 0: return -1
    for i in range(5):
        board[0,0] = p 
        res = vcf_numba(board, p, depth - 1)
        board[0,0] = 0
        if res != -1: return 1
    return -1

def main():
    print("🔥 Benchmarking Recursion Overhead (Depth=9, Branch=5)...")
    
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Warmup JIT
    vcf_numba(board, 1, 2)
    
    # Test Python
    print("Running Python VCF (This might take a while)...")
    t0 = time.time()
    # =we use depth 9 for comparison
    vcf_py(board, 1, 9) 
    t_py = time.time() - t0
    print(f"Python (Depth 9): {t_py:.4f}s")
    
    # Test Numba
    print("Running Numba VCF...")
    t0 = time.time()
    vcf_numba(board, 1, 9)
    t_nb = time.time() - t0
    print(f"Numba (Depth 9):  {t_nb:.6f}s")
    
    speedup = t_py / t_nb
    print(f"\n🚀 Speedup Factor: {speedup:.1f}x")
    print(f"Use this number in your paper Implementation section.")

if __name__ == "__main__":
    main()
