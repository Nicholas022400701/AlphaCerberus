import torch
import time
import numpy as np
import logging
from model import AlphaCerberusNet
from config import CONF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

def benchmark_inference():
    print("=" * 60)
    print("🔥 Alpha-Cerberus: Parallelism Economy Benchmark (AMP Mode)")
    print("=" * 60)

    # 1. Hardware environment check
    if not torch.cuda.is_available():
        print("⚠️ Warning: GPU not detected! Cannot verify parallelism economy.")
        return
    
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"✅ GPU detected: {props.name}")
    print(f"   VRAM: {props.total_memory / 1e9:.2f} GB")

    # 2. Model initialization (keep FP32 weights, let AMP auto-manage precision)
    print("\n[Init] Loading AlphaCerberusNet...")
    model = AlphaCerberusNet().to(device)
    model.eval()
    
    # 3. Prepare data (keep FP32 input)
    C, H, W = CONF.INPUT_CHANNELS, CONF.BOARD_SIZE, CONF.BOARD_SIZE
    
    # Batch = 1 (simulate single inference)
    input_b1 = torch.randn(1, C, H, W, device=device)

    # Batch = 8 (simulate TTA inference: original + 3 rotations + 4 flips)
    input_b8 = torch.randn(8, C, H, W, device=device)

    # 4. Warmup
    print("[Warmup] Warming up GPU (50 iterations)...")
    with torch.inference_mode():
        # Use AMP context
        with torch.amp.autocast('cuda'):
            for _ in range(50):
                _ = model(input_b8)
    torch.cuda.synchronize()

    # 5. Define timing function
    def run_test(tensor, iterations=1000, label=""):
        torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.inference_mode():
            for _ in range(iterations):
                # Key fix: use autocast instead of manual .half()
                with torch.amp.autocast('cuda'):
                    _ = model(tensor)
                
        torch.cuda.synchronize()
        dt = time.time() - t0
        
        avg_ms = (dt / iterations) * 1000
        print(f"   👉 {label:<20} | {iterations} iters | Avg Latency: {avg_ms:.3f} ms")
        return avg_ms

    # 6. Run tests
    print("\n[Testing] Starting benchmark test (N=1000)...")
    latency_b1 = run_test(input_b1, 1000, "Single (B=1)")
    latency_b8 = run_test(input_b8, 1000, "TTA (B=8)")

    # 7. Result analysis
    print("-" * 60)
    print("📊 Experimental Results Analysis:")
    print(f"   • Single inference time : {latency_b1:.3f} ms")
    print(f"   • TTA parallel time : {latency_b8:.3f} ms")
    
    delta = latency_b8 - latency_b1
    ratio = latency_b8 / latency_b1
    
    print(f"\n   📈 Marginal Cost: +{delta:.3f} ms")
    print(f"   🚀 Throughput Scaling: computation increased 8.0x -> time only increased {ratio:.2f}x")
    
    if ratio < 2.0:
        print("\n✅ Verification successful: Parallelism Economy established.")
        print("   Conclusion: On RTX 4060, using TTA to enhance model robustness is nearly 'free'.")
    else:
        print("\n⚠️ Verification questionable: Parallel advantage not significant.")
    
    print("=" * 60)

if __name__ == "__main__":
    benchmark_inference()