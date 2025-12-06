import torch
import time
import numpy as np
import logging
from model import AlphaCerberusNet
from config import CONF

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

def benchmark_inference():
    print("=" * 60)
    print("🔥 Alpha-Cerberus: Parallelism Economy Benchmark (AMP Mode)")
    print("=" * 60)

    # 1. 硬件环境检查
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到 GPU！无法验证并行经济学。")
        return
    
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"✅ 检测到 GPU: {props.name}")
    print(f"   VRAM: {props.total_memory / 1e9:.2f} GB")

    # 2. 模型初始化 (保持 FP32 权重，让 AMP 自动管理精度)
    print("\n[Init] 加载 AlphaCerberusNet...")
    model = AlphaCerberusNet().to(device)
    model.eval()
    
    # 3. 准备数据 (保持 FP32 输入)
    C, H, W = CONF.INPUT_CHANNELS, CONF.BOARD_SIZE, CONF.BOARD_SIZE
    
    # Batch = 1 (模拟单次推理)
    input_b1 = torch.randn(1, C, H, W, device=device)

    # Batch = 8 (模拟 TTA 推理：原图 + 3次旋转 + 4次镜像)
    input_b8 = torch.randn(8, C, H, W, device=device)

    # 4. 预热 (Warmup)
    print("[Warmup] 正在预热 GPU (50 次迭代)...")
    with torch.inference_mode():
        # 使用 AMP 上下文
        with torch.amp.autocast('cuda'):
            for _ in range(50):
                _ = model(input_b8)
    torch.cuda.synchronize()

    # 5. 定义测速函数
    def run_test(tensor, iterations=1000, label=""):
        torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.inference_mode():
            for _ in range(iterations):
                # 关键修复：使用 autocast 替代手动的 .half()
                with torch.amp.autocast('cuda'):
                    _ = model(tensor)
                
        torch.cuda.synchronize()
        dt = time.time() - t0
        
        avg_ms = (dt / iterations) * 1000
        print(f"   👉 {label:<20} | {iterations} iters | Avg Latency: {avg_ms:.3f} ms")
        return avg_ms

    # 6. 执行测试
    print("\n[Testing] 开始基准测试 (N=1000)...")
    latency_b1 = run_test(input_b1, 1000, "Single (B=1)")
    latency_b8 = run_test(input_b8, 1000, "TTA (B=8)")

    # 7. 结果分析
    print("-" * 60)
    print("📊 实验结果分析:")
    print(f"   • 单次推理耗时 : {latency_b1:.3f} ms")
    print(f"   • TTA 并行耗时 : {latency_b8:.3f} ms")
    
    delta = latency_b8 - latency_b1
    ratio = latency_b8 / latency_b1
    
    print(f"\n   📈 边际成本 (Marginal Cost): +{delta:.3f} ms")
    print(f"   🚀 吞吐量倍增 (Scaling): 计算量增加 8.0x -> 耗时仅增加 {ratio:.2f}x")
    
    if ratio < 2.0:
        print("\n✅ 验证成功: 并行经济学 (Parallelism Economy) 成立。")
        print("   结论：在 RTX 4060 上，利用 TTA 增强模型鲁棒性几乎是‘免费’的。")
    else:
        print("\n⚠️ 验证存疑: 并行优势不明显。")
    
    print("=" * 60)

if __name__ == "__main__":
    benchmark_inference()