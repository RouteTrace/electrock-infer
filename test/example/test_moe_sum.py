import torch
import time
# 确保您已经编译并安装了您的自定义扩展
from electrock_infer import _C

# --- 1. 配置测试参数 ---
# Mixtral-8x7B 的典型尺寸
NUM_TOKENS = 1024
HIDDEN_SIZE = 4096 
TOP_K = 2
DTYPE = torch.float16 # 使用半精度以模拟真实场景

# 测试运行配置
WARMUP_RUNS = 10
TIMING_RUNS = 100

def benchmark(func, description):
    """一个精确测量 GPU 函数执行时间的辅助函数"""
    print(f"Benchmarking {description}...".ljust(40), end="")
    
    # 预热
    for _ in range(WARMUP_RUNS):
        func()
    torch.cuda.synchronize()

    # 精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(TIMING_RUNS):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / TIMING_RUNS
    
    print(f"Avg Time: {avg_time_ms:.6f} ms")
    return avg_time_ms

if __name__ == "__main__":
    print("="*60)
    print("Starting MoE Sum Performance Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: {NUM_TOKENS=}, {HIDDEN_SIZE=}, {TOP_K=}, DType={DTYPE}")
    print("="*60)

    # --- 2. 创建输入数据 ---
    device = 'cuda'
    
    # a) 用于 moe_sum 和 torch.sum 的标准输入
    # Shape: [num_tokens, top_k, hidden_size]
    input_default = torch.randn(
        (NUM_TOKENS, TOP_K, HIDDEN_SIZE), 
        dtype=DTYPE, 
        device=device
    )
    
    # b) 用于 moe_sum_shufl 的转置输入
    # Shape: [num_tokens, hidden_size, top_k]
    # .contiguous() 确保内存布局是连续的，这对 CUDA kernel 至关重要
    input_shfl = input_default.transpose(1, 2).contiguous()

    # --- 3. 验证正确性 ---
    print("\nVerifying correctness...")
    
    # 使用 PyTorch 原生 sum 作为黄金标准
    expected_output = torch.sum(input_default, dim=1)
    print(f"Shape : {expected_output.shape}")
    # 验证 moe_sum
    output_custom = torch.empty_like(expected_output)
    _C.moe_sum(input_default, output_custom)
    is_correct1 = torch.allclose(expected_output, output_custom, atol=1e-2)
    print(f"Correctness of 'moe_sum': {'OK' if is_correct1 else 'FAIL'}")

    # 验证 moe_sum_shufl
    output_shfl = torch.empty_like(expected_output)
    _C.moe_sum_efficient(input_shfl, output_shfl)
    # is_correct2 = torch.allclose(expected_output, output_shfl, atol=1e-2)
    # print(f"Correctness of 'moe_sum_efficient': {'OK' if is_correct2 else 'FAIL'}")

    if not (is_correct1):
        print("\nERROR: One of the custom kernels produced incorrect results. Aborting benchmark.")
        exit()

    # --- 4. 运行性能基准测试 ---
    print("\nRunning benchmarks...")
    
    # a) PyTorch 原生实现
    benchmark(lambda: torch.sum(input_default, dim=1), "PyTorch (torch.sum)")
    
    # b) 您的第一个自定义 Kernel
    output_bench1 = torch.empty_like(expected_output)
    benchmark(lambda: _C.moe_sum(input_default, output_bench1), "Custom Kernel (moe_sum)")

    # # c) 您的 vector Kernel
    # output_bench2 = torch.empty_like(expected_output)
    # benchmark(lambda: _C.moe_sum_efficient(input_shfl, output_bench2), "Custom Kernel (vector)")
    
    print("\nBenchmark finished.")
    print("="*60)