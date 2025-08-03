import torch
from torch import nn
import time
from torch.utils.cpp_extension import load
from electrock_infer import _C


# -------------------------------------------------------------------
# 步骤 2: 编写与内核逻辑一致的 PyTorch "Naive" 版本
# -------------------------------------------------------------------

def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """
    不带残差连接的 RMSNorm 的 PyTorch 实现。
    """
    # 为了精度稳定性，使用 float32 进行中间计算
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    
    # 将结果转换回原始数据类型
    return (x * inv_rms).to(x.dtype) * weight

def torch_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """
    带残差连接的 RMSNorm 的 PyTorch 实现。
    注意：这个函数需要模拟内核的 in-place 行为：
    1. 计算 x = x + residual
    2. 对新的 x 进行 RMSNorm
    3. 将原始的 residual 更新为 x + residual 的结果
    """
    # 1. 计算 x + residual
    x_plus_res = x + residual
    
    # 2. 对相加后的结果进行 RMSNorm
    variance = x_plus_res.to(torch.float32).pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    norm_x = (x_plus_res * inv_rms).to(x.dtype) * weight
    
    # 3. In-place 更新原始张量
    x.copy_(norm_x)
    residual.copy_(x_plus_res)


# -------------------------------------------------------------------
# 步骤 1: 动态加载 C++/HIP/CUDA 扩展
# -------------------------------------------------------------------
# PyTorch 会调用系统的 C++ 编译器 (如 g++) 和设备编译器 (如 hipcc 或 nvcc)
# 来编译和链接这个源文件。
# verbose=True 会在编译时打印详细的命令和输出，非常有助于排查编译错误。
# try:
#     custom_kernels = load(
#         name="custom_rms_norm",
#         sources=["rms_norm_kernel.cpp"],
#         verbose=True
#     )
#     print("✅ C++ extension loaded successfully.")
# except Exception as e:
#     print(f"❌ Failed to load C++ extension: {e}")
#     exit()

# -------------------------------------------------------------------
# 步骤 3: 设置测试参数并执行测试
# -------------------------------------------------------------------

def run_test():
    # --- 测试参数 ---
    NUM_TOKENS = 60 * 1024
    HIDDEN_SIZE = 4096 # 内核中断言了这个值
    DTYPE = torch.half  # 可以改为 torch.bfloat16 或 torch.float32
    EPSILON = 1e-5
    DEVICE = 'cuda'

    # --- 性能测试迭代次数 ---
    warmup_iter = 10
    test_iter = 100

    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm device not found. Aborting.")
        return

    print("\n" + "="*50)
    print(f"Test Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Num Tokens: {NUM_TOKENS}")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Data Type: {DTYPE}")
    print("="*50 + "\n")

    # --- 测试 `rms_norm` (不带残差) ---
    print("--- 🔬 Testing `rms_norm` (without residual) ---")
    
    # 创建输入数据
    input_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight_tensor = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    
    # 为自定义内核和torch版本创建副本，以保证公平比较
    input_custom = input_tensor.clone()
    input_torch = input_tensor.clone()
    
    # 运行 torch naive 版本并验证
    output_torch = torch_rms_norm(input_torch, weight_tensor, EPSILON)
    torch.cuda.synchronize()
    # 运行自定义内核版本
    _C.rms_norm(input_custom, EPSILON, weight_tensor)
    torch.cuda.synchronize()
    # 验证正确性
    # atol (absolute tolerance) 设为 1e-2 是因为 half/bfloat16 精度较低
    is_correct = torch.allclose(input_custom, output_torch, atol=1e-2)
    print(f"Correctness Check: {'PASS ✅' if is_correct else 'FAIL ❌'}")
    print("Max difference:", (input_custom - output_torch).abs().max())

    # 性能评测
    # Warm-up
    for _ in range(warmup_iter):
        _C.rms_norm(input_custom, EPSILON, weight_tensor)
        _ = torch_rms_norm(input_torch, weight_tensor, EPSILON)
    torch.cuda.synchronize()

    # Custom Kernel
    start_time = time.perf_counter()
    for _ in range(test_iter):
        _C.rms_norm(input_custom, EPSILON, weight_tensor)
    torch.cuda.synchronize()
    custom_time = (time.perf_counter() - start_time) / test_iter * 1000 # ms
    print(f"Custom Kernel Time: {custom_time:.4f} ms")

    # PyTorch Naive
    start_time = time.perf_counter()
    for _ in range(test_iter):
        _ = torch_rms_norm(input_torch, weight_tensor, EPSILON)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start_time) / test_iter * 1000 # ms
    print(f"PyTorch Naive Time: {torch_time:.4f} ms")
    print(f"🚀 Speedup: {torch_time / custom_time:.2f}x")

    print("\n" + "-"*50 + "\n")

    # --- 测试 `add_rms_norm` (带残差) ---
    print("--- 🔬 Testing `add_rms_norm` (with residual) ---")
    
    # 创建输入数据
    input_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    residual_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight_tensor = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    # 创建副本
    input_custom = input_tensor.clone()
    residual_custom = residual_tensor.clone()
    input_torch = input_tensor.clone()
    residual_torch = residual_tensor.clone()

    # 运行 torch naive 版本
    torch_add_rms_norm(input_torch, residual_torch, weight_tensor, EPSILON)
    torch.cuda.synchronize()
    # 运行自定义内核版本
    _C.add_rms_norm(input_custom, residual_custom, EPSILON, weight_tensor)
    torch.cuda.synchronize()
    # 验证正确性
    # 验证两个输出：input 和 residual
    is_correct_input = torch.allclose(input_custom, input_torch, atol=1e-2)
    is_correct_residual = torch.allclose(residual_custom, residual_torch, atol=1e-2)
    print(f"Correctness Check (input): {'PASS ✅' if is_correct_input else 'FAIL ❌'}")
    print(f"Correctness Check (residual): {'PASS ✅' if is_correct_residual else 'FAIL ❌'}")

    print("Max input difference:", (input_custom - input_torch).abs().max())
    print("Max residual difference:", (residual_custom - residual_torch).abs().max())
    # 性能评测
    # Warm-up
    for _ in range(warmup_iter):
       _C.add_rms_norm(input_custom, residual_custom, EPSILON, weight_tensor)
       torch_add_rms_norm(input_torch, residual_torch, weight_tensor, EPSILON)
    torch.cuda.synchronize()

    # Custom Kernel
    start_time = time.perf_counter()
    for _ in range(test_iter):
        _C.add_rms_norm(input_custom, residual_custom, EPSILON, weight_tensor)
    torch.cuda.synchronize()
    custom_time = (time.perf_counter() - start_time) / test_iter * 1000 # ms
    print(f"Custom Kernel Time: {custom_time:.4f} ms")

    # PyTorch Naive
    start_time = time.perf_counter()
    for _ in range(test_iter):
        torch_add_rms_norm(input_torch, residual_torch, weight_tensor, EPSILON)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start_time) / test_iter * 1000 # ms
    print(f"PyTorch Naive Time: {torch_time:.4f} ms")
    print(f"🚀 Speedup: {torch_time / custom_time:.2f}x")


if __name__ == "__main__":
    run_test()