import torch
import triton
import triton.language as tl
# --------- Naive PyTorch (作为性能基准) ---------

def store_kvcache_naive_torch(key: torch.Tensor, value: torch.Tensor, k_cache_4d: torch.Tensor, v_cache_4d: torch.Tensor,
                               slot_mapping: torch.Tensor):
    """
    一个纯 PyTorch 的、性能较低的“笨拙”实现，用于性能对比的基准。
    """
    num_tokens = key.shape[0]
    block_size = k_cache_4d.shape[1]

    for i in range(num_tokens):
        # 获取 token i 应该写入的物理槽位
        slot_idx = slot_mapping[i].item()
        
        # 将一维的槽位索引转换成二维的 (物理块ID, 块内偏移)
        block_id = slot_idx // block_size
        offset_in_block = slot_idx % block_size
        
        # 从输入中获取 token i 的 k 和 v
        k_token = key[i]  # Shape: [num_kv_heads, head_dim]
        v_token = value[i] # Shape: [num_kv_heads, head_dim]
        
        # 将 k 和 v 写入 cache 的指定位置
        k_cache_4d[block_id, offset_in_block] = k_token
        v_cache_4d[block_id, offset_in_block] = v_token

# ===================================================================
# 1. 参考实现 (Triton Kernel - 保持不变)
#    Kernel 本身处理逻辑视图，不需要修改
# ===================================================================
"""
该kernel专门负责把新计算的k,v存入物理上的k_cache和v_cache
"""
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx) # slot 代表kv cache中每个token的槽位 -->k/v cache = (num_slot, num_kv_head, head_dim)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache_triton(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_kvheads, head_dim = key.shape
    D = num_kvheads * head_dim
    # print(f"{key.shape=} , D = {num_kvheads*head_dim} N={N}, {slot_mapping.shape=}")
     # --- 关键的修正 ---
    # 将 4D 物理 cache 和 3D 输入都展平成 2D 逻辑视图

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# ===================================================================
# 2. 你的 CUDA Kernel 的导入和调用
# ===================================================================
# --- 修改后的自定义 Wrapper ---
def store_kvcache_custom_kernel(key: torch.Tensor, value: torch.Tensor, k_cache_4d: torch.Tensor,
                                v_cache_4d: torch.Tensor, slot_mapping: torch.Tensor):
    """
    你的自定义 CUDA/HIP kernel 的包装器。
    """
    try:
        from electrock_infer import _C  # <--- !!! 请将此行替换成你的模块名 !!!

        # 假设你的 C++ 接口可以直接处理 4D cache
        _C.paged_store_kvcache(key, value, k_cache_4d, v_cache_4d, slot_mapping)

    except ImportError:
        print("\n\n" + "=" * 80)
        print("错误: 无法导入你的自定义模块。")
        print("在性能测试中将跳过此实现。请确保已正确编译，并更新模块名。")
        print("=" * 80 + "\n\n")
        # 返回一个特殊值，表示此实现不可用
        return None
    return True # 表示成功执行



# ===================================================================
# 2. 性能测试工具
# ===================================================================
def benchmark(func, args, warmup_runs=20, timed_runs=100):
    """
    使用 torch.cuda.Event 精确测量 GPU 函数执行时间的工具。
    """
    # 预热
    for _ in range(warmup_runs):
        func(*args)
    torch.cuda.synchronize()

    # 正式计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(timed_runs):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / timed_runs
    return avg_time_ms

# ===================================================================
# 3. 主测试逻辑
# ===================================================================
if __name__ == "__main__":
    # --- 测试配置 ---
    NUM_TOKENS = (1*1024)
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    BLOCK_SIZE = 32
    TOTAL_BLOCKS = 1024
    DTYPE = torch.bfloat16
    DEVICE = 'cuda'
    
    TOTAL_CACHE_SLOTS = TOTAL_BLOCKS * BLOCK_SIZE

    print("===== Test Configuration (Correctness & Performance) =====")
    print(f"Number of new tokens: {NUM_TOKENS}")
    print(f"KV Heads: {NUM_KV_HEADS}, Head Dim: {HEAD_DIM}")
    print(f"Cache Blocks: {TOTAL_BLOCKS}, Block Size: {BLOCK_SIZE} -> Total Slots: {TOTAL_CACHE_SLOTS}")
    print(f"DType: {DTYPE}, Device: {DEVICE}")
    print("========================================================\n")

    # --- 生成测试数据 ---
    key_input = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    value_input = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    cache_shape_4d = (TOTAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cache_initial = torch.zeros(cache_shape_4d, dtype=DTYPE, device=DEVICE)
    v_cache_initial = torch.zeros(cache_shape_4d, dtype=DTYPE, device=DEVICE)
    possible_slots = torch.randperm(TOTAL_CACHE_SLOTS, device=DEVICE)
    slot_mapping = possible_slots[:NUM_TOKENS].to(torch.int32)

    # --- 1. 正确性验证 ---
    print("\n--- Running Correctness Verification ---")
    k_cache_ref = k_cache_initial.clone()
    v_cache_ref = v_cache_initial.clone()
    # triton kernel
    # store_kvcache_triton(key_input, value_input, k_cache_ref, v_cache_ref, slot_mapping)
    k_cache_custom = k_cache_initial.clone()
    v_cache_custom = v_cache_initial.clone()
    # naive torch 
    store_kvcache_naive_torch(key_input, value_input, k_cache_ref, v_cache_ref, slot_mapping)
    # 检查自定义kernel是否可用
    custom_kernel_available = store_kvcache_custom_kernel(key_input, value_input, k_cache_custom, v_cache_custom, slot_mapping)
    
    if custom_kernel_available:
        k_close = torch.allclose(k_cache_ref, k_cache_custom, atol=1e-3, rtol=1e-3)
        v_close = torch.allclose(v_cache_ref, v_cache_custom, atol=1e-3, rtol=1e-3)
        if k_close and v_close:
            print("✅ Correctness Test Passed: Your custom kernel matches the Triton reference.")
        else:
            print("❌ Correctness Test Failed: Outputs do NOT match.")
    else:
        print("Skipping correctness test for custom kernel as it could not be imported.")


    # --- 2. 性能比较 ---
    print("\n\n--- Running Performance Comparison ---")
    
    # 准备用于性能测试的参数
    args = (key_input, value_input, k_cache_initial.clone(), v_cache_initial.clone(), slot_mapping)

    # 测试 Naive PyTorch
    print("Benchmarking Naive PyTorch...")
    time_naive = benchmark(store_kvcache_naive_torch, args)
    print(f"  -> Average time: {time_naive:.6f} ms")

    # 测试 Triton Kernel
    # print("Benchmarking Triton Kernel...")
    # time_triton = benchmark(store_kvcache_triton, args)
    # print(f"  -> Average time: {time_triton:.6f} ms")

    # 测试 Custom Kernel
    time_custom = float('inf')
    if custom_kernel_available:
        print("Benchmarking Custom Kernel...")
        time_custom = benchmark(store_kvcache_custom_kernel, args)
        print(f"  -> Average time: {time_custom:.6f} ms")
    
    # --- 3. 结果总结 ---
    print("\n\n--- Performance Summary ---")
    print(f"Naive PyTorch : {time_naive:9.6f} ms (Baseline)")
    # print(f"Triton Kernel : {time_triton:9.6f} ms ({time_naive / time_triton:6.2f}x speedup vs Naive)")
    if custom_kernel_available:
        # print(f"Custom Kernel : {time_custom:9.6f} ms ({time_naive / time_custom:6.2f}x speedup vs Naive, {time_triton / time_custom:6.2f}x vs Triton)")
        print(f"Custom Kernel : {time_custom:9.6f} ms ({time_naive / time_custom:6.2f}x speedup vs Naive")
    print("---------------------------\n")

# # ===================================================================
# # 3. 主测试逻辑
# # ===================================================================
# if __name__ == "__main__":
#     # --- 测试配置 (已更新) ---
#     NUM_TOKENS = 128
#     NUM_KV_HEADS = 8
#     HEAD_DIM = 128

#     # --- 新增配置 ---
#     BLOCK_SIZE = 32  # 每个物理块的大小
#     TOTAL_BLOCKS = 256  # 物理块的总数

#     DTYPE = torch.float16
#     DEVICE = 'cuda'

#     # 计算出的总槽位数
#     TOTAL_CACHE_SLOTS = TOTAL_BLOCKS * BLOCK_SIZE

#     print("===== Test Configuration (Updated for 4D Cache) =====")
#     print(f"Number of new tokens: {NUM_TOKENS}")
#     print(f"KV Heads: {NUM_KV_HEADS}, Head Dim: {HEAD_DIM}")
#     print(f"Cache Blocks: {TOTAL_BLOCKS}, Block Size: {BLOCK_SIZE} -> Total Slots: {TOTAL_CACHE_SLOTS}")
#     print(f"DType: {DTYPE}, Device: {DEVICE}")
#     print("====================================================\n")

#     # --- 生成测试数据 (已更新) ---
#     print("--- Generating test data ---")

#     key_input = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
#     value_input = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
#     print(f"Shape of new key/value: {key_input.shape}")

#     # --- 关键改动 ---
#     # 创建 4D 的物理 K/V Cache
#     cache_shape_4d = (TOTAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
#     k_cache = torch.zeros(cache_shape_4d, dtype=DTYPE, device=DEVICE)
#     v_cache = torch.zeros(cache_shape_4d, dtype=DTYPE, device=DEVICE)
#     print(f"Shape of K/V Cache: {k_cache.shape}")

#     # Slot Mapping 逻辑不变，它仍然是 token 到一个扁平化索引的映射
#     possible_slots = torch.randperm(TOTAL_CACHE_SLOTS, device=DEVICE)
#     slot_mapping = possible_slots[:NUM_TOKENS].to(torch.int32)
#     print(f"Shape of slot_mapping: {slot_mapping.shape}")

#     # --- 运行并验证 ---

#     print("\n--- Running reference (Triton) implementation ---")
#     k_cache_ref = k_cache.clone()
#     v_cache_ref = v_cache.clone()
#     store_kvcache(key_input, value_input, k_cache_ref, v_cache_ref, slot_mapping)
#     print("Reference implementation finished.")

#     print("\n--- Running your custom CUDA implementation ---")
#     k_cache_custom = k_cache.clone()
#     v_cache_custom = v_cache.clone()
#     try:
#         store_kvcache_custom_cuda(key_input, value_input, k_cache_custom, v_cache_custom, slot_mapping)
#         print("Custom CUDA implementation finished.")

#         print("\n--- Verifying correctness ---")
#         k_close = torch.allclose(k_cache_ref, k_cache_custom, atol=1e-3, rtol=1e-3)
#         v_close = torch.allclose(v_cache_ref, v_cache_custom, atol=1e-3, rtol=1e-3)

#         if k_close and v_close:
#             print("✅ Test Passed: Your CUDA kernel output matches the reference implementation.")
#         else:
#             print("❌ Test Failed: Outputs do NOT match.")
#             if not k_close:
#                 print("Mismatch found in K_CACHE.")
#                 max_abs_err = (k_cache_ref - k_cache_custom).abs().max().item()
#                 print(f"  Max absolute error: {max_abs_err}")
#             if not v_close:
#                 print("Mismatch found in V_CACHE.")
#                 max_abs_err = (v_cache_ref - v_cache_custom).abs().max().item()
#                 print(f"  Max absolute error: {max_abs_err}")

#     except (ImportError, AttributeError, TypeError) as e:
#         print(f"An error occurred while running the custom kernel: {e}")
#         print("Please make sure you have compiled your kernel and updated the placeholder function.")