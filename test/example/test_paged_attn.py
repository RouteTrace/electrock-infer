import torch
import math
import time

# ===================================================================
# 1. 导入你编译好的自定义 Kernel 模块
# !!! 请将 'electrock_infer' 替换成你 setup.py 中定义的包名 !!!
# ===================================================================
try:
    from electrock_infer import _C
except ImportError:
    print("错误：无法导入自定义模块 'electrock_infer'。")
    print("请确保你已经成功编译并安装了你的C++扩展，并且这里的包名与 setup.py 中的 'name' 一致。")
    # 在这种情况下我们仍然可以运行CPU黄金标准，但会跳过自定义kernel
    _C = None

# 辅助函数
def div_ceil(a, b):
    return (a + b - 1) // b

# =================================================================================================
# 2. CPU黄金标准Attention (保持不变)
# =================================================================================================
def cpu_attention_golden(
    q_in: torch.Tensor,
    k_in_contiguous: torch.Tensor,
    v_in_contiguous: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float
) -> torch.Tensor:
    assert not q_in.is_cuda, "Golden reference must run on CPU"
    
    batch_size, q_head_num, head_dim = q_in.shape
    _, kv_head_num, _, _ = k_in_contiguous.shape
    group_size = q_head_num // kv_head_num

    o_out = torch.empty_like(q_in, dtype=torch.float32)

    for b in range(batch_size):
        cur_seq_len = context_lens[b].item()
        for h in range(q_head_num):
            kv_h = h // group_size
            q = q_in[b, h]
            k = k_in_contiguous[b, kv_h, :cur_seq_len, :]
            v = v_in_contiguous[b, kv_h, :cur_seq_len, :]

            s = torch.matmul(q.unsqueeze(0).float(), k.transpose(0, 1).float()) * softmax_scale
            p = torch.softmax(s, dim=-1)
            o = torch.matmul(p, v.float())
            o_out[b, h] = o.squeeze(0)
            
    return o_out.to(q_in.dtype)

# ===================================================================
# 3. 参数化的主测试函数
# ===================================================================
def verify_paged_attn(dtype_str: str):
    """
    一个完整的验证和性能测试函数，数据类型由 dtype_str 参数控制。
    """
    
    # --- <--- 关键改动: Dtype 映射和检查 ---
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"不支持的数据类型: {dtype_str}. 请选择 {list(dtype_map.keys())}")
    
    dtype = dtype_map[dtype_str]
    device = torch.device("cuda")

    # 检查硬件是否支持 bf16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"--- 跳过 bfloat16 测试: 当前GPU不支持 bfloat16 ---")
        return

    # 动态设置验证的容差
    tolerances = {
        "float16": {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 1.6e-2, "rtol": 1e-2}, # bf16 精度较低，容差稍大
        "float32": {"atol": 1e-5, "rtol": 1e-5},
    }[dtype_str]
    # --- 关键改动结束 ---

    # --- 1. 设置测试参数 ---
    BATCH_SIZE = 64
    Q_HEAD_NUM = 16
    KV_HEAD_NUM = 4
    HEAD_DIM = 128
    BLOCK_SIZE = 32
    MAX_CONTEXT_LEN = 100
    TOTAL_PHYSICAL_BLOCKS = BATCH_SIZE * div_ceil(MAX_CONTEXT_LEN, BLOCK_SIZE)

    WARMUP_ITER = 5
    TEST_ITER = 20

    context_lens_vec = [MAX_CONTEXT_LEN] * BATCH_SIZE
    context_lens = torch.tensor(context_lens_vec, dtype=torch.int32)
    
    max_block_num = div_ceil(MAX_CONTEXT_LEN, BLOCK_SIZE)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

    print("\n" + "="*60)
    print(f"🚀 开始测试: DType = {dtype_str}")
    print("="*60)

    # --- 2. 创建张量 (使用参数化的 dtype) ---
    options = {"device": device, "dtype": dtype}
    q_tensor = torch.randn(BATCH_SIZE, Q_HEAD_NUM, HEAD_DIM, dtype=dtype)
    max_len_k_v = max_block_num * BLOCK_SIZE
    k_contiguous = torch.randn(BATCH_SIZE, KV_HEAD_NUM, max_len_k_v, HEAD_DIM, dtype=dtype)
    v_contiguous = torch.randn(BATCH_SIZE, KV_HEAD_NUM, max_len_k_v, HEAD_DIM, dtype=dtype)

    block_table = torch.empty(BATCH_SIZE, max_block_num, dtype=torch.int32)
    k_cache = torch.empty(TOTAL_PHYSICAL_BLOCKS, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM, **options)
    v_cache = torch.empty(TOTAL_PHYSICAL_BLOCKS, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM, **options)

    physical_block_counter = 0
    for b in range(BATCH_SIZE):
        len_ = context_lens_vec[b]
        num_logical_blocks = div_ceil(len_, BLOCK_SIZE)
        for i in range(num_logical_blocks):
            physical_idx = physical_block_counter
            physical_block_counter += 1
            block_table[b, i] = physical_idx
            start_token, end_token = i * BLOCK_SIZE, min((i + 1) * BLOCK_SIZE, len_)
            num_tokens = end_token - start_token
            k_cache[physical_idx, :num_tokens] = k_contiguous[b, :, start_token:end_token, :].transpose(0, 1)
            v_cache[physical_idx, :num_tokens] = v_contiguous[b, :, start_token:end_token, :].transpose(0, 1)
    
    # --- 3. 正确性验证 ---
    print(f"[{dtype_str}] 正在进行正确性验证...")
    q_gpu = q_tensor.contiguous().to(device)
    k_cache_gpu = k_cache.contiguous().to(device)
    v_cache_gpu = v_cache.contiguous().to(device)
    block_table_gpu = block_table.contiguous().to(device)
    context_lens_gpu = context_lens.contiguous().to(device)
    
    o_gpu = torch.Tensor
    if _C:
        o_gpu = _C.paged_attn_varlen(q_gpu, k_cache_gpu, v_cache_gpu, max_len_k_v, context_lens_gpu, block_table_gpu, softmax_scale)
        torch.cuda.synchronize()

        o_ref = cpu_attention_golden(q_tensor, k_contiguous, v_contiguous, context_lens, softmax_scale)
        o_gpu_cpu = o_gpu.cpu()

        is_close = torch.allclose(o_ref, o_gpu_cpu, **tolerances)
        max_diff = (o_ref - o_gpu_cpu).abs().max().item()
        if is_close:
            print(f"✅ [{dtype_str}] Correctness check PASSED!")
            print(f"   - Maximum absolute difference: {max_diff}")
        else:
            print(f"❌ [{dtype_str}] Correctness check FAILED!")
            print(f"   - Maximum absolute difference: {max_diff}")
    else:
        print(f"⚠️ [{dtype_str}] Skipped correctness check because custom module is not available.")

    # --- 4. 性能评测 ---
    if _C:
        print(f"[{dtype_str}] 正在进行性能评测...")
        # 预热
        for _ in range(WARMUP_ITER):
            _C.paged_attn_varlen(q_gpu, k_cache_gpu, v_cache_gpu, max_len_k_v, context_lens_gpu, block_table_gpu, softmax_scale)
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(TEST_ITER):
            _C.paged_attn_varlen(q_gpu, k_cache_gpu, v_cache_gpu, max_len_k_v, context_lens_gpu, block_table_gpu, softmax_scale)
        stop_event.record()
        torch.cuda.synchronize()
        
        custom_kernel_ms = start_event.elapsed_time(stop_event) / TEST_ITER
        print(f"✅ [{dtype_str}] Benchmark complete. Average time: {custom_kernel_ms:.4f} ms")
    else:
        print(f"⚠️ [{dtype_str}] Skipped benchmark because custom module is not available.")


if __name__ == "__main__":
    # --- <--- 关键改动: 循环测试多种数据类型 ---
    # 你可以在这个列表中添加或删除你想测试的类型
    # dtypes_to_test = ["float16", "bfloat16"]
    dtypes_to_test = ["bfloat16"]
    
    for dtype_str in dtypes_to_test:
        try:
            verify_paged_attn(dtype_str)
        except Exception as e:
            print(f"\n❌ [{dtype_str}] An error occurred during the test: {e}")