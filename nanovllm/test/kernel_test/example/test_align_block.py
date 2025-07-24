import torch
import torch.nn.functional as F
import time
# 确保您已经编译并安装了您的自定义扩展
import electrock_infer
def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int
):
    """
    使用纯 PyTorch 实现 moe_align_block_size 的逻辑。
    
    Args:
        topk_ids (torch.Tensor): 1D Tensor, shape [num_tokens * top_k], 包含每个 token-expert 对被分配到的专家ID。
        num_experts (int): 专家总数。
        block_size (int): 对齐的块大小。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        - sorted_token_ids: 排序并填充后的 token-expert 对的原始索引。
        - expert_ids: 每个 padded block 对应的专家 ID。
        - num_tokens_post_pad: 填充后的总 token 数。
    """
    device = topk_ids.device
    
    # 1. 统计每个专家的 token 数量 (等价于 CUDA kernel 的并行计数)
    tokens_per_expert = torch.bincount(topk_ids, minlength=num_experts).long()

    # 2. 计算每个专家按 block_size 向上取整（补齐）后的 token 数
    padded_tokens_per_expert = torch.ceil(tokens_per_expert / block_size).long() * block_size
    
    # 3. 计算每个专家在最终排序数组中的起始偏移量 (等价于 CUDA kernel 的 cumsum)
    #    F.pad 在开头补 0, 和 kernel 的 cumsum[0]=0 逻辑一致
    expert_offsets = F.pad(torch.cumsum(padded_tokens_per_expert, dim=0)[:-1], (1, 0))
    
    # 4. 计算填充后的总 token 数
    num_tokens_post_pad = torch.sum(padded_tokens_per_expert).reshape(1)

    # 5. 生成 expert_ids 映射表
    #    这个张量告诉我们，在最终的排序数组中，每个 block 属于哪个专家
    num_blocks_per_expert = padded_tokens_per_expert // block_size
    expert_ids = torch.arange(num_experts, device=device).repeat_interleave(num_blocks_per_expert)

    # 6. 数据重排 (Permutation), 这是最关键也最慢的一步
    #    模拟 CUDA Kernel 将每个 token 索引散射 (scatter) 到正确位置的过程
    sorted_token_ids = torch.full((num_tokens_post_pad.item(),), -1, dtype=torch.int32, device=device)
    
    # 维护一个计数器，用于计算每个 token 在其专家组内的局部排名 (rank)
    # 这等价于 CUDA Kernel 中每个线程私有的 tokens_cnts 数组的功能
    current_expert_counts = torch.zeros_like(tokens_per_expert)

    # 遍历每一个原始的 token-expert 对
    for i in range(topk_ids.numel()):
        expert_id = topk_ids[i].item()
        
        # 获取该 token 在其专家组内的局部排名
        rank = current_expert_counts[expert_id].long()
        
        # 计算最终位置 = 该专家的全局起始位置 + 局部排名
        final_pos = expert_offsets[expert_id] + rank
        
        # 将原始索引 i 放入计算出的最终位置
        sorted_token_ids[final_pos] = i
        
        # 更新计数器
        current_expert_counts[expert_id] += 1
        
    return sorted_token_ids, expert_ids, num_tokens_post_pad


# ============================================================================
# 性能测试辅助函数
# ============================================================================
def benchmark(is_torch, description, num_runs=1, warmup_runs=10):
    print(f"Benchmarking {description}...".ljust(40), end="")
    if is_torch:
        func = lambda: moe_align_block_size_torch(topk_ids_input, NUM_EXPERTS, BLOCK_SIZE)
    else:
        func = lambda: electrock_infer.moe_align_block_size(
                  topk_ids_input, NUM_EXPERTS, BLOCK_SIZE, 
                  sorted_ids_custom, expert_ids_custom, post_pad_custom)
    # 预热
    for _ in range(warmup_runs):
        func()

    # 精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_runs
    
    print(f"Avg Time: {avg_time_ms:.6f} ms")
    return avg_time_ms

# ============================================================================
# 主测试逻辑
# ============================================================================
if __name__ == "__main__":
    # --- 1. 配置测试参数 ---
    NUM_TOKENS = 1024 * 64
    TOP_K = 2
    NUM_EXPERTS = 8
    BLOCK_SIZE = 64
    DTYPE = torch.int32 # topk_ids 通常是整数类型
    DEVICE = 'cuda'

    print("="*60)
    print("Starting moe_align_block_size Correctness & Performance Test")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: {NUM_TOKENS=}, {TOP_K=}, {NUM_EXPERTS=}, {BLOCK_SIZE=}")
    print("="*60)
    
    # --- 2. 创建输入数据 ---
    # 模拟 gating 后的 top-k 专家索引
    topk_ids_input = torch.randint(
        0, NUM_EXPERTS, 
        (NUM_TOKENS * TOP_K,), 
        dtype=DTYPE, 
        device=DEVICE
    )

    # --- 3. 运行 PyTorch 版本以获取黄金标准 (Ground Truth) ---
    print("\nRunning PyTorch version to get ground truth...")
    sorted_ids_torch, expert_ids_torch, post_pad_torch = moe_align_block_size_torch(
        topk_ids_input, NUM_EXPERTS, BLOCK_SIZE
    )
    print("PyTorch version finished.")

    # --- 4. 运行您的自定义 CUDA Kernel ---
    print("\nRunning custom CUDA kernel...")
    # 准备输出张量
    # 注意：必须预先分配足够大的空间，特别是 sorted_token_ids
    total_padded_size = post_pad_torch.item()
    num_blocks = total_padded_size // BLOCK_SIZE
    
    sorted_ids_custom = torch.empty((total_padded_size,), dtype=torch.int32, device=DEVICE)
    expert_ids_custom = torch.empty((num_blocks,), dtype=torch.int32, device=DEVICE)
    post_pad_custom = torch.empty((1,), dtype=torch.int32, device=DEVICE)
    
    electrock_infer.moe_align_block_size(
        topk_ids_input,
        NUM_EXPERTS,
        BLOCK_SIZE,
        sorted_ids_custom,
        expert_ids_custom,
        post_pad_custom
    )
    print("Custom kernel finished.")

    # --- 5. 验证正确性 ---
    print("\nVerifying correctness...")
    
    # 逐个比较输出张量
    correct1 = torch.equal(post_pad_torch, post_pad_custom)
    correct2 = torch.equal(expert_ids_torch, expert_ids_custom)
    # 对于 sorted_ids，由于填充部分是-1，可能存在未被写入的区域。
    # 我们只比较那些被实际写入的位置
    mask = sorted_ids_torch != -1
    correct3 = torch.equal(sorted_ids_torch[mask], sorted_ids_custom[mask])

    print(f"Correctness of 'num_tokens_post_pad': {'OK' if correct1 else 'FAIL'}")
    print(f"Correctness of 'expert_ids': {'OK' if correct2 else 'FAIL'}")
    print(f"Correctness of 'sorted_token_ids': {'OK' if correct3 else 'FAIL'}")

    if not (correct1 and correct2 and correct3):
        print("\nERROR: Custom kernel produced incorrect results. Aborting benchmark.")
        # 如果结果不一致，可以打印出来进行调试
        # print("PyTorch sorted_ids:", sorted_ids_torch[mask])
        # print("Custom sorted_ids: ", sorted_ids_custom[mask])
        exit()
        
    # # --- 6. 运行性能基准测试 ---
    # print("\nRunning benchmarks...")
    
    # benchmark(True, 
    #           "PyTorch Version")
              
    # benchmark(False,
    #           "Custom CUDA Kernel")
              
    print("\nTest finished.")
    print("="*60)


""" 
============================================================
Starting moe_align_block_size Correctness & Performance Test
Device: NVIDIA A800-SXM4-80GB
Config: NUM_TOKENS=4096, TOP_K=2, NUM_EXPERTS=8, BLOCK_SIZE=64
============================================================

Running PyTorch version to get ground truth...
PyTorch version finished.

Running custom CUDA kernel...
Custom kernel finished.

Verifying correctness...
Correctness of 'num_tokens_post_pad': OK
Correctness of 'expert_ids': OK
Correctness of 'sorted_token_ids': OK

Running benchmarks...
Benchmarking PyTorch Version...         Avg Time: 438.339219 ms
Benchmarking Custom CUDA Kernel...      Avg Time: 0.089446 ms

Test finished.
============================================================ 
"""