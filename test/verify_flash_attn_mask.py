import torch
from flash_attn import flash_attn_varlen_func
def naive_attention_varlen_manual_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    mask_type: str  # 新增参数: 'top-left' 或 'bottom-right'
):
    """
    一个朴素注意力实现，可以手动选择因果掩码的对齐方式。
    """
    output_list = []
    
    # 这个验证脚本只处理 batch_size=1 的情况，以简化逻辑
    assert len(cu_seqlens_q) == 2, "This verification script assumes batch_size=1"
    
    # 由于 batch_size=1, q, k, v 就是单个序列的张量
    # 为了计算方便，将 head 维度放到前面
    q_i = q.transpose(0, 1) # -> [num_heads, seqlen_q, head_dim]
    k_i = k.transpose(0, 1) # -> [num_heads, seqlen_k, head_dim]
    v_i = v.transpose(0, 1) # -> [num_heads, seqlen_k, head_dim]

    # GQA/MQA 支持: 重复 K, V 的头以匹配 Q
    if q_i.shape[0] != k_i.shape[0]:
        num_q_heads = q_i.shape[0]
        num_kv_heads = k_i.shape[0]
        num_groups = num_q_heads // num_kv_heads
        k_i = k_i.repeat_interleave(num_groups, dim=0)
        v_i = v_i.repeat_interleave(num_groups, dim=0)

    attn_scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale

    seqlen_q = q_i.size(1)
    seqlen_k = k_i.size(1)

    # ==================== 核心：根据指令构建不同的掩码 ====================
    if mask_type:
        print(f"    - Naive attention is using '{mask_type}' causal mask.")
        mask = torch.zeros(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool)
        if mask_type == 'top-left':
            # v2.0 行为：简单的左上角对齐因果关系
            # 只有当 j > i 时才屏蔽
            mask = torch.triu(torch.ones_like(mask), diagonal=1)
        elif mask_type == 'bottom-right':
            # v2.1 行为：右下角对齐
            # 假设 q 是序列的最后一部分
            # 只有当 j > i + (k的长度 - q的长度) 时才屏蔽
            offset = seqlen_k - seqlen_q
            row_indices = torch.arange(seqlen_q, device=q.device).view(-1, 1)
            col_indices = torch.arange(seqlen_k, device=q.device).view(1, -1)
            mask = col_indices > row_indices + offset
        
        attn_scores.masked_fill_(mask, float('-inf'))
    # =================================================================

    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v_i).transpose(0, 1).contiguous()
    
    return output

def run_verification():
    # --- 1. 参数设置 ---
    # 使用您例子中的尺寸
    BATCH_SIZE = 1
    SEQ_LEN_Q = 2
    SEQ_LEN_K = 5
    NUM_HEADS = 8
    # 为了简单，我们用标准注意力，非GQA
    NUM_KV_HEADS = NUM_HEADS 
    HEAD_DIM = 64
    DTYPE = torch.float16 # flash-attn 需要半精度
    DEVICE = 'cuda'

    print("--- Verification Setup ---")
    print(f"Batch Size: {BATCH_SIZE}, SeqLen Q: {SEQ_LEN_Q}, SeqLen K: {SEQ_LEN_K}")
    print(f"Num Heads: {NUM_HEADS}, Head Dim: {HEAD_DIM}, DType: {DTYPE}\n")

    # --- 2. 创建输入数据 ---
    # 使用随机数据，但固定种子以保证每次运行结果一致
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # total_tokens = BATCH_SIZE * SEQ_LEN
    total_q = BATCH_SIZE * SEQ_LEN_Q
    total_k = BATCH_SIZE * SEQ_LEN_K

    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = torch.randn(total_k, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)
    v = torch.randn(total_k, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, requires_grad=True)

    # 创建 cu_seqlens
    cu_seqlens_q = torch.tensor([0, SEQ_LEN_Q], dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.tensor([0, SEQ_LEN_K], dtype=torch.int32, device=DEVICE)
    
    # softmax_scale
    scale = 1.0 / (HEAD_DIM ** 0.5)

    # --- 3. 运行三种不同的计算路径 ---

    # 路径 A: 调用官方 flash_attn_varlen_func
    print("--- Path A: Running official flash_attn_varlen_func with causal=True ---")
    output_flash = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, SEQ_LEN_Q, SEQ_LEN_K,
        softmax_scale=scale, causal=True
    )
    print("    - Done.\n")

    # 路径 B: 朴素实现，模拟 v2.0 (左上角对齐)
    print("--- Path B: Running Naive Attention with 'top-left' mask (v2.0 behavior) ---")
    output_naive_v2_0 = naive_attention_varlen_manual_mask(
        q, k, v, cu_seqlens_q, cu_seqlens_k, scale, mask_type='top-left'
    )
    print("    - Done.\n")

    # 路径 C: 朴素实现，模拟 v2.1 (右下角对齐)
    print("--- Path C: Running Naive Attention with 'bottom-right' mask (v2.1 behavior) ---")
    output_naive_v2_1 = naive_attention_varlen_manual_mask(
        q, k, v, cu_seqlens_q, cu_seqlens_k, scale, mask_type='bottom-right'
    )
    print("    - Done.\n")


    # --- 4. 对比结果并得出结论 ---
    print("--- Final Comparison & Conclusion ---")

    # 容忍度可以根据你的 DTYPE 调整
    atol = 1e-2 if DTYPE == torch.float16 else 1e-3

    # 对比 FlashAttention 和 v2.0 的行为
    is_like_v2_0 = torch.allclose(output_flash, output_naive_v2_0, atol=atol)
    print(f"[*] Is flash-attn behavior like v2.0 ('top-left')? -> {is_like_v2_0}")

    # 对比 FlashAttention 和 v2.1 的行为
    is_like_v2_1 = torch.allclose(output_flash, output_naive_v2_1, atol=atol)
    print(f"[*] Is flash-attn behavior like v2.1 ('bottom-right')? -> {is_like_v2_1}")

    print("\n---")
    if is_like_v2_1:
        print("✅ 结论: 您当前安装的 flash-attn 版本在处理不等长序列的因果关系时，遵循 v2.1 及更高版本的“右下角对齐”逻辑。这对于解码场景是正确的。")
    elif is_like_v2_0:
        print("⚠️ 结论: 您当前安装的 flash-attn 版本遵循 v2.0 的“左上角对齐”逻辑。这在解码场景下行为不符合直觉，需要您手动处理或升级库。")
    else:
        print("❌ 结论: flash-attn 的输出与两种朴素实现都不匹配。这可能意味着存在其他问题（如GQA实现细节、数值精度等）或一个未知的Bug。")
    print("---\n")


if __name__ == '__main__':
    # 确保有可用的 CUDA 设备
    if torch.cuda.is_available():
        run_verification()
    else:
        print("CUDA device not found. This script requires a GPU.")
