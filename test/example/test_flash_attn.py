import torch
import math
import numpy as np

# ===================================================================
# 1. 导入你编译好的自定义 Kernel 模块
# !!! 请将 'electrock_infer' 替换成你 setup.py 中定义的包名 !!!
# ===================================================================
try:
    from electrock_infer import _C
except ImportError:
    print("错误：无法导入自定义模块 'electrock_infer'。")
    print("请确保你已经成功编译并安装了你的C++扩展，并且这里的包名与 setup.py 中的 'name' 一致。")
    exit()

# =================================================================================================
# 2. 参考实现 (Golden Standard)
#    使用 PyTorch 的高级 API 来模拟你的 kernel 的行为，用于验证 kernel 结果
# =================================================================================================
def reference_flash_attn(
    Q: torch.Tensor,              # [total_tokens, Q_head, head_dim]
    K: torch.Tensor,              # [total_tokens, KV_head, head_dim]
    V: torch.Tensor,              # [total_tokens, KV_head, head_dim]
    cu_seqlens: torch.Tensor,     # [batch_size + 1]
    softmax_scale: float,
    is_causal: bool
) -> torch.Tensor:
    """
    一个纯 PyTorch 实现，用于模拟 FlashAttention 的 varlen gqa causal 逻辑。
    这个函数的结果将被用作“正确答案”。
    """
    # 为了得到更精确的参考结果，我们在 float32 下进行计算
    Q_ref = Q.to(torch.float32)
    K_ref = K.to(torch.float32)
    V_ref = V.to(torch.float32)

    batch_size = cu_seqlens.size(0) - 1
    q_head_num = Q_ref.size(1)
    kv_head_num = K_ref.size(1)
    head_dim = Q_ref.size(2)
    group_size = q_head_num // kv_head_num

    output_frags = []
    cu_seqlens_cpu = cu_seqlens.cpu().numpy()

    for i in range(batch_size):
        seq_start = cu_seqlens_cpu[i]
        seq_end = cu_seqlens_cpu[i+1]
        seq_len = seq_end - seq_start
        
        if seq_len == 0:
            continue

        # 1. 从 packed tensor 中切片出当前序列
        q_seq = Q_ref[seq_start:seq_end, :, :] # [seq_len, Q_head, head_dim]
        k_seq = K_ref[seq_start:seq_end, :, :] # [seq_len, KV_head, head_dim]
        v_seq = V_ref[seq_start:seq_end, :, :] # [seq_len, KV_head, head_dim]

        # 2. 处理 GQA：重复 K/V head 以匹配 Q head
        k_gqa = k_seq.repeat_interleave(group_size, dim=1) # [seq_len, Q_head, head_dim]
        v_gqa = v_seq.repeat_interleave(group_size, dim=1) # [seq_len, Q_head, head_dim]

        # 3. 计算 Attention Score (S = Q @ K^T)
        # 为了进行批处理矩阵乘法，我们将 head 维度提前
        q_seq_bmm = q_seq.transpose(0, 1) # [Q_head, seq_len, head_dim]
        k_seq_bmm = k_gqa.transpose(0, 1) # [Q_head, seq_len, head_dim]
        
        attn_scores = torch.bmm(q_seq_bmm, k_seq_bmm.transpose(1, 2)) * softmax_scale # [Q_head, seq_len, seq_len]

        # 4. 应用 Causal Mask
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), diagonal=1)
            attn_scores.masked_fill_(mask, float('-inf'))

        # 5. 计算 Softmax (P)
        attn_probs = torch.softmax(attn_scores, dim=-1) # [Q_head, seq_len, seq_len]

        # 6. 计算 Output (O = P @ V)
        v_seq_bmm = v_gqa.transpose(0, 1) # [Q_head, seq_len, head_dim]
        out_seq_bmm = torch.bmm(attn_probs, v_seq_bmm) # [Q_head, seq_len, head_dim]

        # 7. 恢复维度为 [seq_len, Q_head, head_dim] 并添加到列表中
        output_frags.append(out_seq_bmm.transpose(0, 1))

    # 8. 将所有序列的输出拼接回 packed 格式
    if not output_frags:
        return torch.empty_like(Q)
        
    O_ref = torch.cat(output_frags, dim=0)

    # 将最终结果转回与 kernel 输出相同的类型
    return O_ref.to(Q.dtype)


# ===================================================================
# 3. 主测试逻辑
# ===================================================================
if __name__ == "__main__":
    # --- 测试配置 ---
    # 你可以修改这些参数来测试不同的场景
    BATCH_SIZE = 64
    Q_HEAD_NUM = 16
    KV_HEAD_NUM = 4
    HEAD_DIM = 128
    DTYPE = torch.float16
    DEVICE = 'cuda'
    IS_CAUSAL = True
    
    # 使用随机的、不同长度的序列来严格测试 varlen 逻辑
    SEQ_LENS = torch.randint(low=1, high=1024, size=(BATCH_SIZE,), device=DEVICE)
    
    # --- 打印配置 ---
    print("===== Test Configuration =====")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Seq Lengths: {SEQ_LENS.cpu().numpy()}")
    print(f"Q Heads: {Q_HEAD_NUM}, KV Heads: {KV_HEAD_NUM}, Head Dim: {HEAD_DIM}")
    print(f"DType: {DTYPE}, Device: {DEVICE}, Causal: {IS_CAUSAL}")
    print("==============================\n")

    # --- 生成测试数据 ---
    print("--- Generating test data ---")
    total_tokens = torch.sum(SEQ_LENS).item()
    max_seqlen = torch.max(SEQ_LENS).item()
    
    # 创建 cu_seqlens (cumulative sequence lengths)
    cu_seqlens = torch.cat([torch.tensor([0], device=DEVICE, dtype=torch.int32), SEQ_LENS.cumsum(0, dtype=torch.int32)])
    
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # 创建 Q, K, V 张量 (packed format)
    Q = torch.randn((total_tokens, Q_HEAD_NUM, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    K = torch.randn((total_tokens, KV_HEAD_NUM, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    V = torch.randn((total_tokens, KV_HEAD_NUM, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    
    print(f"Total tokens: {total_tokens}, Max seqlen: {max_seqlen}")
    print(f"Shape of Q: {Q.shape}")
    print(f"Shape of K: {K.shape}")
    print(f"Shape of V: {V.shape}")
    print(f"cu_seqlens: {cu_seqlens}")

    # --- 运行并验证 ---
    # 1. 运行参考实现
    print("\n--- Running reference (PyTorch) implementation ---")
    O_ref = reference_flash_attn(Q, K, V, cu_seqlens, softmax_scale, IS_CAUSAL)
    print("Reference implementation finished.")
    
    # 2. 运行你的 HIP Kernel
    print("\n--- Running your custom HIP kernel ---")
    try:
        O_kernel = _C.flash_attn_causal_varlen_gqa_hip(
            Q,
            K,
            V,
            max_seqlen,
            cu_seqlens,
            max_seqlen,
            cu_seqlens,
            softmax_scale,
            IS_CAUSAL
        )
        print("Custom HIP kernel finished.")

        # 3. 对比结果
        print("\n--- Verifying correctness ---")
        
        # 将两个张量都转为 Float32 以进行稳定比较
        O_ref_f32 = O_ref.to(torch.float32)
        O_kernel_f32 = O_kernel.to(torch.float32)

        # 检查 allclose
        # 对于 float16，误差容忍度需要放宽一些
        is_close = torch.allclose(O_ref_f32, O_kernel_f32, atol=1e-2, rtol=1e-2)
        
        if is_close:
            print("✅ Test Passed: Your HIP kernel output is close to the reference implementation.")
        else:
            print("❌ Test Failed: Outputs are NOT close.")

        # 计算并打印详细误差
        diff = (O_ref_f32 - O_kernel_f32).abs()
        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()

        print(f"  Max Absolute Error:  {max_abs_err:.6f}")
        print(f"  Mean Absolute Error: {mean_abs_err:.6f}")

        # 如果误差过大，打印一些值方便调试
        if not is_close:
            print("\n--- Sample values for debugging (first token, first head) ---")
            print("O_ref[0, 0, :8]:   ", O_ref_f32[0, 0, :8])
            print("O_kernel[0, 0, :8]:", O_kernel_f32[0, 0, :8])
            print("Difference[0, 0, :8]:", diff[0, 0, :8])

    except Exception as e:
        print("\n--- An error occurred while running the custom HIP kernel ---")
        print(e)