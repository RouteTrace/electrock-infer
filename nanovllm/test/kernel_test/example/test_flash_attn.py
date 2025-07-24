import torch
import torch.nn.functional as F
import time
import sys

# ==============================================================================
# 重要：请将 'electrock_infer' 替换为您编译的C++扩展的实际名称
# ==============================================================================
try:
    # 假设您的绑定模块名为 electrock_infer
    import electrock_infer as custom_ops
except ImportError:
    print("❌ 无法导入自定义C++扩展。")
    print("   请确保已经成功编译，并且这里的模块名称与您setup.py中定义的一致。")
    sys.exit(1)

def repeat_kv(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    辅助函数，用于GQA中重复K和V张量，使其与MHA兼容。
    """
    if group_size == 1:
        return x
    
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    
    return (
        x.unsqueeze(2)  # [B, h_kv, 1, S, D]
         .expand(batch_size, num_kv_heads, group_size, seq_len, head_dim)  # [B, h_kv, G, S, D]
         .reshape(batch_size, num_kv_heads * group_size, seq_len, head_dim)  # [B, h_kv*G, S, D]
    )

def verify_gqa_python_preallocated_output():
    # --- 1. GQA 参数设置 ---
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
    
    BATCH_SIZE = 64
    SEQ_LEN = 128
    IS_CAUSAL = True

    # 性能评测参数
    WARMUP_ITER = 5
    TEST_ITER = 20

    if not torch.cuda.is_available():
        print("❌ 未找到CUDA设备，程序中止。")
        return

    device = "cuda"
    dtype = torch.half

    print("===========================================================")
    print("🚀 开始 (Python) GQA FlashAttention 验证与基准测试")
    print(f"配置: B={BATCH_SIZE}, H_Q={NUM_Q_HEADS}, H_KV={NUM_KV_HEADS}, N={SEQ_LEN}, d={HEAD_DIM}")
    print(f"分组大小 (Group Size): {GROUP_SIZE}")
    print("===========================================================")

    # --- 2. 创建GQA输入数据 ---
    # Q的形状使用 NUM_Q_HEADS
    q = torch.randn(BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    # K和V的形状使用 NUM_KV_HEADS
    k = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    
    # --- 关键改动：预先分配好用于接收结果的输出张量 O ---
    o_custom = torch.empty_like(q)

    # --- 3. 正确性验证 ---
    print("\n[1/4] 正在执行正确性检查...")
    
    # PyTorch 官方参考实现
    k_repeat = repeat_kv(k, GROUP_SIZE)
    v_repeat = repeat_kv(v, GROUP_SIZE)
    o_torch_ref = F.scaled_dot_product_attention(
        q, k_repeat, v_repeat, is_causal=IS_CAUSAL
    )

    # 您的自定义核函数计算
    # 注意：现在 o_custom 是作为最后一个参数传入
    if IS_CAUSAL:
        custom_ops.flash_attn_simplified_causal_varlen_gqa(q, k, v, o_custom)
    else:
        # 假设您的非因果GQA核函数也已绑定。如果没有，这里需要替换成对应的调用
        # 这里我们继续使用 causal 变体作为演示
        custom_ops.flash_attn_simplified_causal_varlen_gqa(q, k, v, o_custom)

    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(o_custom, o_torch_ref, rtol=1e-3, atol=1e-2)
        print("✅ 正确性检查通过！")
        diff = (o_custom - o_torch_ref).abs()
        print(f"   - 最大绝对误差: {diff.max().item():.6f}")
    except AssertionError as e:
        print("❌ 正确性检查失败！输出不匹配。")
        diff = (o_custom - o_torch_ref).abs()
        print(f"   - 最大绝对误差: {diff.max().item():.6f}")
        return

    # --- 4. 性能评测 ---
    print("\n[2/4] 正在预热 (Warmup) 内核...")
    for _ in range(WARMUP_ITER):
        # 预热您的核函数
        custom_ops.flash_attn_simplified_causal_varlen_gqa(q, k, v, o_custom)
        # 预热PyTorch参考实现
        k_rep, v_rep = repeat_kv(k, GROUP_SIZE), repeat_kv(v, GROUP_SIZE)
        _ = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=IS_CAUSAL)
    torch.cuda.synchronize()
    print("✅ 预热完成。")

    # a) 评测您的自定义GQA内核
    print("\n[3/4] 正在评测您的自定义GQA内核...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(TEST_ITER):
        custom_ops.flash_attn_simplified_causal_varlen_gqa(q, k, v, o_custom)
    end_event.record()
    
    torch.cuda.synchronize()
    custom_kernel_ms = start_event.elapsed_time(end_event)
    avg_custom_ms = custom_kernel_ms / TEST_ITER
    print("✅ 自定义GQA内核评测完成。")

    # b) 评测PyTorch官方GQA实现
    print("\n[4/4] 正在评测PyTorch官方GQA实现...")
    start_event.record()
    for _ in range(TEST_ITER):
        k_rep = repeat_kv(k, GROUP_SIZE)
        v_rep = repeat_kv(v, GROUP_SIZE)
        _ = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=IS_CAUSAL)
    end_event.record()

    torch.cuda.synchronize()
    ref_kernel_ms = start_event.elapsed_time(end_event)
    avg_ref_ms = ref_kernel_ms / TEST_ITER
    print("✅ PyTorch官方GQA实现评测完成。")

    # --- 5. 打印性能对比结果 ---
    print("\n===========================================================")
    print("📊 GQA 基准测试结果")
    print("===========================================================")
    print(f"您的自定义GQA内核 : {avg_custom_ms:.4f} ms / 迭代")
    print(f"PyTorch官方参考   : {avg_ref_ms:.4f} ms / 迭代")
    print("-----------------------------------------------------------")
    if avg_custom_ms > 0:
        speedup = avg_ref_ms / avg_custom_ms
        print(f"🚀 加速比 vs PyTorch : {speedup:.2f}x")
    print("===========================================================")


if __name__ == "__main__":
    verify_gqa_python_preallocated_output()