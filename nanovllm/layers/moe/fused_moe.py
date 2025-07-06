import torch
import triton
import triton.language as tl
import functools
from typing import Optional, Tuple
import electrock_infer
from nanovllm.layers.moe.moe_align_block_size import moe_align_block_size

import sys

# ----------------------------------------------------------------------------
# 1. 简化的 Triton 内核 (Simplified Triton Kernel)
#    只保留了 FP16/BF16 的计算路径，移除了所有量化相关的复杂逻辑。
# ----------------------------------------------------------------------------
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, 
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM, num_valid_tokens,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, # 64
    BLOCK_SIZE_N: tl.constexpr, # 64
    BLOCK_SIZE_K: tl.constexpr, # 32
    GROUP_SIZE_M: tl.constexpr, # 8
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Triton Kernel for Fused MoE.
    此内核执行一个稀疏矩阵乘法: C = A @ B_expert。
    A: 输入的 token 张量 (hidden_states)。
    B: 所有专家的权重堆叠起来的张量。
    C: 计算结果的输出张量。
    关键在于，它通过 sorted_token_ids 和 expert_ids 来高效地索引，
    确保每个 token 只与它被分配到的专家权重相乘。
    """
    # 1.1. 计算当前程序实例（pid）负责计算的块
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M) # 每个BLOCK都对应一个expert
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # N=2*I, 计算N方向需要多少个block
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # 
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    #pid_m 决定该实例对应哪个expert
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    #pid_n 决定了当前程序实例负责计算输出的哪一个N维度的块（BLOCK_SIZE_N列
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 1.2. 创建指向 A 和 B 矩阵的初始指针
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
        
    # 获取经过排序和填充后的 token ID
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    # token_mask = offs_token < num_valid_tokens # 被padding的位置的值都是=num_valid_tokens
    token_mask = offs_token != -1

    
    # 获取当前块对应的专家 ID
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64) # pid_m对应一个block的专家id
    # 如果专家 ID 为 -1，表示这个专家不在当前 GPU 上（专家并行的情况）
    # 在我们简化的场景下，这通常不会发生，但这是完备性检查
    # if off_experts == -1:
    #     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    #     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    #     c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    #     c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    #     tl.store(c_ptrs, accumulator, mask=c_mask)
    #     return

    # 准备 A 和 B 的指针
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_token // top_k :定位hidden_sates的行号， * stride_am：指向对应行的首位索引，将前BLOCK_SIZE_K的块取出
    # BUG fix：如果offs_token = PAD ，会出现越界情况
    safe_offs_token = tl.where(token_mask, offs_token, 0)
    a_ptrs = a_ptr + (safe_offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak) # 指向stored中选定的hidden_states的行，[BLOCK_SIZE_M, BLOCK_SIZE_K]，其中
    
    #off_experts * stride_be 让指针直接跳到B这个三维张量中专家权重矩阵的起始地址。 offs_bn 会进一步偏移，指向所选专家矩阵的特定列块
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) # 最终得到 [BLOCK_SIZE_K, BLOCK_SIZE_N] 列块的指针

    # 1.3. 迭代计算 C 矩阵的一个块
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 和 B 的块
        # mask：广播后得到 [BLOCK_SIZE_M, BLOCK_SIZE_K] 的 mask，只有 token 有效且 K 维度未越界的位置为 True，其它为 False。
        # 只加载有效 token 且 K 维度未越界的 hidden_states 数据，其余位置填 0.0。
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0) 
        # mask： 广播后，mask 形状为 [BLOCK_SIZE_K, BLOCK_SIZE_N]。
        # 只加载 K 维度未越界的专家权重数据，其余位置填 0.0。
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 核心计算：点积累加
        accumulator += tl.dot(a, b)
        
        # 移动指针到下一个 K 块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 1.4. (可选) 乘以路由权重
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + safe_offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # 1.5. 写回结果
    c_ptrs = c_ptr + stride_cm * safe_offs_token[:, None] + stride_cn * offs_bn[None, :]
    c_mask = token_mask[:, None] & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# ----------------------------------------------------------------------------
# 2. 专家选择函数 (Expert Selection Function)
# ----------------------------------------------------------------------------
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据门控输出选择 top-k 个专家。
    """
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    # ops.topk_softmax 是 vLLM 提供的自定义 CUDA 算子，用于高效执行 softmax+topk
    M, _ = gating_output.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=gating_output.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)
    # token_expert_indicies = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)

    use_torch = True
    if use_torch:
        from nanovllm.layers.moe.softmax_topk import topk_softmax_torch
        topk_weights, topk_ids = topk_softmax_torch(gating_output, topk)
    else:
        # TODO: 已验证，需要优化
        electrock_infer.topk_softmax(
            topk_weights,
            topk_ids,
            gating_output.float(),
        )
    

    # 重新归一化 top-k 的权重，使其和为1
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# ----------------------------------------------------------------------------
# 3. 主入口函数 (Main Entrypoint)
# ----------------------------------------------------------------------------
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool=True,
    inplace: bool = False,
    activation: str = "silu",
) -> torch.Tensor:
    """
    核心的 Fused MoE 计算函数（无量化版本）。
    
    流程: 
    1. 路由: 根据 gating_output 选择 top-k 专家。
    2. 计算: 执行两次块稀疏矩阵乘法 (w1 和 w2)，中间穿插激活函数。

    Args:
        hidden_states: 输入张量，形状为 (num_tokens, hidden_size)。
        w1: 融合后的 gate_proj 和 up_proj 权重，形状为 (E, 2 * I, H)。
        w2: down_proj 权重，形状为 (E, H, I)。
        gating_output: 门控网络的输出，形状为 (num_tokens, num_experts)。
        topk: 每个 token 选择的专家数量。
        renormalize: 是否对 topk 权重进行归一化。
        inplace: 是否原地修改 hidden_states。
        activation: 激活函数，通常是 "silu"。
    """
    # 3.1. 约束检查与参数准备
    num_tokens, hidden_size = hidden_states.shape
    # print(f"F : hidden_state[0:10]:{hidden_states[0,:10]}")
    num_experts, intermediate_size_x2, _ = w1.shape
    intermediate_size = intermediate_size_x2 // 2
    # print(f"F : router_logits[0:]:{gating_output[0,:]}")
    # 3.2. 专家选择 (topk_softmax)
    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)
    # print(f"F : topk_weights[0:]:{topk_weights[0,:]} ; topk_ids[0:] :{topk_ids[0:]}")
    # 3.3. 数据对齐与准备
    # 为了让 Triton Kernel 高效读取，需要对 token 进行重排
    # `moe_align_block_size` 是一个关键的预处理步骤
    # 它返回排序后的 token id，以及每个计算块应该使用哪个 expert id
    # 这是实现高性能块稀疏计算的核心
    config = _get_moe_config(num_tokens, num_experts, intermediate_size, topk)
    # print(f"F : get_moe_config : {config}\n")

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], num_experts
    )
    # torch.set_printoptions(threshold=float('inf'))
    # print(f"sorted_token_ids{sorted_token_ids}, shape: {sorted_token_ids.shape} \n",
    #       f"expert_ids: {expert_ids}, shape: {expert_ids.shape}\n",
    #       f"num_tokens_post_padded : {num_tokens_post_padded}")
    # 3.4. 准备中间结果的缓存区
    # 第一个 GEMM (w1) 的输出
    intermediate_cache1 = torch.empty(
        (num_tokens, topk, intermediate_size_x2),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )
    # 激活函数后的输出
    intermediate_cache2 = torch.empty(
        (num_tokens, topk, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )
    # 第二个 GEMM (w2) 的输出
    intermediate_cache3 = torch.empty(
        (num_tokens, topk, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )
    
    # 准备输出张量
    if inplace:
        output = hidden_states
    else:
        output = torch.empty_like(hidden_states)

    # 3.5. 执行计算
    # 调用 Triton Kernel 计算: intermediate_cache1 = A @ W1
    # A 是 hidden_states, W1 是 w1
    _invoke_fused_moe_kernel(
        hidden_states, w1, intermediate_cache1, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        apply_router_weight_on_input=False, # 通常在最后应用权重
        top_k=topk, config=config, compute_type=tl.bfloat16
    )
    # # ------ Verify --------
    # print(f"intermediate_cache1 shape{intermediate_cache1.shape}")
    # out0 = hidden_states @ w1[2].T
    # out1 = hidden_states @ w1[1].T
    # out = torch.stack([out0,out1],dim=1)
    # print(f"out shape {out.shape}\n",
    #       f"out[0][0][:10]:{out[0][0][:10]}\n",
    #       f"out[0][1][:10]:{out[0][1][:10]}\n")
    # print(f"intermediate_cache1[0][0][:10]:{intermediate_cache1[0][0][:10]}\n")
    # print(f"intermediate_cache1[0][1][:10]:{intermediate_cache1[0][1][:10]}\n")
    
    # 执行激活函数: intermediate_cache2 = silu(intermediate_cache1)
    if activation == "silu":
        # 使用 silu_and_mul 算子 (output_ptr, input_ptr, )
        # input: (num_tokens * topk, intermediate_size_x2)
        # output: (num_topkens, topk2, intermediate_size)
        electrock_infer.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, intermediate_size_x2))
    else:
        # 如果需要其他激活函数，可以在这里添加
        raise NotImplementedError(f"Activation {activation} not supported.")
        
    # 调用 Triton Kernel 计算: intermediate_cache3 = A' @ W2
    # A' 是 intermediate_cache2, W2 是 w2
    # BUG FIX: 预期输入维度是2D，需要展平
    intermediate_cache2_flat = intermediate_cache2.view(-1, intermediate_size)

    _invoke_fused_moe_kernel(
        intermediate_cache2_flat, w2, intermediate_cache3, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        apply_router_weight_on_input=True, # 在第二次GEMM后应用路由权重
        top_k=1, config=config, compute_type=tl.bfloat16
    )

    # 3.6. 聚合结果
    # ops.moe_sum 是一个自定义算子，用于将稀疏的结果高效地聚合回密集张量
    electrock_infer.moe_sum(intermediate_cache3, output)
    
    return output


def _invoke_fused_moe_kernel(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    topk_weights: torch.Tensor, sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor,
    apply_router_weight_on_input: bool, top_k: int,
    config: dict, compute_type: tl.dtype
):
    """一个辅助函数，用于配置并启动 Triton kernel。"""
    # 
    grid = (
        lambda META: (
            triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M']) * 
            triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']),
        )
    )

    fused_moe_kernel[grid](
        A, B, C, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.shape[1], A.shape[1], # N, K
        sorted_token_ids.shape[0], # EM
        A.shape[0] * top_k, # num_valid_tokens
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        MUL_ROUTED_WEIGHT=apply_router_weight_on_input,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

@functools.lru_cache
def _get_moe_config(num_tokens, num_experts, intermediate_size, topk) -> dict:
    """返回一个默认的、通常性能不错的 Triton Kernel 配置。"""
    if num_tokens <= num_experts:
        return {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1
        }
    else:
        return {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8
        }