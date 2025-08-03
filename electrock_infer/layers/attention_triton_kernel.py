import torch
import triton
import triton.language as tl

import torch

def paged_attention_decode_naive(
    q: torch.Tensor,           # Shape: [batch_size, num_heads, head_dim]
    k_cache: torch.Tensor,    # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,    # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor, # Shape: [batch_size, max_num_blocks]
    context_lens: torch.Tensor, # Shape: [batch_size]
    scale: float,
    num_kv_heads: int,
):
    """
    一个使用纯 PyTorch 实现的、用于解码阶段的 Paged Attention "黄金标准"。
    该函数为验证目的而设计，性能较低。
    
    *** 已更新：中间计算过程强制使用 float32 以保证数值稳定性 ***
    """
    batch_size, num_heads, head_dim = q.shape
    _, block_size, _, _ = k_cache.shape
    num_groups = num_heads // num_kv_heads

    # <--- 关键改动 1: 保存原始数据类型 ---
    original_dtype = q.dtype
    
    output_list = []

    # 逐个处理 batch 中的每个序列
    for i in range(batch_size):
        # 1. 提取当前序列的 Q, 上下文长度, 和 block_table
        # <--- 关键改动 2: 将 q_i 提前转换为 float32 ---
        q_i = q[i].to(torch.float32)  # Shape: [num_heads, head_dim]
        q_i = q_i.unsqueeze(1) # Shape: [num_heads, 1, head_dim]
        
        seq_len = context_lens[i].item()
        block_table_i = block_tables[i] # Shape: [max_num_blocks]

        # 2. 手动从 Paged KV Cache 中 gather 历史的 K 和 V
        k_for_seq = []
        v_for_seq = []
        if seq_len > 0:
            for token_idx in range(seq_len):
                logical_block_idx = token_idx // block_size
                offset_in_block = token_idx % block_size
                physical_block_id = block_table_i[logical_block_idx]
                
                # 从 cache 中取出对应的 token 向量
                # <--- 关键改动 3: 将 k/v token 转换为 float32 ---
                k_token = k_cache[physical_block_id, offset_in_block].to(torch.float32)
                v_token = v_cache[physical_block_id, offset_in_block].to(torch.float32)
                k_for_seq.append(k_token)
                v_for_seq.append(v_token)
            
            k_i = torch.stack(k_for_seq, dim=0) # Shape: [seq_len, num_kv_heads, head_dim], Dtype: float32
            v_i = torch.stack(v_for_seq, dim=0) # Shape: [seq_len, num_kv_heads, head_dim], Dtype: float32
        else:
            # 如果是首个 token，历史 K/V 为空
            # <--- 关键改动 4: 创建空的 float32 张量 ---
            k_i = torch.empty(0, num_kv_heads, head_dim, dtype=torch.float32, device=k_cache.device)
            v_i = torch.empty(0, num_kv_heads, head_dim, dtype=torch.float32, device=v_cache.device)
            
        # 3. 执行标准的注意力计算 (现在全程在 float32 下进行)
        k_i = k_i.transpose(0, 1) # -> [num_kv_heads, seq_len, head_dim]
        v_i = v_i.transpose(0, 1) # -> [num_kv_heads, seq_len, head_dim]

        if num_groups > 1:
            k_i = k_i.repeat_interleave(num_groups, dim=0)
            v_i = v_i.repeat_interleave(num_groups, dim=0)
            
        attn_scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale

        # <--- 关键改动 5: 移除多余的 .float() 和类型转换 ---
        # attn_scores 已经是 float32，所以不再需要 .float()
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # 不再转换回低精度，保持 attn_weights 为 float32
        
        output_i = torch.matmul(attn_weights, v_i) # Shape: [num_heads, 1, head_dim]
        
        output_i = output_i.squeeze(1) # Shape: [num_heads, head_dim]
        output_list.append(output_i)

    final_output = torch.stack(output_list, dim=0) # Shape: [batch_size, num_heads, head_dim]
    
    # <--- 关键改动 6: 在返回前，将最终结果转换回原始数据类型 ---
    return final_output.to(original_dtype)
# def paged_attention_decode_naive(
#     q: torch.Tensor,           # Shape: [batch_size, num_heads, head_dim]
#     k_cache: torch.Tensor,    # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
#     v_cache: torch.Tensor,    # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
#     block_tables: torch.Tensor, # Shape: [batch_size, max_num_blocks]
#     context_lens: torch.Tensor, # Shape: [batch_size]
#     scale: float,
#     num_kv_heads: int,
# ):
#     """
#     一个使用纯 PyTorch 实现的、用于解码阶段的 Paged Attention "黄金标准"。
#     该函数为验证目的而设计，性能较低。
#     """
#     batch_size, num_heads, head_dim = q.shape
#     _, block_size, _, _ = k_cache.shape
#     num_groups = num_heads // num_kv_heads

#     output_list = []

#     # 逐个处理 batch 中的每个序列
#     for i in range(batch_size):
#         # 1. 提取当前序列的 Q, 上下文长度, 和 block_table
#         q_i = q[i]  # Shape: [num_heads, head_dim]
#         # 解码阶段，每个序列只有一个 query token，所以 seq_len_q = 1
#         q_i = q_i.unsqueeze(1) # Shape: [num_heads, 1, head_dim]
        
#         seq_len = context_lens[i].item()
#         block_table_i = block_tables[i] # Shape: [max_num_blocks]

#         # 2. 手动从 Paged KV Cache 中 gather 历史的 K 和 V
#         # 这是 Paged Attention 的核心逻辑
#         k_for_seq = []
#         v_for_seq = []
#         if seq_len > 0:
#             for token_idx in range(seq_len):
#                 # 计算逻辑块索引和块内偏移
#                 logical_block_idx = token_idx // block_size
#                 offset_in_block = token_idx % block_size
                
#                 # 从 block_table 中查找物理块的 ID
#                 physical_block_id = block_table_i[logical_block_idx].item()
                
#                 # 从 cache 中取出对应的 token 向量
#                 k_token = k_cache[physical_block_id, offset_in_block] # [num_kv_heads, head_dim]
#                 v_token = v_cache[physical_block_id, offset_in_block] # [num_kv_heads, head_dim]
#                 k_for_seq.append(k_token)
#                 v_for_seq.append(v_token)
            
#             # 将 gather 到的 token 拼接成连续的 K 和 V 张量
#             k_i = torch.stack(k_for_seq, dim=0) # Shape: [seq_len, num_kv_heads, head_dim]
#             v_i = torch.stack(v_for_seq, dim=0) # Shape: [seq_len, num_kv_heads, head_dim]
#         else:
#             # 如果是首个 token，历史 K/V 为空
#             k_i = torch.empty(0, num_kv_heads, head_dim, dtype=k_cache.dtype, device=k_cache.device)
#             v_i = torch.empty(0, num_kv_heads, head_dim, dtype=v_cache.dtype, device=v_cache.device)
            
#         # 3. 执行标准的注意力计算
#         # 为了计算方便，将 head 维度放到前面
#         k_i = k_i.transpose(0, 1) # -> [num_kv_heads, seq_len, head_dim]
#         v_i = v_i.transpose(0, 1) # -> [num_kv_heads, seq_len, head_dim]

#         # 处理 GQA (Grouped-Query Attention)
#         if num_groups > 1:
#             k_i = k_i.repeat_interleave(num_groups, dim=0)
#             v_i = v_i.repeat_interleave(num_groups, dim=0)
            
#         # 计算注意力得分
#         # q_i: [num_heads, 1, head_dim], k_i: [num_heads, seq_len, head_dim]
#         attn_scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale

#         # 使用混合精度以保证数值稳定性
#         stable_weights = torch.softmax(attn_scores.float(), dim=-1)
#         attn_weights = stable_weights.to(q_i.dtype)
        
#         # 计算输出
#         # attn_weights: [num_heads, 1, seq_len], v_i: [num_heads, seq_len, head_dim]
#         output_i = torch.matmul(attn_weights, v_i) # Shape: [num_heads, 1, head_dim]
        
#         # 移除多余的 seq_len=1 维度
#         output_i = output_i.squeeze(1) # Shape: [num_heads, head_dim]
#         output_list.append(output_i)

#     # 4. 将 batch 中所有序列的结果拼接起来
#     final_output = torch.stack(output_list, dim=0) # Shape: [batch_size, num_heads, head_dim]
#     return final_output

def naive_attention_varlen_mixed_precision(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    scale: float,
    causal: bool = True
):
    """
    一个使用混合精度思想实现的、最精确的朴素注意力版本。
    输入和输出为 bfloat16，但内部的 softmax 计算使用 float32 以保证稳定性。
    """
    # 假设输入 q, k, v 都是 bfloat16
    output_list = []
    num_groups = num_q_heads // num_kv_heads

    for i in range(len(cu_seqlens_q) - 1):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        q_i = q[start_q:end_q] # bfloat16
        k_i = k[start_k:end_k] # bfloat16
        v_i = v[start_k:end_k] # bfloat16

        q_i = q_i.transpose(0, 1)
        k_i = k_i.transpose(0, 1)
        v_i = v_i.transpose(0, 1)

        if num_groups > 1:
            k_i = k_i.repeat_interleave(num_groups, dim=0)
            v_i = v_i.repeat_interleave(num_groups, dim=0)

        # Matmul 在 float32 下进行
        attn_scores = torch.matmul(q_i.float(), k_i.transpose(-1, -2).float()) * scale

        if causal:
            seqlen_q = q_i.size(1)
            seqlen_k = k_i.size(1)
            mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool), diagonal=1)
            attn_scores.masked_fill_(mask, float('-inf'))

        # ==================== 混合精度核心 ====================
        # 1. 将 scores 提升到 float32 进行 softmax
        stable_weights = torch.softmax(attn_scores.float(), dim=-1)
        
        # =====================================================

        output_i = torch.matmul(stable_weights, v_i.float()).transpose(0, 1).contiguous()
        output_list.append(output_i)

    final_output = torch.cat(output_list, dim=0)
    # 确保最终输出是 bfloat16
    return final_output.to(q.dtype)


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


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_kvheads, head_dim = key.shape
    D = num_kvheads * head_dim
    # print(f"{key.shape=} , D = {num_heads*head_dim}")
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)