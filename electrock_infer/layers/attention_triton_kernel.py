import torch
import triton
import triton.language as tl

import torch

"""Naive KVCache
该kernel负责把新计算的k,v存入物理上[连续]的k_cache和v_cache
"""
def store_naive_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    # 这个参数表示每个序列 *已经* 在cache中的长度
    past_context_lens: torch.Tensor, 
    # 这个参数表示本次新batch的token总数量
    new_context_lens: torch.Tensor, 
    # 这个参数表示每个序列应该写入cache的哪个 "行" (slot)
    cache_batch_idx: torch.Tensor,
    token_offsets: torch.Tensor
):
    """
    将unbatched的key和value张量，根据指定的偏移量和索引，
    高效地存入物理上连续的KV Cache中。

    Args:
        key (torch.Tensor): 形状为 (total_tokens, num_kv_heads, head_dim)
        value (torch.Tensor): 形状为 (total_tokens, num_kv_heads, head_dim)
        k_cache (torch.Tensor): 形状为 (max_seq_num, max_tokens_num, num_kv_heads, head_dim)
        v_cache (torch.Tensor): 形状为 (max_seq_num, max_tokens_num, num_kv_heads, head_dim)
        past_context_lens (torch.Tensor): 每个序列在cache中已有的长度, shape (batch_size,)
        new_context_lens (torch.Tensor): 这个参数表示本次新batch的token总数量, shape (batch_size,)
        cache_batch_idx (torch.Tensor): 写入cache的目标行索引, shape (batch_size,)
    """
    batch_size, num_kv_heads, head_dim = new_context_lens.shape[0], key.shape[1], key.shape[2]
    
    # D是每个token的所有head拼接后的总维度
    D = num_kv_heads * head_dim
    
    # --- 断言检查 ---
    assert key.shape == value.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1, "head_dim 维度必须是连续的"
    assert key.stride(1) == head_dim and value.stride(1) == head_dim, "num_heads 维度必须是连续的"
    
    # 假设cache的布局是 (max_seq_num, max_tokens_num, D_total)
    # 并且 (num_kv_heads, head_dim) 在逻辑上是连续的, 物理上等于D
    assert k_cache.shape[-2:] == (num_kv_heads, head_dim)
    assert v_cache.shape[-2:] == (num_kv_heads, head_dim)
    
    assert past_context_lens.numel() == batch_size
    assert new_context_lens.numel() == batch_size
    assert cache_batch_idx.numel() == batch_size

    grid = (batch_size,)  # 启动batch_size个程序，每个程序负责一个序列的拷贝
    
    store_naive_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        past_context_lens,
        new_context_lens,
        token_offsets,
        cache_batch_idx,
        k_cache.stride(0),  # stride to jump to the next slot in cache
        k_cache.stride(1),  # stride to jump to the next token in cache
        key.stride(0),      # stride to jump to the next token in source
        D,                  # total feature dimension per token
        # 使用常量来提升性能和可读性
        BLOCK_D=triton.next_power_of_2(D), 
        BLOCK_S=64
    )
@triton.jit
def store_naive_kvcache_kernel(
    key_ptr,            # Pointer to source key tensor (total_tokens, num_heads, head_dim)
    value_ptr,          # Pointer to source value tensor (total_tokens, num_heads, head_dim)
    k_cache_ptr,        # Pointer to destination K cache (max_seq_num, max_tokens, ...)
    v_cache_ptr,        # Pointer to destination V cache (max_seq_num, max_tokens, ...)
    past_context_lens_ptr, # Pointer to tensor with past lengths of each sequence
    new_context_lens_ptr,  # Pointer to tensor with new lengths to be added
    token_offsets_ptr,  # Pointer to tensor with start offsets in key_ptr/value_ptr
    cache_batch_idx_ptr,   # Pointer to tensor with target slot index in cache
    cache_stride_b,     # Stride to move one slot down in the cache batch dimension
    cache_stride_s,     # Stride to move one token down in the cache sequence dimension
    key_stride_s,       # Stride to move one token down in the source key tensor
    D: tl.constexpr,    # Total feature dimension (num_heads * head_dim)
    BLOCK_D: tl.constexpr, # Block size for feature dimension D, must be power of 2
    BLOCK_S: tl.constexpr  # Block size for sequence dimension S
):
    """
    Each program in the grid is responsible for one sequence in the batch.
    """
    # 1. 获取当前程序负责的序列索引 (0 to batch_size-1)
    batch_idx = tl.program_id(0)

    # 2. 加载该序列的所有元数据
    # 要写入的目标物理槽位
    dest_slot_idx = tl.load(cache_batch_idx_ptr + batch_idx)
    # 在cache中已经存在的长度，也是我们写入的起始位置
    past_len = tl.load(past_context_lens_ptr + batch_idx)
    # 计算本次需要新写入的token数量
    new_seq_len = tl.load(new_context_lens_ptr + batch_idx)
    new_len = new_seq_len - past_len
    # 在源张量中的起始token偏移量
    token_offset = tl.load(token_offsets_ptr + batch_idx)

    # 3. 计算基准指针 (Base Pointers)
    # a. 源指针: 指向当前序列在 key_ptr 和 value_ptr 中的第一个token
    key_src_base_ptr = key_ptr + token_offset * key_stride_s
    value_src_base_ptr = value_ptr + token_offset * key_stride_s # value和key有相同的shape和stride
    
    # b. 目标指针: 指向当前序列在 k_cache_ptr 和 v_cache_ptr 中的写入区域的起始位置
    k_dest_base_ptr = k_cache_ptr + dest_slot_idx * cache_stride_b + past_len * cache_stride_s
    v_dest_base_ptr = v_cache_ptr + dest_slot_idx * cache_stride_b + past_len * cache_stride_s

    # 4. 以块为单位进行数据拷贝
    # a. 创建维度 D (feature dimension) 的偏移量
    offs_d = tl.arange(0, BLOCK_D)
    
    # b. 创建一个循环，以 BLOCK_S 为步长遍历所有新token
    for s_start in range(0, new_len, BLOCK_S):
        # -- a. 计算当前序列块的偏移量和掩码 --
        offs_s = s_start + tl.arange(0, BLOCK_S)
        s_mask = offs_s < new_len
        
        # -- b. 计算源和目标的完整指针 --
        #    源指针: key_src_base_ptr + offs_s[:, None] * key_stride_s + offs_d[None, :]
        #    由于stride_s是1（对于token维度），可以简化
        key_src_ptr = key_src_base_ptr + (offs_s[:, None] * key_stride_s + offs_d[None, :])
        value_src_ptr = value_src_base_ptr + (offs_s[:, None] * key_stride_s + offs_d[None, :])
        
        #    目标指针:
        k_dest_ptr = k_dest_base_ptr + (offs_s[:, None] * cache_stride_s + offs_d[None, :])
        v_dest_ptr = v_dest_base_ptr + (offs_s[:, None] * cache_stride_s + offs_d[None, :])

        # -- c. 从源加载数据，带掩码防止越界读取 --
        #    mask需要同时考虑序列维度和特征维度
        load_mask = (offs_s[:, None] < new_len) & (offs_d[None, :] < D)
        
        key_block = tl.load(key_src_ptr, mask=load_mask, other=0.0)
        value_block = tl.load(value_src_ptr, mask=load_mask, other=0.0)
        
        # -- d. 将数据写入目标cache，带掩码防止越界写入 --
        tl.store(k_dest_ptr, key_block, mask=load_mask)
        tl.store(v_dest_ptr, value_block, mask=load_mask)


"""Paged KVCache
该kernel专门负责把新计算的kv存入物理上[离散]的k_cache和v_cache, 
通过slot_mapping去选择位置
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