import torch
import triton
import triton.language as tl

@triton.jit
def _unpage_kv_kernel_fixed(
    # Pointers to source Paged Caches
    K_cache_ptr,
    V_cache_ptr,
    # Pointers to destination contiguous Tensors
    K_out_ptr,
    V_out_ptr,
    # Pointers to metadata
    block_table_ptr,
    cu_seqlens_k_ptr,
    # Tensor dimensions for stride calculations
    num_total_tokens,
    num_kv_heads,
    head_dim,
    block_size,
    stride_k_cache_block, stride_k_cache_token, stride_k_cache_head,
    stride_v_cache_block, stride_v_cache_token, stride_v_cache_head,
    stride_k_out_token, stride_k_out_head,
    stride_v_out_token, stride_v_out_head,
    max_num_blocks_per_seq,
    batch_size,
    # Triton program metadata
    BLOCK_SIZE_HD: tl.constexpr,
    MAX_BATCH_SIZE: tl.constexpr,
):
    """
    (Corrected Version) Triton kernel to gather scattered KV blocks.
    Parallelized over tokens and heads.
    """
    # 1. Get 2D program IDs: one for token, one for head
    pid_token = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    # Early exit if program is out of bounds
    if pid_token >= num_total_tokens:
        return

    # 2. Vectorized, loop-free method to find which sequence the token belongs to
    # A compile-time constant, must be >= your max batch size
    seq_idx_offsets = tl.arange(0, MAX_BATCH_SIZE)
    
    cu_seqlens_k_vals = tl.load(
        cu_seqlens_k_ptr + 1 + seq_idx_offsets,
        mask=seq_idx_offsets < batch_size,
        other=num_total_tokens + 1
    )
    is_token_in_seq = pid_token < cu_seqlens_k_vals
    seq_idx = tl.argmin(is_token_in_seq, axis=0)

    # 3. Calculate the token's logical position within its sequence
    start_token_idx = tl.load(cu_seqlens_k_ptr + seq_idx)
    token_idx_in_seq = pid_token - start_token_idx

    # 4. Use the block_table to find the physical block
    block_idx_in_table = token_idx_in_seq // block_size
    offset_in_block = token_idx_in_seq % block_size
    physical_block_id = tl.load(block_table_ptr + seq_idx * max_num_blocks_per_seq + block_idx_in_table)

    # 5. Calculate source pointers for the current head
    head_dim_offsets = tl.arange(0, BLOCK_SIZE_HD)
    head_dim_mask = head_dim_offsets < head_dim

    src_k_ptr = K_cache_ptr + (physical_block_id * stride_k_cache_block +
                               offset_in_block * stride_k_cache_token +
                               pid_head * stride_k_cache_head)
    src_v_ptr = V_cache_ptr + (physical_block_id * stride_v_cache_block +
                               offset_in_block * stride_v_cache_token +
                               pid_head * stride_v_cache_head)

    # 6. Calculate destination pointers for the current head
    dest_k_ptr = K_out_ptr + (pid_token * stride_k_out_token +
                              pid_head * stride_k_out_head)
    dest_v_ptr = V_out_ptr + (pid_token * stride_v_out_token +
                              pid_head * stride_v_out_head)

    # 7. Load and store the head_dim vector
    k_vec = tl.load(src_k_ptr + head_dim_offsets, mask=head_dim_mask)
    v_vec = tl.load(src_v_ptr + head_dim_offsets, mask=head_dim_mask)
    
    tl.store(dest_k_ptr + head_dim_offsets, k_vec, mask=head_dim_mask)
    tl.store(dest_v_ptr + head_dim_offsets, v_vec, mask=head_dim_mask)
def unpage_kv_cache_fixed(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    block_tables, # Changed from list to torch.Tensor
    cu_seqlens_k
):
    """
    (Corrected Version) Python wrapper to launch the _unpage_kv_kernel_fixed.
    """

    # Ensure tensors are on CUDA
    # assert all(t.is_cuda for t in [k_cache, v_cache, k_out, v_out, block_tables_tensor, cu_seqlens_k_tensor])
    
    # Extract dimensions
    total_k_tokens, num_kv_heads, head_dim = k_out.shape
    block_size = k_cache.shape[1]
    batch_size, max_num_blocks_per_seq = block_tables.shape

    # Set up a 2D grid for parallelization over tokens and heads
    grid = (total_k_tokens, num_kv_heads)
    
    # A tunable parameter for kernel performance
    BLOCK_SIZE_HD = 128 if head_dim > 64 else 64
    MAX_BATCH_SIZE_CONFIG = 128
    _unpage_kv_kernel_fixed[grid](
        k_cache, v_cache,
        k_out, v_out,
        block_tables, cu_seqlens_k,
        total_k_tokens, num_kv_heads, head_dim, block_size,
        # Strides for pointer arithmetic
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_out.stride(0), k_out.stride(1),
        v_out.stride(0), v_out.stride(1),
        max_num_blocks_per_seq,
        batch_size,
        BLOCK_SIZE_HD=BLOCK_SIZE_HD,
        MAX_BATCH_SIZE=MAX_BATCH_SIZE_CONFIG
    )
    # Note: The original `unpage_kv_cache` needs to be updated to convert the list of block_tables to a tensor
    # before calling this function. e.g., block_tables_tensor = torch.tensor(block_tables, device='cuda', dtype=torch.int32)

@triton.jit
def _unpage_kv_kernel(
    # Pointers to source Paged Caches
    K_cache_ptr,
    V_cache_ptr,
    # Pointers to destination contiguous Tensors
    K_out_ptr,
    V_out_ptr,
    # Pointers to metadata
    block_table_ptr,
    cu_seqlens_k_ptr, # Cumulative sequence lengths of the keys in the cache
    # Tensor dimensions for stride calculations
    block_size,
    stride_k_cache_block, stride_k_cache_token, stride_k_cache_head,
    stride_v_cache_block, stride_v_cache_token, stride_v_cache_head,
    stride_k_out_token, stride_k_out_head,
    stride_v_out_token, stride_v_out_head,
    batch_size,
    max_num_blocks_per_seq,
    # Triton program metadata
    D: tl.constexpr, # hidden_size
):
    """
    Triton kernel to gather scattered KV blocks from a paged cache into
    contiguous tensors.
    """
    #期望每个实例搬运一个token相关的kv cache
    pid = tl.program_id(0)
    #确定在哪一个seq_id上, 逻辑第几个block块上，块内的相对位置
    seq_id = 0
    logic_block_idx = 0
    cur_seq_len = 0
    inner_block_offset = 0

    for i in range(1, batch_size+1):
        if pid < tl.load(cu_seqlens_k_ptr+i):
            start, end = tl.load(cu_seqlens_k_ptr+i-1) , tl.load(cu_seqlens_k_ptr+i)
            cur_seq_len = end - start
            inner_block_offset = (pid - start) % block_size
            logic_block_idx = (pid - start) // block_size
            break
    # 拿到真实kv cache的块号 并计算cache offset
    real_block_idx = tl.load( block_table_ptr + (seq_id * max_num_blocks_per_seq) + logic_block_idx)
    key_cache_offset = (real_block_idx * stride_k_cache_block) + (inner_block_offset * stride_k_cache_token) + tl.arange(0, D)
    value_cache_offset = (real_block_idx * stride_v_cache_block) + (inner_block_offset * stride_v_cache_token) + tl.arange(0, D)
    # 取出对应token的 hidden
    key = tl.load(K_cache_ptr + key_cache_offset)
    value = tl.load(V_cache_ptr + value_cache_offset)
    # 计算output_offset
    key_out_offset = pid * stride_k_out_token + tl.arange(0, D)
    value_out_offset = pid * stride_v_out_token + tl.arange(0, D)
    #搬运
    tl.store(K_out_ptr + key_out_offset, key)
    tl.store(V_out_ptr + value_out_offset, value)


def unpage_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    block_tables: list,
    cu_seqlens_k: torch.Tensor,
    num_tokens
):
    """
    Python wrapper to launch the _unpage_kv_kernel.
    Gathers KV pairs from paged cache into contiguous output tensors.
    """
    # Extract dimensions
    total_k_tokens, num_kv_heads, head_dim = k_out.shape
    D = num_kv_heads * head_dim
    assert num_tokens == total_k_tokens
    block_size = k_cache.shape[1] 

    batch_size, max_num_blocks_per_seq = len(block_tables), len(block_tables[0])

    # Triton grid setup
    grid = (total_k_tokens,)
    

    # Launch the kernel
    _unpage_kv_kernel[grid](
        k_cache, v_cache,
        k_out, v_out,
        block_tables, cu_seqlens_k,
        block_size,
        # Strides for pointer arithmetic in the kernel
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_out.stride(0), k_out.stride(1),
        v_out.stride(0), v_out.stride(1),
        batch_size, max_num_blocks_per_seq, 
        D
    )



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