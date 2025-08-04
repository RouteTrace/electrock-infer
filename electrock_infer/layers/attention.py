import torch
from torch import nn
import triton
import triton.language as tl
from electrock_infer import _C
# from flash_attn import flash_attn_varlen_func
from electrock_infer.utils.context import get_context
from electrock_infer.layers.attention_triton_kernel import store_kvcache, paged_attention_decode_naive, naive_attention_varlen_mixed_precision

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.current_total_tokens = 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context() # 获取forward的相关信息
        if not context.enable_block_table:
            # 兼容旧版本flash_attn,以支持paged_attention中的block_table功能
            return self._forward_old_flash_attn(q, k, v, context)
        
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        k_cache = self.k_cache # (total_block_num, block_size, num_kv_heads, head_dim)
        v_cache = self.v_cache # (total_block_num, block_size, num_kv_heads, head_dim)
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # TODO
            # 1. Not support MMA_bf16, manually convert to fp16.
            # 2. Not support prefix caching.
            o = _C.flash_attn_causal_varlen_gqa_hip(
                q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), 
                max_seqlen_q=ctx.max_seqlen_q, cu_seqlens_q=ctx.cu_seqlens_q,
                max_seqlen_k=ctx.max_seqlen_k, cu_seqlens_k=ctx.cu_seqlens_k,
                softmax_scale=self.scale, causal=True).to(torch.bfloat16)
        else:    # decode phrase
            # TODO: flash decoding
            o = _C.paged_attn_varlen(q, k_cache, v_cache, 0, ctx.context_lens, ctx.block_tables, self.scale)
            
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
