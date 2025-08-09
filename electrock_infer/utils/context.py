from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    cache_batch_idx: torch.Tensor | None = None
    cache_batch_seqlen: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill, 
    cu_seqlens_q=None, 
    cu_seqlens_k=None, 
    max_seqlen_q=0, 
    max_seqlen_k=0, 
    slot_mapping=None, 
    context_lens=None, 
    block_tables=None, 
    cache_batch_idx=None,
    cache_batch_seqlen=None
):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                        slot_mapping, context_lens, block_tables, cache_batch_idx, cache_batch_seqlen)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
