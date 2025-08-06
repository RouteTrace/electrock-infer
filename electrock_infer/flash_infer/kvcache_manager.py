from collections import deque
import numpy as np
from electrock_infer.engine.sequence import Sequence
'''
    KVCache.shape = (max_num_seqs, max_tokens_num, num_kv_heads, head_dim)
'''
class KVCacheManager:

    def __init__(self, max_num_seqs: int, max_model_len: int):
        assert max_num_seqs > 0
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.free_cache_ids: deque[int] = deque(range(max_num_seqs))
        self.used_cache_ids: set[int] = set()


    def allocate(self, seq: Sequence):
        assert len(self.free_cache_ids) > 0
        cache_id = self.free_cache_ids[0]
        self.free_cache_ids.remove(cache_id)
        self.used_cache_ids.add(cache_id)
        seq.cache_id = cache_id


    def deallocate(self, seq: Sequence):
        assert len(self.used_cache_ids) > 0
        self.used_cache_ids.remove(seq.cache_id)
        self.free_cache_ids.append(seq.cache_id)

