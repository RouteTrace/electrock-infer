# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple



# -----------------------------------------------------------------------------
# 核心 RoPE 计算 - 只保留 NeoX 风格实现
# -----------------------------------------------------------------------------

def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX 风格的旋转：将维度对半切分后交换位置并取反。"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_emb_neox_style_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    一个特化版本的、只使用纯 PyTorch 实现 NeoX 风格 RoPE 的函数。
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    
    # 直接使用 NeoX 风格的对半切分
    x1, x2 = torch.chunk(x, 2, dim=-1)
        
    # RoPE 核心公式
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    
    # 直接返回 NeoX 风格的合并结果
    return torch.cat((o1, o2), dim=-1)


# -----------------------------------------------------------------------------
# 简化的 RotaryEmbedding 核心类 (专为 NeoX 风格)
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    基础旋转位置编码的 NeoX 专用实现。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 核心简化：硬编码为 NeoX 风格
        self.is_neox_style: bool = True
        dtype = torch.get_default_dtype()

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """计算频率向量的倒数 (inverse frequency)"""
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """预计算并缓存 cos 和 sin 值 (逻辑不变)"""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE 的纯 PyTorch 前向传播实现"""
        positions_flat = positions.flatten()
        cos_sin = self.cos_sin_cache.index_select(0, positions_flat)
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 对 Query 应用旋转
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        # 调用特化后的 NeoX 风格函数
        query_rot = _apply_rotary_emb_neox_style_torch(query_rot, cos, sin)
        query_new = torch.cat((query_rot, query_pass), dim=-1)

        # 对 Key 应用旋转
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        # 调用特化后的 NeoX 风格函数
        key_rot = _apply_rotary_emb_neox_style_torch(key_rot, cos, sin)
        key_new = torch.cat((key_rot, key_pass), dim=-1)
        
        return query_new, key_new

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE 的高性能 CUDA 实现"""


        if self.cos_sin_cache.device != query.device or self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

        # 调用 CUDA 算子，self.is_neox_style 硬编码为 True
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style 
        )
        return query, key
    

# -----------------------------------------------------------------------------
# 简化的 RoPE 工厂函数 (Factory Function)
# -----------------------------------------------------------------------------


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb