from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import MixtralConfig
from torch import distributed as dist


from electrock_infer.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from electrock_infer.layers.layernorm import RMSNorm
from electrock_infer.layers.linear import (ReplicatedLinear, RowParallelLinear, ColumnParallelLinear,
                                    MergedColumnParallelLinear, QKVParallelLinear)
from electrock_infer.layers.moe.mixtral_moe import FusedMoE
from electrock_infer.layers.rotary_embedding import RotaryEmbedding, get_rope
from electrock_infer.layers.attention import Attention


"""Inference-only Mixtral model."""
class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 intermediate_size: int,
                 params_dtype: Optional[torch.dtype] = None,
                 tp_size: Optional[int] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(hidden_size, 
                                     num_experts,
                                     bias=False)
        
        self.experts = FusedMoE(num_experts=num_experts,
                                top_k=top_k,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                                params_dtype=params_dtype,
                                reduce_results=True,
                                renormalize=True,
                                tp_size=tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        # print(f"{orig_shape=}, {hidden_states=}")
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        # print(f"MixtralMoe_{router_logits=}, shape:{router_logits.shape}")
        final_hidden_states = self.experts(hidden_states, router_logits)
        # print(f"MixtralMoe_{final_hidden_states=}")
        return final_hidden_states.view(orig_shape)


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hf_config : MixtralConfig,
        hidden_size: int,
        num_heads: int, # num_attention_heads ： 32
        num_kv_heads: int, # 8
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MixtralConfig has an optional head_dim argument
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            # is_neox_style=True,  # optimizer for cuda_rotaryembed
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        # Fix: only singe value will be return 
        output = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        hf_config: MixtralConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hf_config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(hf_config, "rope_theta", 10000)
        self.tp_size = dist.get_world_size() # only for single machine and tensor parallel
        self.self_attn = MixtralAttention(
            hf_config=hf_config,
            hidden_size=self.hidden_size,
            num_heads=hf_config.num_attention_heads,
            max_position=hf_config.max_position_embeddings,
            num_kv_heads=hf_config.num_key_value_heads,
            rope_theta=rope_theta,
            prefix=f"{prefix}.self_attn")
        self.block_sparse_moe = MixtralMoE(
            num_experts=hf_config.num_local_experts,
            top_k=hf_config.num_experts_per_tok,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            tp_size=self.tp_size,
            prefix=f"{prefix}.block_sparse_moe")
        
        self.input_layernorm = RMSNorm(hf_config.hidden_size,
                                       eps=hf_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hf_config.hidden_size,
                                                eps=hf_config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(self, hf_config, prefix: str = ""):
        super().__init__()

        self.hf_config = hf_config
        self.vocab_size = hf_config.vocab_size 
        # embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hf_config.hidden_size
        )
        # Decoder layers
        self.layers = nn.ModuleList([MixtralDecoderLayer(hf_config) for _ in range(hf_config.num_hidden_layers)])
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> Union[torch.Tensor]:

        residual = None
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    experts_modules_mapping={
        "w1":("w13_weight", "w1"),
        "w3":("w13_weight", "w3"),
        "w2":("w2_weight", "w2"),
    }
    def __init__(self, hf_config, prefix: str = ""):
        super().__init__()


        self.model = MixtralModel(hf_config)
        
        self.lm_head = ParallelLMHead(
            hf_config.vocab_size,
            hf_config.hidden_size
        )

        if hf_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        # intermediate_tensors: Optional[IntermediateTensors] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

