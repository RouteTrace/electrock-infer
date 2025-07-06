import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import re
# fused_experts 是一个底层的、用 Triton 或 cuBLAS 编写的高性能 CUDA 核
from nanovllm.layers.moe.fused_moe import fused_experts
import torch.distributed as dist
class FusedMoE(nn.Module):
    """
    一个简化的、仅支持张量并行 (TP) 的 FusedMoE 层。

    此实现移除了 Expert/Data Parallelism、多硬件支持和量化逻辑
    专注于在 CUDA 上运行的未量化 Mixtral MoE 核心功能。

    Args:
        num_experts: 专家总数。
        top_k: 每个 token 选择的专家数量。
        hidden_size: 输入隐藏层维度。
        intermediate_size: 专家网络中间层的维度。
        tp_size: 张量并行的大小 (world size)。
        params_dtype: 模型参数的数据类型。
        renormalize: 是否在 top-k 选择后对路由权重进行重新归一化。
        reduce_results: 是否在计算后对所有 TP rank 的结果进行 AllReduce。
                         对于 MoE 通常在外部处理 默认为 False。
    """
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        reduce_results: bool = False,
    ):
        super().__init__()

        # 获取并行配置
        self.tp_size = tp_size 
        self.tp_rank = dist.get_rank()
        
        # 核心参数
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.reduce_results = reduce_results
        self.params_dtype = params_dtype or torch.get_default_dtype()

        # 验证 intermediate_size 可以被 tp_size 整除
        if self.intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )
        self.intermediate_size_per_partition = self.intermediate_size // self.tp_size

        # 创建权重
        # w13_weight 融合了 gate_proj (w1) 和 up_proj (w3)
        # 形状: (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # 这是一个 ColumnParallelLinear 类型的权重
        self.w13_weight = nn.Parameter(torch.empty(
            self.num_experts,
            2 * self.intermediate_size_per_partition,
            self.hidden_size,
            dtype=self.params_dtype
        ), requires_grad=False)

        # w2_weight 对应 down_proj
        # 形状: (num_experts, hidden_size, intermediate_size_per_partition)
        # 这是一个 RowParallelLinear 类型的权重
        self.w2_weight = nn.Parameter(torch.empty(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size_per_partition,
            dtype=self.params_dtype
        ), requires_grad=False)

        self.w13_weight.weight_loader = self.weight_loader
        self.w2_weight.weight_loader = self.weight_loader
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        # # 1. 选择专家
        # # `fused_topk` 是一个优化的 CUDA kernel，用于执行 softmax 和 top-k 操作
        # # topk_weights代表所选专家的概率，topk_ids代表所选专家的全局id
        # # topk_weights, topk_ids  shape = (num_tokens, topk)
        # topk_weights, topk_ids = fused_topk(
        #     hidden_states=hidden_states, #(num_tokens, hidden_state)
        #     gating_output=router_logits, # (num_tokens, n_experts)
        #     topk=self.top_k,
        #     renormalize=self.renormalize
        # )

        #  调用核心的 MoE CUDA kernel 进行计算  ( 把 softmax_topk放入了fused_experts中)
        # `fused_experts` 将所有计算（包括索引、矩阵乘法、激活函数等）融合在一起
        # print(f"fusedmoe_{hidden_states=}, shape: {hidden_states.shape}")
        # print(f"fusedmoe_{self.w13_weight=}, shape: {self.w13_weight.shape}")
        # print(f"fusedmoe_{self.w2_weight=}, shape: {self.w2_weight.shape}")
        final_hidden_states = fused_experts(
            hidden_states=hidden_states, # (num_tokens, hidden_state)
            w1=self.w13_weight,
            w2=self.w2_weight,
            gating_output=router_logits, # (num_tokens, n_experts)
            topk=self.top_k,
            renormalize=self.renormalize,
            inplace=True, # 原地修改
            activation="silu" # Mixtral 使用 silu (SwiGLU)
        )
        # print(f"fusedmoe_{final_hidden_states=}, shape: {final_hidden_states.shape}")
        # 3. 如果需要，对 TP 的结果进行聚合
        # 在标准的 MoE 实现中，这一步通常在更高层处理，但保留该选项
        # dist.all_reduce(input_, group=self.device_group)
        if self.reduce_results and self.tp_size > 1:
            # BUG: dist.all_reduce()不能用返回值接收
            dist.all_reduce(final_hidden_states)

        # print(f"fusedmoe_reduce_{final_hidden_states=}, shape: {final_hidden_states.shape}")
        return final_hidden_states

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                       shard_id: str, expert_id: int):
        """
        自定义的权重加载器，用于将标准 Checkpoint 的权重加载到我们融合且分片的参数中。
        """
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        
        # 目标参数 , 对应的专家 
        param_data = param.data[expert_id]

        if shard_id in ("w1", "w3"): # 对应 ColumnParallelLinear 的 w13
            # 分割点在 intermediate_size 维度 (dim=0)
            shard_size = self.intermediate_size_per_partition
            # 加载当前 rank 对应的权重分片
            loaded_shard = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
            
            if shard_id == "w1": # gate_proj
                # 加载到 w13 的前半部分
                param_data[:shard_size, :].copy_(loaded_shard)
            else: # up_proj
                # 加载到 w13 的后半部分
                param_data[shard_size:, :].copy_(loaded_shard)

        elif shard_id == "w2": # 对应 RowParallelLinear 的 w2
            # 分割点在 intermediate_size 维度 (dim=1)
            shard_size = self.intermediate_size_per_partition
            # 加载当前 rank 对应的权重分片
            loaded_shard = loaded_weight.narrow(1, shard_size * tp_rank, shard_size)
            param_data.copy_(loaded_shard)
        else:
            raise ValueError(f"未知的 shard_id: {shard_id}")

