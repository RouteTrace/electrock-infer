import torch
from typing import Tuple
def topk_softmax_torch(
    gating_output: torch.Tensor,
    topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用纯 PyTorch 实现 topk + softmax 的逻辑。

    Args:
        gating_output (torch.Tensor): 门控网络的原始输出，形状为 [num_tokens, num_experts]。
        topk (int): 需要选择的专家数量。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
        - topk_weights: top-k 的权重（概率），形状为 [num_tokens, topk]，类型为 float32。
        - topk_indices: top-k 的专家索引，形状为 [num_tokens, topk]，类型为 int32。
    """
    
    # --- 步骤 1: 执行 Softmax ---
    # 为了数值稳定性，通常在 float32 类型下执行 softmax 计算，即使输入是半精度。
    # softmax 操作是在最后一个维度（专家维度）上进行的。
    gating_probs = torch.softmax(gating_output.float(), dim=-1)

    # --- 步骤 2: 执行 Top-k ---
    # torch.topk 会返回两个张量：k 个最大的值（即权重）和它们的索引。
    topk_weights, topk_indices = torch.topk(gating_probs, k=topk, dim=-1)

    return topk_weights, topk_indices.to(torch.int32)
