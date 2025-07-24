#pragma once
#include <torch/torch.h>
#include <string>
namespace electrock_infer{

    void topk_softmax(
        torch::Tensor &topk_weights,         // [num_tokens, topk=2]
        torch::Tensor &topk_indices,         // [num_tokens, topk=2]
        torch::Tensor &gating_output);

    void silu_and_mul(
        torch::Tensor &out,  // 输出张量 [..., d]
        torch::Tensor &input // 输入张量 [..., 2 * d]
    );
    void moe_sum(
        torch::Tensor &input, // [num_tokens, topk=2, hidden_size]
        torch::Tensor &output // [num_tokens, hidden_size]
    );
    void moe_sum_efficient(
        torch::Tensor &input,
        torch::Tensor &output);

    void moe_align_block_size(
        torch::Tensor topk_ids,
        int64_t num_experts,
        int64_t block_size,
        torch::Tensor sorted_token_ids,
        torch::Tensor expert_ids,
        torch::Tensor num_tokens_post_pad);
        
    void flash_attn_simplified_causal_varlen_gqa(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);
    // void rms_norm(
    //     torch::Tensor &input,   // (num_tokens, hidden_size)
    //     double epsilon, // 1e-5
    //     torch::Tensor weight); // self.weight.shape = (hidden_size,) = (4096)
    // void add_rms_norm(
    //     torch::Tensor &input,   // (num_tokens, hidden_size)
    //     torch::Tensor &residual,  // if not None, residual.shape = input.shape (num_tokens, hidden_size)
    //     double epsilon, // 1e-5
    //     torch::Tensor weight); // self.weight.shape = (hidden_size,) = (4096)

}