/*
 *
 * 目标: 只保留 Mixtral (experts=8, topk=2) 所需的核心功能，并使代码结构更清晰。
 *
 * 简化措施:
 * 1. 在 `moe_align_block_size` 中，移除了对全局内存和 uint16 作为后备方案的复杂判断，
 * 因为对于 experts=8 的情况，共享内存总是足够的。同时移除了 SGL-MoE 的特化代码。
 * 2. 在 `moe_sum` 中，移除了 switch 结构，只保留了 topk=2 的实现路径。
 */
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "ops.h"
#include "cuda_utils.cuh"
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace electrock_infer {

// ============================================================================
// moe_sum: 聚合专家输出
// ============================================================================

/**
 * 这是一个通用的 CUDA 内核，用于将 top-k 个专家的输出逐元素相加。
 * 它通过模板参数 `TOPK` 在编译时展开循环，以获得最高性能。
 */
template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,         // 输出: [num_tokens, hidden_size]
    const scalar_t* __restrict__ input, // 输入: [num_tokens, topk, hidden_size]
    const int64_t d
) {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        scalar_t x = 0.0;
        // #pragma unroll 会在编译时将这个循环展开，避免运行时的循环开销
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            x += input[token_idx * TOPK * d + k * d + idx];
        }
        out[token_idx * d + idx] = x;
    }
}


/**
 * C++ Host 端的启动函数。
 * 对于 Mixtral (topk=2)，我们只保留 case 2 的执行路径。
 */
void moe_sum(
    torch::Tensor& input,   // [num_tokens, topk=2, hidden_size]
    torch::Tensor& output   // [num_tokens, hidden_size]
) {
    const int hidden_size = input.size(-1);
    const int num_tokens = output.numel() / hidden_size;
    const int topk = input.size(1);
    
    TORCH_CHECK(topk == 2, "This simplified function is for top_k=2");

    // 设置 CUDA 执行配置
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 根据数据类型分发并启动内核
    DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        // 直接调用模板特化为 TOPK=2 的内核
        electrock_infer::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            hidden_size
        );
    });
}


} // namespace electrock_infer