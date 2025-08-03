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
#include <cuda_fp16.h> // 使用 half2 需要这个头文件
#include "ops.h"
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace electrock_infer {

// ============================================================================
// moe_sum: 聚合专家输出
// ============================================================================

template <typename scalar_t>
__global__ void moe_sum_kernel_efficient(
    scalar_t* __restrict__ out,         // 输出: [num_tokens, hidden_size]
    const scalar_t* __restrict__ input, // 输入: [num_tokens, hidden_size, topk=2]
    const int num_total_elements        // 总元素数: num_tokens * hidden_size
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_total_elements) {
        // 使用 if constexpr 根据 scalar_t 的类型在编译时选择不同的代码路径
        if constexpr (std::is_same_v<scalar_t, float>) {
            // --- float 路径 ---
            const float2* input_vec = reinterpret_cast<const float2*>(input);
            const float2 two_experts = input_vec[i];
            out[i] = two_experts.x + two_experts.y;
        } 
        else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            // --- half 路径 ---
            const half2* input_vec = reinterpret_cast<const half2*>(input);
            const half2 two_experts = input_vec[i];
            // 使用 __hadd2 进行高效的 half2 向量加法
            // 这里 two_experts.x + two_experts.y 也可以，但 __hadd2 更直接
            out[i] = __hadd(two_experts.x, two_experts.y);
        }
    }
}


// --- 高效版本的 C++ 启动函数 ---
void moe_sum_efficient(
    torch::Tensor& input, // 输入: [num_tokens, hidden_size, topk=2]
    torch::Tensor& output // 输出: [num_tokens, hidden_size]
) {
    const int64_t hidden_size = input.size(1);
    const int64_t num_tokens = output.numel() / hidden_size;
    const int topk = input.size(-1);

    TORCH_CHECK(topk == 2, "This function is hard-coded for topk=2");

    const int64_t num_total_elements = num_tokens * hidden_size;

    // --- 启动配置：1D Grid ---
    // 每个线程负责一个输出元素，这是最高效的模式
    const int block_size = 256;
    const int grid_size = CEILDIV(num_total_elements, block_size);
    dim3 grid(grid_size);
    dim3 block(block_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "moe_sum_efficient_kernel", [&] {
        electrock_infer::moe_sum_kernel_efficient<scalar_t><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            num_total_elements
        );
    });
}

} // namespace electrock_infer