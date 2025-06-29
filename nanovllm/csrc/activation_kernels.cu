/*
 * vLLM activation_kernels.cu 的终极简化版本。
 *
 * 目标：以最清晰、最直接的方式展示 silu_and_mul 的核心计算。
 *
 * 简化措施：
 * 1. 移除了所有通用模板和宏，创建了一个专用的CUDA核函数。
 * 2. 将 SiLU(x) * y 的数学逻辑直接内联到核函数中。
 * 3. 将 CUDA 内核的启动逻辑直接放在最终的C++函数体内。
 */
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"      // 假设此文件包含 VLLM_LDG 等宏
#include "dispatch_utils.h"   // 假设此文件包含 VLLM_DISPATCH_FLOATING_TYPES 宏

namespace vllm {

// 1. 专用的、一体化的 SiLU & Mul CUDA 核函数
template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,          // 输出: [..., d]
    const scalar_t* __restrict__ input,  // 输入: [..., 2 * d]
    const int d) {
  
  // 每个 CUDA block 负责一个 token 的计算
  const int64_t token_idx = blockIdx.x;
  
  // block 内的线程并行处理 hidden_dim 维度
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // 从输入张量加载 gate 和 up 的值
    const scalar_t gate_val = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t up_val = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);

    // --- 核心计算逻辑，直接内联 ---
    // 计算 SiLU(gate_val)
    const float gate_float = static_cast<float>(gate_val);
    const float silu_gate_float = gate_float / (1.0f + expf(-gate_float));
    const scalar_t silu_gate = static_cast<scalar_t>(silu_gate_float);
    
    // 计算 SiLU(gate_val) * up_val
    out[token_idx * d + idx] = silu_gate * up_val;
  }
}

} // namespace vllm


// 2. 最终暴露给 Python 层的、经过完全简化的 C++ 函数
void silu_and_mul_simplified(
    torch::Tensor& out,    // 输出张量 [..., d]
    torch::Tensor& input   // 输入张量 [..., 2 * d]
) {
    // --- CUDA 内核启动逻辑，不再隐藏于宏中 ---

    // 1. 计算维度信息
    // d 是 FFN 的中间维度 (intermediate_size)
    const int d = input.size(-1) / 2;
    const int64_t num_tokens = input.numel() / (2 * d);

    // 2. 设置 CUDA 执行配置
    // Grid 维度：等于 token 的数量，让每个 block 处理一个 token
    const dim3 grid(num_tokens);
    // Block 维度：线程数，通常取 d 和 1024 中的较小值
    const dim3 block(std::min(d, 1024));

    // 3. 获取当前 CUDA 环境
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 4. 根据输入的数据类型 (float, half, bfloat16) 分发并启动专用内核
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "silu_and_mul_kernel", [&] {
            vllm::silu_and_mul_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    d
                );
        });
}