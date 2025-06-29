/*
 * vLLM activation_kernels.cu 的简化版本。
 * 完整保留了 silu_and_mul 的核心实现，移除了其他所有激活函数。
 * 这种设计模式（通用核 + 具体实现）展示了 vLLM 如何高效地重用代码。
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"      // 假设此文件包含 VLLM_LDG 等宏
#include "dispatch_utils.h"   // 假设此文件包含 VLLM_DISPATCH_FLOATING_TYPES 宏

namespace vllm {

// 1. SiLU激活函数的数学定义 (设备端函数)
// 这是一个 __device__ 函数，意味着它只能被 GPU 上的其他函数（如核函数）调用。
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // SiLU(x) = x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}


// 2. 通用的“先激活、后相乘”计算模板 (设备端函数)
// ACT_FN 是一个函数指针模板参数，ACT_FIRST 是一个布尔模板参数。
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x, const scalar_t& y) {
  // 根据 act_first 的值决定是对 x 还是对 y 应用激活函数
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}


// 3. 通用的门控激活 CUDA 核函数 (Gated Activation CUDA Kernel)
// 这个核函数是通用的，它通过模板参数接收具体的激活函数。
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // 输出: [..., d]
    const scalar_t* __restrict__ input,  // 输入: [..., 2 * d]
    const int d) {

  // 每个 CUDA block 处理一个 token 的计算
  const int64_t token_idx = blockIdx.x;
  
  // block内的线程并行处理 hidden_dim 维度
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // 从输入张量加载门控单元(gate)和上行投影(up-projection)的值
    // input 张量被看作是 [gate, up] 两个部分拼接而成
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);

    // 调用计算模板，执行 act_fn(x) * y 或 x * act_fn(y)
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

}  // namespace vllm


// 4. C++ Host端的内核启动宏 (Launcher Macro)
// 这个宏封装了启动 CUDA 核函数的通用逻辑，如计算 grid/block 维度、设置流等。
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });


// 5. 最终暴露给 Python 层的 C++ 函数
void silu_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input   // [..., 2 * d]
) {
  // 调用启动宏，并传入具体的参数：
  // KERNEL = vllm::silu_kernel (使用SiLU激活函数)
  // ACT_FIRST = true (对输入的前半部分[gate]应用SiLU)
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

// 保留 mul_and_silu 作为对比，它使用了相同的内核但模板参数不同
void mul_and_silu(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input   // [..., 2 * d]
) {
  // ACT_FIRST = false (对输入的后半部分[up]应用SiLU)
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}