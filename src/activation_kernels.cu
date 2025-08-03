/*
 *
 * 目标：以最清晰、最直接的方式展示 silu_and_mul 的核心计算。
 *
 * 简化措施：
 * 1. 移除了所有通用模板和宏，创建了一个专用的CUDA核函数。
 * 2. 将 SiLU(x) * y 的数学逻辑直接内联到核函数中。
 * 3. 将 CUDA 内核的启动逻辑直接放在最终的C++函数体内。
 */
#include <torch/all.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include "ops.h"
#include "cuda_utils.cuh"
#include <cmath>
namespace electrock_infer {
#define LDG(args) __ldg(args)

// 1. 专用的、一体化的 SiLU & Mul CUDA 核函数
template <typename scalar_t>
__global__ __launch_bounds__(1024) void silu_and_mul_kernel(
    scalar_t* __restrict__ out,          // 输出: [..., d]
    const scalar_t* __restrict__ input,  // 输入: [..., 2 * d]
    const int d) {
  
  // 每个 CUDA block 负责一个 token 的计算
  const int64_t bid = blockIdx.x;
  
  // block 内的线程并行处理 hidden_dim 维度
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // 从输入张量加载 gate 和 up 的值
    const scalar_t gate_val = input[bid * 2 * d + idx];
    const scalar_t up_val = input[bid * 2 * d + d + idx];

    // --- 核心计算逻辑，直接内联 ---
    // 计算 SiLU(gate_val)
    const float gate_float = static_cast<float>(gate_val);
    const float silu_gate_float = gate_float / (1.0f + __expf(-gate_float));
    const scalar_t silu_gate = static_cast<scalar_t>(silu_gate_float);
    
    // 计算 SiLU(gate_val) * up_val
    out[bid * d + idx] = silu_gate * up_val;
  }
}

__global__ void silu_and_mul_d7168_float4_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    const int d) {
  
  // --- 1. 将 float 指针转换为 float4 指针 ---
  // 这是向量化优化的关键步骤。
  const float4* input_vec = reinterpret_cast<const float4*>(input);
  float4* out_vec = reinterpret_cast<float4*>(out);

  // --- 2. 计算向量化后的维度 ---
  // 由于 d=7168, d_vec = 7168 / 4 = 1792
  const int d_vec = d / 4; 
  const int d_input_vec = 2 * d_vec;

  // 每个 CUDA block 负责一个 token (一行) 的计算
  const int64_t token_idx = blockIdx.x;
  
  // 计算当前 token 在输入和输出张量中的偏移量（以 float4 为单位）
  const int64_t input_offset = token_idx * d_input_vec;
  const int64_t output_offset = token_idx * d_vec;
  
  // --- 3. Block 内的线程并行处理一行数据 ---
  // 每个线程负责处理多个 float4 向量
  for (int64_t i = threadIdx.x; i < d_vec; i += blockDim.x) {
    // --- 4. 一次性加载 4 个 gate 和 4 个 up 的值 ---
    const float4 gate_vec = input_vec[input_offset + i];
    const float4 up_vec = input_vec[input_offset + d_vec + i]; // gate 和 up 相隔 d (即 d_vec 个 float4)

    // --- 5. 核心计算逻辑，对 float4 内的每个元素独立计算 ---
    // 编译器会自动将这部分展开，实现指令级并行
    const float s_x = gate_vec.x / (1.0f + expf(-gate_vec.x));
    const float s_y = gate_vec.y / (1.0f + expf(-gate_vec.y));
    const float s_z = gate_vec.z / (1.0f + expf(-gate_vec.z));
    const float s_w = gate_vec.w / (1.0f + expf(-gate_vec.w));
    
    // 将 SiLU 结果与 up 值相乘，并打包成 float4
    const float4 result_vec = make_float4(
        s_x * up_vec.x,
        s_y * up_vec.y,
        s_z * up_vec.z,
        s_w * up_vec.w
    );
    
    // --- 6. 一次性写回 4 个 float 结果 ---
    out_vec[output_offset + i] = result_vec;
  }
}

// 最终暴露给 Python 层的、经过完全简化的 C++ 函数
void silu_and_mul(
    torch::Tensor &out,  // 输出张量 [..., d]
    torch::Tensor &input // 输入张量 [..., 2 * d]
)
{
  // --- 1. Tensor Validation ---
  TORCH_CHECK(out.is_cuda(), "Output tensor must be on a CUDA device!");
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device!");
  TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions!");

  // 2. 计算维度信息
  const int d = input.size(-1) / 2;
  TORCH_CHECK(input.size(-1) == 2 * d, "Input last dimension must be 2*d!");
  TORCH_CHECK(out.size(-1) == d, "Output last dimension must be d!");

  const int64_t num_tokens = input.numel() / (2 * d);

  // 3. 设置 CUDA 执行配置
  const dim3 grid(num_tokens);
  const dim3 block(std::min(d, 1024));

  // 4. 获取当前 CUDA 环境 (device_guard is good practice)

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // 5. 根据输入的数据类型分发并启动专用内核
  // 这是关键部分：使用 AT_DISPATCH_... 宏来处理类型分发
  if (input.scalar_type() == torch::kFloat32)
  {
    // --- 路径A：调用高度优化的 float4 Kernel ---
    const dim3 grid(num_tokens);
    const dim3 block(1024); // 使用为 float4 kernel 优化的 block size

    electrock_infer::silu_and_mul_d7168_float4_kernel<<<grid, block, 0, stream>>>(
        out.data_ptr<float>(),
        input.data_ptr<float>(),
        d );
  }else{
  
        // 使用 AT_DISPATCH 宏自动处理 float, half, bfloat16 等类型
        DISPATCH_FLOATING_TYPES(
            input.scalar_type(), 
            "silu_and_mul_generic_dispatch", 
            ([&] {
                electrock_infer::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    d
                );
            })
        );

  }
} // namespace 
}