/**
 * @file cuda_utils.cuh
 * @brief 一个通用的CUDA工具和接口声明头文件
 *
 * 包含了标准的CUDA头文件、一个强大的错误检查宏、
 * 以及通用的设备端辅助函数和主机端接口声明。
 *
 * 使用 .cuh 后缀名以明确表示此头文件包含CUDA特定代码，
 * 如 __device__ 函数或CUDA API相关的宏。
 */
#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

// 1. ================== 核心库包含 (Core Includes) ==================
// 核心CUDA运行时库，包含了大部分API函数如 cudaMalloc, cudaMemcpy等
// #include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <ATen/dtk_macros.h>
// #include <ATen/cuda/CUDAContext.h>
#include <ATen/hip/HIPContext.h>
// #include <c10/cuda/CUDAGuard.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
// 包含了核函数中使用的内置变量，如 threadIdx, blockIdx等
// #include <device_launch_parameters.h>

// 用于主机端进行输出和错误处理
#include <iostream>
#include <string>
#include <stdexcept> // 用于抛出异常

// 2. ================== CUDA错误检查宏 (Error Checking Macro) ==================
/**
 * @brief 一个强大的宏，用于检查CUDA API调用的返回值。
 *
 * 如果API调用返回的不是 cudaSuccess，它会构造一个详细的错误信息
 * (包括文件名、行号和错误描述) 并抛出一个 std::runtime_error 异常。
 * 这比在每个API调用后手动写 if/else 判断要简洁和健壮得多。
 *
 * @param call 要执行的CUDA API调用，例如 CUDA_CHECK(cudaMalloc(...));
 */
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::string error_msg = "CUDA Error in " + std::string(__FILE__) +   \
                                    " at line " + std::to_string(__LINE__) +     \
                                    ": " + cudaGetErrorString(err);              \
            throw std::runtime_error(error_msg);                                 \
        }                                                                        \
    } while (0)

// #define WARP_SIZE 32
#define WARP_SIZE 64
#define DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

#define DISPATCH_FLOATING_AND_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                               \
                     DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(__VA_ARGS__))

#define DISPATCH_CASE_INTEGRAL_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

// #define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
//     cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
// HIP (ROCm) 平台
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
// 3. ================== 设备端辅助函数 (Device-side Helper Functions) ==================
//
// 使用 __device__ __forceinline__ 可以在头文件中安全地定义设备端函数。
// __forceinline__ 建议编译器将函数内联，以获得最佳性能，适合短小的函数。

/**
 * @brief 在两个值之间进行线性插值 (Linear Interpolation)。
 * @tparam T 数据类型 (例如 float, double)。
 * @param a 起始值。
 * @param b 结束值。
 * @param t 插值因子 (通常在 0.0 到 1.0 之间)。
 * @return 插值结果。
 */
template <typename T>
__device__ __forceinline__ T lerp(T a, T b, T t) {
    return a + t * (b - a);
}


// 4. ================== 主机端接口声明 (Host-side API Declarations) ==================
//
// 这里是你项目的主要部分。声明那些作为CUDA功能入口的"包装函数"。
// 这些函数将在 .cu 文件中被实现，并可以被其他 .cpp 或 .cu 文件调用。
// 它们是连接 C++ 世界和 CUDA 世界的“桥梁”。

/**
 * @brief 示例函数：在GPU上对一个浮点数数组的每个元素乘以一个常数。
 *
 * @param data 指向主机端输入/输出数据的指针。函数内部会负责主机与设备间的数据传输。
 * @param num_elements 数组中的元素数量。
 * @param factor 要乘以的常数因子。
 */
void multiply_array_on_gpu(float* data, int num_elements, float factor);

// 你可以在这里添加更多的函数声明...
// void another_cool_gpu_function(int* params, ...);


#endif // CUDA_UTILS_CUH