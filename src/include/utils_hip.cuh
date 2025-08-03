#pragma once
#include <algorithm>
#include <ATen/dtk_macros.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
// #include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "hip/hip_ext.h"
#include <float.h>
// #include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #include <torch/all.h>
// #include <torch/extension.h>
// #include <torch/types.h>
#include <vector>


// HIP 错误检查宏
#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::string error_msg = "HIP error in file '" + std::string(__FILE__) + "' on line " + std::to_string(__LINE__) + ": " + std::string(hipGetErrorString(e)); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)

#define WARP_SIZE 64
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
// #define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<hip_bfloat16 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
typedef float float4_ __attribute__((ext_vector_type(4)));
typedef float float2_ __attribute__((ext_vector_type(2)));
typedef float float8_  __attribute__((ext_vector_type(8)));
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef uint32_t UINT2  __attribute__((ext_vector_type(2)));
typedef uint32_t UINT4  __attribute__((ext_vector_type(4)));
typedef uint32_t UINT8  __attribute__((ext_vector_type(8)));
typedef long BB  __attribute__((ext_vector_type(2)));
// mma
#define LDMATRIX_32x8_B32(dst,src)                                                       \
    asm volatile(                                                                           \
            "DS_READ_M32x8_B32  %0, %1 offset: 0;"                                         \
            : "=v"(dst) : "{v1}"(src))

#define HMMA16168(D,A,B,C)                                                                       \
    asm volatile(                                                                           \
            "v_mmac_16x16x8_f32 %0 ,%1 ,%2 ,%0 \n"                                     \
            :"=v"(D)                                                                        \
            :"v"(A),"v"(B),"v"(C))

#define HMMA161616F32(C,A,B) \
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t" :"+v"(C) : "v"(A), "v"(B));

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  if (((T2).size(0) != (T1).size(0)) || ((T2).size(1) != (T1).size(1)) ||      \
      ((T2).size(2) != (T1).size(2)) || ((T2).size(3) != (T1).size(3))) {      \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

#define DIV_CEIL(a,b) ((a+b-1) / b)


template <typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
    val += __shfl_xor(val, mask, kWarpSize);
  }
  return val;
}

template <typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
    val = max(val, __shfl_xor(val, mask, kWarpSize));

  }
  return val;
}

template <typename T, int M, const int N, const int K = 2>
DEVICE_INLINE void fill_3D_regs(T (&R)[M][N][K], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        R[i][j][k] = val;
      }
    }
  }
}

template <typename T, int M, const int N = 2>
DEVICE_INLINE void fill_2D_regs(T (&R)[M][N], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      R[i][j] = val;
    }
  }
}

template <typename T, int M> DEVICE_INLINE void fill_1D_regs(T (&S)[M], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
    S[i] = val;
  }
}

template <typename T, int M>
DEVICE_INLINE void fill_1D_smem(T (&R)[M], T val, int tid) {
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      R[i] = val;
    }
  }
}

#define INFHALF __float2half(65536.0f)
#define ZEROHALF __float2half(0.0f)


// DISPATCH DATATYPE
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
  
#define DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
