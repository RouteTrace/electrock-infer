#include "hip/hip_runtime.h"
#include "utils_hip.cuh"
#include <torch/torch.h>
#include <torch/all.h>
#include <c10/cuda/CUDAMathCompat.h>
namespace electrock_infer {
    template <typename T, int N>
    struct alignas(sizeof(T) * N) Vector {
    T val[N];
};

template <typename scalar_t, typename cache_t, int Vec_Size>
__global__ void store_page_kv_cache(
        const scalar_t* __restrict__ key,
        const scalar_t* __restrict__ value,
        cache_t* __restrict__ key_cache,
        cache_t* __restrict__ value_cache, // (total_block_num, block_size, num_kv_heads, head_dim)
        const int32_t* __restrict__ slot_mapping,
        const int num_heads,
        const int head_size,
        const int block_size,
        const int total_elements_per_token
) {

    using cache_vector_t = Vector<cache_t, Vec_Size>;
    const int32_t token_idx = blockIdx.x;
    const int32_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
        return;
    }
    const int32_t block_idx = slot_idx / block_size;
    const int32_t block_offset = slot_idx % block_size;


    for (int i = threadIdx.x; i < total_elements_per_token / Vec_Size; i += blockDim.x) {

        const int vec_offset = i * Vec_Size;
        const int32_t src_key_idx = token_idx * total_elements_per_token + vec_offset;
        const int32_t src_value_idx = token_idx * total_elements_per_token + vec_offset;

        const cache_vector_t key_vec =
                *(reinterpret_cast<const cache_vector_t*>(&key[src_key_idx]));
        const cache_vector_t value_vec =
                *(reinterpret_cast<const cache_vector_t*>(&value[src_value_idx]));

        // cache layout: [num_blocks, block_size, num_heads, head_size]
        const int head_idx = vec_offset / head_size;
        const int head_offset = vec_offset & (head_size - 1);

        const int32_t block_stride = block_size * num_heads * head_size;
        const int32_t token_stride = num_heads * head_size;
        const int32_t head_stride = head_size;

        const int32_t tgt_idx = block_idx * block_stride +
                                block_offset * token_stride +
                                head_idx * head_stride +
                                head_offset;

        *(reinterpret_cast<cache_vector_t*>(&key_cache[tgt_idx])) =
                key_vec;
        *(reinterpret_cast<cache_vector_t*>(&value_cache[tgt_idx])) =
                value_vec;
    }
}
// Host端的启动函数
void paged_store_kvcache(
        torch::Tensor K, // [tokens_num, num_kv_heads, head_dim]
        torch::Tensor V, // [tokens_num, num_kv_heads, head_dim]
        torch::Tensor K_cache,  // (total_block_num, block_size, num_kv_heads, head_dim)
        torch::Tensor V_cache,  // (total_block_num, block_size, num_kv_heads, head_dim)
        torch::Tensor slot_mapping // (tokens_num)
){

//     CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf);
//     CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf);
//     CHECK_TORCH_TENSOR_DTYPE(K_cache, torch::kHalf);
//     CHECK_TORCH_TENSOR_DTYPE(V_cache, torch::kHalf);

    const int num_tokens = K.size(0);
    const int head_dim = K.size(2);

    const int num_kv_heads = K_cache.size(2);
    const int block_size = K_cache.size(1);
    constexpr bool scale = false;
    const int total_elements_per_token = head_dim * num_kv_heads;

    TORCH_CHECK(V.size(0) == num_tokens && V.size(1) == num_kv_heads && V.size(2) == head_dim, "V tensor shape mismatch");
    TORCH_CHECK(slot_mapping.numel() == num_tokens, "slot_mapping must have num_tokens elements");
//     TORCH_CHECK(total_elements_per_token % 1024 == 0, "Total elements per token must be divisible by 1024");

    DISPATCH_FLOATING_TYPES(
        K.scalar_type(),
        "store_page_kv_cache_wrapper",
        [&]
        {
                constexpr int vec_size = 8;
                const dim3 grid(num_tokens);
                const dim3 block(std::min(1024, (total_elements_per_token + vec_size - 1) / vec_size));
                store_page_kv_cache<scalar_t, scalar_t, vec_size><<<grid, block>>>(
                    K.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    K_cache.data_ptr<scalar_t>(),
                    V_cache.data_ptr<scalar_t>(),
                    slot_mapping.data_ptr<int32_t>(),
                    num_kv_heads,
                    head_dim,
                    block_size,
                    total_elements_per_token // Stride
                );
        });
}
}