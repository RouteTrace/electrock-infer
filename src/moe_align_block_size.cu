#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "ops.h"
#include "cuda_utils.cuh"
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace electrock_infer {
namespace {
// 辅助函数，用于在共享内存中计算2D数组的索引
__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  return row * total_col + col;
}
}  // namespace


// ============================================================================
//  moe_align_block_size: 数据对齐与重排
// ============================================================================
template <typename scalar_t>
__global__ void moe_align_block_size_global_mem_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* tokens_cnts, int32_t* cumsum) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }
}

/**
 * 这是 `moe_align_block_size` 的核心CUDA内核。
 * 它使用共享内存（shared memory）来高效地完成 token 的计数和重排。
 * 算法分为几个阶段：并行计数 -> 并行前缀和 -> 并行重排。
 */
template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel(
    scalar_t *__restrict__ topk_ids, // 输入: 类型由模板决定
    int32_t *sorted_token_ids,       // 输出: 固定为 int32_t*
    int32_t *expert_ids,             // 输出: 固定为 int32_t*
    int32_t *total_tokens_post_pad,  // 输出: 固定为 int32_t*
    int32_t num_experts,             // 参数: 固定为 int32_t
    int32_t block_size,              // 参数: 固定为 int32_t
    size_t numel)
{
    // 动态分配共享内存
    extern __shared__ int32_t shared_mem[];
    // 将共享内存划分给 cumsum 和 tokens_cnts 两个数组使用
    int32_t* cumsum = shared_mem; // 前num_experts + 1
    // 剩下的 (num_thread+1) * num_experts， 每个线程一行，外加一行用于前缀和归约
    token_cnts_t* tokens_cnts = (token_cnts_t*)(shared_mem + num_experts + 1); 

    // --- 阶段1: 并行计数 ---
    // 每个线程负责一部分 token，统计这些 token 分别要去哪个专家
    
    // 计算每个thread负责多少tokens
    const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
    // 每一个线程负责的token的起始索引
    const size_t start_idx = threadIdx.x * tokens_per_thread; 
    // 把共享内存初始化为0，从第1行开始的，空出了第0行
    for (int i = 0; i < num_experts; ++i) {
        tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
    }
    // 统计thread对应的token中，要去哪个专家，然后累加
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
        ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
    }
    __syncthreads(); // 此时tokens_cnts中每一行(每个thread)都统计好了每个专家负责的token数量，下一步就是要计算总和

    // --- 阶段2: 并行前缀和 (Parallel Prefix Sum) ---
    // 对每个专家的计数值，在所有线程间进行累加，得到每个专家的总 token 数
    if (threadIdx.x < num_experts) {
        tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
        for (int i = 1; i <= blockDim.x; ++i) {
            tokens_cnts[index(num_experts, i, threadIdx.x)] += tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
        }
    }
    __syncthreads(); // 至此，每一行的每一个expert的统计数量，都累加到了最后一行 tokens_cnts[num_thread][num_expert]

    // --- 阶段3: 计算最终填充和偏移量 ---
    // 由单个线程(threadIdx.x == 0)完成最终的全局累加
    if (threadIdx.x == 0) {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i) {
            // 对每个专家的 token 数按 block_size 向上取整（补齐），然后计算累积和
            cumsum[i] = cumsum[i - 1] +
                        CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)], block_size) * block_size;
        }
        *total_tokens_post_pad = cumsum[num_experts];
    }
    __syncthreads(); // 至此，cumsum[num_expert] 累计了每一个专家所需要的槽位（按block_size为分配单位）；同时cumsum[id]代表起始点

    // --- 阶段4: 数据重排 (Permutation) ---
    // 为每个 block 写入其对应的 expert_id
    if (threadIdx.x < num_experts) {
        // 初始化 i 为 对应专家的blocks的起始索引，然后按block为单位分配expert_id； 每个线程只处理自己的blocks
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // 每个线程根据自己第一步的计数和第三步的全局偏移量，将 token 索引写入到排序后数组的正确位置
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
        int32_t expert_id = topk_ids[i];
        // 最终位置 = 该专家的全局起始位置 + 该 token 在当前线程中属于该专家的局部排名
        int32_t rank_post_pad = tokens_cnts[index(num_experts, threadIdx.x, expert_id)] + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        // 更新局部排名，为下一个相同专家的 token 做准备
        ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
    }
}

/**
 * C++ Host 端的启动函数 (Launcher)。
 * 对于 experts=8 的情况，共享内存总是足够的，因此我们直接选择最优路径。
 */
// 修正后的 C++ Host 启动函数 (伪代码)
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
//   cudaDeviceGetAttribute(&device_max_shared_mem,
//                          cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
// hip
  hipDeviceGetAttribute(&device_max_shared_mem,
                      hipDeviceAttributeMaxSharedMemoryPerBlock, // <-- HIP 版本
                      dev);
  const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);
  const int32_t shared_mem_i32 =
      ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
  const int32_t shared_mem_i16 =
      ((num_thread + 1) * num_experts) * sizeof(uint16_t) +
      (num_experts + 1) * sizeof(int32_t);

  bool use_global_memory = false;
  bool use_i16 = false;  // Use uint16_t for shared memory token counts
  if (shared_mem_i32 < device_max_shared_mem) {
    // Do nothing in this case. We're all set to use int32_t token counts
  } else if (shared_mem_i16 < device_max_shared_mem &&
             topk_ids.numel() <= 65535) {
    // when nelements of topk_ids is smaller than 65535 (max value of uint16),
    // element value of token_cnts would also smaller than 65535,
    // so we can use uint16 as dtype of token_cnts
    use_i16 = true;
  } else {
    use_global_memory = true;
  }

  if (use_global_memory) {
    DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_global_mem_kernel", [&] {
          // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
          // tensors
          const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);

          auto options_int = torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(topk_ids.device());
          torch::Tensor token_cnts_buffer =
              torch::empty({(num_experts + 1) * num_experts}, options_int);
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);

          auto kernel =
              electrock_infer::moe_align_block_size_global_mem_kernel<scalar_t>;
          kernel<<<1, num_thread, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), token_cnts_buffer.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>());
        });
  } else if (use_i16) {
    DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
          // set dynamic shared mem
          auto kernel =
              electrock_infer::moe_align_block_size_kernel<scalar_t, uint16_t>;
          AT_CUDA_CHECK(DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i16));
          kernel<<<1, num_thread, shared_mem_i16, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        });
  } else {
    DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
          auto kernel =
              electrock_infer::moe_align_block_size_kernel<scalar_t, int32_t>;
          AT_CUDA_CHECK(DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i32));
          kernel<<<1, num_thread, shared_mem_i32, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        });
  }
}
}
