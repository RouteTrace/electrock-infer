#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


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

/**
 * 这是 `moe_align_block_size` 的核心CUDA内核。
 * 它使用共享内存（shared memory）来高效地完成 token 的计数和重排。
 * 算法分为几个阶段：并行计数 -> 并行前缀和 -> 并行重排。
 */
template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids,      // 输入: [num_tokens * top_k]
    int32_t* sorted_token_ids,            // 输出: 排序后的 token 索引
    int32_t* expert_ids,                  // 输出: 每个 block 对应的专家 ID
    int32_t* total_tokens_post_pad,       // 输出: 填充后的总 token 数
    int32_t num_experts,
    int32_t block_size,
    size_t numel
) {
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
        *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
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
void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad
) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(num_experts == 8, "This simplified function is for num_experts=8");

    // 启动配置：线程数取 experts 和 warp size 中的较大者
    constexpr int32_t warpSize = 32;
    const int32_t num_thread = max((int32_t)num_experts, warpSize);
    
    // 计算所需的动态共享内存大小：内核中会分成两个使用, [num_experts+1] 以及 [num_thred+1][num_experts] 
    const int32_t shared_mem_size =
        ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
    
    // 直接以int32_t类型调用kernel
    using scalar_t = int32_t;
    auto kernel = electrock_infer::moe_align_block_size_kernel<scalar_t, int32_t>;

    // 启动内核，只使用一个 block，但 block 内有 num_thread 个线程
    kernel<<<1, num_thread, shared_mem_size, stream>>>(
        topk_ids.data_ptr<scalar_t>(),
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens_post_pad.data_ptr<int32_t>(),
        num_experts,
        block_size,
        topk_ids.numel()
    );
}
}