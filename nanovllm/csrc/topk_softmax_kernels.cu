/*
 * vLLM moe_kernels.cu 的核心代码，专为 Mixtral (num_experts=8) 优化。
 *
 * 保留了从 C++ 入口 -> 内核启动器 -> 专用 CUDA 内核的完整调用链。
 * 移除了与 num_experts=8 无关的备用内核和 switch case，以保持简洁。
 */
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h" // 假设此文件定义了 VLLM_SHFL_XOR_SYNC_WIDTH 等宏

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

// 内核依赖的对齐数组定义
template <typename T, int N, int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
    float data[N];
};


// ====================== 核心：专为2的幂专家数设计的融合内核 ===============================

/*
  这是一个高度优化的 Top-K Gating Softmax 内核。
  它将 softmax 和 top-k 搜索融合在一次 GPU Kernel Launch 中，
  并利用 warp 内的 shuffle 指令进行高效的并行计算，避免使用共享内存。
  对于 num_experts=8，vLLM 正是调用这个内核以获得最高性能。
*/
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const float* input, const bool* finished, float* output, const int num_rows, int* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert)
{
    // --- 编译时常量计算，用于规划线程如何协作 ---
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static constexpr int ROWS_PER_WARP = (WARP_SIZE * VPT) / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // --- 运行时计算 ---
    // 计算当前线程负责处理哪一行 token 的 logits
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // --- 数据加载 ---
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = input + thread_row * ELTS_PER_ROW + first_elt_read_by_thread;

    using AccessType = AlignedArray<float, ELTS_PER_LDG>;
    float row_chunk[VPT];
    #pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        (reinterpret_cast<AccessType*>(&row_chunk))[ii] = (reinterpret_cast<const AccessType*>(thread_read_ptr))[ii * THREADS_PER_ROW];
    }

    // --- Fused Softmax 计算 ---
    // 1. Warp 内并行求最大值
    float thread_max = row_chunk[0];
    #pragma unroll
    for (int ii = 1; ii < VPT; ++ii) { thread_max = max(thread_max, row_chunk[ii]); }
    #pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW)); }

    // 2. Warp 内并行计算 exp(x - max) 并求和
    float row_sum = 0;
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii) { row_chunk[ii] = expf(row_chunk[ii] - thread_max); row_sum += row_chunk[ii]; }
    #pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW); }

    // 3. 计算 Softmax 概率值
    const float reciprocal_row_sum = 1.f / row_sum;
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii) { row_chunk[ii] *= reciprocal_row_sum; }


    // --- Fused Top-K 计算 (k=2) ---
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    // 循环 k 次来找到 top-k
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // 1. Warp 内并行执行 ArgMax
        float max_val = row_chunk[0];
        int expert = start_col;
        #pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
            #pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val) { max_val = val; expert = col + ii; }
            }
        }
        #pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // 2. 主线程将结果写回全局内存
        if (thread_group_idx == 0)
        {
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            // 在我们的简化场景下, start_expert=0, end_expert=8
            indices[idx] = should_process_row ? expert : NUM_EXPERTS; 
            if (source_rows) source_rows[idx] = k_idx * num_rows + thread_row;
        }

        // 3. 将已找到的最大值屏蔽，为下一次 top-k 迭代做准备
        if (k_idx + 1 < k)
        {
            if (thread_group_idx == ( (expert / ELTS_PER_LDG) % THREADS_PER_ROW) )
            {
                row_chunk[(expert / COLS_PER_GROUP_LDG) * ELTS_PER_LDG + (expert % ELTS_PER_LDG)] = -10000.f;
            }
        }
    }
}

// ====================== C++ Host端的启动器 (Launcher) ===============================
// 这是内核启动器的辅助结构体和函数，用于计算模板参数
namespace detail
{
template <int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(const float* input, const bool* finished, float* output, int* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert);
}

// 这是vLLM中的宏，我们保留它以展示原始的调用结构
#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB)                       \
    topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB>(         \
        gating_output, nullptr, topk_weights, topk_indicies,            \
        token_expert_indices, num_tokens, topk, 0, num_experts,         \
        stream);

// 这是C++ Host端的总分发函数
void topkGatingSoftmaxKernelLauncher(
    const float* gating_output,
    float* topk_weights,
    int* topk_indicies,
    int* token_expert_indices,
    float* softmax_workspace, // 在 expert=8 时不需要
    const int num_tokens,
    const int num_experts,
    const int topk,
    cudaStream_t stream) {
        
    static constexpr int WARPS_PER_TB = 4;
    // 原始的 switch 结构，我们只保留 num_experts = 8 的情况
    switch (num_experts) {
        // ... 其他 case 已被移除 ...
        case 8:
            LAUNCH_SOFTMAX(8, WARPS_PER_TB);
            break;
        // ... 其他 case 已被移除 ...
        default: {
            // 在 expert=8 的情况下，永远不会进入这个分支
            TORCH_CHECK(false, "This simplified launcher only supports num_experts=8");
        }
    }
}

} // namespace moe
} // namespace vllm


// ====================== Python调用的C++入口函数 ===============================
void topk_softmax(
    torch::Tensor& topk_weights,                // [num_tokens, topk=2]
    torch::Tensor& topk_indices,                // [num_tokens, topk=2]
    torch::Tensor& token_expert_indices,        // [num_tokens, topk=2]
    torch::Tensor& gating_output)               // [num_tokens, num_experts=8]
{
    const int num_experts = gating_output.size(-1);
    const int num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    // 检查输入是否符合我们特化的版本
    TORCH_CHECK(num_experts == 8, "This function is compiled for num_experts=8");
    TORCH_CHECK(topk == 2, "This function assumes top_k=2 for Mixtral");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 对于 expert=8, workspace是不需要的
    torch::Tensor softmax_workspace = torch::empty({0}, gating_output.options());

    vllm::moe::topkGatingSoftmaxKernelLauncher(
        gating_output.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        token_expert_indices.data_ptr<int>(),
        softmax_workspace.data_ptr<float>(),
        num_tokens,
        num_experts,
        topk,
        stream);
}