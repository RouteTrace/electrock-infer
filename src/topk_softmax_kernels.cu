#include <torch/all.h>
#include "cuda_utils.cuh"
#include "ops.h"
#define CEIL(x,y) ((x+y-1) / y)
namespace electrock_infer
{
    // M 行 N 列，每一行计算softmax and topk=2
    __global__ void topkGatingSoftmax(const float *logits, float *weight_output, int *idx_output, const int topk, const int M, const int N)
    {
        int input_offset = blockIdx.x * blockDim.y * N; // 每个block的偏移
        int output_offset = blockIdx.x * blockDim.y * topk;
        int tid_x = threadIdx.x;
        int tid_y = threadIdx.y;

        int global_row = tid_y + blockIdx.x * blockDim.y; // 计算当前线程对应第几行，用于边界判断
        __shared__ float smem[4][8]; // 一个block处理4行8列的计算
        //搬运数据
        smem[tid_y][tid_x] = logits[input_offset + tid_y * N + tid_x];
        //online safe softmax
        if(tid_x == 0 && global_row < M){
            float max_val = -INFINITY;
            float pre_max_val = 0.f;
            float sum = 0.f;

            //用于追踪topk=2
            float top1_val = -INFINITY;
            int top1_idx = -1;
            float top2_val = -INFINITY;
            int top2_idx = -1;
#pragma unroll
            for(int i = 0; i<N;i++){
                //caculate online softmax
                float current_val = smem[tid_y][i];
                max_val = max_val >  current_val ? max_val : current_val;
                sum = expf(pre_max_val - max_val) * sum + expf(current_val - max_val);
                pre_max_val = max_val;

                // --- 更新 Top-K 追踪 ---
                if (current_val > top1_val)
                {
                    // 当前值成为新的 Top-1，旧的 Top-1 成为 Top-2
                    top2_val = top1_val;
                    top2_idx = top1_idx;
                    top1_val = current_val;
                    top1_idx = i;
                }
                else if (current_val > top2_val)
                {
                    // 当前值成为新的 Top-2
                    top2_val = current_val;
                    top2_idx = i;
                }
            }
            //理论上除以sum之前的topk_id已经是最终结果
            idx_output[output_offset+ tid_y * topk] = top1_idx;
            idx_output[output_offset + tid_y * topk + 1] = top2_idx;

            // 省略非top2的exp计算，直接把top2的数值写入output
            weight_output[output_offset + tid_y * topk] = expf(smem[tid_y][top1_idx] - max_val) / sum;
            weight_output[output_offset + tid_y * topk + 1] = expf(smem[tid_y][top2_idx] - max_val) / sum;
        }

    }
    // 这是C++ Host端的总分发函数
    void topkGatingSoftmaxKernelLauncher(
        const float *gating_output,
        float *topk_weights,
        int *topk_indicies,
        const int num_tokens, 
        const int num_experts,
        const int topk,
        cudaStream_t stream)
    {
        constexpr int BLOCK_COLUMN_SIZE = 8;
        constexpr int BLOCK_ROW_SIZE = WARP_SIZE / BLOCK_COLUMN_SIZE; 

        dim3 block_dim(BLOCK_COLUMN_SIZE, BLOCK_ROW_SIZE, 1); // block_size = (8, 4,)
        dim3 grid_dim(CEIL(num_tokens, BLOCK_ROW_SIZE), 1, 1); // grid_size = (num_blocks,)

        topkGatingSoftmax<<<grid_dim, block_dim, 0, stream>>>(
            gating_output, topk_weights, topk_indicies, topk, num_tokens, num_experts);
    }

    // ====================== Python调用的C++入口函数 ===============================
    void topk_softmax(
        torch::Tensor &topk_weights,         // [num_tokens, topk=2]
        torch::Tensor &topk_indices,         // [num_tokens, topk=2]
        torch::Tensor &gating_output)        // [num_tokens, num_experts=8]
    {
        const int num_experts = gating_output.size(-1);
        const int num_tokens = gating_output.numel() / num_experts;
        const int topk = topk_weights.size(-1);

        // 检查输入是否符合我们特化的版本
        TORCH_CHECK(num_experts == 8, "This function is compiled for num_experts=8");
        TORCH_CHECK(topk == 2, "This function assumes top_k=2 for Mixtral");

        //安全地、临时地切换当前 CUDA 设备
        const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
        
        topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            num_tokens,
            num_experts,
            topk,
            stream);
    }


} // namespace electrock_infer
