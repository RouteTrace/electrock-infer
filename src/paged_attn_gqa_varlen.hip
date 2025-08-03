#include "utils_hip.cuh"
#include <torch/extension.h>
#include "ops.h"
namespace electrock_infer{


// =================================================================================================
// Paged Attention （结合FlashAttention-2）
// BLOCK_SIZE = 32, HEAD_DIM = 128
// 支持变长 kv_seq_len 
// =================================================================================================
#define DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
template <const int BLOCK_SIZE, const int HEAD_DIM, typename scalar_t>
__global__ void paged_attn_varlen_kernel(
        scalar_t *Q, scalar_t *K_cache, scalar_t *V_cache, scalar_t * O, 
        int32_t *block_table,
        int32_t *context_lens,
        int Q_head,
        int KV_head,
        int row_max_block_num,
        float softmax_scale
    ){
    const int tid = threadIdx.x;
    const int group_size = Q_head / KV_head; // GQA的分组
    const int batch_id = blockIdx.x / Q_head;
    const int Q_head_id = blockIdx.x % Q_head;
    const int KV_head_id = Q_head_id / group_size;
    const int kv_seq_len = context_lens[batch_id];
    // K-loop
    const int Bc = BLOCK_SIZE; 
    const int Tc = div_ceil(kv_seq_len, Bc); // 主循环次数
    // warp
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    // kv cache block_table
    const int block_num = div_ceil(kv_seq_len, BLOCK_SIZE); // block_num == Tc
    const int block_table_offset = batch_id * row_max_block_num;
    // gmem offset
    const int Q_gmem_offset = (batch_id * Q_head + Q_head_id) * HEAD_DIM;
    const int O_gmem_offset = Q_gmem_offset;
    // allocate shared memory 
    __shared__ scalar_t Q_batch_smem[HEAD_DIM]; // [HEAD_DIM]
    __shared__ scalar_t K_batch_smem[BLOCK_SIZE * HEAD_DIM]; // [BLOCK_SIZE, HEAD_DIM] 32 * 128 * 2 = 8KB
    __shared__ scalar_t V_batch_smem[BLOCK_SIZE * HEAD_DIM]; // [BLOCK_SIZE, HEAD_DIM] 
    __shared__ scalar_t S_V[4][8][16];
    __shared__ float partial_maxes[4]; 
    __shared__ float partial_sum[4]; 
    __shared__ float final_block_max_smem;
    __shared__ float final_block_sum_smem;
    
    // register 
    float row_max_old = -INFINITY;
    float row_sum_old = 0.f;
    scalar_t R_V[16];
    float R_O = 0.f;
    float R_D = 0.f;
    fill_1D_regs<scalar_t, 16>(R_V, 0);
    // load Q, gmem to smem
    if (tid < HEAD_DIM) {
        Q_batch_smem[tid] = Q[Q_gmem_offset + tid];
    }

    
    // 主循环 K-loop : 每次计算一个block
    #pragma unroll 1
    for(int K_tile_loop = 0; K_tile_loop < Tc; ++K_tile_loop){
        //load K and V cache, gmem to smem, BLOCK_SIZE个token
        int KV_cache_block_id = block_table[block_table_offset + K_tile_loop];// 当前要搬运的block全局ID
        int KV_cache_offset = (KV_cache_block_id * BLOCK_SIZE * KV_head * HEAD_DIM);// [BLOCK_SIZE, KV_head, HEAD_DIM]
        int thread_num_per_token = blockDim.x / BLOCK_SIZE; // 搬一个token的HEAD_DIM的线程数 --> 256 / 32 = 8
        int inner_token_id = tid / thread_num_per_token; // 当前线程负责块内第几个token(tid / 8)
        int batch_token_id = K_tile_loop * BLOCK_SIZE + inner_token_id; // 当前线程处理当前seq中第几个token
        int element_num_per_thread = HEAD_DIM / thread_num_per_token; // 每个线程要搬运的数量 --> 128 / 8 = 16
        int dim_start_index = (tid % thread_num_per_token) * element_num_per_thread; // 0, 16, 32, 48, ..., 96
        int KV_cache_per_thread_offset = KV_cache_offset + (inner_token_id * KV_head * HEAD_DIM) + (KV_head_id * HEAD_DIM) + dim_start_index;
        bool is_valid_kv_token = (batch_token_id < kv_seq_len); // Feature: support varlen
        // load KV, gmem-->smem
        #pragma unroll
        for(int i = 0; i < element_num_per_thread; i++){
            // 每个线程要搬运16个
            K_batch_smem[inner_token_id * HEAD_DIM + dim_start_index + i] = is_valid_kv_token ? K_cache[KV_cache_per_thread_offset + i] : static_cast<scalar_t>(0);
            V_batch_smem[inner_token_id * HEAD_DIM + dim_start_index + i] = is_valid_kv_token ? V_cache[KV_cache_per_thread_offset + i] : static_cast<scalar_t>(0);
        }
        __syncthreads(); 
        // compute Q@K^T : [128] * [32, 128]^T ：8个线程分别每个token的点积
        scalar_t sum_partial = static_cast<scalar_t>(0);
        float row_max_new = -INFINITY;
        float row_sum_new = 0.f;
        #pragma unroll
        for(int i = 0; i < 16; i++){
            sum_partial += is_valid_kv_token ? (Q_batch_smem[dim_start_index + i] * K_batch_smem[inner_token_id * HEAD_DIM + dim_start_index + i]) : static_cast<scalar_t>(0);
        }
        float temp_sum_fp32 = static_cast<float>(sum_partial);
        temp_sum_fp32 = warp_reduce_sum<float, 8>(temp_sum_fp32); 
        // Fix: mask invalid row
        float s_val_partial = is_valid_kv_token ? temp_sum_fp32 : -INFINITY; // 0~7, 8~15, 16~31, ..., 56~63 每个warp计算得到8个s_val

        {
            // compute row_max,  warp reduce --> block reduce
            float warp_max_partial = warp_reduce_max<float, WARP_SIZE>(s_val_partial);
            if(lane_id==0){
                partial_maxes[warp_id] = warp_max_partial;
            }
            __syncthreads();
            float tmp_reduce_max_val = 0;
            if (warp_id == 0) {
                tmp_reduce_max_val = (lane_id < 4) ? partial_maxes[lane_id] : 0.f;
                tmp_reduce_max_val = warp_reduce_max<float, 4>(tmp_reduce_max_val); 
            }
            if (tid == 0) {
                final_block_max_smem = static_cast<float>(tmp_reduce_max_val); // 由Warp 0的0号线程将最终值写入smem
            }
            __syncthreads();//所有线程持有final_block_max_smem

            // update new max
            row_max_new = max(row_max_old, final_block_max_smem * softmax_scale);
        }
        
        // compute P
        // Fix: support varlen
        float p_val_partial = is_valid_kv_token ? __expf(__fmaf_rn(s_val_partial, softmax_scale, -row_max_new)) : 0.f ; // 0~7, 8~16, .. 56~63 区间内的P相同
        float block_row_max_old = (K_tile_loop > 0) ? row_max_old : row_max_new; // 防止溢出(if K_tile_loop == 0 , row_max_old=-INFINITY)
        {
            //compute row_sum, warp reduce --> block reduce
            float tmp_reduce_sum_val = (lane_id % 8 ==0 ) ? p_val_partial : 0.f;
            float warp_sum_partial = warp_reduce_sum<float, WARP_SIZE>(tmp_reduce_sum_val);
            if(lane_id == 0){
                partial_sum[warp_id] = warp_sum_partial;
            }
            __syncthreads();
            float block_sum = 0.f;
            if(warp_id == 0){
                block_sum = (lane_id < 4) ? partial_sum[lane_id] : 0.f;
                block_sum = warp_reduce_sum<float, 4>(block_sum);
            }
            if(tid == 0){
                final_block_sum_smem = block_sum;
            }
            __syncthreads();//所有线程持有 final_block_sum_smem
            // update new sum
            row_sum_new = __expf(block_row_max_old - row_max_new) * row_sum_old + final_block_sum_smem;
        }
        // P@V 
        #pragma unroll
        for(int i = 0; i<16; i++){
            R_V[i] = V_batch_smem[inner_token_id * HEAD_DIM + dim_start_index + i];
        }
        #pragma unroll
        for(int i = 0; i<16; i++){
            R_V[i] = R_V[i] * static_cast<scalar_t>(p_val_partial); // 每8个线程与部分行进行计算
        }
        // ---------------------------------Warp 0----------------------------------
        // tid    |  t0  |    t1 |    t2 |    t3 |    t4 |   t5 |  t6    |      t7 |  
        // TOKEN_0| 0~15 | 16~31 | 32~47 | 48~64 | 64~79 | 80~95| 96~111 | 112~ 127|  
        // -------------------------------------------------------------------------
        // tid    | t8   | t9    | t10   | t11   | t12   | t13  | t14    |   t15   |  
        // TOKEN_1| 0~15 | 16~31 | 32~47 | 48~64 | 64~79 | 80~95| 96~111 | 112~ 127|  
        // -------------------------------------------------------------------------
        // ....    ....     ....    ....    ....    ....    ....  ....      ....   |
        // -------------------------------------------------------------------------
        // tid    | t56  | t57   | t58   | t59   | t60   | t61  | t62    |   t63   |  
        // TOKEN_7| 0~15 | 16~31 | 32~47 | 48~64 | 64~79 | 80~95| 96~111 | 112~ 127|  
        // -------------------------------------------------------------------------
        // 将每一列的数累加，4个warp的前8个lane都得到了对应128dim的中间值，还需要跨warp进行累加才得到最终值
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float val = static_cast<float>(R_V[i]); 
            val += __shfl_xor(val, 8, 64);
            val += __shfl_xor(val, 16, 64);
            val += __shfl_xor(val, 32, 64);
            R_V[i] = static_cast<scalar_t>(val);
        } // 每个warp的前8个thread的寄存器存放了HEAD_DIM的中间值

        if(lane_id < 8){
            #pragma unroll
            for(int i = 0; i < 16; i++){
                S_V[warp_id][lane_id][i] = R_V[i];
            }
        }
        __syncthreads();
        // 每2个thread计算一个最终值(128个组)并写回
        const int group_id = tid / 2; // 0~127
        scalar_t PV_val = 0; 
        PV_val += (S_V[(tid % 2) * 2 + 0][group_id / 16][group_id % 16]);
        PV_val += (S_V[(tid % 2) * 2 + 1][group_id / 16][group_id % 16]);
        PV_val += __shfl_xor(static_cast<float>(PV_val), 1, WARP_SIZE);
        //compute O
        R_O = __expf(block_row_max_old - row_max_new) * R_D + static_cast<float>(PV_val);
        // update m, l, O
        R_D = R_O;
        row_max_old = row_max_new;
        row_sum_old = row_sum_new;

    } // 主循环结束

    R_O = __frcp_rn(row_sum_old) * R_D;
    // write back to O
    if(tid % 2 == 0){
        O[O_gmem_offset + (tid / 2)] = static_cast<scalar_t>(R_O);
    }

}

// Host端的启动函数
torch::Tensor paged_attn_varlen(
        torch::Tensor Q, // [batch_size, head_num, head_dim]
        torch::Tensor K_cache,  // (total_block_num, block_size, num_kv_heads, head_dim)
        torch::Tensor V_cache,  // (total_block_num, block_size, num_kv_heads, head_dim)
        int max_seqlen_k,
        torch::Tensor context_lens, // [batch_size]
        torch::Tensor block_table, // [batch_size, max_block_num]
        float softmax_scale
    ){
    // CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf);
    // CHECK_TORCH_TENSOR_DTYPE(K_cache, torch::kHalf);
    // CHECK_TORCH_TENSOR_DTYPE(V_cache, torch::kHalf);
    // CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf);

    const int batch_size = Q.size(0);
    const int Q_head = Q.size(1);
    const int HEAD_DIM = Q.size(2);
    const int KV_head = K_cache.size(2);
    const int BLOCK_SIZE = K_cache.size(1);
    const int row_max_block_num = block_table.size(1);

    assert(Q_head % KV_head == 0); // GQA必须被整除
    assert(HEAD_DIM == 128); // Only support mixtral or other models with 128 head_dim.
    assert(block_table.size(0) == batch_size); // check 

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    torch::Tensor O = torch::empty_like(Q);
    switch (BLOCK_SIZE)
    {
    case 32:
        DISPATCH_FLOATING_TYPES(
                Q.scalar_type(),
                "paged_attn_varlen_kernel",
                [&] {
                    const dim3 grid(batch_size * Q_head);
                    const dim3 block(WARP_SIZE * 4);
                    paged_attn_varlen_kernel<32, 128, scalar_t><<<grid, block, 0, stream>>>(
                                Q.data_ptr<scalar_t>(),
                                K_cache.data_ptr<scalar_t>(),
                                V_cache.data_ptr<scalar_t>(),
                                O.data_ptr<scalar_t>(),
                                block_table.data_ptr<int32_t>(),
                                context_lens.data_ptr<int32_t>(),
                                Q_head,
                                KV_head,
                                row_max_block_num,
                                softmax_scale
                );
                }
        );
        break;
    default:
        throw std::runtime_error("The kvcache block size only supports 32.");
        break;
    }
    return O;
}
} // namespace

