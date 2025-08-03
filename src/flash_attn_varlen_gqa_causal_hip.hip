#include "utils_hip.cuh"
#include "ops.h"
#include <iostream>
#include <cmath>
#include <torch/all.h>
namespace electrock_infer{

// =================================================================================================
// Flash Attention 2
// HEAD_DIM = 128
// 支持变长 qkv_seq_len 
// 暂时不支持prefix caching, 如果支持则需要重构
// =================================================================================================
// block_dim 256, grid( DIV_CEIL(max_seq_len, Br), batch_size * q_head)
template <const int Br, const int Bc, const int PAD, const int HEAD_DIM>
__global__ void flash_attn_causal_varlen_gqa_hip_kernel(
    _Float16 *Q, _Float16 *K, _Float16 *V, _Float16 *O,
    int *cu_seqlens_q, 
    int *cu_seqlens_k, 
    int total_tokens_num,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    int Q_head, 
    int KV_head, 
    int group_size)
{   
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.y / Q_head;
    const int Q_head_id = blockIdx.y % Q_head;
    const int KV_head_id = Q_head_id / group_size;
    const int Q_tile_id = blockIdx.x;
    const int O_tile_id = Q_tile_id;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    // 暂时不支持prefix caching, 因此qkv的start和end是一样的
    const int seq_start_q = cu_seqlens_q[batch_id]; 
    const int seq_end_q = cu_seqlens_q[batch_id + 1];
    const int cur_seq_len = seq_end_q - seq_start_q; // [seq_start_q, seq_end_q) 左闭右开

    //offset
    const int start_token_index = seq_start_q + Q_tile_id * Br; // 起始token 索引
    const int Q_tile_gmem_offset = (start_token_index * Q_head * HEAD_DIM); //[start_token:, q_head, head_dim]
    const int Q_head_offset = Q_head_id * HEAD_DIM;
    const int O_tile_gmem_offset = Q_tile_gmem_offset;

    // 大于cur_seq的block提前退出, 还要考虑block内的无效token
    if(start_token_index >= seq_end_q){
        return;
    }
    // 考虑causal mask(大于Q索引的K就没必要计算Q@K了), 计算K_seq_len循环次数上限
    const int q_block_end_idx = min(start_token_index + Br, seq_end_q); // 当前block处理有效token的末尾索引
    const int causal_attention_key_token_nums =  q_block_end_idx - seq_start_q; // 当前block计算attention_score时，真正能注意到的key_token_num
    const int Tc = div_ceil(causal_attention_key_token_nums, Bc); // 修改前用cur_seq_len, 但是Query只能跟“之前的token”进行Q@K, 因此过滤掉后面的无效计算, 加速kernel
    // smem allocate
    extern __shared__ _Float16 smem[];
    constexpr int Q_tile_size = Br * (HEAD_DIM + PAD);
    constexpr int KV_tile_size = Bc * (HEAD_DIM + PAD);
    
    _Float16 *Q_tile_smem = smem; // [br, head_dim + pad]
    _Float16 *K_tile_smem = Q_tile_smem + Q_tile_size; // [bc, head_dim + pad]
    _Float16 *V_tile_smem = K_tile_smem + KV_tile_size; // [bc, head_dim + pad]
    
    // register allocate
    float16x4 R_Q = {0.0};
    float16x4 R_K = {0.0};
    float16x4 R_P = {0.0};
    float16x4 R_V = {0.0};
    float4_ R_S = {0.0};
    float4_ R_O[8] = {0.0};
    float4_ R_D[8] = {0.0};
    float global_pre_sum = 0.0f;
    float global_pre_max = -INFINITY;
    // load Q, gmem --> smem
    {   
        // 256个线程搬 64 * 128个数据, 每4个线程搬一行(每个线程搬32个)
        int temp_q_smem_start_row_id = tid / 4;
        int temp_q_smem_start_col_id = (tid % 4) * 32 ;
        int current_token_idx = start_token_index + temp_q_smem_start_row_id;
        bool is_valid_q_token = current_token_idx < seq_end_q; // 判断当前token是否有效
        #pragma unroll
        for(int i = 0; i<32;i++){
            int current_gmem_offset = Q_tile_gmem_offset + (temp_q_smem_start_row_id * Q_head * HEAD_DIM) + Q_head_offset + temp_q_smem_start_col_id + i;
            int current_smem_offset = temp_q_smem_start_row_id * (HEAD_DIM + PAD) + temp_q_smem_start_col_id + i;
            Q_tile_smem[current_smem_offset] = is_valid_q_token ? Q[current_gmem_offset] : static_cast<_Float16>(0);
        }
    }
    // 主循环： KV-loop
    #pragma unroll 1
    for(int K_tile_id = 0; K_tile_id < Tc; ++K_tile_id){
        
        // load K V , gmem --> smem
        {
            int KV_tile_start_token_index = seq_start_q + K_tile_id * Bc; // 起始token 索引
            int KV_tile_gmem_offset = KV_tile_start_token_index * KV_head * HEAD_DIM; // [start_token : , kv_head, head_dim]
            int KV_head_offset = KV_head_id * HEAD_DIM;
            // load K, 16 * 128 , 每16个线程搬一行(每个线程搬8个)
            int temp_kv_smem_start_row_id = tid / 16;
            int temp_kv_smem_start_col_id = (tid % 16) * 8;
            // 判断当前数据是否有效(序列末尾不足Bc的情况)
            int current_k_token_idx = KV_tile_start_token_index + temp_kv_smem_start_row_id;
            bool is_vaild_kv_token = current_k_token_idx < seq_end_q;
            int temp_kv_smem_offset = temp_kv_smem_start_row_id * (HEAD_DIM + PAD) + temp_kv_smem_start_col_id;
            int temp_kv_gmem_offset = KV_tile_gmem_offset + (temp_kv_smem_start_row_id * KV_head * HEAD_DIM) + KV_head_offset + temp_kv_smem_start_col_id;
            #pragma unroll
            for(int i = 0; i < 8; ++i){
                K_tile_smem[temp_kv_smem_offset + i] = is_vaild_kv_token ? K[temp_kv_gmem_offset + i] : static_cast<_Float16>(0);
            }
            //load V, 16 * 128, 同上
            #pragma unroll
            for(int i = 0; i < 8; ++i){
                V_tile_smem[temp_kv_smem_offset + i] = is_vaild_kv_token ? V[temp_kv_gmem_offset + i] : static_cast<_Float16>(0);
            }
        }
        __syncthreads(); // FIX：不能删
        R_S = {0.0};//重置
        // compute Q@K^T, 每个warp的计算是独立的, 每次迭代计算HMMA161616
        #pragma unroll
        for(int K_mma_tile = 0; K_mma_tile < (HEAD_DIM / 16); K_mma_tile++){
            // Q_smem --> R_Q
            {
                int lane_row_index = (warp_id * 16) + (lane_id % 16); // 每个warp负责16行
                int lane_col_index = (K_mma_tile * 16) + (lane_id / 16) * 4 ;
                R_Q[0] = Q_tile_smem[lane_row_index * (HEAD_DIM + PAD) + lane_col_index];
                R_Q[1] = Q_tile_smem[lane_row_index * (HEAD_DIM + PAD) + lane_col_index + 1];
                R_Q[2] = Q_tile_smem[lane_row_index * (HEAD_DIM + PAD) + lane_col_index + 2];
                R_Q[3] = Q_tile_smem[lane_row_index * (HEAD_DIM + PAD) + lane_col_index + 3];
            }
            // K_smem --> R_K
            {
                int lane_smem_K_Bc = (lane_id % 16); // 每个warp共享同一个K_tile
                int lane_smem_K_d = (K_mma_tile * 16) + (lane_id / 16) * 4;
                R_K[0] = K_tile_smem[lane_smem_K_Bc * (HEAD_DIM + PAD) + lane_smem_K_d];
                R_K[1] = K_tile_smem[lane_smem_K_Bc * (HEAD_DIM + PAD) + lane_smem_K_d + 1];
                R_K[2] = K_tile_smem[lane_smem_K_Bc * (HEAD_DIM + PAD) + lane_smem_K_d + 2];
                R_K[3] = K_tile_smem[lane_smem_K_Bc * (HEAD_DIM + PAD) + lane_smem_K_d + 3];
            }
            // 计算HMMA161616
            HMMA161616F32(R_S, R_Q, R_K);
        } // R_S[Br, Bc]
        // ======================= Masking for invalid Q-rows =======================
        // {
        //     // 每个线程需要知道自己负责的Q行的全局索引
        //     int q_token_idx_base = start_token_index + warp_id * 16 + (lane_id % 16);
        //     bool is_valid_q_row = q_token_idx_base < seq_end_q;

        //     // 如果当前行是无效的(padding)，则将其所有S值设为-inf
        //     // 这样它在后续的max和exp计算中就不会有任何贡献
        //     R_S[0] = is_valid_q_row ? R_S[0] : -INFINITY;
        //     R_S[1] = is_valid_q_row ? R_S[1] : -INFINITY;
        //     R_S[2] = is_valid_q_row ? R_S[2] : -INFINITY;
        //     R_S[3] = is_valid_q_row ? R_S[3] : -INFINITY;
        // }
        // ======================= Causal Mask Start =======================
        {
            // 每个线程需要知道自己负责计算的S[q_idx, k_idx]的全局索引
            int q_token_idx_base = start_token_index + warp_id * 16 + (lane_id % 16);
            int k_token_idx_base = seq_start_q + (K_tile_id * Bc);

            // R_S的4个分量对应4列的索引(参考MMA161616的C_layout), eg. T0: 0, 4, 8, 12 ; T16: 1, 5, 9, 13
            float s0 = R_S[0], s1 = R_S[1], s2 = R_S[2], s3 = R_S[3];
            int k_token_idx0 = k_token_idx_base + (lane_id/16) + 0;
            int k_token_idx1 = k_token_idx_base + (lane_id/16) + 4;
            int k_token_idx2 = k_token_idx_base + (lane_id/16) + 8;
            int k_token_idx3 = k_token_idx_base + (lane_id/16) + 12;

            // 如果k的索引 > q的索引，就进行屏蔽
            if (k_token_idx0 > q_token_idx_base) s0 = -INFINITY;
            if (k_token_idx1 > q_token_idx_base) s1 = -INFINITY;
            if (k_token_idx2 > q_token_idx_base) s2 = -INFINITY;
            if (k_token_idx3 > q_token_idx_base) s3 = -INFINITY;

            R_S[0] = s0; R_S[1] = s1; R_S[2] = s2; R_S[3] = s3;
        }
        // ======================== Causal Mask End ========================

        // Online-softmax
        float lane_row_max_new = -INFINITY;
        float lane_row_sum_new = 0.f;
        // compute cur_row_max  , P = exp(S-m_new) , sum(P)
        {   
            // max
            float tmp_max_0 = max(R_S[0], R_S[1]);
            float tmp_max_1 = max(R_S[2], R_S[3]);
            float cur_thread_row_max = max(tmp_max_0, tmp_max_1); 
            cur_thread_row_max = max(cur_thread_row_max, __shfl_xor(cur_thread_row_max, 16, 64)); // t, t+16, t+32, t+48
            cur_thread_row_max = max(cur_thread_row_max, __shfl_xor(cur_thread_row_max, 32, 64)); // t, t+32
            lane_row_max_new = max(global_pre_max, cur_thread_row_max * softmax_scale);
            

            // P 
            R_S[0] = __expf(__fmaf_rn(R_S[0], softmax_scale, -lane_row_max_new));
            R_S[1] = __expf(__fmaf_rn(R_S[1], softmax_scale, -lane_row_max_new));
            R_S[2] = __expf(__fmaf_rn(R_S[2], softmax_scale, -lane_row_max_new));
            R_S[3] = __expf(__fmaf_rn(R_S[3], softmax_scale, -lane_row_max_new));
            // sum
            float cur_thread_rowsum = R_S[0] + R_S[1] + R_S[2] + R_S[3];
            cur_thread_rowsum += __shfl_xor(cur_thread_rowsum, 16, 64);
            cur_thread_rowsum += __shfl_xor(cur_thread_rowsum, 32, 64);
            lane_row_sum_new = cur_thread_rowsum;
        }
        //整理R_S以符合MMA的layout和数据类型(C_layout_fp32 --> A_layout_fp16)
        // 转置前的布局(C_layout)                                                 转置后的布局（符合A_layout)
        //     寄存器| v0 (列0) | v1 (列1) | v2 (列2) | v3 (列3)             寄存器| v0       | v1       | v2       | v3
        // ---------|----------|----------|----------|----------       ---------|----------|----------|----------|----------
        // 线程 T0  |    A     |    B     |    C     |    D             线程 T0  |    A     |    E     |    I     |    M      
        // 线程 T16 |    E     |    F     |    G     |    H             线程 T16 |    B     |    F     |    J     |    N    
        // 线程 T32 |    I     |    J     |    K     |    L             线程 T32 |    C     |    G     |    K     |    O  
        // 线程 T48 |    M     |    N     |    O     |    P             线程 T48 |    D     |    H     |    L     |    P
        {
            const int group_id = lane_id % 16;
            const int role_id = lane_id / 16;
            // 计算组内四个线程伙伴的lane_id
            const int p0 = group_id + 0;   // T0 角色的伙伴
            const int p1 = group_id + 16;  // T16 角色的伙伴
            const int p2 = group_id + 32;  // T32 角色的伙伴
            const int p3 = group_id + 48;  // T48 角色的伙伴

            // --- 使用临时寄存器保存原始值，防止在计算过程中被污染 ---
            float r0 = R_S[0], r1 = R_S[1], r2 = R_S[2], r3 = R_S[3];

            // --- 阶段一：数据分发 (16次shuffle, 无发散) ---
            // 每个线程都计算出转置后的矩阵的全部4行数据，并暂存起来

            // 计算转置后的第0行 (A, E, I, M)
            float final_row0_v0 = __shfl(r0, p0, 64); // 从T0的r0获取A
            float final_row0_v1 = __shfl(r0, p1, 64); // 从T16的r0获取E
            float final_row0_v2 = __shfl(r0, p2, 64); // 从T32的r0获取I
            float final_row0_v3 = __shfl(r0, p3, 64); // 从T48的r0获取M

            // 计算转置后的第1行 (B, F, J, N)
            float final_row1_v0 = __shfl(r1, p0, 64); // 从T0的r1获取B
            float final_row1_v1 = __shfl(r1, p1, 64); // 从T16的r1获取F
            float final_row1_v2 = __shfl(r1, p2, 64); // 从T32的r1获取J
            float final_row1_v3 = __shfl(r1, p3, 64); // 从T48的r1获取N

            // 计算转置后的第2行 (C, G, K, O)
            float final_row2_v0 = __shfl(r2, p0, 64); // 从T0的r2获取C
            float final_row2_v1 = __shfl(r2, p1, 64); // 从T16的r2获取G
            float final_row2_v2 = __shfl(r2, p2, 64); // 从T32的r2获取K
            float final_row2_v3 = __shfl(r2, p3, 64); // 从T48的r2获取O

            // 计算转置后的第3行 (D, H, L, P)
            float final_row3_v0 = __shfl(r3, p0, 64); // 从T0的r3获取D
            float final_row3_v1 = __shfl(r3, p1, 64); // 从T16的r3获取H
            float final_row3_v2 = __shfl(r3, p2, 64); // 从T32的r3获取L
            float final_row3_v3 = __shfl(r3, p3, 64); // 从T48的r3获取P

            // --- 阶段二：数据选择 (根据角色赋值) ---
            // 每个线程根据自己的角色，从上面计算好的4行完整数据中，认领属于自己的那一行
            if(role_id == 0){
                R_P[0] = static_cast<_Float16>(final_row0_v0);
                R_P[1] = static_cast<_Float16>(final_row0_v1);
                R_P[2] = static_cast<_Float16>(final_row0_v2);
                R_P[3] = static_cast<_Float16>(final_row0_v3);
            }else if(role_id == 1){
                R_P[0] = static_cast<_Float16>(final_row1_v0);
                R_P[1] = static_cast<_Float16>(final_row1_v1);
                R_P[2] = static_cast<_Float16>(final_row1_v2);
                R_P[3] = static_cast<_Float16>(final_row1_v3);
            }else if(role_id == 2){
                R_P[0] = static_cast<_Float16>(final_row2_v0);
                R_P[1] = static_cast<_Float16>(final_row2_v1);
                R_P[2] = static_cast<_Float16>(final_row2_v2);
                R_P[3] = static_cast<_Float16>(final_row2_v3);
            }else if(role_id == 3){
                R_P[0] = static_cast<_Float16>(final_row3_v0);
                R_P[1] = static_cast<_Float16>(final_row3_v1);
                R_P[2] = static_cast<_Float16>(final_row3_v2);
                R_P[3] = static_cast<_Float16>(final_row3_v3);
            }
        }
        // compute P@V
        #pragma unroll
        for(int i = 0; i<8; i++){
            R_O[i] = {0.0}; // 重置
        }

        #pragma unroll
        for(int V_mma_tile = 0; V_mma_tile < (HEAD_DIM / 16); V_mma_tile++){
            // load V, smem --> R_V
            int lane_smem_V_d = V_mma_tile * 16 + lane_id % 16;
            int lane_smem_V_Bc = (lane_id / 16) * 4;

            R_V[0] = V_tile_smem[(lane_smem_V_Bc + 0) * (HEAD_DIM + PAD) + lane_smem_V_d];
            R_V[1] = V_tile_smem[(lane_smem_V_Bc + 1) * (HEAD_DIM + PAD) + lane_smem_V_d];
            R_V[2] = V_tile_smem[(lane_smem_V_Bc + 2) * (HEAD_DIM + PAD) + lane_smem_V_d];
            R_V[3] = V_tile_smem[(lane_smem_V_Bc + 3) * (HEAD_DIM + PAD) + lane_smem_V_d];

            __syncthreads(); // FIX：不能删
            HMMA161616F32(R_O[V_mma_tile], R_P, R_V);
        }
        //更新 O_old, l_old
        float block_row_max_old = K_tile_id > 0 ? global_pre_max : lane_row_max_new;
        float rescale_o_factor = __expf(block_row_max_old - lane_row_max_new);
        #pragma unroll
        for(int j = 0; j<8; j++){
            float4_ reg_O = R_O[j];
            float4_ reg_D = R_D[j];
            R_D[j][0] = __fmaf_rn(rescale_o_factor, reg_D[0], reg_O[0]);
            R_D[j][1] = __fmaf_rn(rescale_o_factor, reg_D[1], reg_O[1]);
            R_D[j][2] = __fmaf_rn(rescale_o_factor, reg_D[2], reg_O[2]);
            R_D[j][3] = __fmaf_rn(rescale_o_factor, reg_D[3], reg_O[3]);
        }
        global_pre_max = lane_row_max_new;
        global_pre_sum = __fmaf_rn(rescale_o_factor, global_pre_sum, lane_row_sum_new);
    }// 主循环结束
    // __syncthreads();
    float rescale_factor = __frcp_rn(global_pre_sum);
    //最终缩放 O = (1 / l) * D
    #pragma unroll
    for(int j = 0; j<8; j++){
        R_D[j][0] = rescale_factor * R_D[j][0];
        R_D[j][1] = rescale_factor * R_D[j][1];
        R_D[j][2] = rescale_factor * R_D[j][2];
        R_D[j][3] = rescale_factor * R_D[j][3];
    }
    // 将O写回gmem
    int store_token_idx = start_token_index + (warp_id * 16) + (lane_id % 16);
    int store_O_gmem_token_offset = store_token_idx * (Q_head * HEAD_DIM) + Q_head_offset; // 行偏移+head偏移 [head_dim]
    // 只写回有效行的数据
    if(store_token_idx < seq_end_q){

        #pragma unroll
        for(int j = 0; j < 8; j++){
            int store_O_gmem_dim_offset = store_O_gmem_token_offset + j * 16;
            O[store_O_gmem_dim_offset + (lane_id / 16) + 0] = R_D[j][0];
            O[store_O_gmem_dim_offset + (lane_id / 16) + 4] = R_D[j][1];
            O[store_O_gmem_dim_offset + (lane_id / 16) + 8] = R_D[j][2];
            O[store_O_gmem_dim_offset + (lane_id / 16) + 12] = R_D[j][3];
        }
    }

}

// Host端的启动函数
torch::Tensor flash_attn_causal_varlen_gqa_hip(
                                    torch::Tensor Q,  
                                    torch::Tensor K, 
                                    torch::Tensor V, 
                                    int max_seqlen_q,
                                    torch::Tensor cu_seqlens_q, // [batch_size + 1]
                                    int max_seqlen_k,
                                    torch::Tensor cu_seqlens_k, //  [batch_size + 1]
                                    float softmax_scale,
                                    bool causal
                                    ) {

    // Mixtral-7x8 : head_dim=128 Q_head=32 KV_head=8 
    const int total_tokens_num = Q.size(0);
    const int Q_head = Q.size(1);
    const int head_dim = Q.size(2); 
    const int KV_head = K.size(1);
    const int group_size = Q_head / KV_head;
    const int batch_size = cu_seqlens_q.size(0) - 1;

    // assert(causal==true)
    // assert(total_tokens_num == K.size(0));
    // assert(total_tokens_num == V.size(0));
    // assert(max_seqlen_q == max_seqlen_k);
    // assert(head_dim == 128);
    
    constexpr int Br = 64;
    constexpr int Bc = 16;
    constexpr int kNumThreads = WARP_SIZE * 4; // 256
    constexpr int kPad = 8;
    at::ScalarType origin_dtype = Q.scalar_type();
    torch::Tensor O = torch::empty_like(Q);
    const int smem_max_size = ((Br * (head_dim + kPad)) + (Bc * (head_dim + kPad)) + (Bc * (head_dim + kPad))) * sizeof(_Float16);
    TORCH_CHECK(smem_max_size <= 65536, "smem overflow");
    const int max_block_per_seq = DIV_CEIL(max_seqlen_q, Br);
    dim3 grid(max_block_per_seq, batch_size * Q_head);
    dim3 block(kNumThreads);
    hipFuncSetAttribute(
        (const void *)flash_attn_causal_varlen_gqa_hip_kernel<Br, Bc, kPad, 128>,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        smem_max_size);
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    flash_attn_causal_varlen_gqa_hip_kernel<Br, Bc, kPad, 128><<<grid, block, smem_max_size, stream>>>(
        reinterpret_cast<_Float16 *>(Q.data_ptr()),
        reinterpret_cast<_Float16 *>(K.data_ptr()),
        reinterpret_cast<_Float16 *>(V.data_ptr()),
        reinterpret_cast<_Float16 *>(O.data_ptr()),
        cu_seqlens_q.data_ptr<int32_t>(),
        cu_seqlens_k.data_ptr<int32_t>(),
        total_tokens_num,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        Q_head,
        KV_head,
        group_size);
    return O.to(origin_dtype);
}
}