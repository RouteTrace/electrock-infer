#include "cuda_utils.cuh"
#include <torch/extension.h>
#include "ops.h"
namespace electrock_infer{


// =================================================================================================
// FlashAttention-2 简化学习版内核
//
// 硬编码配置:
// - kHeadDim: 128 (头的维度)
// - Br: 64 (Q tile的行数)
// - Bc: 16 (K/V tile的行/列数)
// - kStage: 1 (无软件流水线)
// - kNumThreads: 128 (每个块128个线程, 即4个Warp)
// =================================================================================================
__global__ void __launch_bounds__(128)
flash_attn_simplified_kernel_causal_varlen_gqa(half *Q, half *K, half *V, half *O,
                                 int QKV_seqlen, int Q_head, int KV_head, int group_size) {
  
    // --- 固定的常量 ---
    constexpr int kHeadDim = 128;
    constexpr int Br = 64;
    constexpr int Bc = 16;
    constexpr int kMmaAtomK = 16; // MMA指令的K维度
    constexpr int kPad = 8; // 共享内存行填充

    // --- 1. 索引和偏移量计算 ---
    const int Tc = div_ceil(QKV_seqlen, Bc); 
    const float scale = 1.0f / sqrtf((float)kHeadDim);

    const int QKV_batch_id = blockIdx.y / Q_head;
    const int Q_head_id = blockIdx.y % Q_head;
    const int KV_head_id = Q_head_id / group_size;
    const int Q_tile_id = blockIdx.x;
    const int O_tile_id = Q_tile_id;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int Q_gmem_offset = ((QKV_batch_id * Q_head + Q_head_id) * QKV_seqlen * kHeadDim); // Q[seq_len, d]
    const int K_gmem_offset = ((QKV_batch_id * KV_head + KV_head_id) * QKV_seqlen * kHeadDim); // K[seq_len, d]
    const int V_gmem_offset = K_gmem_offset;
    const int O_gmem_offset = Q_gmem_offset;

    // --- 线程到数据加载的映射 ---
    // 128个线程共同加载Q tile (64x128)
    int load_smem_Q_Br = (tid / 2); // 每个线程负责 Q tile 的半行(64列)
    int load_smem_Q_d = (tid % 2) * 64; 
    // 128个线程共同加载K/V tile (16x128)
    int load_smem_KV_Bc = (tid / 8); // 每个线程负责 K/V tile 的1/8行(16列)
    int load_smem_KV_d = (tid % 8) * 16;
    
    int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
    // 线程保护：如果当前线程要加载的Q行超出实际序列长度，则退出
    // if (load_gmem_Q_Br >= QKV_seqlen)
    //     return;

    // block保护: 处理seq尾部的block需要额外处理逻辑
    if (Q_tile_id * Br >= QKV_seqlen) {
        return;
    }

    // --- 2. 共享内存定义 ---
    extern __shared__ half smem[];
    constexpr int Q_tile_size = Br * (kHeadDim + kPad);
    constexpr int KV_tile_size = Bc * (kHeadDim + kPad);
    
    half *Q_tile_smem = smem;
    half *K_tile_smem = Q_tile_smem + Q_tile_size;
    half *V_tile_smem = K_tile_smem + KV_tile_size;
    
    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

    // --- 3. 寄存器定义 ---
    float lane_block_row_max_old[1][2];
    float lane_block_row_sum_old[1][2];
    fill_2D_regs<float, 1, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, 1, 2>(lane_block_row_sum_old, 0.0f);

    uint32_t R_Q[1][4];
    uint32_t R_K[2][2];     // Bc/8 = 16/8 = 2
    uint32_t R_V[16][2];    // kHeadDim/8 = 128/8 = 16
    uint32_t R_S[1][2][2];  // [M_tiles][N_tiles][regs] -> [1][2][2]
    uint32_t R_O[1][16][2]; // [M_tiles][N_tiles][regs] -> [1][16][2]
    uint32_t R_D[1][16][2];
    fill_3D_regs<uint32_t, 1, 2, 2>(R_S, 0);
    fill_3D_regs<uint32_t, 1, 16, 2>(R_D, 0);
    fill_3D_regs<uint32_t, 1, 16, 2>(R_O, 0);

    // uint32_t TYPE_SIZE = sizeof(half);
    // --- 4. 加载Q tile (gmem -> smem) ---
    {   
        // uint32_t load_smem_Q_ptr = (smem_Q_base_ptr + (load_smem_Q_Br * (kHeadDim + kPad) + load_smem_Q_d) * sizeof(half));
        // // 每个线程加载64个元素(128字节), 分16次, 每次16字节(cp.async.cg一次最多16B)
        // for (int i = 0; i < 64; i += 8) { 
        // CP_ASYNC_CG(load_smem_Q_ptr + i * TYPE_SIZE, &Q[load_gmem_Q_addr + i], 16);
        // }
        // CP_ASYNC_COMMIT_GROUP();
        int load_gmem_Q_addr = (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_smem_Q_d);
        int Q_smem_offset = (load_smem_Q_Br * (kHeadDim + kPad) + load_smem_Q_d);
        // 加入限制，为了正常搬运有效行的数据
        if(load_gmem_Q_Br < QKV_seqlen){
            #pragma unroll
            for(int i = 0; i<64; i+=8){
                LDST128BITS(Q_tile_smem[Q_smem_offset + i]) = LDST128BITS(Q[load_gmem_Q_addr + i]);
            }
        }else{
            // TODO: 大于实际seq_len的Q_tile_smem位置填0
            #pragma unroll
            for(int i = 0; i<64; i+=2){
                HALF2(Q_tile_smem[Q_smem_offset + i]) = __float2half2_rn(0.0f);
            }
        }
        
    }

    // --- 5. 主循环: 遍历K和V的tile ---
    #pragma unroll 1
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
        // --- 5.1 加载K和V (gmem -> smem) ---
        // kStage=1, 无流水线, 每次循环都同步加载当前的K和V
        { // 加载K
            int load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
            int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_KV_Bc;
            int load_gmem_K_d = load_smem_KV_d;
            int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
            // uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_KV_Bc * (kHeadDim + kPad) + load_smem_KV_d) * sizeof(half));
            // // 每个线程加载16个元素(32字节), 分2次
            // for (int i = 0; i < 16; i += 8) {
            //     CP_ASYNC_CG(load_smem_K_ptr + i * TYPE_SIZE, &K[load_gmem_K_addr + i], 16);
            // }
            // CP_ASYNC_COMMIT_GROUP();
            int K_smem_offset = load_smem_KV_Bc * (kHeadDim + kPad) + load_smem_KV_d;
            if(load_gmem_K_Bc < QKV_seqlen){
                for( int i = 0; i<16;i+=8){
                    LDST128BITS(K_tile_smem[K_smem_offset + i]) = LDST128BITS(K[load_gmem_K_addr + i]);
                }
            }else{
                // TODO: 最后一个tile_K不满16个seq，对应K_tile_smem填0
                for(int i=0;i<16;i+=2){
                    HALF2(K_tile_smem[K_smem_offset + i]) = __float2half2_rn(0.0f);
                }
            }

        }
        { // 加载V
            int load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
            int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_KV_Bc;
            int load_gmem_V_d = load_smem_KV_d;
            int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
            // uint32_t load_smem_V_ptr = (smem_V_base_ptr + (load_smem_KV_Bc * (kHeadDim + kPad) + load_smem_KV_d) * sizeof(half));
            // for (int i = 0; i < 16; i += 8) {
            //     CP_ASYNC_CG(load_smem_V_ptr + i * TYPE_SIZE, &V[load_gmem_V_addr + i], 16);
            // }
            // CP_ASYNC_COMMIT_GROUP();
            int V_smem_offset = load_smem_KV_Bc * (kHeadDim + kPad) + load_smem_KV_d;
            if(load_gmem_V_Bc < QKV_seqlen){
                for( int i = 0; i<16;i+=8){
                    LDST128BITS(V_tile_smem[V_smem_offset + i]) = LDST128BITS(V[load_gmem_V_addr + i]);
                }
            }else{
                // TODO: 最后一个tile_V不满16个seq，对应V_tile_smem填0
                for(int i=0;i<16;i+=2){
                    HALF2(V_tile_smem[V_smem_offset + i]) = __float2half2_rn(0.0f);
                }
            }
        }

        // 等待Q(第一次循环)和当前K加载完成；忽略最近的GROUP(1) 也就是V的commit
        // CP_ASYNC_WAIT_GROUP(1); 
        __syncthreads();

        // --- 5.2 计算S = Q@K^T ---
        fill_3D_regs<uint32_t, 1, 2, 2>(R_S, 0);
    #pragma unroll
        for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
            // 从smem加载Q分片到寄存器
            {
                int warp_smem_Q_Br = warp_id * 16; // 4个warp, 每个负责16行
                int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
                int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // 每个lane负责8个half元素，对应16B-->4个寄存器R_Q[0][0~3]
                uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr + (lane_smem_Q_Br * (kHeadDim + kPad) + lane_smem_Q_d) * sizeof(half));
                LDMATRIX_X4(R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], lane_smem_Q_ptr);
            }
            // 从smem加载K分片到寄存器
        #pragma unroll
            for (int j = 0; j < 2; ++j) { // kWarpTileSeqLenK = 2
                int warp_smem_K_Bc = j * 8;
                int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
                int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8; // 这里蕴含了 warpTileK 转置, 将ldmatrix所需的起始地址设为了column主序,搬到寄存器上就是转置后的了
                uint32_t lane_smem_K_ptr = (smem_K_base_ptr + (lane_smem_K_Bc * (kHeadDim + kPad) + lane_smem_K_d) * sizeof(half));
                LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
            }
            // 执行MMA指令
        #pragma unroll
            for (int j = 0; j < 2; ++j) { // kWarpTileSeqLenK = 2
                // R_S = R_Q * R_K + R_S, 注意这里是乘累加
                HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], 
                            R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
            }
        }// 至此，tile_k_seqlen对应的S矩阵计算完毕 tile_S[Br, Bc] = [64, 8*2]

        //根据每个RS寄存器的值的全局索引判断是否填入mask
        int s_row_0 = Q_tile_id * Br + warp_id * 16 + lane_id / 4;
        int s_row_1 = Q_tile_id * Br + warp_id * 16 + 8 + lane_id / 4;
        int s_col = tile_K_seqlen * Bc + (lane_id % 4) * 2;;
        #pragma unroll
        for(int j = 0 ; j < 2; ++j){
            half2 s0 = HALF2(R_S[0][j][0]);
            half2 s1 = HALF2(R_S[0][j][1]);
            s0.x = ((s_col + j * 8) > s_row_0) ? __float2half(-INFINITY) : s0.x;
            s0.y = ((s_col + j * 8 + 1) > s_row_0) ? __float2half(-INFINITY) : s0.y;
            s1.x = ((s_col + j * 8) > s_row_1) ? __float2half(-INFINITY) : s1.x;
            s1.y = ((s_col + j * 8 + 1) > s_row_1) ? __float2half(-INFINITY) : s1.y;
            HALF2(R_S[0][j][0]) = s0;
            HALF2(R_S[0][j][1]) = s1;
        }

        __syncthreads();

        // --- 5.3 在线Softmax ---
        float lane_row_max_new[1][2];
        float lane_row_sum_new[1][2];
        fill_2D_regs<float, 1, 2>(lane_row_max_new, -INFINITY);
        fill_2D_regs<float, 1, 2>(lane_row_sum_new, 0.0f);
        // a. 计算当前块 S_ij 的行最大值 m_new
    #pragma unroll
        for (int j = 0; j < 2; ++j) { // kWarpTileSeqLenK = 2 , 因为有16列，每次循环是比较出8列中最大的值，又因为数据布局都是多个8*8的小矩阵；一个warp对应16*16，所以分两部分+两次循环处理，具体要参考C|D数据布局
            float2 t_reg_S_0 = __half22float2(HALF2(R_S[0][j][0]));
            float2 t_reg_S_1 = __half22float2(HALF2(R_S[0][j][1]));
            float tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
            float tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
            lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
            lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
        }
        lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]); // t0~3,t4~7,t8,...,t28~31 对应row 0,1,2,...,7; 
        lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]); // 对应row 8,9,..,15,的最大值
        __syncthreads();

        // b. 更新全局最大值，并计算 P=exp(S-m_new) 和 sum(P)
        float block_row_max_new_0 = lane_row_max_new[0][0];
        float block_row_max_new_1 = lane_row_max_new[0][1];
        float block_row_max_old_0 = lane_block_row_max_old[0][0];
        float block_row_max_old_1 = lane_block_row_max_old[0][1];
        block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
        block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

    #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float2 t_reg_S_0 = __half22float2(HALF2(R_S[0][j][0]));
            float2 t_reg_S_1 = __half22float2(HALF2(R_S[0][j][1]));
            t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, -block_row_max_new_0));
            t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, -block_row_max_new_0));
            t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, -block_row_max_new_1));
            t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, -block_row_max_new_1));
            lane_row_sum_new[0][0] += (t_reg_S_0.x + t_reg_S_0.y);
            lane_row_sum_new[0][1] += (t_reg_S_1.x + t_reg_S_1.y);
            HALF2(R_S[0][j][0]) = __float22half2_rn(t_reg_S_0);
            HALF2(R_S[0][j][1]) = __float22half2_rn(t_reg_S_1);
        }
        lane_row_sum_new[0][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
        lane_row_sum_new[0][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
        __syncthreads();
        
        // --- 5.4 计算 P@V 并更新 O, m, l ---
        CP_ASYNC_WAIT_GROUP(0); // 确保V已经加载完毕
        __syncthreads();

        fill_3D_regs<uint32_t, 1, 16, 2>(R_O, 0);
    #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) { // Bc=16, kMmaAtomK=16, so loop runs once
        #pragma unroll
            for (int j = 0; j < 16; ++j) { // kWarpTileHeadDimV = 16
                int warp_smem_V_d = j * 8;
                int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16; //所有warp共享V_tile
                uint32_t lane_smem_V_ptr = (smem_V_base_ptr + (lane_smem_V_Bc * (kHeadDim + kPad) + warp_smem_V_d) * sizeof(half));
                LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
            }
            
            int w = tile_V_Bc * 2; // w=0
        #pragma unroll
            for (int j = 0; j < 16; ++j) { // kWarpTileHeadDimV = 16
                HMMA16816(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1],
                            R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1],
                            R_O[0][j][0], R_O[0][j][1]);
            }
        }
        __syncthreads();

        // 更新 O_old, l_old
        block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
        block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

        float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
        float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);

    #pragma unroll
        for (int j = 0; j < 16; ++j) { // kWarpTileHeadDimV = 16
            float2 t_reg_O_0 = __half22float2(HALF2(R_O[0][j][0]));
            float2 t_reg_O_1 = __half22float2(HALF2(R_O[0][j][1]));
            float2 t_reg_D_0 = __half22float2(HALF2(R_D[0][j][0]));
            float2 t_reg_D_1 = __half22float2(HALF2(R_D[0][j][1]));
            
            t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
            t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
            t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
            t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
            HALF2(R_D[0][j][0]) = __float22half2_rn(t_reg_D_0);
            HALF2(R_D[0][j][1]) = __float22half2_rn(t_reg_D_1);
        }
        
        float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
        float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
        lane_block_row_sum_old[0][0] = __fmaf_rn(rescale_o_factor_0, block_row_sum_old_0, lane_row_sum_new[0][0]);
        lane_block_row_sum_old[0][1] = __fmaf_rn(rescale_o_factor_1, block_row_sum_old_1, lane_row_sum_new[0][1]);
        
        lane_block_row_max_old[0][0] = block_row_max_new_0;
        lane_block_row_max_old[0][1] = block_row_max_new_1;

    } // 主循环结束
    __syncthreads();

    // --- 6. 最终缩放并写回全局内存 ---
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
    #pragma unroll
    for (int j = 0; j < 16; ++j) { // kWarpTileHeadDimV = 16
        float2 t_reg_D_0 = __half22float2(HALF2(R_D[0][j][0]));
        float2 t_reg_D_1 = __half22float2(HALF2(R_D[0][j][1]));
        t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
        t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
        t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
        t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
        HALF2(R_D[0][j][0]) = __float22half2_rn(t_reg_D_0);
        HALF2(R_D[0][j][1]) = __float22half2_rn(t_reg_D_1);
    }

    // "集体存储"：从寄存器直接写入全局内存
    // TODO：只对有效行的数据进行搬运
    #pragma unroll
    for (int j = 0; j < 16; ++j) { // kWarpTileHeadDimV = 16
        uint32_t R_Z[2][4]; // 用来收集数据，并以向量化方式读写
        R_Z[0][0] = R_D[0][j][0];
        R_Z[1][0] = R_D[0][j][1];
        R_Z[0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
        R_Z[0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
        R_Z[0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
        R_Z[1][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
        R_Z[1][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
        R_Z[1][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);

        if (lane_id % 4 == 0) {
            int store_warp_regs_O_Br = warp_id * 16;
            int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
            int store_warp_regs_O_d = j * 8;
            if(store_lane_gmem_O_Br < QKV_seqlen){
                int store_gmem_O_addr_0 = (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_warp_regs_O_d);
                LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
            }
            if(store_lane_gmem_O_Br + 8 < QKV_seqlen){
                int store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_warp_regs_O_d);
                LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
            }
            
        }
    }
}

// Host端的启动函数
void flash_attn_simplified_causal_varlen_gqa(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
    CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf);

    const int QKV_batch = Q.size(0);
    const int Q_head = Q.size(1);
    const int QKV_seqlen = Q.size(2);
    const int d = Q.size(3);
    const int KV_head = K.size(1);
    const int group_size = Q_head / KV_head;

    assert(KV_head == V.size(1));
    // 检查输入维度是否符合我们简化的版本
    if (d != 128) {
        throw std::runtime_error("This simplified kernel only supports head_dim=128.");
    }


    constexpr int Br = 64;
    constexpr int Bc = 16;
    constexpr int kNumThreads = 128;
    constexpr int kPad = 8;
    constexpr int kStage = 1;
    constexpr int kHeadDim = 128;

    const int smem_max_size = ((Br * (kHeadDim + kPad)) + (kStage * Bc * (kHeadDim + kPad)) + (Bc * (kHeadDim + kPad))) * sizeof(half);

    dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * Q_head);
    dim3 block(kNumThreads);

    cudaFuncSetAttribute(flash_attn_simplified_kernel_causal_varlen_gqa, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    flash_attn_simplified_kernel_causal_varlen_gqa<<<grid, block, smem_max_size>>>(
        reinterpret_cast<half *>(Q.data_ptr()),
        reinterpret_cast<half *>(K.data_ptr()),
        reinterpret_cast<half *>(V.data_ptr()),
        reinterpret_cast<half *>(O.data_ptr()),
        QKV_seqlen,
        Q_head, KV_head, group_size);
}

} // namespace electrockinfer
