#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAMathCompat.h>
// #include <c10/cuda/complex.h>

//hip
#define WARP_SIZE 64
//cuda
//#define WARP_SIZE 32
#define DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
namespace electrock_infer {

    template<typename scalar_t, typename T_ACC, int Seme_size, int num_per_Threads>
    __global__ __launch_bounds__(1024) void add_rms_norm_kernel(scalar_t *input, scalar_t *residual, scalar_t *weihgt, int hiddsize, T_ACC eps) {

        __shared__ T_ACC smem_val[Seme_size];
        __shared__ T_ACC s_rms_val;

        int col = threadIdx.x;
        int row = blockIdx.x;
        int laneid = col & (WARP_SIZE - 1);
        int wid = col / WARP_SIZE;

        unsigned int idx = row * hiddsize + col * num_per_Threads;

        //using Ldg_num_per_Threads = at::native::memory::aligned_vector<scalar_t, num_per_Threads>;
        unsigned int col_idx =  col * num_per_Threads;
        scalar_t reg_input[num_per_Threads];
        scalar_t reg_res[num_per_Threads];

        T_ACC r_rms_val;
        T_ACC val = (T_ACC) 0;
       // if (col_idx < hiddsize) {
            FETCH_FLOAT4(reg_input[0]) = FETCH_FLOAT4(input[idx]);
            FETCH_FLOAT4(reg_res[0]) = FETCH_FLOAT4(residual[idx]);
            for (int i = 0; i < num_per_Threads; i++) {
                reg_res[i] += reg_input[i];
                val += static_cast<T_ACC>(reg_res[i]) * static_cast<T_ACC>(reg_res[i]);
            }
        //}
        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            val += __shfl_down(val, stride, WARP_SIZE);
        }

        if (laneid == 0) {
            smem_val[wid] = val;
        }
        __syncthreads();

        val = (col < Seme_size) ? smem_val[col] : 0;
        if(wid == 0){
            for (int stride = Seme_size / 2; stride > 0; stride >>= 1) {
                val += __shfl_down(val, stride, WARP_SIZE);
            }
        }

        if (col == 0) {
            s_rms_val = c10::cuda::compat::rsqrt(val / hiddsize + eps);
        }
        __syncthreads();
        r_rms_val = s_rms_val;

        scalar_t reg_weihgt[num_per_Threads];

       // if (col_idx < hiddsize) {
            FETCH_FLOAT4(reg_weihgt[0]) = FETCH_FLOAT4(weihgt[col_idx]);
            for (int i = 0; i < num_per_Threads; i++) {
                reg_input[i] = static_cast<T_ACC>(reg_res[i]) * static_cast<T_ACC>(reg_weihgt[i]) * r_rms_val;
            }
            FETCH_FLOAT4(residual[idx]) = FETCH_FLOAT4(reg_res[0]);
            FETCH_FLOAT4(input[idx]) = FETCH_FLOAT4(reg_input[0]);
      //  }
    }

    template<typename scalar_t, typename T_ACC, int Seme_size, int num_per_Threads>
    __global__ __launch_bounds__(1024)  void rms_norm_kernel(scalar_t *input, scalar_t *weihgt, int hiddsize, T_ACC eps) {


        __shared__ T_ACC smem_val[Seme_size];
        __shared__ T_ACC s_rms_val;

        int col = threadIdx.x;
        int row = blockIdx.x;
        int laneid = col & (WARP_SIZE - 1);
        int wid = col / WARP_SIZE;

        unsigned int idx = row * hiddsize + col * num_per_Threads;
        unsigned int col_idx =  col * num_per_Threads;
        // using Ldg_num_per_Threads = at::native::memory::aligned_vector<scalar_t, num_per_Threads>;

        scalar_t reg_input[num_per_Threads];
        T_ACC r_rms_val;
        T_ACC val = (T_ACC) 0;

        //if  ( col_idx < hiddsize)  {
            FETCH_FLOAT4(reg_input[0]) = FETCH_FLOAT4(input[idx]);
            for (int i = 0; i < num_per_Threads; i++) {
                val += static_cast<T_ACC>(reg_input[i]) * static_cast<T_ACC>(reg_input[i]);
            }
     //   }

        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            val += __shfl_down(val, stride, WARP_SIZE);
        }

        if (laneid == 0) {
            smem_val[wid] = val;
        }
        __syncthreads();

        val = (col < Seme_size) ? smem_val[col] : 0;
        if(wid == 0){
            for (int stride = Seme_size / 2; stride > 0; stride >>= 1) {
                val += __shfl_down(val, stride, WARP_SIZE);
            }
        }
        if (col == 0) {
            s_rms_val = c10::cuda::compat::rsqrt(val / hiddsize + eps);
        }
        __syncthreads();
        r_rms_val = s_rms_val;

        scalar_t reg_weihgt[num_per_Threads];

     //   if (col_idx < hiddsize) {

            FETCH_FLOAT4(reg_weihgt[0]) = FETCH_FLOAT4(weihgt[col_idx]);
            for (int i = 0; i < num_per_Threads; i++) {
                reg_input[i] = static_cast<T_ACC>(reg_input[i]) * static_cast<T_ACC>(reg_weihgt[i]) * r_rms_val;
            }
            FETCH_FLOAT4(input[idx]) = FETCH_FLOAT4(reg_input[0]);
            // *(Ldg_num_per_Threads *) (input + idx) = *(Ldg_num_per_Threads *) reg_input;
        //}
    }


// 带残差连接，同时residual也要原地修改( x = x + residual ; residual = x;)
    void add_rms_norm(
            torch::Tensor &input,   // (num_tokens, hidden_size)
            torch::Tensor &residual,  // if not None, residual.shape = input.shape (num_tokens, hidden_size)
            double epsilon, // 1e-5
            torch::Tensor weight // self.weight.shape = (hidden_size,) = (4096)
    ) {

        const int hidden_size = input.size(1);
        const int num_tokens = input.size(0);
        TORCH_CHECK(hidden_size == 4096, " input.shape dimention may not be 2D");
        TORCH_CHECK(residual.size(1) == 4096, " residual.shape dimention may not be 2D");
        TORCH_CHECK(hidden_size % 8 == 0, " hidden_size.shape dimention may not fit for this kernal");
//        TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
//        TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
        // 设置 CUDA 执行配置
        // 设置 CUDA 执行配置
        // 根据数据类型分发并启动内核
        AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                input.scalar_type(),
                "rms_norm_kernel",
                [&] {
                    // using T_ACC = at::acc_type<scalar_t, true>;
                    using T_ACC = float;
                    T_ACC eps = epsilon;
                    scalar_t *input_data = input.expect_contiguous()->data_ptr<scalar_t>();
                    scalar_t *weight_data = weight.expect_contiguous()->data_ptr<scalar_t>();
                    scalar_t *res_data = residual.expect_contiguous()->data_ptr<scalar_t>();
                    if (hidden_size <= 1024) {
                        add_rms_norm_kernel<scalar_t, T_ACC, (128 / WARP_SIZE), 8><<<num_tokens, 128>>>(input_data,
                                res_data,
                                weight_data,
                                hidden_size, eps);
                    } else if (hidden_size <= 2048) {
                        add_rms_norm_kernel<scalar_t, T_ACC, (256 / WARP_SIZE), 8><<<num_tokens, 256>>>(input_data,
                                res_data,
                                weight_data,
                                hidden_size, eps);
                    } else if (hidden_size <= 4096) {
                        add_rms_norm_kernel<scalar_t, T_ACC, (512 / WARP_SIZE), 8><<<num_tokens, 512>>>(input_data,
                                res_data,
                                weight_data,
                                hidden_size, eps);
                    } else {
                        add_rms_norm_kernel<scalar_t, T_ACC, (1024 / WARP_SIZE), 8><<<num_tokens, 1024>>>(input_data,
                                res_data,
                                weight_data,
                                hidden_size, eps);
                    }
                }
        );
    }


// 不带残差连接的版本
    void rms_norm(
            torch::Tensor &input,   // (num_tokens, hidden_size)
            double epsilon, // 1e-5
            torch::Tensor &weight // self.weight.shape = (hidden_size,) = (4096)
    ) {
        const int hidden_size = input.size(1);
        const int num_tokens = input.size(0);
        TORCH_CHECK(hidden_size == 4096, " input.shape dimention may not be 2D");
        TORCH_CHECK(weight.size(0) == 4096, " weight.shape dimention may not be 1D");
        TORCH_CHECK(hidden_size % 8 == 0, " hidden_size.shape dimention may not fit for this kernal");
//        TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
//        TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    //    printf("hidden_size = %d \n", hidden_size);
     //   fflush(stdout);
        // 设置 CUDA 执行配置
        // 根据数据类型分发并启动内核
        AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                input.scalar_type(),
                "rms_norm_kernel",
                [&] {
                    // using T_ACC = at::acc_type<scalar_t, true>;
                    using T_ACC = float;
                    T_ACC eps = epsilon;
                    scalar_t *input_data = input.expect_contiguous()->data_ptr<scalar_t>();
                    scalar_t *weight_data = weight.expect_contiguous()->data_ptr<scalar_t>();
                    if (hidden_size <= 1024) {
                        rms_norm_kernel<scalar_t, T_ACC, (128 / WARP_SIZE), 8><<<num_tokens, 128>>>(input_data, weight_data,
                                hidden_size, eps);
                    } else if (hidden_size <= 2048) {
                        rms_norm_kernel<scalar_t, T_ACC, (256 / WARP_SIZE), 8><<<num_tokens, 256>>>(input_data, weight_data,
                                hidden_size, eps);
                    } else if (hidden_size <= 4096) {
                        rms_norm_kernel<scalar_t, T_ACC, (512 / WARP_SIZE), 8><<<num_tokens, 512>>>(input_data, weight_data,
                                4096, eps);
                    } else {
                        rms_norm_kernel<scalar_t, T_ACC, (1024 / WARP_SIZE), 8><<<num_tokens, 1024>>>(input_data,
                                weight_data,
                                hidden_size, eps);
                    }
                }
        );

    }
}
