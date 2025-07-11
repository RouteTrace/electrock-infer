#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#define DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
__global__ void add_rms_norm_kernel(

) { 
    
}

template <typename T>
__global__ void rms_norm_kernel(
    
) { 
    
}


// 带残差连接，同时residual也要原地修改( x = x + residual ; residual = x;)
void add_rms_norm(
    torch::Tensor& input,   // (num_tokens, hidden_size) 
    torch::Tensor& residual,  // if not None, residual.shape = input.shape (num_tokens, hidden_size)
    torch::Tensor eps, // 1e-5
    torch::Tensor weight, // self.weight.shape = (hidden_size,) = (4096)
){

    const int hidden_size = input.size(1);
    const int num_tokens = intput.size(0);
    TORCH_CHECK(hidden_size == 4096, " input.shape dimention may not be 2D");
    TORCH_CHECK(residual.size(1) == 4096, " residual.shape dimention may not be 2D");
    // 设置 CUDA 执行配置


    // 根据数据类型分发并启动内核
    DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_rms_norm_kernel", [&] {
        add_rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            eps,
            weight.data_ptr<scalar_t>()
        );
    });
}


// 不带残差连接的版本
void rms_norm(
    torch::Tensor& input,   // (num_tokens, hidden_size) 
    torch::Tensor eps, // 1e-5
    torch::Tensor weight, // self.weight.shape = (hidden_size,) = (4096)
){
   const int hidden_size = input.size(1);
    const int num_tokens = intput.size(0);
    TORCH_CHECK(hidden_size == 4096, " input.shape dimention may not be 2D");
    TORCH_CHECK(weight.size(0) == 4096, " weight.shape dimention may not be 1D");

    // 设置 CUDA 执行配置


    // 根据数据类型分发并启动内核
    DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
        rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            eps
        );
    });
    
}
""" implementation of torch.naive 

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

-------------------------------------------------------------------
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual
"""
