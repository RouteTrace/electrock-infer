#include <torch/torch.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include "ops.h"

namespace py = pybind11;

// PYBIND11_MODULE 宏定义了模块的入口点
// 模块名 `my_rms_norm_ext` 必须与 CMake 中定义的目标名一致
PYBIND11_MODULE(electrock_infer, m) {
      m.doc() = "CUDA extension for My Project"; // 模块文档

      // 使用 m.def() 将 C++ 函数暴露给 Python
      // 第一个参数 是在 Python 中调用的函数名
      // 第二个参数是 C++ 函数的地址

      m.def("silu_and_mul",
            &electrock_infer::silu_and_mul,
            "silu_and_mul",
            py::arg("out"),
            py::arg("input"));

      m.def("topk_softmax",
            &electrock_infer::topk_softmax,
            "topk_softmax",
            py::arg("topk_weights"),
            py::arg("topk_indices"),
            py::arg("gating_output"));

      m.def("moe_sum",
            &electrock_infer::moe_sum,
            "moe_sum",
            py::arg("input"),
            py::arg("output"));

      m.def("moe_sum_efficient",
            &electrock_infer::moe_sum_efficient,
            "moe_sum_efficient",
            py::arg("input"),
            py::arg("output"));

      m.def("moe_align_block_size",
            &electrock_infer::moe_align_block_size,
            "moe_align_block_size",
            py::arg("topk_ids"),
            py::arg("num_experts"),
            py::arg("block_size"),
            py::arg("sorted_token_ids"),
            py::arg("expert_ids"),
            py::arg("num_tokens_post_pad"));

      m.def("flash_attn_simplified_causal_varlen_gqa",
            &electrock_infer::flash_attn_simplified_causal_varlen_gqa,
            "flash_attn_simplified_causal_varlen_gqa",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("O"));

      // m.def("add_rms_norm",
      //       &electrock_infer::add_rms_norm,
      //       "add_rms_norm",
      //       py::arg("input"),
      //       py::arg("residual"),
      //       py::arg("eps"),
      //       py::arg("weight"));

      // m.def("rms_norm",
      //       &electrock_infer::rms_norm,
      //       "rms_norm",
      //       py::arg("input"),
      //       py::arg("eps"),
      //       py::arg("weight"));


}