from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch
# 获取 PyTorch 的头文件路径
pytorch_include_dir = os.path.join(os.path.dirname(torch.__file__), 'include')
setup(
    name="electrock_infer", # package name
    include_dirs=["include",
                  pytorch_include_dir,
                  os.path.join(pytorch_include_dir, 'torch', 'csrc', 'api', 'include')],
    ext_modules=[
        CUDAExtension(
            "electrock_infer",
            ["pytorch_bind.cpp",
            "src/activation_kernels.cu",
            "src/moe_sum.cu",
            "src/moe_sum_efficient.cu",
            "src/moe_align_block_size.cu",
            "src/topk_softmax_kernels.cu",
            "src/flash_attn_mma_split_q_stage1_causal_varlen_gqa.cu"
            ]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)