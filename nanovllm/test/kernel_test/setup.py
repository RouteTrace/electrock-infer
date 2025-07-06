from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="electrock_infer", # package name
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "electrock_infer",
            ["pytorch_bind.cpp",
            "src/activation_kernels.cu",
            "src/moe_sum.cu",
            "src/moe_sum_efficient.cu",
            "src/moe_align_block_size.cu",
            "src/topk_softmax_kernels.cu"
            ]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)