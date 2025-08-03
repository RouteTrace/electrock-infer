import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ['CXX'] = 'hipcc'

source_dir = "src"
cpp_files = glob.glob("*.cpp")
hip_files = glob.glob(os.path.join(source_dir, "*.hip"))
source_files = cpp_files + hip_files

# --- 配置编译参数和DCU架构 ---
targets = ['gfx928'] # 硬编码编译架构, 需要用到MMA指令
compile_args = ['-O3', '-std=c++17', '-Wno-unused-variable']
for target in targets:
    compile_args.append(f'--offload-arch={target}')

setup(
    name="electrock_infer",
    version="1.0.0",
    author="Shengguang Zhu", 
    description="a lightweight MoE inference engine implementation built from scratch", 
    python_requires=">=3.10,<3.13", 
    install_requires=[            
        "torch>=2.1.0",
        "triton>=2.1.0",
        "transformers>=4.51.0",
        "xxhash",
    ],
    packages=find_packages(),

    # C++ 扩展
    ext_modules=[
        CppExtension(
            name="electrock_infer._C", 
            sources=source_files,
            include_dirs=["src/include"], 
            libraries=['amdhip64'],
            extra_compile_args=compile_args
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)