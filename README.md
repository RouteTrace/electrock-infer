# Elect-Rock-Infer

A lightweight Inference Engine implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Fast Moe Engine for Mixtral.
* 📖 **Readable codebase** - Clean architectures.
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, customized kernel for MoE layer, etc.

## Install from source

```bash
conda create -n electrock python=3.10
conda activate electrock
cd electrock-infer-xdb

pip install https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.0/torch-2.1.0+das1.0+git00661e0.abi0.dtk2404-cp310-cp310-manylinux2014_x86_64.whl 
pip install https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.0/triton-2.1.0+das1.0+git3841f975.abi0.dtk2404-cp310-cp310-manylinux2014_x86_64.whl

pip install -e .
```

## Quick Start

See `example.py` for usage. 
```python
import os
from electrock_infer import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("/work/share/data/XDZS2025/Mixtral-8x7B-v0.1")
llm = LLM(path, enforce_eager=True, tensor_parallel_size=2, gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0.9, max_tokens=256)
prompts = [
    "Hello my name is"
]
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")
```

## Benchmark

See `bench.py` for benchmark.
```python

```

**Test Configuration:**
- Hardware: K100-AI (64GB) * 2
- Model: Mixtral-8x7B-Instruct-v0.1
- Total Requests: 100 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100-256 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) | Request(req/s)|
|----------------|-------------|----------|-----------------------|----|
| baseline  |  none    |   none  |      none        |noen
| Elect-Rock-Infer| none     | none   | none               |none
| ERI(flash_attn 2.0.4)|133,966| 111.79| 1198.42tok/s|2.29


