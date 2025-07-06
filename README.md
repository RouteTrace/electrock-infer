# Elect-Rock-Infer

A lightweight Inference Engine implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Fast Moe Engine for Mixtral.
* 📖 **Readable codebase** - Clean architectures.
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, customized kernel for MoE layer, etc.

## Installation

```bash
cd electrock-infer
pip install -e .
```

## Manual download

```bash

```

## Quick Start

See `example.py` for usage. 
```python

```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: A800 (80GB) * 2
- Model: Mixtral-8x7B-Instruct-v0.1
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) | Request(req/s)|
|----------------|-------------|----------|-----------------------|----|
| vLLM (v0)          |   133,966   |  83.54s   | 1603.57             |3.07
| Elect-Rock-Infer| 133,966     | 79.33s   | 1688.80               |3.23|

