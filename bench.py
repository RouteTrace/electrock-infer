import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 128
    max_input_len = 1024 
    max_ouput_len = 1024 

    path = os.path.expanduser("~/zhushengguang/models/Qwen3-0.6B/")
    # path = os.path.expanduser("~/zhushengguang/models/Mixtral-8x7B-Instruct-v0.1/")
    llm = LLM(path, enforce_eager=True, max_model_len=4096, tensor_parallel_size=2,  )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # warmup
    llm.generate(["Benchmark: "], SamplingParams())
    print("Warmup done")

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
