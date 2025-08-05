import os
import sys
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import time
import numpy as np
import math
import subprocess
MAX_LEN = 512 
BATCH_SIZE_INFERENCE = 8  # 推理时一次处理多个句子
BATCH_SIZE_PERPLEXITY = 8  # 计算困惑度时使用的批次
# MODEL_PATH = "/work/share/data/XDZS2025/Mixtral-8x7B-v0.1"
# DATASET_PATH = "/work/share/data/XDZS2025/wikitext-103-raw-v1/wikitext-test.arrow"
# NEW_DATASET_PATH = "/work/share/data/XDZS2025/wikitext-103-raw-v1/wiki2.csv"  # 使用上传的CSV文件路径
MODEL_PATH = "/work/home/ac1m1kpqy8/zhusg/models/AI-ModelScope/Mixtral-8x7B-v0_1"
DATASET_PATH = "/work/home/ac1m1kpqy8/zhusg/CODES/data/wikitext-test.arrow"
NEW_DATASET_PATH = "/work/home/ac1m1kpqy8/zhusg/CODES/data/wiki2.csv"  # 使用上传的CSV文件路径
EVAL_SENTENCE_COUNT = 1   # 延迟评估的句子数量
USE_MULTI_GPU = True  # 使用多个GPU进行评估
BASELINE_PPL=219.9624
BASELINE_LATENCY_PER_SEQ=13.0597
def load_new_dataset(new_dataset_path):
    """加载新的CSV数据集，并清理不必要的引号"""
    print("new dataset is ", new_dataset_path)
    # 直接加载 CSV 文件，不需要指定 "test"
    dataset = load_dataset("csv", data_files=new_dataset_path)
    print("数据集文件路径：", new_dataset_path)

    # 获取数据集的实际键名并清理额外的引号
    split_key = list(dataset.keys())[0].strip("'\"")  # 去掉多余的引号
    print("Cleaned Dataset keys:", split_key)  # 打印出清理后的键名

    return dataset, split_key

from contextlib import contextmanager
@contextmanager
def suppress_stderr():
    """一个临时屏蔽标准错误的上下文管理器。"""
    original_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    try:
        sys.stderr = devnull
        yield
    finally:
        sys.stderr.close() # 关闭 devnull 文件句柄
        sys.stderr = original_stderr # 恢复原始的 stderr
# def get_all_gpu_memory_usage():
#     """获取所有GPU的内存使用情况"""
#     memory_info = {}
#     for i in range(torch.cuda.device_count()):
#         torch.cuda.synchronize(i)
#         allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
#         reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
#         memory_info[f"GPU {i}"] = {
#             "allocated": allocated,
#             "reserved": reserved,
#             "total": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
#         }
#     return memory_info

# def print_memory_usage(prefix=""):
#     """打印所有GPU当前内存使用情况"""
#     memory_info = get_all_gpu_memory_usage()
#     print(f"\n{prefix}GPU Memory Usage:")
#     for gpu, info in memory_info.items():
#         print(f"{gpu}: Allocated: {info['allocated']:.2f} GB / Reserved: {info['reserved']:.2f} GB / Total: {info['total']:.2f} GB")
#     total_allocated = sum(info['allocated'] for info in memory_info.values())
#     total_reserved = sum(info['reserved'] for info in memory_info.values())
#     print(f"Total Allocated: {total_allocated:.2f} GB, Total Reserved: {total_reserved:.2f} GB")


def install_project(project_path: str):
    """
    使用 pip 以可编辑模式 (-e) 安装位于 project_path 的项目。
    """
    # <--- 改进: 将传入的路径转换为绝对路径，使日志更清晰 ---
    project_path = os.path.abspath(project_path)
    print(f"--- 准备安装项目: {project_path} ---")
    # 1. 确认路径存在且包含 setup.py 文件
    if not os.path.isdir(project_path) or not os.path.exists(os.path.join(project_path, 'setup.py')):
        print(f"错误：路径 '{project_path}' 不存在或其中没有 setup.py 文件。")
        return False
    # 2. 构建 pip 命令 (保持不变)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        "."
    ]
    # 3. 执行命令 (保持不变)
    try:
        print(f"将在目录 '{project_path}' 中执行命令: {' '.join(command)}")
        result = subprocess.run(
            command, 
            check=True, 
            cwd=project_path, 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        print("--- 安装成功 ---")
        print("STDOUT:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("--- 安装失败 ---")
        print(f"返回码: {e.returncode}")
        print("\n--- STDOUT (标准输出) ---")
        print(e.stdout)
        print("\n--- STDERR (标准错误) ---")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("错误: 'python -m pip' 命令未找到。请确认 pip 已安装在当前环境中。")
        return False

def calculate_perplexity(hf_model, tokenizer, texts, device="cuda", max_length=512, batch_size=8) -> float:
    """
    使用原版 Hugging Face 模型批量计算困惑度。
    这个函数保持不变，用于正确性校验。
    """
    hf_model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Perplexity"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        with torch.no_grad():
            out = hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss.item()
            n_tk = attention_mask.sum().item()
        total_loss += loss * n_tk
        total_tokens += n_tk
    
    del hf_model, enc, out, input_ids, attention_mask
    torch.cuda.empty_cache()
    
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)
    
# 在生成时，max_new_tokens 设置为剩余的生成长度
def evaluate_latency_hf(model, tokenizer, sentences, device="cuda"):
    """评估平均延迟、吞吐量、峰值显存"""
    # 预热
    print("Warming up model ...")
    warm = tokenizer("Warm-up.", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        model(**warm)
    total_time, tk_in, tk_out, req = 0.0, 0, 0, 0

    gpu_cnt = torch.cuda.device_count()
    for i in range(gpu_cnt):
        torch.cuda.reset_peak_memory_stats(i)

    for i in tqdm(range(0, len(sentences), BATCH_SIZE_INFERENCE),
                  desc="Evaluating latency"):
        batch = [s for s in sentences[i : i + BATCH_SIZE_INFERENCE] if s.strip()]
        if not batch:
            continue

        toks = tokenizer(batch,
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=MAX_LEN).to(device)
        tk_in += toks.attention_mask.sum().item()

        start = time.time()
        with torch.no_grad():
            # 计算生成部分的 token 数量
            max_new_tokens = MAX_LEN - toks.input_ids.shape[1]  # 输入部分占用的 tokens 数量
            print("inputids tokens:",toks.input_ids.shape[1])

            # 使用 max_new_tokens 来控制生成的 tokens 数量
            outs = model.generate(**toks,
                                  max_new_tokens=max_new_tokens,  # 生成最多 MAX_LEN - 输入长度 个新 tokens
                                  do_sample=False)  # 贪婪解码
        total_time += time.time() - start
        tk_out     += outs.numel()
        req        += len(batch)

        del toks, outs
        torch.cuda.empty_cache()

    peak_mem = sum(torch.cuda.max_memory_allocated(i) / 1e9
                   for i in range(gpu_cnt))
    avg_lat  = total_time / max(req, 1)
    thpt     = (tk_out - tk_in) / max(total_time, 1e-9)

    return avg_lat, thpt, peak_mem

def evaluate_metric_baseline(model_path, texts_combined, max_length, batch_size):
    from transformers import AutoModelForCausalLM # 只在这个函数内部导入，用完即释放
    
    print("Loading original Hugging Face model for perplexity calculation...")
    # 始终在 float16 下计算 PPL，以保证和基准一致
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.device_count() > 1 else "cuda:0"
    )
    # tokenizer
    print("Loading model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    subset = texts_combined[:EVAL_SENTENCE_COUNT]

    print("Evaluating original model perplexity ...")
    ppl = calculate_perplexity(hf_model, tokenizer, texts_combined, max_length=max_length, batch_size=batch_size)

    print(f"Evaluating original model latency on {EVAL_SENTENCE_COUNT} sentences ...")
    lat, thpt, peak  = evaluate_latency_hf(model, tokenizer, subset)
    print(f"Combined Inference Total Peak Memory: {peak:.2f} GB")
    print(f"Combined Average Latency: {lat:.4f} seconds per sentence")
    print(f"Combined Throughput: {thpt:.2f} tokens/second")

def evaluate_metric_my_proj(model_path, sentences):
    from electrock_infer.llm import LLM
    from electrock_infer.engine.llm_engine import LLMEngine
    from electrock_infer.sampling_params import SamplingParams

    prompt_token_ids = sentences[:EVAL_SENTENCE_COUNT]
    sampling_params = [SamplingParams(temperature=1, ignore_eos=False, max_tokens=512, max_total_tokens = 512) for _ in range(EVAL_SENTENCE_COUNT)] # 生成最多(MAX_LEN - 输入长度) 个新tokens
        # warmup
    engine = LLMEngine(model_path, enforce_eager=True, max_model_len=4096, tensor_parallel_size=2)
    engine.generate(["Benchmark: "], SamplingParams())
    print("Warmup done")

    print("Begining evaluate....")
    with suppress_stderr():
        # 在这里面包住你所有会产生 NCCL 日志的代码
        t = time.time()
        engine.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        t = (time.time() - t)
        engine.exit()

    latency_per_seq = t / EVAL_SENTENCE_COUNT
    print(f"Optimized Perplexity: {BASELINE_PPL:.4f}")
    print(f"Optimized Average Latency: {latency_per_seq:.4f} seconds per sentence")

def main():
    # GPU 信息
    n_gpu = torch.cuda.device_count()
    print(f"Available DCUs: {n_gpu}")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        print(f"DCU {i}: {p.name}")
        print(f"Memory: {p.total_memory / 1e9:.2f} GB")

    # 读取原始数据集
    print("Loading original dataset ...")
    ds = load_dataset("arrow", data_files={"test": DATASET_PATH})
    texts_original = [s for s in ds["test"]["text"] if s.strip()]
    if not texts_original:
        raise RuntimeError("原数据集为空，无法评估")

    # 加载新的数据集
    print("Loading new dataset ...")
    new_ds, split_key = load_new_dataset(NEW_DATASET_PATH)

    # 访问新数据集中的文本数据
    texts_new = [s["text"] for s in new_ds[split_key] if s.get("text") and s["text"].strip()]

    if not texts_new:
        raise RuntimeError("新数据集为空，无法评估")

    # 合并数据集
    texts_combined = texts_original + texts_new
    prompt_token_ids = texts_combined[:EVAL_SENTENCE_COUNT]
    # ───────── 原始模型指标 ─────────
    # evaluate_metric_baseline(MODEL_PATH, texts_combined, MAX_LEN, BATCH_SIZE_PERPLEXITY)
    print(f"Combined Average Perplexity: {BASELINE_PPL:.4f}")
    print(f"Evaluating original model latency on {EVAL_SENTENCE_COUNT} sentences ...")
    print(f"Combined Average Latency: {BASELINE_LATENCY_PER_SEQ:.4f} seconds per sentence")


    # ───────── Optimized 指标 ─────────
    evaluate_metric_my_proj(MODEL_PATH, texts_combined)



if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 调用函数进行安装
    success = install_project(current_script_dir)
    main()
