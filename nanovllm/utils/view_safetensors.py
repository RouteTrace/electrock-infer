import os
import glob
from safetensors import safe_open
import torch
from collections import Counter
import logging
import re

# --- 配置 ---
# --- 请修改为您的 safetensors 文件所在的目录 ---
TARGET_DIRECTORY = os.path.expanduser("~/zhushengguang/models/Mixtral-8x7B-Instruct-v0.1/")
# 设置日志级别，避免不必要的警告
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_safetensors_file(filepath, output_file=None):
    """
    分析单个 safetensors 文件并打印详细信息，并可写入文件。
    """
    try:
        # 1. 打印文件基本信息
        filename = os.path.basename(filepath)
        lines = []
        lines.append("=" * 80)
        lines.append(f"正在分析文件: {filename}")
        
        file_size_bytes = os.path.getsize(filepath)
        file_size_mb = file_size_bytes / (1024 * 1024)
        lines.append(f"文件大小: {file_size_mb:.2f} MB")
        
        # 2. 初始化统计变量
        total_params = 0
        dtype_counts = Counter()
        tensor_infos = []

        # 3. 使用 safe_open 安全地打开文件进行分析
        # device="cpu" 确保张量加载到CPU，避免占用显存
        with safe_open(filepath, framework="pt", device="cpu") as f:
            # 打印元数据
            metadata = f.metadata()
            if metadata:
                lines.append("\n--- 文件元数据 ---")
                for key, value in metadata.items():
                    lines.append(f"  {key}: {value}")
            # 遍历所有张量以收集信息
            for key in f.keys():
                tensor = f.get_tensor(key)
                num_params = tensor.numel() # 获取张量中的元素总数
                total_params += num_params
                dtype_counts[tensor.dtype] += 1
                tensor_infos.append({
                    "key": key,
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                    "num_params": num_params
                })
        # 4. 打印文件摘要信息
        lines.append("\n--- 文件摘要 ---")
        lines.append(f"总张量数量: {len(tensor_infos)}")
        lines.append(f"总参数量: {total_params:,}")
        lines.append(f"数据类型分布:")
        for dtype, count in dtype_counts.items():
            lines.append(f"  - {str(dtype):>15}: {count} 个张量")
        # 5. 打印详细的张量列表
        lines.append("\n--- 张量详细列表 ---")
        tensor_infos.sort(key=lambda x: x['key'])
        for i, info in enumerate(tensor_infos):
            name = info['key']
            shape = str(info['shape'])
            dtype = str(info['dtype'])
            params = f"{info['num_params']:,}"
            lines.append(f"  {i+1: >3}. {name:<60} | 形状: {shape:<25} | 类型: {dtype:<15} | 参数量: {params}")
    except Exception as e:
        logging.error(f"处理文件 '{filepath}' 时发生错误: {e}")
        lines.append(f"处理文件 '{filepath}' 时发生错误: {e}")
    finally:
        lines.append("=" * 80 + "\n")
        # 打印到屏幕
        for l in lines:
            print(l)
        # 写入文件
        if output_file is not None:
            with open(output_file, "a", encoding="utf-8") as f:
                for l in lines:
                    f.write(l + "\n")


if __name__ == "__main__":
    output_file = "output.txt"
    if not os.path.isdir(TARGET_DIRECTORY):
        msg = f"错误: 目录 '{TARGET_DIRECTORY}' 不存在或不是一个有效的目录。"
        print(msg)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print("请修改脚本中的 TARGET_DIRECTORY 变量。")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("请修改脚本中的 TARGET_DIRECTORY 变量。\n")
    else:
        search_pattern = os.path.join(TARGET_DIRECTORY, "**", "*.safetensors")
        safetensors_files = glob.glob(search_pattern, recursive=True)
        if not safetensors_files:
            msg = f"在目录 '{TARGET_DIRECTORY}' (包括子目录) 中没有找到 .safetensors 文件。"
            print(msg)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        else:
            # 按文件名中的数字顺序排序
            def extract_number(filename):
                match = re.search(r'(\d+)', os.path.basename(filename))
                return int(match.group(1)) if match else -1
            safetensors_files.sort(key=extract_number)
            msg = f"在 '{TARGET_DIRECTORY}' 中找到了 {len(safetensors_files)} 个 safetensors 文件。开始分析...\n"
            print(msg)
            all_lines = [msg]
            for filepath in safetensors_files:
                # 收集每个文件的分析信息
                file_lines = []
                def analyze_and_collect(filepath):
                    try:
                        filename = os.path.basename(filepath)
                        lines = []
                        lines.append("=" * 80)
                        lines.append(f"正在分析文件: {filename}")
                        file_size_bytes = os.path.getsize(filepath)
                        file_size_mb = file_size_bytes / (1024 * 1024)
                        lines.append(f"文件大小: {file_size_mb:.2f} MB")
                        total_params = 0
                        dtype_counts = Counter()
                        tensor_infos = []
                        with safe_open(filepath, framework="pt", device="cpu") as f:
                            metadata = f.metadata()
                            if metadata:
                                lines.append("\n--- 文件元数据 ---")
                                for key, value in metadata.items():
                                    lines.append(f"  {key}: {value}")
                            for key in f.keys():
                                tensor = f.get_tensor(key)
                                num_params = tensor.numel()
                                total_params += num_params
                                dtype_counts[tensor.dtype] += 1
                                tensor_infos.append({
                                    "key": key,
                                    "shape": tuple(tensor.shape),
                                    "dtype": tensor.dtype,
                                    "num_params": num_params
                                })
                        lines.append("\n--- 文件摘要 ---")
                        lines.append(f"总张量数量: {len(tensor_infos)}")
                        lines.append(f"总参数量: {total_params:,}")
                        lines.append(f"数据类型分布:")
                        for dtype, count in dtype_counts.items():
                            lines.append(f"  - {str(dtype):>15}: {count} 个张量")
                        lines.append("\n--- 张量详细列表 ---")
                        tensor_infos.sort(key=lambda x: x['key'])
                        for i, info in enumerate(tensor_infos):
                            name = info['key']
                            shape = str(info['shape'])
                            dtype = str(info['dtype'])
                            params = f"{info['num_params']:,}"
                            lines.append(f"  {i+1: >3}. {name:<60} | 形状: {shape:<25} | 类型: {dtype:<15} | 参数量: {params}")
                    except Exception as e:
                        logging.error(f"处理文件 '{filepath}' 时发生错误: {e}")
                        lines.append(f"处理文件 '{filepath}' 时发生错误: {e}")
                    finally:
                        lines.append("=" * 80 + "\n")
                        for l in lines:
                            print(l)
                        return lines
                file_lines = analyze_and_collect(filepath)
                all_lines.extend(file_lines)
            # 最后一次性写入txt，保证顺序
            with open(output_file, "w", encoding="utf-8") as f:
                for l in all_lines:
                    f.write(l + "\n")