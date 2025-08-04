import subprocess
import sys
import os

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

# --- 使用示例 ---
if __name__ == "__main__":
    

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 调用函数进行安装
    success = install_project(current_script_dir)

    if success:
        print("\n项目已成功安装，现在可以在代码中 import 它了。")