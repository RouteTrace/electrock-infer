import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import functools
from typing import Optional, Tuple

#
# ============================================================================
#
#               第一部分: PyTorch 参考模型 (您提供的代码)
#
# ============================================================================
#
# ACT2FN 是一个激活函数的映射，我们这里为了简单直接定义
# 在实际 Transformers 库中，这是一个字典
class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)

ACT2FN = {"silu": SiLU()}


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, hidden_act):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """纯 PyTorch 实现的 MoE 模块，作为“黄金标准”"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_size']
        self.ffn_dim = config['intermediate_size']
        self.num_experts = config['num_experts']
        self.top_k = config['top_k']

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MixtralBlockSparseTop2MLP(self.hidden_dim, self.ffn_dim, config['hidden_act']) 
            for _ in range(self.num_experts)
        ])
        self.jitter_noise = 0 # 推理和验证时禁用

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        print(f"M : hidden_state[0:10]:{hidden_states_reshaped[0,:10]}; hidden_states_reshaped {hidden_states_reshaped.shape}")
        router_logits = self.gate(hidden_states_reshaped)
        print(f"M : router_logits[0:]:{router_logits[0,:]}")
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        print(f"M : routing_weights[0:]:{routing_weights[0,:]} ; selected_experts[0:] :{selected_experts[0:]}")

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states_reshaped[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx].unsqueeze(1)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
            
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

#
# ============================================================================
#
#               第二部分: 您的 Fused MoE 实现 (您提供的代码)
#
# ============================================================================
#
import electrock_infer
from nanovllm.layers.moe.moe_align_block_size import moe_align_block_size
# from nanovllm.test.kernel_test.example.verify.fused_moe import fused_experts
from nanovllm.layers.moe.fused_moe import fused_experts
# @triton.jit def fused_moe_kernel(...): ...
# def fused_topk(...): ...
# def fused_experts(...): ...
# def _invoke_fused_moe_kernel(...): ...
# def _get_moe_config(...): ...
# (以上函数请确保已在此文件或导入的文件中定义)


#
# ============================================================================
#
#               第三部分: 主验证逻辑
#
# ============================================================================
#

def run_verification():
    """执行端到端的正确性验证"""
    
    # --- 1. 配置测试参数 ---
    BATCH_SIZE = 1
    NUM_TOKENS = 1 # 验证时通常用 1
    HIDDEN_SIZE = 4096
    INTERMEDIATE_SIZE = 14336 // 2 # 修正：应该是 14336/2
    NUM_EXPERTS = 8
    TOP_K = 2
    DTYPE = torch.bfloat16
    DEVICE = 'cuda'

    config = {
        'hidden_size': HIDDEN_SIZE,
        'intermediate_size': INTERMEDIATE_SIZE,
        'num_experts': NUM_EXPERTS,
        'top_k': TOP_K,
        'hidden_act': 'silu'
    }
    
    print("="*80)
    print("Starting End-to-End MoE Correctness Verification")
    print(f"Config:  {BATCH_SIZE=} {NUM_TOKENS=}, {HIDDEN_SIZE=}, {INTERMEDIATE_SIZE=}, {NUM_EXPERTS=}, {TOP_K=}, DType={DTYPE}")
    print("="*80)

    # --- 2. 创建参考模型并设置权重 ---
    torch.manual_seed(42) # 固定随机种子以保证每次权重一致
    reference_model = MixtralSparseMoeBlock(config).to(DEVICE, dtype=DTYPE).eval()

    # --- 3. 准备完全相同的输入 ---
    hidden_states = torch.randn((BATCH_SIZE, NUM_TOKENS, HIDDEN_SIZE), dtype=DTYPE, device=DEVICE)

    # --- 4. 运行参考模型，得到“黄金标准”结果 ---
    print("Running PyTorch reference implementation...")
    with torch.no_grad():
        expected_output, expected_logits = reference_model(hidden_states)
    print("Reference implementation finished.")
    expected_output = expected_output.reshape(-1, HIDDEN_SIZE)
    print(f"Reference reuslt shapre : {expected_output.shape}")
    # --- 5. 准备 Fused MoE 的输入，关键是统一权重 ---
    print("\nPreparing inputs for Fused MoE implementation...")
    # a. 门控网络的输出就是参考模型的 logits
    gating_output = expected_logits.view(-1, NUM_EXPERTS)

    # b. 堆叠专家权重
    # w1 和 w3 合并为 w1_stacked
    w1_stacked_list = [torch.cat([expert.w1.weight, expert.w3.weight], dim=0) for expert in reference_model.experts]
    w1_stacked = torch.stack(w1_stacked_list)
    print(f"w1_stacked shape {w1_stacked.shape}\n")
    # w2 堆叠为 w2_stacked
    w2_stacked_list = [expert.w2.weight for expert in reference_model.experts]
    w2_stacked = torch.stack(w2_stacked_list)
    print(f"w2_stacked shape {w2_stacked.shape}\n")
    print("Stacked weights prepared.")
    
    # --- 6. 运行您的 Fused MoE 实现 ---
    print("\nRunning your Fused MoE implementation...")
    with torch.no_grad():
        actual_output = fused_experts(
            hidden_states=hidden_states.view(-1, HIDDEN_SIZE),
            w1=w1_stacked,
            w2=w2_stacked,
            gating_output=gating_output,
            topk=TOP_K,
            renormalize=True, # 必须与参考模型逻辑一致
            inplace=True,
        )
    print("Your implementation finished.")
    print(f"Your finished reuslt shapre : {actual_output.shape}")
    # --- 7. 对比最终结果 ---
    print("\nComparing final outputs...")
    # 将输出 reshape 成相同形状以便比较
    actual_output_reshaped = actual_output.view(-1, HIDDEN_SIZE)
    
    # 使用 torch.allclose 进行浮点数比较
    is_correct = torch.allclose(expected_output, actual_output_reshaped, atol=1e-1, rtol=1e-2)

    if is_correct:
        print("\n✅✅✅ Verification PASSED! ✅✅✅")
        print("Your `fused_experts` implementation is numerically consistent with the reference PyTorch model.")
        print("\nExpected output (first token, first 5 values):")
        print(expected_output[0, :20])
        print("\nActual output (first token, first 5 values):")
        print(actual_output_reshaped[0, :20])
    else:
        print("\n❌❌❌ Verification FAILED! ❌❌❌")
        max_diff = torch.max(torch.abs(expected_output - actual_output_reshaped))
        print(f"Maximum absolute difference between outputs: {max_diff.item()}")
        
        # 打印一些值用于调试
        print("\nExpected output (first token, first 5 values):")
        print(expected_output[0, :20])
        print("\nActual output (first token, first 5 values):")
        print(actual_output_reshaped[0, :20])

    print("="*80)

def run_performance_analysis(num_tokens=1024, num_repeats=1, warmup_steps=0):
    """执行简化的性能分析，只关注耗时。"""
    
    # --- 1. 配置与环境准备 ---
    BATCH_SIZE = 64
    HIDDEN_SIZE = 4096
    INTERMEDIATE_SIZE = 14336 // 2
    NUM_EXPERTS = 8
    TOP_K = 2
    DTYPE = torch.bfloat16
    DEVICE = 'cuda'

    config = {
        'hidden_size': HIDDEN_SIZE,
        'intermediate_size': INTERMEDIATE_SIZE,
        'num_experts': NUM_EXPERTS,
        'top_k': TOP_K,
        'hidden_act': 'silu'
    }

    print("\n" + "="*80)
    print("开始性能分析 (仅计时)")
    print(f"配置: num_tokens={num_tokens}, 重复次数={num_repeats}, 预热次数={warmup_steps}")
    print("="*80)

    # --- 2. 准备模型与输入数据 ---
    torch.manual_seed(1000)
    reference_model = MixtralSparseMoeBlock(config).to(DEVICE, dtype=DTYPE).eval()
    hidden_states = torch.randn((BATCH_SIZE, num_tokens, HIDDEN_SIZE), dtype=DTYPE, device=DEVICE)
    
    with torch.no_grad():
        _, gating_logits = reference_model(hidden_states)
    gating_output = gating_logits.view(-1, NUM_EXPERTS)
    
    w1_stacked = torch.stack([torch.cat([expert.w1.weight, expert.w3.weight], dim=0) for expert in reference_model.experts])
    w2_stacked = torch.stack([expert.w2.weight for expert in reference_model.experts])
    hidden_states_flat = hidden_states.view(-1, HIDDEN_SIZE).clone()
    
    # --- 3. 初始化CUDA事件用于精确计时 ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # --- 4. 测试 PyTorch 参考模型 ---
    print("\n--- 测试 PyTorch 参考模型 ---")
    with torch.no_grad():
        for _ in range(warmup_steps): # 预热
            _ = reference_model(hidden_states)


        start_event.record() # 开始计时
        for _ in range(num_repeats):
            _ = reference_model(hidden_states)
        end_event.record() # 结束计时

    ref_avg_latency = start_event.elapsed_time(end_event) / num_repeats
    print(f"平均耗时: {ref_avg_latency:.4f} ms")

    # --- 5. 测试 Fused MoE 实现 ---
    print("\n--- 测试 Fused MoE 实现 ---")
    with torch.no_grad():
        for _ in range(warmup_steps): # 预热
            _ = fused_experts(hidden_states_flat.clone(), w1_stacked, w2_stacked, gating_output, TOP_K, inplace=False)

        torch.cuda.synchronize()
        start_event.record() # 开始计时
        for _ in range(num_repeats):
            _ = fused_experts(hidden_states_flat.clone(), w1_stacked, w2_stacked, gating_output, TOP_K, inplace=False)
        end_event.record() # 结束计时
        torch.cuda.synchronize()

    fused_avg_latency = start_event.elapsed_time(end_event) / num_repeats
    print(f"平均耗时: {fused_avg_latency:.4f} ms")

    # --- 6. 总结 ---
    print("\n" + "-"*40)
    print("性能总结")
    print("-"*40)
    speedup = ref_avg_latency / fused_avg_latency
    print(f"🚀 加速比: Fused MoE 实现速度是 PyTorch 参考的 {speedup:.2f} 倍。")
    print("="*80)

if __name__ == "__main__":
    # 确保所有需要的函数和类都已定义或导入
    run_verification()
    # run_performance_analysis()