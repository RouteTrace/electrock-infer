import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
from electrock_infer.config import Config
from electrock_infer.sampling_params import SamplingParams
from electrock_infer.engine.sequence import Sequence
from electrock_infer.engine.scheduler import Scheduler
from electrock_infer.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 配置参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # config.__post_init__()
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # 根据tensor_parallel_size启动子进程，range从1开始
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        

        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        self.scheduler = Scheduler(config)

        # atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        # try:
        #     self.model_runner.call("exit")
        # except Exception as e:
        #     print(f"主进程: 发送 'exit' 指令时发生错误: {e}")
        # shutdown_timeout = 5  # 等待10秒
        # print(f"主进程: 将等待最多 {shutdown_timeout} 秒让子进程自行退出...")

        # for i, p in enumerate(self.ps):
        #     rank = i + 1  # 我们的子进程 rank 是从1开始的
        #     # 等待，但有超时限制
        #     p.join(timeout=shutdown_timeout)
        #     # 检查子进程是否仍然存活
        #     if p.is_alive():
        #         # 如果超时后进程仍然存活，就强制终止它
        #         print(f"警告: Rank {rank} (PID: {p.pid}) 未能在 {shutdown_timeout} 秒内正常退出，正在强制终止...")
        #         p.terminate()  # 发送 SIGTERM 终止信号
        #         p.join()       # 等待终止完成
        #         print(f"主进程: Rank {rank} 已被强制终止。")
        #     else:
        #         print(f"主进程: Rank {rank} (PID: {p.pid}) 已成功退出。")
        
        # print("主进程: 所有子进程均已清理完毕。")
        # # 清理主进程自己的资源
        # if self.model_runner.world_size > 1 and self.model_runner.rank == 0:
        #     try:
        #         self.model_runner.shm.close()
        #         self.model_runner.shm.unlink() # 主进程负责删除共享内存
        #     except Exception as e:
        #         print(f"主进程: 已清理共享内存")
        # # 在所有进程都结束后，再尝试销毁通信组
        # # 注意：如果子进程是被强制kill的，这步可能会失败，但这是清理的最后一步了
        # try:
        #     if dist.is_initialized():
        #          dist.destroy_process_group()
        # except Exception as e:
        #     print(f"主进程: 销毁通信组时出错: {e}")

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        #将batch加入到scheduler中， prompt可以是token_ids 也可以是str
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)


        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
