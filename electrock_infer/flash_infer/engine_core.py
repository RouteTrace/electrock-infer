import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
from electrock_infer.config import Config
from electrock_infer.sampling_params import SamplingParams
from electrock_infer.engine.sequence import Sequence, SequenceStatus
from electrock_infer.engine.scheduler import Scheduler
from electrock_infer.flash_infer.model_executer import ModelExecuter
from electrock_infer.flash_infer.kvcache_manager import KVCacheManager


class EngineCore:

    def __init__(self, model, **kwargs):
        # 配置参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # 根据tensor_parallel_size启动子进程，range从1开始
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelExecuter, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)


        self.model_runner = ModelExecuter(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.config = config
        self.kvcache_manager = KVCacheManager(config.max_num_seqs, config.max_model_len)
        self.scheduled_seqs: list[Sequence] = []
        atexit.register(self.exit)

    def step(self, is_prefill: bool):
        # seqs, is_prefill = self.scheduler.schedule()
        seqs = self.scheduled_seqs
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        outputs = self.postprocess(seqs, token_ids)
        return outputs

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[str]:
        use_tqdm = use_tqdm or not (self.config.disable_tqdm)
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        #将batch加入到scheduler中， prompt可以是token_ids 也可以是str
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        is_prefill = True
        while not self.is_finished():
            output = self.step(is_prefill)
            is_prefill = False
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
            
            
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        finished_outputs = []

        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.config.eos) or seq.num_completion_tokens == seq.max_tokens or seq.max_total_tokens == len(seq):
                seq.status = SequenceStatus.FINISHED
                self.kvcache_manager.deallocate(seq)
                self.scheduled_seqs.remove(seq)
                finished_outputs.append((seq.seq_id, seq.completion_token_ids))

        return finished_outputs

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # BUG: 把大于max_model_len过滤掉之后，输出的顺序可能会对应不上prompts
        if len(prompt) >= self.config.max_model_len: 
            return
        seq = Sequence(prompt, sampling_params)
        self.kvcache_manager.allocate(seq)
        self.scheduled_seqs.append(seq)

    def is_finished(self):
        return not self.scheduled_seqs

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()