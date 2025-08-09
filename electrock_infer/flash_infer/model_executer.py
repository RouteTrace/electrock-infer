import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from electrock_infer.config import Config
from electrock_infer.engine.sequence import Sequence
from electrock_infer.layers.sampler import Sampler
from electrock_infer.utils.context import set_context, get_context, reset_context
from electrock_infer.utils.loader import load_model
from electrock_infer.flash_infer.mixtral import MixtralForCausalLM


class ModelExecuter:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = True # config.enforce_eager 暂不支持cudagraph
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        dist.init_process_group("nccl", "tcp://localhost:2025", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = MixtralForCausalLM(self.hf_config) # 选择了支持NaiveKVcache的Attention版本
        load_model(self.model, config.model, self.rank, self.config)
        self.sampler = Sampler()
        # TODO:
        self.allocate_kv_cache(config)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="electrock", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="electrock")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear() 
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank #必须是主进程
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size
        self.shm.buf[0:4] = n.to_bytes(4, "little") #前4字节，表示data的字节长度
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_kv_cache(self, config: Config):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        num_kv_heads = self.hf_config.num_key_value_heads // self.world_size
        # BUG fix: head_dim may not exist in hf_config
        head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
        # 计算一个最长句子的kvcache所需要的字节数, 其中max_model_len是限制最大长度
        seq_bytes = 2 * self.hf_config.num_hidden_layers * config.max_model_len * num_kv_heads * head_dim * self.hf_config.torch_dtype.itemsize
        assert (total - used)>0 , "no memory leaved for allocating kv cache."
        #TODO: expend batch_size to max capacity on DCU.
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used) // seq_bytes
        print(f"最大支持{config.num_kvcache_blocks}个sequence.")
        assert config.num_kvcache_blocks > config.max_num_seqs, "config.max_num_seqs exceeds the upper limit that kvcache can allocate"
        self.kv_cache = torch.zeros(2, self.hf_config.num_hidden_layers, config.max_num_seqs, config.max_model_len, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        context_lens = []
        max_seqlen = 0
        cu_seqlens = [0]
        cache_batch_idx = []
        cache_batch_seqlen = []
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            context_lens.append(seqlen)
            cu_seqlens.append(cu_seqlens[-1] + seqlen) # 记录每个seq_q的长度，进行累加并append——>记录了每个seq的起始索引[ 0, q1, q2, ...]
            max_seqlen = max(seqlen, max_seqlen) # 获得该batch中最长seq长度
            cache_batch_idx.append(seq.cache_id)
            cache_batch_seqlen.append(seq.num_cached_tokens)
            seq.num_cached_tokens = seqlen
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cache_batch_idx = torch.tensor(cache_batch_idx, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cache_batch_seqlen = torch.tensor(cache_batch_seqlen, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, context_lens=context_lens, cache_batch_idx=cache_batch_idx, cache_batch_seqlen=cache_batch_seqlen)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        context_lens = []
        cache_batch_idx = []
        cache_batch_seqlen = []
        for seq in seqs:
            seqlen = len(seq)
            # 都是对单个token进行记录(tensor[1])
            input_ids.append(seq.last_token) 
            positions.append(len(seq))
            context_lens.append(len(seq))
            cache_batch_idx.append(seq.cache_id)
            #TODO record every seq's num_cached_token
            cache_batch_seqlen.append(seq.num_cached_tokens)
            seq.num_cached_tokens = seqlen
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cache_batch_idx = torch.tensor(cache_batch_idx, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cache_batch_seqlen = torch.tensor(cache_batch_seqlen, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, context_lens=context_lens, cache_batch_idx=cache_batch_idx, cache_batch_seqlen=cache_batch_seqlen)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None
    
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state
