import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import torch.distributed
from nanovllm.config import Config
from transformers import AutoTokenizer
# Only-for test
from nanovllm.models.mixtral import MixtralForCausalLM
import torch.multiprocessing as mp
import tqdm

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, rank : int):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    experts_modules_mapping = getattr(model, "experts_modules_mapping", {})
    files = glob(os.path.join(path, "*.safetensors"))
    for file in tqdm.tqdm(files, desc=f"Rank:{rank} Loading safetensors files"):
        with safe_open(file, "pt", "cuda") as f:
            weight_names = list(f.keys())
            for weight_name in tqdm.tqdm(weight_names, desc=f"Rank:{rank} Loading weights from {os.path.basename(file)}", leave=False):
                is_loaded = False # ensure every weight just be  load once
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v) #将字符串中的"k"换成"v",并返回新字符串
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        is_loaded = True
                        break
                for w in experts_modules_mapping:
                    # load experts weight customized for tensor parallel and stacked w13/w2.
                    if w in weight_name and not is_loaded:
                        # eg. modules_list = \
                        # ['model', 'layers', '0', 'block_sparse_moe', 'experts', '0', 'w1', 'weight']
                        modules_list = weight_name.split('.')
                        expert_id = int(modules_list[-3])
                        v, shard_id = experts_modules_mapping[w]
                        param_name = ".".join(modules_list[:-3]) + "." + v # eg: model.layers.0.block_sparse_moe.experts.w13_weight
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param,"weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id, expert_id)
                        is_loaded = True
                        break
                if not is_loaded:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


class ModelLoader:
    def __init__(self, config : Config, rank, event) -> None:
        hf_config = config.hf_config
        self.event = event
        self.rank = rank
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        torch.distributed.init_process_group("nccl", "tcp://localhost:2333", rank=rank, world_size=config.tensor_parallel_size)
        self.model = MixtralForCausalLM(hf_config)
        print(f"Model(rank:{self.rank}) have loaded.")
        load_model(self.model, config.model, self.rank)

        self.loop()
    def loop(self):

        while(True):
            print("Weight have loaded, enter process loop..\n")
            print(f"current device:{torch.cuda.current_device()}. Rank: {self.rank}")


if __name__ == "__main__":

    path = os.path.expanduser("~/zhushengguang/models/Mixtral-8x7B-Instruct-v0.1/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = Config(model=path,
                    tensor_parallel_size=2,
                    enforce_eager=True)
    hf_config = config.hf_config

    ctx = mp.get_context("spawn")
    events = []
    ps = []
    for i in range(1, config.tensor_parallel_size):
        event = ctx.Event()
        process = ctx.Process(target=ModelLoader, args=(config, i, event))
        process.start()
        events.append(event)
        ps.append(process)
    
    modelloder = ModelLoader(config, rank=0, event=events)

