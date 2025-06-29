import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from nanovllm.config import Config
from transformers import AutoTokenizer, MixtralForCausalLM

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


if __name__ == "__main__":
    path = os.path.expanduser("~/zhushengguang/models/Mixtral-8x7B-Instruct-v0.1/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = MixtralForCausalLM.from_pretrained(path)

    config = Config(model=path)
    hf_config = config.hf_config
