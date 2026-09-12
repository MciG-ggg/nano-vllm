import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, prefix: str = ""):
    """Load weights from safetensors into model.

    ``prefix`` is prepended to every weight name (matches vllm-omni's
    ``init_vllm_registered_model(prefix=...)``). Multimodal shells hold
    vision_tower + connector + language_model as siblings; the prefix
    lets one checkpoint's ``model.text_model.layers.0...`` land in
    ``self.language_model.layers.0...``. Empty prefix (no change) keeps
    minimind / qwen3 / standalone models working.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                full_name = f"{prefix}{weight_name}" if prefix else weight_name
                for k in packed_modules_mapping:
                    if k in full_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = full_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(full_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
