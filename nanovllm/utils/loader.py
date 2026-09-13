import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _resolve_packed_modules_mapping(
    model: nn.Module, full_name: str
) -> dict | None:
    """Pick the right packed_modules_mapping for ``full_name``.

    Multimodal shells (SmolVLM, MiniMax Omni, etc.) hold submodules
    with different conventions -- the text backbone fuses q/k/v into
    ``qkv_proj`` but the SigLIP vision tower does not. We resolve
    per-key by walking the module path until we find a submodule whose
    own ``packed_modules_mapping`` declares a match for ``full_name``.
    Top-level mapping is the fallback for standalone models.
    """
    parts = full_name.split(".")
    # Walk the module path piece by piece; for each prefix, check
    # whether that submodule declares a packed_modules_mapping.
    for end in range(len(parts), 0, -1):
        prefix_path = ".".join(parts[:end])
        try:
            sub = model.get_submodule(prefix_path)
        except AttributeError:
            continue
        sub_mapping = getattr(sub, "packed_modules_mapping", None)
        if sub_mapping is None:
            continue
        for k in sub_mapping:
            if k in full_name[len(prefix_path) + 1 :]:
                return sub_mapping
        # Found a submodule with a mapping but it doesn't match this
        # key; stop searching because nested mappings are not
        # expected to coexist (single convention per family).
        return None
    return getattr(model, "packed_modules_mapping", None) or None


def load_model(model: nn.Module, path: str, prefix: str = ""):
    """Load weights from safetensors into model.

    ``prefix`` is prepended to every weight name (matches vllm-omni's
    ``init_vllm_registered_model(prefix=...)``). Multimodal shells hold
    vision_tower + connector + language_model as siblings; the prefix
    lets one checkpoint's ``model.text_model.layers.0...`` land in
    ``self.language_model.layers.0...``. Empty prefix (no change) keeps
    minimind / qwen3 / standalone models working.

    packed_modules_mapping is resolved per-key from the closest
    submodule so a shell's text-backbone fused projections don't
    collide with the vision tower's per-head projections.
    """
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f:
                full_name = f"{prefix}{weight_name}" if prefix else weight_name
                mapping = _resolve_packed_modules_mapping(model, full_name)
                if mapping:
                    for k in mapping:
                        if k in full_name:
                            v, shard_id = mapping[k]
                            param_name = full_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = param.weight_loader
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        continue  # mapping exists but no key match — treat as plain
                    continue
                param = model.get_parameter(full_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))
