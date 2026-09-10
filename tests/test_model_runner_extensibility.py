"""
Minimal test: ModelRunner accepts a custom model_class.

No GPU, no dist, no transformers/triton/flash_attn — just verifies dispatch logic.
Run:
    KMP_DUPLICATE_LIB_OK=TRUE python tests/test_model_runner_extensibility.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# ── stub every heavy third-party dep before nanovllm is imported ──
_tl = MagicMock()
_tl.constexpr = type
_triton_mock = MagicMock()
_triton_mock.jit = lambda fn: fn
_triton_mock.language = _tl

sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("triton", _triton_mock)
sys.modules.setdefault("triton.language", _tl)
sys.modules.setdefault("flash_attn", MagicMock())
sys.modules.setdefault("safetensors", MagicMock())

# torch.compile triggers triton.backends → stub it before torch.compile is ever called
import torch as _torch
_torch.compile = lambda fn, **kw: fn  # @torch.compile becomes a no-op

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch
import torch
import torch.distributed as _real_dist
from torch import nn

import nanovllm.engine.model_runner as _mr
import nanovllm.models.qwen3 as _qwen3_mod


class FakeModel(nn.Module):
    """Satisfies the fork contract: forward(input_ids, positions) -> hidden_states."""

    def __init__(self, hf_config):
        super().__init__()
        self.linear = nn.Linear(hf_config.hidden_size, hf_config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.zeros(input_ids.size(0), self.linear.in_features, device=input_ids.device))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden_states.size(0), 100, device=hidden_states.device)

    packed_modules_mapping = {}


def _make_config():
    hf_config = SimpleNamespace(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, dtype=torch.float32,
    )
    return SimpleNamespace(
        hf_config=hf_config, model="fake-model", enforce_eager=True,
        tensor_parallel_size=1, max_num_batched_tokens=128, max_model_len=128,
        max_num_seqs=4, gpu_memory_utilization=0.9, kvcache_block_size=16,
    )


_dist_mock = MagicMock()
_dist_mock.get_rank.return_value = 0
_dist_mock.get_world_size.return_value = 1

_qwen3_mock_instance = MagicMock(
    packed_modules_mapping={}, forward=MagicMock(), compute_logits=MagicMock(),
)
_qwen3_cls_mock = MagicMock(return_value=_qwen3_mock_instance)


def _patch_all(with_qwen3_mock=False):
    patches = [
        patch.object(_mr, "dist", _dist_mock),
        patch.object(_real_dist, "get_rank",       _dist_mock.get_rank),
        patch.object(_real_dist, "get_world_size", _dist_mock.get_world_size),
        patch.object(torch.cuda, "set_device", MagicMock()),
        patch.object(torch, "set_default_device", MagicMock()),
        patch.object(torch, "set_default_dtype", MagicMock()),
        patch.object(_mr, "load_model", MagicMock()),
        patch.object(_mr, "Sampler", MagicMock(return_value=MagicMock())),
        patch.object(_mr.ModelRunner, "warmup_model",      MagicMock()),
        patch.object(_mr.ModelRunner, "allocate_kv_cache", MagicMock()),
    ]
    # Patch the *source* of Qwen3ForCausalLM so the name resolved in __init__ sees the mock.
    if with_qwen3_mock:
        # Patch both the source module AND the name binding in model_runner
        patches.append(patch.object(_qwen3_mod, "Qwen3ForCausalLM", _qwen3_cls_mock))
        patches.append(patch.object(_mr, "Qwen3ForCausalLM", _qwen3_cls_mock))
    for p in patches:
        p.start()
    return patches


def _unpatch_all(patches):
    for p in reversed(patches):
        p.stop()


# ── tests ──

def test_custom_model_class():
    patches = _patch_all()
    try:
        runner = _mr.ModelRunner(_make_config(), rank=0, event=MagicMock(), model_class=FakeModel)
        assert isinstance(runner.model, FakeModel), f"got {type(runner.model).__name__}"
        print("PASS  custom model_class accepted")
    finally:
        _unpatch_all(patches)


def test_default_is_qwen3():
    patches = _patch_all(with_qwen3_mock=True)
    try:
        runner = _mr.ModelRunner(_make_config(), rank=0, event=MagicMock())
        _qwen3_cls_mock.assert_called_once()
        assert runner.model is _qwen3_mock_instance, "model should be the mocked Qwen3 instance"
        print("PASS  default falls back to Qwen3ForCausalLM")
    finally:
        _unpatch_all(patches)


if __name__ == "__main__":
    test_custom_model_class()
    test_default_is_qwen3()
    print("all passed")
