"""
Focused unit test: Sampler top-p nucleus filter.

Verifies the ``top_p`` path added for HF MiniMind parity math only —
no GPU, no dist, no model. Follows the repo's stub-everything pattern:

    KMP_DUPLICATE_LIB_OK=TRUE python tests/test_sampler_top_p.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# ── stub every heavy third-party dep before nanovllm is imported ──
_triton_mock = MagicMock()
_triton_mock.jit = lambda fn: fn
_triton_mock.language = MagicMock()

sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("triton", _triton_mock)
sys.modules.setdefault("triton.language", _triton_mock.language)
sys.modules.setdefault("flash_attn", MagicMock())
sys.modules.setdefault("safetensors", MagicMock())

# torch.compile triggers triton.backends → stub it before torch.compile is ever called
import torch as _torch

_torch.compile = lambda fn, **kw: fn  # @torch.compile becomes a no-op

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from nanovllm.layers.sampler import Sampler  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402


def _sample_ids(logits: torch.Tensor, top_p: float, trials: int = 400) -> set[int]:
    s = Sampler()
    temps = torch.full((logits.shape[0],), 1.0)
    out: set[int] = set()
    for _ in range(trials):
        out.update(s(logits.clone(), temps, top_p=top_p).tolist())
    return out


def test_top_p_keeps_only_nucleus():
    # Row: token 0 dominates (p≈0.90), token 1 is the tail filler.
    logits = torch.tensor([[10.0, 0.0, -100.0, -100.0]])
    seen = _sample_ids(logits, top_p=0.9)
    assert 0 in seen, f"top-1 must survive the filter, got {seen}"
    assert 1 not in seen, f"tail token leaked through top_p=0.9: {seen}"
    print(f"PASS  top_p=0.9 admits {sorted(seen)} only")


def test_top_p_keeps_top1_when_everything_is_cut():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    probs = logits.softmax(-1)
    # cumsum hits 0.25 / 0.5 / 0.75 / 1.0 → top_p=0.1 keeps only token 0.
    assert probs[0, 0].item() > 0.1
    seen = _sample_ids(logits, top_p=0.1)
    assert seen == {0}, f"expected only top-1, got {seen}"
    print("PASS  degenerate top_p=0.1 never collapses to empty")


def test_top_p_default_is_noop():
    params = SamplingParams()
    assert params.top_p == 1.0 and params.top_k == 0
    # top_p=1.0 must admit the whole tail eventually.
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    seen = _sample_ids(logits, top_p=1.0)
    assert len(seen) == 4, f"top_p=1.0 filtered the tail: {seen}"
    print("PASS  top_p=1.0 default passes everything")


def test_topk_topp_compose():
    # top_k=2 keeps {0, 1}; top_p=0.9 then drops the tail token 1.
    s = Sampler()
    logits = torch.tensor([[10.0, 0.0, -100.0, -100.0]])
    temps = torch.full((1,), 1.0)
    seen: set[int] = set()
    for _ in range(400):
        seen.update(s(logits.clone(), temps, top_k=2, top_p=0.9).tolist())
    assert seen == {0}, f"expected top_k∩top_p={{0}}, got {seen}"
    print("PASS  top_k=2 + top_p=0.9 compose to {0}")


def test_module_stubs_untouched():
    # Guard: this file must not have polluted sys.modules for real deps.
    for name in ("transformers", "triton", "flash_attn", "safetensors"):
        assert isinstance(sys.modules[name], (MagicMock, ModuleType)), name
    print("PASS  sys.modules stubs intact")


if __name__ == "__main__":
    test_top_p_keeps_only_nucleus()
    test_top_p_keeps_top1_when_everything_is_cut()
    test_top_p_default_is_noop()
    test_topk_topp_compose()
    test_module_stubs_untouched()
    print("all passed")
