"""Public fork API with no engine work at package import time.

Importing ``nanovllm`` must be cheap: constructing ``LLM`` starts CUDA,
loads Qwen3, and initializes distributed state. Keep those side effects behind
an explicit ``from nanovllm import LLM`` or ``LLM(...)`` call.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLM", "SamplingParams"]
