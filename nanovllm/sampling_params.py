from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # Nucleus / top-k hints routed to ``Sampler`` per-call by the
    # stage that owns the loop (thinker wraps ``runner.sampler`` and
    # threads them through; talker passes its own ``top_k``).
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k >= 0, "top_k must be non-negative"
