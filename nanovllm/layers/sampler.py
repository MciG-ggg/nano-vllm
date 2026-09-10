import torch
from torch import nn


@torch.compile
def _gumbel_max_sample(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-max tail: argmax of (logits + Gumbel(0,1) noise).

    Kept as a free function so the hot path stays compileable while
    optional kwargs (top_k / repetition_penalty) are handled outside.
    """
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


class Sampler(nn.Module):
    """Sample one token per row of ``logits``.

    Default path (``top_k=0``, ``top_p=1.0``, ``repetition_penalty=1.0``)
    is identical to the pre-extension behaviour: temperature scaling +
    Gumbel-max.

    Optional kwargs:
      - ``top_k > 0``: keep only the top-k logits before sampling.
      - ``top_p < 1.0``: nucleus filter — drop positions whose running
        softmax cumulative sum (in descending-logit order) exceeds
        ``top_p``; the top-1 position is always kept so the filter can
        never collapse to empty. Matches the ``transformers`` / vLLM
        convention used by HF ``MiniMindOmni.stream_generate``.
      - ``repetition_penalty != 1.0``: divide any history token's logit by
        ``repetition_penalty`` (HF transformers convention; <1 amplifies,
        >1 penalises).

    ``top_k`` and ``top_p`` compose: the smaller candidate set wins.
    """

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        history: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        logits = logits.float()
        if repetition_penalty != 1.0 and history is not None:
            uniq = torch.unique(history.to(logits.device))
            logits[..., uniq] /= repetition_penalty
        logits = logits.div_(temperatures.unsqueeze(dim=1))
        if top_k and top_k < logits.shape[-1]:
            top_v, top_i = logits.topk(top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, top_i, top_v)
            logits = masked
        if top_p < 1.0:
            sorted_l, sorted_i = logits.sort(dim=-1, descending=True)
            sorted_probs = sorted_l.softmax(dim=-1)
            cumsum_probs = sorted_probs.cumsum(dim=-1)
            # Drop position i if the cumsum strictly before i already
            # exceeds ``top_p``; keep the top-1 always so the filter
            # never collapses to empty. Mirrors the vLLM/transformers
            # recipe used by the HF MiniMind reference.
            remove = cumsum_probs > top_p
            remove[..., 0] = False
            remove[..., 1:] = remove[..., :-1].clone()
            logits = logits.scatter(
                -1, sorted_i, sorted_l.masked_fill(remove, float("-inf"))
            )
        return _gumbel_max_sample(logits)