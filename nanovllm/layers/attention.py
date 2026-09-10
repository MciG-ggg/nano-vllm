import torch
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    _HAS_FLASH_ATTN = False

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    d = tl.arange(0, D_PAD)
    mask = d < D
    key_offsets = idx * key_stride + d
    value_offsets = idx * value_stride + d
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + d
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def _next_pow2(n: int) -> int:
    """Smallest power of 2 ≥ n. 1 for n ≤ 1."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    D_PAD = _next_pow2(D)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D, D_PAD)


def _gather_paged_kv_decode(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged K/V for decode (one new token per sequence).

    Returns contiguous [B, H_kv, S_max, D] tensors suitable for torch SDPA.
    Positions beyond context_lens are zeroed so they don't leak.
    """
    B = block_tables.shape[0]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    S_max = int(context_lens.max().item())
    device = k_cache.device

    # Vectorised gather: token_idx[b, s] = block_tables[b, s//block_size] * block_size + s%block_size
    token_idx = torch.arange(S_max, device=device).view(1, S_max)             # [1, S_max]
    block_in_seq = (token_idx // block_size).clamp(max=block_tables.shape[1] - 1)
    slot_in_block = token_idx % block_size
    block_ids = block_tables[:, block_in_seq.squeeze(0)]                       # [B, S_max]
    flat_slots = (block_ids * block_size + slot_in_block).long()               # [B, S_max]
    valid = token_idx < context_lens.view(B, 1)                                # [B, S_max]

    kv_flat = k_cache.view(-1, num_kv_heads, head_dim)
    k = kv_flat[flat_slots].transpose(1, 2).contiguous()                         # [B, S_max, H, D] → [B, H, S_max, D]
    v_flat = v_cache.view(-1, num_kv_heads, head_dim)
    v = v_flat[flat_slots].transpose(1, 2).contiguous()
    k = k * valid.view(B, 1, S_max, 1)
    v = v * valid.view(B, 1, S_max, 1)
    return k, v


def _gather_paged_kv_prefill(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged K/V for prefix-cache prefill.

    Returns flat [total_k_tokens, H_kv, D] matching varlen layout.
    """
    B = block_tables.shape[0]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    device = k_cache.device
    cu_cpu = cu_seqlens_k.cpu().tolist()
    total = cu_cpu[-1]

    out_k = torch.empty((total, num_kv_heads, head_dim), dtype=k_cache.dtype, device=device)
    out_v = torch.empty_like(out_k)
    kv_k = k_cache.view(-1, num_kv_heads, head_dim)
    kv_v = v_cache.view(-1, num_kv_heads, head_dim)

    for b in range(B):
        s, e = cu_cpu[b], cu_cpu[b + 1]
        if s == e:
            continue
        s_len = e - s
        token_idx = torch.arange(s_len, device=device)
        block_in_seq = (token_idx // block_size).clamp(max=block_tables.shape[1] - 1)
        slot_in_block = token_idx % block_size
        block_ids = block_tables[b, block_in_seq]
        flat_slots = (block_ids * block_size + slot_in_block).long()
        out_k[s:e] = kv_k[flat_slots]
        out_v[s:e] = kv_v[flat_slots]
    return out_k, out_v


def _sdpa_varlen_prefill(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, scale: float,
    num_heads: int, num_kv_heads: int,
) -> torch.Tensor:
    """SDPA fallback for varlen prefill (no paged cache).

    q,k,v are flat [N, H, D] with cu_seqlens indicating per-sequence boundaries.
    Runs SDPA per sequence; concatenates outputs.
    """
    n_q, n_k, n_v = q.shape[0], k.shape[0], v.shape[0]
    assert n_k == n_v, f"k/v length mismatch: {n_k} vs {n_v}"
    # Clamp stale bucket edges when Q/K/V disagree in length.
    cu = [min(int(c), n_q, n_k) for c in cu_seqlens_q.cpu().tolist()]
    rep = num_heads // num_kv_heads
    outs = []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        if s >= e:
            continue
        qi = q[s:e].transpose(0, 1)                       # [H_q, S, D]
        ki = k[s:e].transpose(0, 1)                       # [H_kv, S, D]
        vi = v[s:e].transpose(0, 1)
        if rep != 1:
            ki = ki.repeat_interleave(rep, dim=0)
            vi = vi.repeat_interleave(rep, dim=0)
        oi = torch.nn.functional.scaled_dot_product_attention(
            qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
            is_causal=True, scale=scale)
        outs.append(oi.squeeze(0).transpose(0, 1).contiguous())   # [S, H_q, D]
    if not outs:
        return torch.empty((0, num_heads, q.shape[-1]), dtype=q.dtype, device=q.device)
    return torch.cat(outs, dim=0)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # Prefix cache path: K/V come from the paged cache.
            # Non-prefix path: K/V are the freshly-computed prefill K/V tensors.
            paged_prefill = context.block_tables is not None

            if (
                _HAS_FLASH_ATTN
                and q.is_cuda
                and q.is_contiguous()
                and k.is_cuda
                and k.is_contiguous()
                and v.is_cuda
                and v.is_contiguous()
                and q.dim() == 3
                and k.dim() == 3
                and v.dim() == 3
            ):
                if paged_prefill:
                    # flash_attn reads from cache directly via block_table.
                    args_kv = (k_cache, v_cache)
                else:
                    args_kv = (k, v)
                o = flash_attn_varlen_func(q, *args_kv,
                                           max_seqlen_q=context.max_seqlen_q,
                                           cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k,
                                           cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True,
                                           block_table=context.block_tables)
            else:
                # SDPA fallback path. Requires paged KV to be gathered into
                # contiguous tensors first. Block size is fixed by the engine
                # (see engine/config); default 256 matches nano-vllm's default.
                block_size = getattr(context, 'block_size', 256)
                if paged_prefill:
                    k_g, v_g = _gather_paged_kv_prefill(
                        k_cache, v_cache, context.block_tables,
                        context.cu_seqlens_k, block_size)
                else:
                    k_g, v_g = k, v
                o = _sdpa_varlen_prefill(
                    q, k_g, v_g, context.cu_seqlens_q,
                    self.scale, self.num_heads, self.num_kv_heads)
        else:    # decode (one new token per sequence)
            if _HAS_FLASH_ATTN:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens,
                                            block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            else:
                block_size = getattr(context, 'block_size', 256)
                k_cts, v_cts = _gather_paged_kv_decode(
                    k_cache, v_cache, context.block_tables,
                    context.context_lens, block_size)
                # q is [B, H_q, D]; reshape to [B, 1, H_q, D] for SDPA
                B = q.shape[0]
                q4 = q.view(B, self.num_heads, 1, self.head_dim)
                rep = self.num_heads // self.num_kv_heads
                if rep != 1:
                    k_cts = k_cts.repeat_interleave(rep, dim=1)
                    v_cts = v_cts.repeat_interleave(rep, dim=1)
                o4 = torch.nn.functional.scaled_dot_product_attention(
                    q4, k_cts, v_cts, is_causal=False, scale=self.scale)
                # o4 is [B, H_q, 1, D] → squeeze seq dim → [B, H_q, D]
                o = o4.squeeze(2)
        return o
