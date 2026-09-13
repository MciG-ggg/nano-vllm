import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    trust_remote_code: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        # Multimodal shells (SmolVLM = Idefics3Config) hold the text
        # backbone config under ``.text_config``; ``hidden_size`` etc.
        # live there, not the top level. Materialise them as plain
        # attributes on ``hf_config`` so downstream model_runner code
        # (allocate_kv_cache, capture_cudagraph) can read them without
        # walking into the nested structure. text_config takes
        # precedence so shell vs nested views agree after __post_init__.
        if hasattr(self.hf_config, "text_config"):
            tc = self.hf_config.text_config
            for k in (
                "hidden_size",
                "vocab_size",
                "max_position_embeddings",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "rope_theta",
                "rope_scaling",
                "head_dim",
                "tie_word_embeddings",
                "dtype",
            ):
                v = getattr(tc, k, None)
                if v is not None:
                    setattr(self.hf_config, k, v)
        if hasattr(self.hf_config, "max_position_embeddings"):
            self.max_model_len = min(
                self.max_model_len, int(self.hf_config.max_position_embeddings)
            )
