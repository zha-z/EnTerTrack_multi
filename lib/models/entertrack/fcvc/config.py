from dataclasses import dataclass


@dataclass(frozen=True)
class FCVCConfig:
    enabled: bool = False
    token_dim: int = 192
    embed_dim: int = 128
    template_tokens: int = 64
    search_tokens: int = 256
    grid_size: int = 16
    num_queries: int = 8
    num_senders: int = 2
    num_heads: int = 4
    samples_per_sender: int = 4
    mid_tap_after_block: int = 3
    high_tap_after_block: int = 6
    replay_start_block: int = 4
    replay_end_block: int = 6
    residual_norm_bound: float = 0.0
    null_bias: float = 2.0


def default_fcvc_config() -> FCVCConfig:
    return FCVCConfig()
