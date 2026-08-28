"""Target-semantic prompt collaboration for the Plain ViT baseline.

The extractor is deterministic and parameter-free.  The adapter is a fresh,
standalone V2 module; it intentionally does not inherit from or alter
``PlainCollaborationV1``.
"""

import math

import torch
from torch import nn


class TargetPromptExtractor(nn.Module):
    """Gather fixed top-k sender-local search tokens using the local response."""

    def __init__(self, prompt_k=8, token_dim=192):
        super().__init__()
        if prompt_k <= 0:
            raise ValueError("prompt_k must be positive")
        self.prompt_k = int(prompt_k)
        self.token_dim = int(token_dim)

    def _validate(self, search_tokens, score_map):
        if search_tokens.dim() != 3:
            raise ValueError("search_tokens must have shape [B,L,C]")
        if score_map.dim() != 4 or score_map.shape[1] != 1:
            raise ValueError("score_map must have shape [B,1,H,W]")
        if search_tokens.shape[0] != score_map.shape[0]:
            raise ValueError("search token and score-map batch sizes differ")
        if search_tokens.shape[-1] != self.token_dim:
            raise ValueError("unexpected token dimension")
        score_count = score_map.shape[-2] * score_map.shape[-1]
        if search_tokens.shape[1] != score_count:
            raise ValueError("score map and search token counts differ")
        if self.prompt_k > score_count:
            raise ValueError("prompt_k exceeds the search token count")

    def extract_with_metadata(self, search_tokens, score_map):
        self._validate(search_tokens, score_map)
        flat_scores = score_map.flatten(1)
        finite_scores = torch.isfinite(flat_scores)
        safe_scores = torch.where(
            finite_scores, flat_scores,
            torch.full_like(flat_scores, -torch.inf))
        topk_scores, topk_indices = torch.topk(
            safe_scores, k=self.prompt_k, dim=1)
        gather_index = topk_indices.unsqueeze(-1).expand(
            -1, -1, search_tokens.shape[-1])
        prompt = torch.gather(search_tokens, dim=1, index=gather_index)
        valid = (
            finite_scores.sum(dim=1) >= self.prompt_k
        ) & torch.isfinite(prompt).flatten(1).all(dim=1)
        return {
            "prompt": prompt,
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "valid": valid,
        }

    def forward(self, search_tokens, score_map):
        return self.extract_with_metadata(search_tokens, score_map)["prompt"]


class TargetPromptCollaboration(nn.Module):
    """Cross-attend local search queries to compact prompts per sender."""

    def __init__(self, token_dim=192, num_heads=3, enabled=False,
                 residual_init_scale=0.01, residual_scale_max=0.25,
                 relative_norm_cap=0.25, aggregation="mean"):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if aggregation != "mean":
            raise ValueError("E3 only supports mean sender aggregation")
        if residual_scale_max <= 0 or relative_norm_cap <= 0:
            raise ValueError("residual bounds must be positive")
        if abs(residual_init_scale) >= residual_scale_max:
            raise ValueError("initial residual scale must be below its maximum")
        self.enabled = bool(enabled)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.aggregation = str(aggregation)
        self.residual_scale_max = float(residual_scale_max)
        self.relative_norm_cap = float(relative_norm_cap)
        self.local_norm = nn.LayerNorm(self.token_dim)
        self.remote_norm = nn.LayerNorm(self.token_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.token_dim, self.num_heads, batch_first=True)
        normalized_init = float(residual_init_scale) / self.residual_scale_max
        self.residual_logit = nn.Parameter(torch.tensor(
            math.atanh(normalized_init), dtype=torch.float32))

    @staticmethod
    def _bypass(local_tokens):
        zero = local_tokens.new_zeros(())
        return {
            "search_tokens": local_tokens,
            "used_remote": False,
            "valid_remote_count": local_tokens.new_zeros(
                local_tokens.shape[0], dtype=torch.long),
            "remote_weights": None,
            "residual_norm": zero,
            "relative_residual_norm": zero,
            "residual_scale": zero,
        }

    def _validate(self, local_tokens, remote_prompts, remote_valid):
        if local_tokens.dim() != 3:
            raise ValueError("local_tokens must have shape [B,L,C]")
        if remote_prompts.dim() != 4:
            raise ValueError("remote_prompts must have shape [B,R,K,C]")
        if remote_prompts.shape[0] != local_tokens.shape[0]:
            raise ValueError("local and remote batch sizes differ")
        if remote_prompts.shape[-1] != self.token_dim:
            raise ValueError("unexpected remote token dimension")
        if local_tokens.shape[-1] != self.token_dim:
            raise ValueError("unexpected local token dimension")
        if remote_prompts.shape[2] <= 0:
            raise ValueError("remote prompt K must be positive")
        if remote_valid is not None and tuple(remote_valid.shape) != tuple(
                remote_prompts.shape[:2]):
            raise ValueError("remote_valid must have shape [B,R]")

    def forward(self, local_tokens, remote_prompts=None, remote_valid=None):
        if not self.enabled or remote_prompts is None:
            return self._bypass(local_tokens)
        if isinstance(remote_prompts, (tuple, list)):
            if not remote_prompts:
                return self._bypass(local_tokens)
            remote_prompts = torch.stack(list(remote_prompts), dim=1)
        if remote_prompts.shape[1] == 0:
            return self._bypass(local_tokens)
        self._validate(local_tokens, remote_prompts, remote_valid)
        if not bool(torch.isfinite(local_tokens).all().item()):
            return self._bypass(local_tokens)

        finite = torch.isfinite(remote_prompts).flatten(2).all(dim=2)
        valid = finite if remote_valid is None else (
            finite & remote_valid.to(device=local_tokens.device,
                                     dtype=torch.bool))
        weights = valid.to(local_tokens.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(
            weight_sum > 0,
            weights / weight_sum.clamp_min(torch.finfo(weights.dtype).eps),
            torch.zeros_like(weights))

        query = self.local_norm(local_tokens)
        sender_deltas = []
        for sender_index in range(remote_prompts.shape[1]):
            sender = remote_prompts[:, sender_index].to(
                device=local_tokens.device, dtype=local_tokens.dtype)
            sender = torch.where(
                valid[:, sender_index, None, None], sender,
                torch.zeros_like(sender))
            sender = self.remote_norm(sender)
            delta, _ = self.cross_attention(
                query=query, key=sender, value=sender, need_weights=False)
            sender_deltas.append(delta)
        aggregate = (
            torch.stack(sender_deltas, dim=1)
            * weights[:, :, None, None]
        ).sum(dim=1)

        scale = self.residual_scale_max * torch.tanh(self.residual_logit)
        residual = aggregate * scale.to(dtype=aggregate.dtype)
        local_norm = local_tokens.flatten(1).norm(dim=1).clamp_min(1e-12)
        residual_norm = residual.flatten(1).norm(dim=1).clamp_min(1e-12)
        cap_factor = torch.minimum(
            torch.ones_like(residual_norm),
            self.relative_norm_cap * local_norm / residual_norm)
        residual = residual * cap_factor[:, None, None]
        residual_norm = residual.flatten(1).norm(dim=1)
        relative_norm = residual_norm / local_norm
        any_valid = valid.any(dim=1)
        fused = torch.where(
            any_valid[:, None, None], local_tokens + residual, local_tokens)
        return {
            "search_tokens": fused,
            "used_remote": bool(any_valid.any().item()),
            "valid_remote_count": valid.sum(dim=1),
            "remote_weights": weights,
            "residual_norm": residual_norm.mean(),
            "relative_residual_norm": relative_norm.mean(),
            "residual_scale": scale,
        }


def build_target_prompt_collaboration(cfg, token_dim):
    model_cfg = cfg.MODEL.TARGET_PROMPT_COLLABORATION
    return TargetPromptCollaboration(
        token_dim=token_dim,
        num_heads=int(model_cfg.NUM_HEADS),
        enabled=bool(model_cfg.ENABLED),
        residual_init_scale=float(model_cfg.RESIDUAL_INIT_SCALE),
        residual_scale_max=float(model_cfg.RESIDUAL_SCALE_MAX),
        relative_norm_cap=float(model_cfg.RELATIVE_NORM_CAP),
        aggregation=str(model_cfg.AGGREGATION).lower())


def build_target_prompt_extractor(cfg, token_dim):
    model_cfg = cfg.MODEL.TARGET_PROMPT_COLLABORATION
    return TargetPromptExtractor(
        prompt_k=int(model_cfg.PROMPT_K), token_dim=token_dim)
