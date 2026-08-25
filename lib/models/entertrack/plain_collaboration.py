"""Minimal final-search feature collaboration for the Plain ViT baseline.

The module is deliberately independent from PCUM, C3R, FCVC and token
pruning.  It consumes complete final search-token maps from synchronized
remote views and returns a bounded, local-first residual.  Disabled and
no-remote calls return the exact input tensor object.
"""

import math

import torch
from torch import nn


class PlainCollaborationV1(nn.Module):
    """Content-only cross-view attention at the search-token/head boundary."""

    def __init__(self, token_dim=192, num_heads=3, enabled=False,
                 residual_init_scale=0.01, residual_scale_max=0.25,
                 relative_norm_cap=0.25, aggregation="mean"):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if residual_scale_max <= 0:
            raise ValueError("residual_scale_max must be positive")
        if relative_norm_cap <= 0:
            raise ValueError("relative_norm_cap must be positive")
        if aggregation not in ("mean", "external"):
            raise ValueError("aggregation must be mean or external")
        if abs(residual_init_scale) >= residual_scale_max:
            raise ValueError("residual_init_scale must be smaller than its maximum")

        self.enabled = bool(enabled)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.residual_scale_max = float(residual_scale_max)
        self.relative_norm_cap = float(relative_norm_cap)
        self.aggregation = str(aggregation)

        self.local_norm = nn.LayerNorm(self.token_dim)
        self.remote_norm = nn.LayerNorm(self.token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        normalized_init = float(residual_init_scale) / self.residual_scale_max
        self.residual_logit = nn.Parameter(torch.tensor(
            math.atanh(normalized_init), dtype=torch.float32))

    @staticmethod
    def _as_remote_tensor(remote_tokens):
        if isinstance(remote_tokens, (list, tuple)):
            if not remote_tokens:
                return None
            return torch.stack(list(remote_tokens), dim=1)
        return remote_tokens

    @staticmethod
    def _bypass(local_tokens):
        zero = local_tokens.new_zeros(())
        return {
            "search_tokens": local_tokens,
            "used_remote": False,
            "valid_remote_count": local_tokens.new_zeros(
                local_tokens.shape[0]),
            "remote_weights": None,
            "residual_norm": zero,
            "relative_residual_norm": zero,
            "residual_scale": zero,
        }

    def _validate_shapes(self, local_tokens, remote_tokens, remote_valid,
                         remote_weights):
        if local_tokens.dim() != 3:
            raise ValueError("local_tokens must have shape [B,L,C]")
        if remote_tokens.dim() != 4:
            raise ValueError("remote_tokens must have shape [B,R,L,C]")
        if remote_tokens.shape[0] != local_tokens.shape[0]:
            raise ValueError("local and remote batch sizes differ")
        if remote_tokens.shape[2:] != local_tokens.shape[1:]:
            raise ValueError("local and remote token shapes differ")
        if local_tokens.shape[-1] != self.token_dim:
            raise ValueError("unexpected token dimension")
        expected_weight_shape = remote_tokens.shape[:2]
        if remote_valid is not None and tuple(remote_valid.shape) != tuple(
                expected_weight_shape):
            raise ValueError("remote_valid must have shape [B,R]")
        if remote_weights is not None and tuple(remote_weights.shape) != tuple(
                expected_weight_shape):
            raise ValueError("remote_weights must have shape [B,R]")

    def forward(self, local_tokens, remote_tokens=None, remote_valid=None,
                remote_weights=None):
        if not self.enabled or remote_tokens is None:
            return self._bypass(local_tokens)
        remote_tokens = self._as_remote_tensor(remote_tokens)
        if remote_tokens is None or remote_tokens.shape[1] == 0:
            return self._bypass(local_tokens)
        self._validate_shapes(
            local_tokens, remote_tokens, remote_valid, remote_weights)
        if not bool(torch.isfinite(local_tokens).all().item()):
            return self._bypass(local_tokens)

        batch_size, remote_count = remote_tokens.shape[:2]
        finite_remote = torch.isfinite(remote_tokens).flatten(2).all(dim=2)
        if remote_valid is None:
            valid = finite_remote
        else:
            valid = remote_valid.to(
                device=local_tokens.device, dtype=torch.bool) & finite_remote

        if remote_weights is not None and self.aggregation != "external":
            raise ValueError(
                "remote_weights require aggregation='external'")
        if self.aggregation == "external":
            if remote_weights is None:
                raise ValueError("external aggregation requires remote_weights")
            weights = remote_weights.to(
                device=local_tokens.device, dtype=local_tokens.dtype)
            weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
            weights = weights.clamp_min(0.0) * valid.to(weights.dtype)
        else:
            weights = valid.to(local_tokens.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(
            weight_sum > 0,
            weights / weight_sum.clamp_min(torch.finfo(weights.dtype).eps),
            torch.zeros_like(weights),
        )

        query = self.local_norm(local_tokens)
        sender_deltas = []
        for sender_index in range(remote_count):
            sender = remote_tokens[:, sender_index].to(
                device=local_tokens.device, dtype=local_tokens.dtype)
            sender = torch.where(
                valid[:, sender_index, None, None], sender,
                torch.zeros_like(sender))
            sender = self.remote_norm(sender)
            delta, _ = self.cross_attention(
                query=query, key=sender, value=sender,
                need_weights=False)
            sender_deltas.append(delta)
        stacked_delta = torch.stack(sender_deltas, dim=1)
        aggregate_delta = (
            stacked_delta * weights[:, :, None, None]).sum(dim=1)

        scale = self.residual_scale_max * torch.tanh(self.residual_logit)
        residual = aggregate_delta * scale.to(dtype=aggregate_delta.dtype)
        local_norm = local_tokens.flatten(1).norm(dim=1).clamp_min(1e-12)
        residual_norm = residual.flatten(1).norm(dim=1).clamp_min(1e-12)
        cap = self.relative_norm_cap * local_norm
        cap_factor = torch.minimum(
            torch.ones_like(residual_norm), cap / residual_norm)
        residual = residual * cap_factor[:, None, None]
        residual_norm = residual.flatten(1).norm(dim=1)
        relative_norm = residual_norm / local_norm

        fused_tokens = local_tokens + residual
        any_valid = valid.any(dim=1)
        fused_tokens = torch.where(
            any_valid[:, None, None], fused_tokens, local_tokens)
        return {
            "search_tokens": fused_tokens,
            "used_remote": bool(any_valid.any().item()),
            "valid_remote_count": valid.sum(dim=1),
            "remote_weights": weights,
            "residual_norm": residual_norm.mean(),
            "relative_residual_norm": relative_norm.mean(),
            "residual_scale": scale,
        }


def build_plain_collaboration(cfg, token_dim):
    model_cfg = cfg.MODEL.PLAIN_COLLABORATION
    return PlainCollaborationV1(
        token_dim=token_dim,
        num_heads=int(model_cfg.NUM_HEADS),
        enabled=bool(model_cfg.ENABLED),
        residual_init_scale=float(model_cfg.RESIDUAL_INIT_SCALE),
        residual_scale_max=float(model_cfg.RESIDUAL_SCALE_MAX),
        relative_norm_cap=float(model_cfg.RELATIVE_NORM_CAP),
        aggregation=str(model_cfg.AGGREGATION).lower(),
    )
