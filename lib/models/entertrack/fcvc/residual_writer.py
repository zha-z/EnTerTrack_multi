import torch
import torch.nn as nn


class ResidualWriter(nn.Module):
    def __init__(self, token_dim=192, embed_dim=128, num_heads=4,
                 residual_norm_bound=0.0):
        super().__init__()
        self.token_query = nn.Linear(token_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.residual_proj = nn.Linear(embed_dim, token_dim, bias=False)
        self.norm = nn.LayerNorm(token_dim)
        self.residual_norm_bound = float(residual_norm_bound)

    def forward(self, template_tokens, search_tokens, collaborative_queries,
                force_zero_residual=False):
        base_search = search_tokens.clone()
        if force_zero_residual:
            residual = torch.zeros_like(base_search)
            fused = base_search
            weights = torch.zeros(
                search_tokens.shape[0], search_tokens.shape[1],
                collaborative_queries.shape[1], device=search_tokens.device,
                dtype=search_tokens.dtype)
        else:
            token_q = self.token_query(search_tokens.float())
            context, weights = self.attn(token_q, collaborative_queries.float(),
                                         collaborative_queries.float(),
                                         need_weights=True,
                                         average_attn_weights=True)
            residual = self.residual_proj(context).to(dtype=search_tokens.dtype)
            if self.residual_norm_bound > 0.0:
                norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                scale = (self.residual_norm_bound / norm).clamp(max=1.0)
                residual = residual * scale
            fused = self.norm((base_search + residual).float()).to(dtype=search_tokens.dtype)
        return {
            "tokens": torch.cat((template_tokens.clone(), fused), dim=1),
            "search_tokens": fused,
            "template_tokens": template_tokens.clone(),
            "residual": residual,
            "assignment": weights,
        }
