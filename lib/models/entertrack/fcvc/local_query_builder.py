import torch
import torch.nn as nn
import torch.nn.functional as F

from .sender_bundle import normalized_position_grid


class ReceiverLocalQueryBuilder(nn.Module):
    def __init__(self, token_dim=192, embed_dim=128, num_queries=8, grid_size=16):
        super().__init__()
        self.num_queries = num_queries
        self.grid_size = grid_size
        self.feature_proj = nn.Linear(token_dim * 3 + 5, embed_dim)

    def _anchors(self, response):
        b, _, h, w = response.shape
        pooled = F.max_pool2d(response, kernel_size=3, stride=1, padding=1)
        score = response.masked_fill(response != pooled, -float("inf")).reshape(b, -1)
        fallback = response.reshape(b, -1)
        top_score, top = torch.topk(score, k=min(self.num_queries, h * w), dim=1)
        if (top_score == -float("inf")).any():
            _, top = torch.topk(fallback, k=min(self.num_queries, h * w), dim=1)
        y = torch.div(top, w, rounding_mode="floor")
        x = top.remainder(w)
        ref = torch.stack(((x.to(response.dtype) + 0.5) / w,
                           (y.to(response.dtype) + 0.5) / h), dim=-1)
        return top, ref

    def forward(self, mid_features, high_features, response, confidence_uncertainty,
                prototype):
        b, n, c = high_features.shape
        top, ref = self._anchors(response)
        gather = top.unsqueeze(-1).expand(-1, -1, c)
        mid = mid_features.gather(1, gather)
        high = high_features.gather(1, gather)
        resp_flat = response.reshape(b, 1, n).transpose(1, 2)
        unc_flat = confidence_uncertainty.reshape(b, 2, n).transpose(1, 2)
        aux = torch.cat((resp_flat.gather(1, top.unsqueeze(-1)),
                         unc_flat.gather(1, top.unsqueeze(-1).expand(-1, -1, 2)),
                         ref), dim=-1).to(dtype=high.dtype)
        proto = prototype.unsqueeze(1).expand(-1, self.num_queries, -1).to(dtype=high.dtype)
        query = self.feature_proj(torch.cat((mid, high, proto, aux), dim=-1).float())
        mask = torch.ones(b, self.num_queries, dtype=torch.bool, device=high.device)
        provenance = {"kind": "prediction_only", "anchor_indices": top.detach(),
                      "reference_points": ref.detach()}
        return {
            "queries": query,
            "reference_points": ref,
            "query_masks": mask,
            "provenance": provenance,
        }
