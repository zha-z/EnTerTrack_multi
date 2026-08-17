import torch
import torch.nn as nn


class FCVCTeacher(nn.Module):
    def __init__(self, token_dim=192, embed_dim=128, num_queries=8):
        super().__init__()
        self.mid_proj = nn.Linear(token_dim, embed_dim)
        self.high_proj = nn.Linear(token_dim, embed_dim)
        self.slot_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.writer = nn.Linear(embed_dim, token_dim, bias=False)

    def forward(self, mid_features, high_features, gt_roi):
        if gt_roi is None:
            raise ValueError("GT ROI is allowed only in teacher training path")
        tokens = []
        for mid, high in zip(mid_features, high_features):
            mask = gt_roi.to(device=mid.device, dtype=mid.dtype).reshape(mid.shape[0], -1, 1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            tokens.append(self.mid_proj((mid * mask).sum(dim=1) / denom.squeeze(1)))
            tokens.append(self.high_proj((high * mask).sum(dim=1) / denom.squeeze(1)))
        sequence = torch.stack(tokens, dim=1)
        slots = self.slot_embed.expand(sequence.shape[0], -1, -1)
        fused = self.fusion(torch.cat((slots, sequence), dim=1))
        return fused[:, :slots.shape[1]]

    def tracking_residual(self, receiver_search, teacher_slots):
        pooled = self.writer(teacher_slots.mean(dim=1)).unsqueeze(1)
        return receiver_search + pooled
