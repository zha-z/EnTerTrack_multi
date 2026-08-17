import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_sender_map(features, coords, grid_size):
    b, n, c = features.shape
    fmap = features.reshape(b, grid_size, grid_size, c).permute(0, 3, 1, 2)
    grid = coords.mul(2.0).sub(1.0).view(b, -1, 1, 2)
    sampled = F.grid_sample(fmap.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
    return sampled.squeeze(-1).transpose(1, 2)


class DeformableCrossViewBlock(nn.Module):
    def __init__(self, token_dim=192, embed_dim=128, num_heads=4,
                 num_senders=2, samples_per_sender=4, grid_size=16):
        super().__init__()
        self.num_senders = num_senders
        self.samples_per_sender = samples_per_sender
        self.grid_size = grid_size
        self.value_proj = nn.Linear(token_dim, embed_dim)
        self.offset = nn.Linear(embed_dim, num_senders * samples_per_sender * 2)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.update = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.null_value = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)

    def forward(self, queries, sender_features, sender_refs, global_context=None,
                force_null=False):
        b, q, d = queries.shape
        offsets = 0.25 * torch.tanh(self.offset(queries))
        offsets = offsets.view(b, q, self.num_senders, self.samples_per_sender, 2)
        samples = []
        coords_by_sender = []
        for slot in range(self.num_senders):
            coords = (sender_refs[:, :, slot].unsqueeze(2) + offsets[:, :, slot]).clamp(0.0, 1.0)
            coords_by_sender.append(coords.detach())
            projected = self.value_proj(sender_features[slot].float())
            sampled = sample_sender_map(projected, coords.reshape(b, q * self.samples_per_sender, 2),
                                        self.grid_size)
            samples.append(sampled.reshape(b, q, self.samples_per_sender, d))
        kv = torch.cat(samples, dim=2).reshape(b, q * self.num_senders * self.samples_per_sender, d)
        null = self.null_value.expand(b, 1, d)
        kv = torch.cat((kv, null), dim=1)
        if force_null:
            context = torch.zeros_like(queries)
            weights = torch.zeros(b, q, kv.shape[1], device=queries.device, dtype=queries.dtype)
            weights[:, :, -1] = 1.0
        else:
            seed = queries if global_context is None else queries + global_context
            context, weights = self.attn(seed, kv, kv, need_weights=True,
                                         average_attn_weights=True)
            non_null = 1.0 - weights[:, :, -1:].to(context.dtype)
            context = context * non_null
        out = queries + self.update(context)
        return {
            "queries": out,
            "context": context,
            "attention_weights": weights,
            "sample_coordinates": torch.stack(coords_by_sender, dim=2),
            "finite_coordinates": torch.isfinite(torch.stack(coords_by_sender)).all(),
        }
