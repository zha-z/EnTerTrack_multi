import torch
import torch.nn as nn


class GlobalSemanticMatcher(nn.Module):
    def __init__(self, token_dim=192, embed_dim=128, num_heads=4,
                 num_senders=2, grid_size=16, null_bias=2.0):
        super().__init__()
        self.num_senders = num_senders
        self.grid_size = grid_size
        self.high_proj = nn.Linear(token_dim, embed_dim)
        self.proto_proj = nn.Linear(token_dim, embed_dim)
        self.pos_proj = nn.Linear(2, embed_dim)
        self.type_embed = nn.Embedding(3, embed_dim)
        self.sender_embed = nn.Embedding(16, embed_dim)
        self.view_embed = nn.Embedding(16, embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.null_bias = float(null_bias)

    def _bank(self, bundles):
        pieces = []
        metadata = []
        for slot, bundle in enumerate(bundles):
            b, n, _ = bundle.high_features.shape
            pos = bundle.position_grid.flatten(2).transpose(1, 2).float()
            view = bundle.view_id.long().clamp(0, 15)
            emb = self.high_proj(bundle.high_features.float())
            emb = emb + self.pos_proj(pos)
            emb = emb + self.type_embed.weight[0].view(1, 1, -1)
            emb = emb + self.sender_embed(view).unsqueeze(1)
            emb = emb + self.view_embed(view).unsqueeze(1)
            conf = bundle.confidence_uncertainty[:, 0].flatten(1).float()
            unc = bundle.confidence_uncertainty[:, 1].flatten(1).float()
            prior = (conf.clamp_min(1e-6).log() - unc).unsqueeze(-1)
            pieces.append(emb + 0.01 * prior)
            metadata.extend([(slot, "spatial", i) for i in range(n)])
            proto = self.proto_proj(bundle.target_prototype.float()).unsqueeze(1)
            proto = proto + self.type_embed.weight[1].view(1, 1, -1)
            proto = proto + self.sender_embed(view).unsqueeze(1)
            proto = proto + self.view_embed(view).unsqueeze(1)
            pieces.append(proto)
            metadata.append((slot, "prototype", -1))
        null = self.null_token.expand(pieces[0].shape[0], -1, -1)
        null = null + self.type_embed.weight[2].view(1, 1, -1)
        pieces.append(null)
        metadata.append((-1, "null", -1))
        return torch.cat(pieces, dim=1), metadata

    def forward(self, queries, bundles):
        bank, metadata = self._bank(bundles)
        matched, weights = self.attn(queries, bank, bank, need_weights=True,
                                     average_attn_weights=True)
        null_attention = weights[:, :, -1]
        b, q, _ = queries.shape
        refs = []
        contribs = []
        offset = 0
        for _slot in range(self.num_senders):
            spatial = weights[:, :, offset:offset + self.grid_size * self.grid_size]
            denom = spatial.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            coords = bundles[_slot].position_grid.flatten(2).transpose(1, 2).float()
            ref = torch.matmul(spatial / denom, coords)
            refs.append(ref.clamp(0.0, 1.0))
            contribs.append(spatial.sum(dim=-1))
            offset += self.grid_size * self.grid_size + 1
        sender_contribution = torch.stack(contribs, dim=-1)
        stats = {
            "attention_weights": weights.detach(),
            "null_attention_ratio": null_attention.detach(),
            "sender_contribution": sender_contribution.detach(),
            "metadata": metadata,
        }
        return {
            "matched": matched,
            "attention_weights": weights,
            "null_attention_ratio": null_attention,
            "sender_contribution": sender_contribution,
            "sender_reference_points": torch.stack(refs, dim=2),
            "diagnostics": stats,
        }
