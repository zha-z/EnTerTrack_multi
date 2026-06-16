import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention


def attention_entropy(attention_weights):
    eps = 1e-10
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-2)
    return entropy.mean(dim=1)


def compute_saliency_scores(attention):
    entropy = attention_entropy(attention)
    min_entropy = entropy.min(dim=1, keepdim=True).values  # [1, 1]
    max_entropy = entropy.max(dim=1, keepdim=True).values  # [1, 1]
    saliency_scores = (entropy - min_entropy) / (max_entropy - min_entropy + 1e-10)
    return saliency_scores

class DeltaEstimator(nn.Module):
    def __init__(self, dim, reduction=32, depth=2):
        super().__init__()
        hidden_dim = max(1, dim // reduction)

        layers = []
        layers.extend([
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        ])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ATP(nn.Module):
    def __init__(self, dim: int, layer_id):
        super().__init__()
        hidden_dim = 256
        self.layer_id = layer_id
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 16),
            nn.LayerNorm(hidden_dim // 16),
            nn.GELU(),
            nn.Linear(hidden_dim // 16, 1)
        )
        if layer_id == 0:
            self.delta_estimator = DeltaEstimator(dim)
        else:
            self.delta_estimator = None

    def forward(self, attn: torch.Tensor, tokens: torch.Tensor, lens_t: int,
                global_index: torch.Tensor, box_mask_z: torch.Tensor, training: bool = True, ce_keep_rate=1.0,
                temperature=20.0):

        if ce_keep_rate >= 0.9:
            return tokens, global_index, None, None, None, None

        tokens_t = tokens[:, :lens_t]
        tokens_s = tokens[:, lens_t:]

        lens_s = attn.shape[-1] - lens_t
        bs, hn, seq_len, _ = attn.shape
        if self.layer_id == 0:
            if ce_keep_rate == 0.7:
                attn_t = (attn[:, :, lens_t:, lens_t:])
            else:
                return tokens, global_index, None, None, None, None
        else:
            attn_t = attn[:, :, :lens_t, lens_t:]
            if box_mask_z is not None:
                box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
                attn_t = attn_t[box_mask_z]
                attn_t = attn_t.view(bs, hn, -1, lens_s)

        redundant_scores = 1 - compute_saliency_scores(attn_t)
        threshold = self.threshold_predictor(redundant_scores)
        threshold = torch.sigmoid(threshold)
        tokens_frozen = None

        if training:
            soft_mask = torch.sigmoid((redundant_scores - threshold) * temperature)
            mask = (soft_mask > 0.5).float()
            ste_mask = (mask - soft_mask).detach() + soft_mask

            full_mask = torch.cat([torch.ones_like(tokens_t[:, :, 0]), mask], dim=1)

            if self.delta_estimator is not None:
                delta_x = self.delta_estimator(tokens_s * (1 - ste_mask.unsqueeze(-1)))
                tokens_frozen = (tokens_s + delta_x) * (1 - ste_mask.unsqueeze(-1))


            tokens_s_new = tokens_s
            tokens_new = torch.cat([tokens_t, tokens_s_new], dim=1)
            return tokens_new, global_index, None, ste_mask, full_mask, tokens_frozen
        else:
            keep_mask = redundant_scores > threshold
            attentive_tokens = torch.masked_select(tokens_s, keep_mask.unsqueeze(-1)).view(1, -1, tokens.shape[-1])
            keep_index = torch.masked_select(global_index, keep_mask).view(1, -1)
            removed_tokens = torch.masked_select(tokens_s, ~keep_mask.unsqueeze(-1)).view(1, -1, tokens.shape[-1])
            removed_index = torch.masked_select(global_index, ~keep_mask).view(1, -1)

            if self.delta_estimator is not None and removed_tokens.shape[1] > 0:
                delta_x = self.delta_estimator(removed_tokens)
                tokens_frozen = removed_tokens + delta_x

            tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

            return tokens_new, keep_index, removed_index, None, None, tokens_frozen


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, layer_id=0.,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_atp = layer_id == 0
        self.layer_id = layer_id
        if self.use_atp:
            self.atp = ATP(dim, layer_id)
        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None,
                training: bool = True, keep_ratio_search=None, temperature=2.0, frozen_token=None):
        lens_t = global_index_template.shape[1]
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)

        token_frozen = frozen_token
        removed_index_search = None
        attn_mask = mask
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
        if self.use_atp:
            x, global_index_search, removed_index_search, atp_mask, attn_mask, token_frozen = self.atp(attn, x, lens_t,
                                                                                                       global_index_search,
                                                                                                       ce_template_mask,
                                                                                                       training,
                                                                                                       keep_ratio_search,
                                                                                                       temperature=temperature)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_atp:
            return x, global_index_search, removed_index_search, attn, atp_mask, attn_mask, token_frozen
        else:
            return x, global_index_search, removed_index_search, attn, None, attn_mask, token_frozen


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x