import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks_arp import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        #self.dist_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc

        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i, layer_id=i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def _get_global_indices(self, B, lens, device):
        return torch.arange(lens, device=device).unsqueeze(0).repeat(B, 1)

    def _recover_frozen_tokens(self, x, frozen_token, atp_mask, removed_index, global_index_s, lens_t, training):
        if frozen_token is None:
            return x, None

        B, _, C = x.shape

        if training:
            x_update = atp_mask.unsqueeze(-1) * x + frozen_token
            return x_update, None
        else:
            if removed_index is not None:
                x_combined = torch.cat([x, frozen_token], dim=1)
                all_indices = torch.cat([global_index_s, removed_index], dim=1)
                total_len = x_combined.shape[1]
                x_recovered = torch.zeros(B, total_len, C, device=x.device, dtype=x.dtype)
                x_recovered = x_recovered.scatter_(
                    dim=1,
                    index=all_indices.unsqueeze(-1).expand(-1, -1, C).long(),
                    src=x_combined
                )
                return x_recovered, None
            return x, global_index_s
    def forward_features(self, z, x, z2=None, z3=None,
                        mask_z=None, mask_x=None,
                        ce_template_mask=None, ce_keep_rate=None,
                        return_last_attn=False,
                        return_atp=False,
                        training: bool = True,
                        temperature=2.0):
        """
        支持两种输入格式：

        1. 单模板单搜索：
            z, x

        2. 三模板单搜索：
            z, z2, z3, x

        注意：
        - 保留 EnTeRTrack / ARP 的 ATP 逻辑；
        - 不使用 CEThreeMDOTBlock，仍然使用当前 attn_blocks_arp.CEBlock；
        - 因为 CEBlock 通过 global_index_t 的长度判断 template/search 分界，
        所以三模板时 global_index_t 必须是 3 * lens_z。
        """
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # ------------------------------------------------------------
        # 1. Patch embedding
        # ------------------------------------------------------------
        x = self.patch_embed(x)
        z = self.patch_embed(z)

        template_tokens = [z]
        if z2 is not None:
            z2 = self.patch_embed(z2)
            template_tokens.append(z2)

        if z3 is not None:
            z3 = self.patch_embed(z3)
            template_tokens.append(z3)

        num_templates = len(template_tokens)

        # ------------------------------------------------------------
        # 2. Attention mask handling
        #    这里仍然保留原逻辑。
        #    一般训练中 mask_z/mask_x 为 None。
        # ------------------------------------------------------------
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(
                mask_z[None].float(),
                scale_factor=1. / self.patch_size
            ).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(
                mask_x[None].float(),
                scale_factor=1. / self.patch_size
            ).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            # 如果是三模板，mask_z 也要扩展成三份
            if num_templates > 1:
                mask_z = mask_z.repeat(1, num_templates, 1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        # ------------------------------------------------------------
        # 3. CLS token
        # ------------------------------------------------------------
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # ------------------------------------------------------------
        # 4. Position embedding
        #    三个模板共享 self.pos_embed_z。
        # ------------------------------------------------------------
        template_tokens = [
            t + self.pos_embed_z for t in template_tokens
        ]

        x = x + self.pos_embed_x

        if self.add_sep_seg:
            template_tokens = [
                t + self.template_segment_pos_embed for t in template_tokens
            ]
            x = x + self.search_segment_pos_embed

        # ------------------------------------------------------------
        # 5. 拼接三模板 token
        #    z_all = [z1 | z2 | z3]
        # ------------------------------------------------------------
        z_all = template_tokens[0]
        for t in template_tokens[1:]:
            z_all = combine_tokens(z_all, t, mode=self.cat_mode)

        # x_all = [z1 | z2 | z3 | search]
        x = combine_tokens(z_all, x, mode=self.cat_mode)

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        # ------------------------------------------------------------
        # 6. 序列长度
        # ------------------------------------------------------------
        lens_z_single = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        lens_z_all = lens_z_single * num_templates

        global_index_t = torch.arange(
            lens_z_all,
            device=x.device
        ).unsqueeze(0).repeat(B, 1)

        global_index_s = torch.arange(
            lens_x,
            device=x.device
        ).unsqueeze(0).repeat(B, 1)

        # ------------------------------------------------------------
        # 7. CE template mask
        #
        # 如果外部只传了第一个模板的 mask: [B, lens_z_single]，
        # 这里自动 repeat 成 [B, 3*lens_z_single]。
        #
        # 更严格的做法是在 actor 里把三个模板 mask cat 后传进来。
        # ------------------------------------------------------------
        if ce_template_mask is not None and num_templates > 1:
            if ce_template_mask.shape[1] == lens_z_single:
                ce_template_mask = ce_template_mask.repeat(1, num_templates)

        # ------------------------------------------------------------
        # 8. Transformer blocks with ARP / ATP
        # ------------------------------------------------------------
        removed_indexes_s = []
        atp_masks = []
        attn_mask = None
        tokens = []
        frozen_token = None
        attn = None

        for i, blk in enumerate(self.blocks):
            x, global_index_s, removed_index_s, attn, atp_mask, attn_mask, frozen_token = \
                blk(
                    x,
                    global_index_t,
                    global_index_s,
                    attn_mask,
                    ce_template_mask,
                    training,
                    ce_keep_rate,
                    temperature=temperature,
                    frozen_token=frozen_token
                )

            if atp_mask is not None:
                atp_masks.append(atp_mask)
                attn_mask = ~(attn_mask.bool())

            if self.ce_loc is not None and i in self.ce_loc and removed_index_s is not None:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)

        lens_x_new = global_index_s.shape[1]
        lens_z_new_all = global_index_t.shape[1]

        # ------------------------------------------------------------
        # 9. 分离 template tokens 和 search tokens
        # ------------------------------------------------------------
        z_all = x[:, :lens_z_new_all]
        x = x[:, lens_z_new_all:]

        # ------------------------------------------------------------
        # 10. 恢复 ATP / CE 被冻结或移除的 search tokens
        # ------------------------------------------------------------
        if frozen_token is not None:
            curr_atp_mask = atp_masks[-1] if atp_masks else None
            curr_removed_idx = removed_indexes_s[-1] if removed_indexes_s else None

            x, _ = self._recover_frozen_tokens(
                x,
                frozen_token,
                curr_atp_mask,
                curr_removed_idx,
                global_index_s,
                lens_z_new_all,
                training
            )

        elif removed_indexes_s and removed_indexes_s[-1] is not None and lens_x_new != lens_x:
            removed_indexes_cat = removed_indexes_s[-1]

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros(
                [B, pruned_lens_x, x.shape[2]],
                device=x.device,
                dtype=x.dtype
            )

            x = torch.cat([x, pad_x], dim=1)

            index_all = torch.cat(
                [global_index_s, removed_indexes_cat],
                dim=1
            )

            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=x
            )

        # ------------------------------------------------------------
        # 11. 恢复 search token 顺序
        # ------------------------------------------------------------
        x = recover_tokens(
            x,
            lens_z_new_all,
            lens_x,
            mode=self.cat_mode
        )

        # ------------------------------------------------------------
        # 12. 重新拼接 [templates | search]
        # ------------------------------------------------------------
        x = torch.cat([z_all, x], dim=1)

        if return_atp:
            aux_dict = {
                "attn": attn,
                "tokens": tokens,
                "removed_indexes_s": removed_indexes_s,
                "atp_masks": atp_masks,
                "atp_layers": torch.tensor([0], device=x.device),
                "num_templates_used": num_templates,
            }
        else:
            aux_dict = {
                "attn": attn,
                "removed_indexes_s": removed_indexes_s,
                "num_templates_used": num_templates,
            }

        return x, aux_dict


    def forward(self, z, x, z2=None, z3=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,
                return_atp=False,
                training: bool = True,
                temperature=2.0):
        """
        支持：
        - 单模板：forward(z, x)
        - 三模板：forward(z, x, z2=z2, z3=z3)
        """
        x, aux_dict = self.forward_features(
            z=z,
            z2=z2,
            z3=z3,
            x=x,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            training=training,
            return_atp=return_atp,
            temperature=temperature,
            return_last_attn=return_last_attn
        )

        return x, aux_dict

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model

def vit_tiny_patch16_224_arp(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=6, num_heads=3, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

