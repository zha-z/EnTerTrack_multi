import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.entertrack.vit_arp_three import vit_tiny_patch16_224_arp
from lib.models.entertrack.vit import vit_tiny_patch16_224,vit_tiny_patch16_224_half
from lib.utils.box_ops import box_xyxy_to_cxcywh


class EnTeRTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self,
            template: torch.Tensor,
            search: torch.Tensor,
            template2: torch.Tensor = None,
            template3: torch.Tensor = None,
            ce_template_mask=None,
            ce_keep_rate=None,
            temperature=2.0,
            return_last_attn=False,
            return_atp=True,
            training=False):
        """
        支持两种模式：

        1. 单模板单搜索：
            template + search

        2. 三模板单搜索：
            template + template2 + template3 + search

        注意：
        这里参考 OSTrackThree 的写法，把 template2/template3
        直接传入 backbone，而不是做 late fusion。
        """

        # ------------------------------------------------------------
        # 三模板模式
        # ------------------------------------------------------------
        if template2 is not None and template3 is not None:
            x, aux_dict = self.backbone(
                z=template,
                z2=template2,
                z3=template3,
                x=search,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
                return_atp=return_atp,
                temperature=temperature,
                training=training
            )

            # 标记当前用了 3 个模板，方便 FLOPs 统计
            aux_dict["num_templates_used"] = 3

        # ------------------------------------------------------------
        # 单模板模式，保持原逻辑
        # ------------------------------------------------------------
        else:
            x, aux_dict = self.backbone(
                z=template,
                x=search,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
                return_atp=return_atp,
                temperature=temperature,
                training=training
            )

            aux_dict["num_templates_used"] = 1

        # ------------------------------------------------------------
        # Forward head
        # ------------------------------------------------------------
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out["backbone_feat"] = feat_last

        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_entertrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    #if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224_arp':
        backbone = vit_tiny_patch16_224_arp(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        backbone = vit_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224_half':
        backbone = vit_tiny_patch16_224_half(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = EnTeRTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'EnTeRTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model