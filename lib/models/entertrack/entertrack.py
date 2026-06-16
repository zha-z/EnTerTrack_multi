import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.entertrack.vit_arp import vit_tiny_patch16_224_arp
from lib.models.entertrack.vit import vit_tiny_patch16_224,vit_tiny_patch16_224_half
from lib.models.entertrack.pcum import build_pcum
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SearchRegionPromptGate(nn.Module):
    """
    Lightweight score-map prompt gate.

    Communication payload is expected to be low-dimensional reliability/scale
    statistics. The spatial prior itself is supplied as a small score-map-sized
    Gaussian map in local search coordinates.
    """

    def __init__(self, input_dim=6, hidden_dim=32, init_scale=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.prompt_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, score_map, prompt_map, gate_input=None):
        if prompt_map is None:
            return score_map, None

        if prompt_map.dim() == 3:
            prompt_map = prompt_map.unsqueeze(1)

        prompt_map = prompt_map.to(device=score_map.device, dtype=score_map.dtype)
        if prompt_map.shape[-2:] != score_map.shape[-2:]:
            prompt_map = torch.nn.functional.interpolate(
                prompt_map,
                size=score_map.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        if gate_input is None:
            gate = torch.ones(score_map.shape[0], 1, 1, 1, device=score_map.device, dtype=score_map.dtype)
        else:
            gate_input = gate_input.to(device=score_map.device, dtype=score_map.dtype)
            gate = torch.sigmoid(self.mlp(gate_input)).view(-1, 1, 1, 1)

        score_map = score_map + gate * self.prompt_scale * prompt_map
        return score_map, gate


class EnTeRTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 use_search_prompt=False, prompt_hidden_dim=32, prompt_init_scale=0.1,
                 pcum=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.pcum = pcum
        self.use_search_prompt = use_search_prompt
        self.search_prompt_gate = None
        if use_search_prompt:
            self.search_prompt_gate = SearchRegionPromptGate(
                input_dim=6,
                hidden_dim=prompt_hidden_dim,
                init_scale=prompt_init_scale
            )

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                temperature = 2.0,
                return_last_attn=False,
                return_atp = True,
                training = False,
                prompt_map=None,
                prompt_gate_input=None,
                remote_prompts=None,
                remote_states=None,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,return_atp = return_atp, temperature=temperature, training=training)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None, prompt_map=prompt_map,
                                prompt_gate_input=prompt_gate_input,
                                remote_prompts=remote_prompts,
                                remote_states=remote_states)

        out.update(aux_dict)
        out['backbone_feat'] = feat_last
        return out

    def forward_head(self, cat_feature, gt_score_map=None, prompt_map=None, prompt_gate_input=None,
                     remote_prompts=None, remote_states=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        pcum_out = None
        if self.pcum is not None:
            template_tokens = cat_feature[:, :-self.feat_len_s]
            pcum_out = self.pcum({
                "search": enc_opt,
                "template": template_tokens,
            }, remote_prompts=remote_prompts, remote_states=remote_states)
            enc_opt = pcum_out["search_tokens"]
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
            if pcum_out is not None:
                out['pcum'] = pcum_out
                out['local_prompt'] = pcum_out.get('local_prompt', None)
                out['aligned_prompt'] = pcum_out.get('aligned_prompt', None)
                out['remote_prompt'] = remote_prompts
            if self.search_prompt_gate is not None and prompt_map is not None:
                score_map, gate = self.search_prompt_gate(score_map, prompt_map, prompt_gate_input)
                out['score_map'] = score_map
                out['prompt_gate'] = gate
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            prompt_gate = None
            if self.search_prompt_gate is not None and prompt_map is not None:
                score_map_ctr, prompt_gate = self.search_prompt_gate(
                    score_map_ctr, prompt_map, prompt_gate_input
                )
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            if pcum_out is not None:
                out['pcum'] = pcum_out
                out['local_prompt'] = pcum_out.get('local_prompt', None)
                out['aligned_prompt'] = pcum_out.get('aligned_prompt', None)
                out['remote_prompt'] = remote_prompts
            if prompt_gate is not None:
                out['prompt_gate'] = prompt_gate
            return out
        else:
            raise NotImplementedError


def build_entertrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrain_file = getattr(cfg.MODEL, "PRETRAIN_FILE", "")
    pretrain_path = ""

    if pretrain_file:
        pretrain_path = pretrain_file if os.path.isabs(pretrain_file) \
            else os.path.join(pretrained_path, pretrain_file)

    # Full tracker checkpoints are loaded after model construction. Backbone-only
    # checkpoints can be passed directly to the ViT constructor.
    if pretrain_file and training and not pretrain_file.endswith(".pth.tar"):
        pretrained = pretrain_path
    else:
        pretrained = ""

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
    pcum = None
    if getattr(getattr(cfg.MODEL, "PCUM", None), "ENABLED", False):
        pcum = build_pcum(cfg, token_dim=hidden_dim)

    model = EnTeRTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        use_search_prompt=getattr(cfg.MODEL, "USE_SEARCH_PROMPT", False),
        prompt_hidden_dim=getattr(cfg.MODEL, "PROMPT_HIDDEN_DIM", 32),
        prompt_init_scale=getattr(cfg.MODEL, "PROMPT_INIT_SCALE", 0.1),
        pcum=pcum,
    )

    if pretrain_file and training and pretrain_path and os.path.isfile(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = None

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("net", None)
            if state_dict is None:
                state_dict = checkpoint.get("model", None)

        if state_dict is not None:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print('Load pretrained tracking model from: ' + pretrain_path)
            print('missing keys: ', missing_keys)
            print('unexpected keys: ', unexpected_keys)

    return model
