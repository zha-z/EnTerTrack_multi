import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.entertrack.vit_arp import vit_tiny_patch16_224_arp
from lib.models.entertrack.vit import vit_tiny_patch16_224,vit_tiny_patch16_224_half
from lib.models.entertrack.pcum import build_pcum
from lib.models.entertrack.c3r import build_c3r
from lib.models.entertrack.plain_collaboration import build_plain_collaboration
from lib.models.entertrack.plain_collaboration_checkpoint import (
    load_plain_collaboration_initialization)
from lib.utils.box_ops import box_xyxy_to_cxcywh


CONTROLLED_B0_ROLE = "controlled_b0"
POSTHOC_B0_PCUM_ROLE = "posthoc_b0_pcum_frozen"
POSTHOC_J0_ADAPT_ROLE = "posthoc_j0_b0_partial_adapt"
POSTHOC_J1_PCUM_ADAPT_ROLE = "posthoc_j1_b0_pcum_partial_adapt"
CONTROLLED_B0_ALLOWED_SOURCE_KEYS = (".atp.",)


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    state_dict = checkpoint.get("net", None)
    if state_dict is None:
        state_dict = checkpoint.get("model", None)
    return state_dict


def load_controlled_b0_initialization(model, checkpoint_path):
    """Strictly load the common non-ATP weights from the EnTeR source checkpoint.

    B0 deliberately removes ATP. The source checkpoint therefore has a small,
    explicitly named ATP-only surplus. All B0 parameters must still be present
    with identical shapes, and no other source key is accepted.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Controlled B0 checkpoint has no state dict: {}".format(
            checkpoint_path))

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }

    model_state = model.state_dict()
    missing_keys = sorted(set(model_state) - set(state_dict))
    source_only_keys = sorted(set(state_dict) - set(model_state))
    disallowed_source_keys = [
        key for key in source_only_keys
        if not any(marker in key for marker in CONTROLLED_B0_ALLOWED_SOURCE_KEYS)
    ]
    shape_mismatches = sorted(
        (key, tuple(state_dict[key].shape), tuple(model_state[key].shape))
        for key in set(model_state).intersection(state_dict)
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    )

    if missing_keys or disallowed_source_keys or shape_mismatches:
        raise RuntimeError(
            "Controlled B0 initialization audit failed: missing={}, "
            "disallowed_source={}, shape_mismatches={}".format(
                missing_keys, disallowed_source_keys, shape_mismatches))

    filtered_state = {
        key: state_dict[key] for key in model_state
    }
    model.load_state_dict(filtered_state, strict=True)
    report = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "loaded_key_count": len(filtered_state),
        "excluded_source_keys": source_only_keys,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "strict_core_load": True,
    }
    model.initialization_audit = report
    return report


def load_posthoc_b0_pcum_initialization(model, checkpoint_path):
    """Load frozen B0 core weights while preserving freshly initialized PCUM.

    The B0 checkpoint must not contain any pcum.* tensors. All non-PCUM model
    tensors must be present and shape-compatible. The final load is strict so
    that the newly initialized PCUM tensors are explicit members of the full
    model state rather than silently accepted missing keys.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("B0+PCUM base checkpoint has no state dict: {}".format(
            checkpoint_path))

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }

    inherited_pcum_keys = sorted(key for key in state_dict if key.startswith("pcum."))
    if inherited_pcum_keys:
        raise RuntimeError(
            "B0+PCUM initialization refuses inherited PCUM parameters: {}".format(
                inherited_pcum_keys[:20]))

    model_state = model.state_dict()
    non_pcum_model_keys = sorted(
        key for key in model_state if not key.startswith("pcum."))
    missing_non_pcum = sorted(set(non_pcum_model_keys) - set(state_dict))
    source_only_keys = sorted(set(state_dict) - set(non_pcum_model_keys))
    shape_mismatches = sorted(
        (key, tuple(state_dict[key].shape), tuple(model_state[key].shape))
        for key in set(non_pcum_model_keys).intersection(state_dict)
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    )

    if missing_non_pcum or source_only_keys or shape_mismatches:
        raise RuntimeError(
            "B0+PCUM frozen-core initialization audit failed: missing_non_pcum={}, "
            "source_only={}, shape_mismatches={}".format(
                missing_non_pcum, source_only_keys, shape_mismatches))

    merged_state = dict(model_state)
    for key in non_pcum_model_keys:
        merged_state[key] = state_dict[key]
    model.load_state_dict(merged_state, strict=True)
    report = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "loaded_non_pcum_key_count": len(non_pcum_model_keys),
        "fresh_pcum_key_count": len([
            key for key in model_state if key.startswith("pcum.")
        ]),
        "inherited_a0_pcum_parameters": len(inherited_pcum_keys),
        "strict_full_load": True,
    }
    model.initialization_audit = report
    return report


def load_c3r_initialization(model, checkpoint_path):
    """Strictly load the frozen local checkpoint and keep explicit fresh C3R keys.

    Every non-C3R tensor must exist with the exact shape.  The merged full state
    is then loaded with ``strict=True``; no missing/unexpected key is tolerated.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("C3R base checkpoint has no state dict: {}".format(
            checkpoint_path))
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }
    inherited_c3r = sorted(key for key in state_dict if key.startswith("c3r."))
    if inherited_c3r:
        raise RuntimeError("C3R initialization refuses inherited C3R keys: {}".format(
            inherited_c3r[:20]))
    model_state = model.state_dict()
    local_keys = sorted(key for key in model_state if not key.startswith("c3r."))
    missing_local = sorted(set(local_keys) - set(state_dict))
    source_only = sorted(set(state_dict) - set(local_keys))
    shape_mismatches = sorted(
        (key, tuple(state_dict[key].shape), tuple(model_state[key].shape))
        for key in set(local_keys).intersection(state_dict)
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    )
    if missing_local or source_only or shape_mismatches:
        raise RuntimeError(
            "C3R frozen-local initialization failed: missing_local={}, "
            "source_only={}, shape_mismatches={}".format(
                missing_local, source_only, shape_mismatches))
    merged_state = dict(model_state)
    for key in local_keys:
        merged_state[key] = state_dict[key]
    model.load_state_dict(merged_state, strict=True)
    report = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "loaded_local_key_count": len(local_keys),
        "fresh_c3r_key_count": len([
            key for key in model_state if key.startswith("c3r.")]),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "strict_full_load": True,
    }
    model.initialization_audit = report
    return report


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
                 pcum=None, c3r=None, c3r_freeze_local=False,
                 plain_collaboration=None,
                 plain_collaboration_freeze_local=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.pcum = pcum
        self.c3r = c3r
        self.c3r_freeze_local = bool(c3r_freeze_local)
        self.plain_collaboration = plain_collaboration
        self.plain_collaboration_freeze_local = bool(
            plain_collaboration_freeze_local)
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
                remote_suppression_override=None,
                c3r_packets=None,
                c3r_context=None,
                collaboration_feature=None,
                plain_remote_tokens=None,
                plain_remote_valid=None,
                plain_remote_weights=None,
                ):
        if collaboration_feature is not None:
            if self.plain_collaboration is None:
                raise RuntimeError("Head-only collaboration requires the V1 adapter")
            if any((
                    self.pcum is not None,
                    self.c3r is not None,
                    self.search_prompt_gate is not None)):
                raise RuntimeError(
                    "Plain collaboration is exclusive with PCUM/C3R/search prompt")
            feat_last = collaboration_feature
            search_tokens = feat_last[:, -self.feat_len_s:]
            collaboration = self.plain_collaboration(
                local_tokens=search_tokens,
                remote_tokens=plain_remote_tokens,
                remote_valid=plain_remote_valid,
                remote_weights=plain_remote_weights,
            )
            fused_feature = torch.cat((
                feat_last[:, :-self.feat_len_s],
                collaboration["search_tokens"],
            ), dim=1)
            out = self.forward_head(fused_feature, None)
            out["plain_collaboration"] = collaboration
            out["local_search_tokens"] = search_tokens
            out["backbone_feat"] = feat_last
            return out

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
                                remote_states=remote_states,
                                remote_suppression_override=remote_suppression_override)

        # C3R is a post-local-head attachment.  Empty/missing packets take the
        # explicit bypass and therefore do not call the CENTER head again.
        if self.c3r is not None and c3r_packets and c3r_context is not None:
            if self.pcum is not None or self.search_prompt_gate is not None:
                raise RuntimeError("C3R requires legacy PCUM/search prompt to be disabled")
            search_tokens = feat_last[:, -self.feat_len_s:]
            collaboration = self.c3r.collaborate(
                local_tokens=search_tokens,
                local_response=out["score_map"],
                packets=c3r_packets,
                receiver_id=int(c3r_context["receiver_id"]),
                sequence_hash=int(c3r_context["sequence_hash"]),
                local_frame_id=int(c3r_context["frame_id"]),
                local_timestamp_ms=int(c3r_context["timestamp_ms"]),
                frame_interval_ms=int(c3r_context["frame_interval_ms"]),
                last_frame_by_sender=c3r_context.get("last_frame_by_sender", None),
            )
            if collaboration["used_remote"]:
                fused_feature = torch.cat((
                    feat_last[:, :-self.feat_len_s],
                    collaboration["search_tokens"],
                ), dim=1)
                collaborative_out = self.forward_head(fused_feature, None)
                collaborative_out["c3r"] = collaboration
                collaborative_out["local_output"] = out
                collaborative_out["local_search_tokens"] = search_tokens
                out = collaborative_out

        out.update(aux_dict)
        out['backbone_feat'] = feat_last
        return out

    def train(self, mode=True):
        super().train(mode)
        if self.c3r is not None and self.c3r_freeze_local:
            self.backbone.eval()
            self.box_head.eval()
            if self.pcum is not None:
                self.pcum.eval()
            if self.search_prompt_gate is not None:
                self.search_prompt_gate.eval()
            self.c3r.train(mode)
        if (self.plain_collaboration is not None
                and self.plain_collaboration_freeze_local):
            self.backbone.eval()
            self.box_head.eval()
            if self.pcum is not None:
                self.pcum.eval()
            if self.search_prompt_gate is not None:
                self.search_prompt_gate.eval()
            self.plain_collaboration.train(mode)
        return self

    def forward_head(self, cat_feature, gt_score_map=None, prompt_map=None, prompt_gate_input=None,
                     remote_prompts=None, remote_states=None,
                     remote_suppression_override=None):
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
            }, remote_prompts=remote_prompts, remote_states=remote_states,
                remote_suppression_override=remote_suppression_override)
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

    plain_enabled = bool(getattr(getattr(
        cfg.MODEL, "PLAIN_COLLABORATION", None), "ENABLED", False))
    b0_checkpoint = getattr(cfg, "B0_CHECKPOINT", "")
    if plain_enabled and b0_checkpoint:
        pretrain_file = b0_checkpoint
        repository_root = os.path.normpath(os.path.join(current_dir, "../../.."))
        pretrain_path = b0_checkpoint if os.path.isabs(b0_checkpoint) \
            else os.path.join(repository_root, b0_checkpoint)
    elif pretrain_file:
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
    c3r = None
    if getattr(getattr(cfg.MODEL, "C3R", None), "ENABLED", False):
        if pcum is not None or getattr(cfg.MODEL, "USE_SEARCH_PROMPT", False):
            raise RuntimeError("C3R configs must disable PCUM and search-prompt fusion")
        c3r = build_c3r(cfg, token_dim=hidden_dim)

    plain_collaboration = None
    if getattr(getattr(
            cfg.MODEL, "PLAIN_COLLABORATION", None), "ENABLED", False):
        if (pcum is not None or c3r is not None
                or getattr(cfg.MODEL, "USE_SEARCH_PROMPT", False)):
            raise RuntimeError(
                "Plain collaboration requires PCUM/C3R/search prompt off")
        if cfg.MODEL.BACKBONE.TYPE != "vit_tiny_patch16_224_half":
            raise RuntimeError(
                "Plain collaboration V1 requires the Plain ViT-Tiny backbone")
        plain_collaboration = build_plain_collaboration(
            cfg, token_dim=hidden_dim)

    model = EnTeRTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        use_search_prompt=getattr(cfg.MODEL, "USE_SEARCH_PROMPT", False),
        prompt_hidden_dim=getattr(cfg.MODEL, "PROMPT_HIDDEN_DIM", 32),
        prompt_init_scale=getattr(cfg.MODEL, "PROMPT_INIT_SCALE", 0.1),
        pcum=pcum,
        c3r=c3r,
        c3r_freeze_local=bool(getattr(
            getattr(cfg.TRAIN, "C3R", None), "FREEZE_LOCAL", False)),
        plain_collaboration=plain_collaboration,
        plain_collaboration_freeze_local=bool(getattr(
            getattr(cfg.TRAIN, "PLAIN_COLLABORATION", None),
            "FREEZE_LOCAL", False)),
    )

    if pretrain_file and training and pretrain_path and os.path.isfile(pretrain_path):
        if plain_collaboration is not None:
            report = load_plain_collaboration_initialization(
                model, pretrain_path)
            print('Load frozen B0 core for Plain Collaboration V1: ' + pretrain_path)
            print('strict full load: ', report["strict_full_load"])
            print('fresh adapter keys: ', report["fresh_adapter_key_count"])
            return model
        if c3r is not None:
            report = load_c3r_initialization(model, pretrain_path)
            print('Load C3R frozen local core from: ' + pretrain_path)
            print('strict full load: ', report["strict_full_load"])
            print('fresh c3r keys: ', report["fresh_c3r_key_count"])
            return model
        if getattr(cfg, "MODEL_ROLE", "") in (
                POSTHOC_B0_PCUM_ROLE, POSTHOC_J1_PCUM_ADAPT_ROLE):
            report = load_posthoc_b0_pcum_initialization(model, pretrain_path)
            print('Load posthoc B0+PCUM frozen core from: ' + pretrain_path)
            print('strict full load: ', report["strict_full_load"])
            print('fresh pcum keys: ', report["fresh_pcum_key_count"])
            print('inherited a0 pcum parameters: ', report["inherited_a0_pcum_parameters"])
            return model
        if getattr(cfg, "MODEL_ROLE", "") in (
                CONTROLLED_B0_ROLE, POSTHOC_J0_ADAPT_ROLE):
            report = load_controlled_b0_initialization(model, pretrain_path)
            print('Load controlled B0 initialization from: ' + pretrain_path)
            print('strict core load: ', report["strict_core_load"])
            print('excluded source keys: ', report["excluded_source_keys"])
            return model
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = _checkpoint_state_dict(checkpoint)

        if state_dict is not None:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print('Load pretrained tracking model from: ' + pretrain_path)
            print('missing keys: ', missing_keys)
            print('unexpected keys: ', unexpected_keys)

    return model
