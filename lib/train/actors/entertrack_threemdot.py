from . import BaseActor

import torch
import torch.nn.functional as F

from lib.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
    giou_loss_details,
    l1_loss_details,
)
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate, adjust_temperature
from lib.models.entertrack.pcum import PromptConsistencyLoss, build_pseudo_remote_prompts
from lib.models.entertrack.c3r import CommunicationPerturbation, gate_ranking_loss
from .plain_collaboration import forward_plain_collaboration
from .target_prompt_collaboration import (
    forward_target_prompt_collaboration)


class EnTeRTrackActorThreeMDOT(BaseActor):
    """
    Actor for fine-tuning single-view EnTeRTrack on ThreeMDOT / MDOT.

    当前版本：
    1. 不使用多机 prompt；
    2. 不使用 teacher model；
    3. 不使用 distillation loss；
    4. 只取第 0 个视角作为单机主视角；
    5. 保留 EnTeRTrack / ARP 的 FLOPs 约束。
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)

        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

        self.flops_weight = 0.0
        self.F_target = 1.0
        self.prompt_consistency_loss = PromptConsistencyLoss(
            stop_gradient_teacher=self._get_cfg_value("TRAIN.PCUM.STOP_GRAD_TEACHER", True)
        )
        self.paired_supervision_enabled = bool(
            self._get_cfg_value("TRAIN.PCUM.PAIRED_SUPERVISION", False)
        )
        self._diagnostics_enabled = bool(
            self._get_cfg_value("TRAIN.PCUM.DIAGNOSTICS_ENABLED", False)
        )
        self._diagnostic_active = False
        self._diagnostic_stage = None
        self._diagnostic_values = {}
        self._diagnostic_local_residual = None
        self._fusion_hook_handle = None
        self._paired_cpu_rng_state = None
        self._paired_cuda_rng_state = None
        self._paired_cuda_device = None
        if self._diagnostics_enabled:
            self._register_fusion_diagnostic_hook()
        self._c3r_iteration = 0
        self.last_arp_sample_diagnostics = []
        c3r_train = getattr(getattr(cfg, "TRAIN", None), "C3R", None)
        self._c3r_perturbation = CommunicationPerturbation(
            enabled=bool(getattr(c3r_train, "PERTURBATIONS_ENABLED", False)),
            dropout_probability=float(getattr(c3r_train, "REMOTE_DROPOUT_PROB", 0.25)),
            delays=getattr(c3r_train, "DELAYS", [0, 1, 2, 4]),
            delay_probabilities=getattr(
                c3r_train, "DELAY_PROBS", [0.50, 0.20, 0.20, 0.10]),
            corruption_probability=float(getattr(
                c3r_train, "SEMANTIC_CORRUPTION_PROB", 0.15)),
            wrong_remote_probability=float(getattr(
                c3r_train, "WRONG_REMOTE_PROB", 0.20)),
            conflict_probability=float(getattr(
                c3r_train, "ONE_GOOD_ONE_BAD_PROB", 0.15)),
            seed=int(getattr(c3r_train, "SEED", 20260716)),
        )

    def __call__(self, data):
        """
        data:
            template_images: [V, B, 3, H, W] or [1, B, 3, H, W] or list
            search_images:   [V, B, 3, H, W] or [1, B, 3, H, W] or list
            template_anno:   [V, B, 4] or [1, B, 4] or list
            search_anno:     [V, B, 4] or [1, B, 4] or list
        """

        self._update_flops_schedule(data["epoch"])

        out_dict = self.forward_pass(self.net, data)

        loss, status = self.compute_losses(out_dict, data)
        backbone_diagnostics = self._multiview_backbone_diagnostics(
            out_dict, data)
        if backbone_diagnostics and not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Controlled ABC backbone produced a non-finite loss")
        status.update(backbone_diagnostics)

        if isinstance(out_dict, dict) and isinstance(out_dict.get("c3r", None), dict):
            diagnostics = out_dict["c3r"]
            c3r_variant = str(self._get_cfg_value("MODEL.C3R.VARIANT", "c1")).lower()
            if c3r_variant in ("c0", "a1"):
                rank_loss = diagnostics["gate_logits"].sum() * 0.0
            else:
                rank_loss = gate_ranking_loss(
                    diagnostics["gate_logits"], diagnostics["gate_labels"])
            budget_loss = diagnostics["residual_budget_loss"]
            rank_weight = float(self._get_cfg_value(
                "TRAIN.C3R.GATE_RANK_WEIGHT", 0.10))
            budget_weight = float(self._get_cfg_value(
                "TRAIN.C3R.RESIDUAL_BUDGET_WEIGHT", 0.05))
            loss = loss + rank_weight * rank_loss + budget_weight * budget_loss
            status["Loss/c3r_gate_rank"] = float(rank_loss.detach().item())
            status["Loss/c3r_residual_budget"] = float(budget_loss.detach().item())
            status["C3R/accepted_packets"] = float(diagnostics["accepted_count"])
            status["Loss/total"] = float(loss.detach().item())

        return loss, status

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------
    def _get_cfg_value(self, path, default):
        """
        安全读取 cfg。
        Example:
            self._get_cfg_value("MODEL.BACKBONE.TEMPERATURE", False)
        """
        node = self.cfg

        for key in path.split("."):
            if not hasattr(node, key):
                return default
            node = getattr(node, key)

        return node

    def _pcum_ranking_enabled(self):
        return bool(self._get_cfg_value(
            "TRAIN.PCUM_RANKING.ENABLED",
            self._get_cfg_value("TRAIN.PCUM.RANKING_ENABLED", False),
        ))

    def _pcum_rank_delay_weight(self):
        return float(self._get_cfg_value(
            "TRAIN.PCUM_RANKING.LAMBDA_DELAY",
            self._get_cfg_value("TRAIN.PCUM.RANK_DELAY_WEIGHT", 0.1),
        ))

    def _pcum_visible_only_ranking(self):
        return bool(self._get_cfg_value("TRAIN.PCUM_RANKING.VISIBLE_ONLY", False))

    def _pcum_remote_suppression_only(self):
        return bool(self._get_cfg_value(
            "TRAIN.PCUM_RANKING.REMOTE_SUPPRESSION_ONLY", False))

    def _pcum_suppress_bce_weight(self):
        return float(self._get_cfg_value(
            "TRAIN.PCUM_RANKING.LAMBDA_SUPPRESS_BCE", 0.10))

    def _pcum_suppress_mean_weight(self):
        return float(self._get_cfg_value(
            "TRAIN.PCUM_RANKING.LAMBDA_SUPPRESS_MEAN", 0.001))

    def _pcum_suppress_label_margin(self):
        return float(self._get_cfg_value(
            "TRAIN.PCUM_RANKING.SUPPRESS_LABEL_MARGIN", 0.0))

    def _pcum_rank_zero_margin(self):
        value = float(self._get_cfg_value("TRAIN.PCUM_RANKING.MARGIN_ZERO", 0.02))
        if value != 0.02:
            return value
        return float(self._get_cfg_value("TRAIN.PCUM.RANK_ZERO_MARGIN", 0.02))

    def _pcum_rank_local_margin(self):
        value = float(self._get_cfg_value("TRAIN.PCUM_RANKING.MARGIN_LOCAL", 0.0))
        if value != 0.0:
            return value
        return float(self._get_cfg_value("TRAIN.PCUM.RANK_LOCAL_MARGIN", 0.0))

    def _pcum_safe_margin(self):
        value = float(self._get_cfg_value("TRAIN.PCUM_RANKING.SAFE_MARGIN", 0.0))
        if value != 0.0:
            return value
        return float(self._get_cfg_value("TRAIN.PCUM.SAFE_MARGIN", 0.0))

    def _pcum_rank_zero_weight(self):
        value = float(self._get_cfg_value("TRAIN.PCUM_RANKING.LAMBDA_ZERO", 0.1))
        if value != 0.1:
            return value
        return float(self._get_cfg_value("TRAIN.PCUM.RANK_ZERO_WEIGHT", 0.1))

    def _pcum_rank_local_weight(self):
        value = float(self._get_cfg_value("TRAIN.PCUM_RANKING.LAMBDA_LOCAL", 0.05))
        if value != 0.05:
            return value
        return float(self._get_cfg_value("TRAIN.PCUM.RANK_LOCAL_WEIGHT", 0.05))

    def _pcum_safe_weight(self):
        value = float(self._get_cfg_value("TRAIN.PCUM_RANKING.LAMBDA_SAFE", 0.0))
        if value != 0.0:
            return value
        return float(self._get_cfg_value("TRAIN.PCUM.SAFE_LOSS_WEIGHT", 0.0))

    def _unwrap_network(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def _register_fusion_diagnostic_hook(self):
        network = self._unwrap_network()
        pcum = getattr(network, "pcum", None) if network is not None else None
        fusion = getattr(pcum, "fusion", None)
        if fusion is None or self._fusion_hook_handle is not None:
            return

        def _hook(_module, inputs, output):
            if not self._diagnostic_active or self._diagnostic_stage is None:
                return
            search_tokens = inputs[0]
            denominator = search_tokens.detach().float().norm().clamp_min(1e-12)
            relative_change = (
                (output.detach().float() - search_tokens.detach().float()).norm()
                / denominator
            ).item()
            key = "%s_feature_relative_change" % self._diagnostic_stage
            self._diagnostic_values[key] = float(relative_change)

            fusion_residual = output.detach() - search_tokens.detach()
            if self._diagnostic_stage == "local":
                self._diagnostic_local_residual = fusion_residual
            elif self._diagnostic_stage == "collaborative":
                local_residual = self._diagnostic_local_residual
                if local_residual is not None and local_residual.shape == fusion_residual.shape:
                    incremental = (
                        (fusion_residual.float() - local_residual.float()).norm()
                        / search_tokens.detach().float().norm().clamp_min(1e-12)
                    ).item()
                    self._diagnostic_values["remote_incremental_feature_change"] = float(incremental)
                self._diagnostic_local_residual = None

        self._fusion_hook_handle = fusion.register_forward_hook(_hook)

    def close_diagnostics(self):
        if self._fusion_hook_handle is not None:
            self._fusion_hook_handle.remove()
            self._fusion_hook_handle = None
        self._diagnostic_local_residual = None

    def begin_paired_iteration(self, data, diagnostics_active=False):
        self._update_flops_schedule(data["epoch"])
        self._diagnostic_active = bool(self._diagnostics_enabled and diagnostics_active)
        self._diagnostic_stage = None
        self._diagnostic_values = {}
        self._diagnostic_local_residual = None
        self._capture_paired_forward_rng()

    def _capture_paired_forward_rng(self):
        self._paired_cpu_rng_state = torch.get_rng_state()
        self._paired_cuda_rng_state = None
        self._paired_cuda_device = None
        network = self._unwrap_network()
        if network is None:
            return
        parameter = next(network.parameters(), None)
        if parameter is not None and parameter.is_cuda:
            self._paired_cuda_device = parameter.device
            self._paired_cuda_rng_state = torch.cuda.get_rng_state(parameter.device)

    def _restore_paired_forward_rng(self, clear=True):
        if self._paired_cpu_rng_state is not None:
            torch.set_rng_state(self._paired_cpu_rng_state)
        if self._paired_cuda_rng_state is not None:
            torch.cuda.set_rng_state(
                self._paired_cuda_rng_state,
                device=self._paired_cuda_device,
            )
        if clear:
            self._paired_cpu_rng_state = None
            self._paired_cuda_rng_state = None
            self._paired_cuda_device = None

    def _set_diagnostic_stage(self, stage):
        self._diagnostic_stage = stage if self._diagnostic_active else None

    def collect_gradient_diagnostics(self):
        if not self._diagnostic_active:
            return {}

        network = self._unwrap_network()
        named_parameters = list(network.named_parameters())

        def _grad_norm(prefix):
            total = 0.0
            found = False
            for name, parameter in named_parameters:
                if name.startswith(prefix) and parameter.grad is not None:
                    value = parameter.grad.detach().float().norm().item()
                    total += value * value
                    found = True
            return total ** 0.5 if found else 0.0

        pcum = getattr(network, "pcum", None)
        fusion = getattr(pcum, "fusion", None)
        values = dict(self._diagnostic_values)
        values.update({
            "Grad/pcum_encoder": _grad_norm("pcum.encoder."),
            "Grad/pcum_aligner": _grad_norm("pcum.aligner."),
            "Grad/pcum_fusion_film": _grad_norm("pcum.fusion.film."),
            "Grad/remote_suppression_gate": _grad_norm(
                "pcum.remote_suppression_gate."),
        })
        frozen_grad_present = False
        for name, parameter in named_parameters:
            if not name.startswith("pcum.remote_suppression_gate.") and parameter.grad is not None:
                if float(parameter.grad.detach().float().abs().max().item()) > 0.0:
                    frozen_grad_present = True
                    break
        values["Grad/frozen_parameter_present"] = float(frozen_grad_present)
        if fusion is not None:
            raw_scale = fusion.residual_scale.detach().float().item()
            effective_scale = fusion._residual_scale().detach().float().item()
            scale_grad = fusion.residual_scale.grad
            values.update({
                "PCUM/raw_residual_scale": float(raw_scale),
                "PCUM/effective_residual_scale": float(effective_scale),
                "Grad/residual_scale": 0.0 if scale_grad is None else float(
                    scale_grad.detach().float().item()
                ),
            })
        self._diagnostic_local_residual = None
        self._diagnostic_stage = None
        self._diagnostic_active = False
        return values

    def _select_first_view(self, value):
        """
        只取第 0 个视角，用于单机 EnTeRTrack 微调。

        支持：
        - list: [view0, view1, ...]
        - tensor [V, B, C, H, W]
        - tensor [1, B, C, H, W]
        - tensor [B, C, H, W]
        - tensor [V, B, 4]
        - tensor [B, 4]
        """
        if isinstance(value, list):
            value = value[0]

        elif isinstance(value, torch.Tensor):
            if value.dim() == 5:
                # [V, B, C, H, W] or [1, B, C, H, W]
                value = value[0]

            elif value.dim() == 3 and value.shape[-1] == 4:
                # [V, B, 4] or [1, B, 4]
                value = value[0]

        return value

    def _select_view(self, value, view_index=0):
        """
        Select one UAV view while preserving the old single-view behavior.
        """
        if isinstance(value, list):
            return value[view_index]

        if isinstance(value, torch.Tensor):
            if value.dim() == 5:
                return value[view_index]
            if value.dim() == 3 and value.shape[-1] == 4:
                return value[view_index]

        return value

    def _select_view_valid(self, data, key, view_index):
        valid = data.get(key, None)
        if valid is None:
            return None

        if not isinstance(valid, torch.Tensor):
            valid = torch.as_tensor(valid)

        # LTRLoader(stack_dim=1) collates per-sample [V] tensors into [V, B].
        if valid.dim() == 2:
            if valid.shape[0] > view_index:
                return valid[view_index].bool()
            if valid.shape[1] > view_index:
                return valid[:, view_index].bool()

        if valid.dim() == 1 and valid.shape[0] > view_index:
            return valid[view_index].view(1).bool()

        return None

    def _view_prompt_valid_mask(self, data, view_index, device=None, dtype=None):
        if not self._get_cfg_value("TRAIN.PCUM.USE_REMOTE_VISIBLE_MASK", False):
            return None

        template_valid = self._select_view_valid(data, "template_view_valid", view_index)
        search_valid = self._select_view_valid(data, "search_view_valid", view_index)

        if template_valid is None and search_valid is None:
            return None
        if template_valid is None:
            mask = search_valid
        elif search_valid is None:
            mask = template_valid
        else:
            mask = template_valid & search_valid

        if device is not None:
            mask = mask.to(device=device)
        if dtype is not None:
            mask = mask.to(dtype=dtype)
        return mask

    def _apply_prompt_mask(self, prompt, mask):
        if prompt is None or mask is None:
            return prompt
        mask = mask.to(device=prompt.device, dtype=prompt.dtype)
        while mask.dim() < prompt.dim():
            mask = mask.unsqueeze(-1)
        return prompt * mask

    def _make_remote_state(self, masks, device=None, dtype=None,
                           metric_states=None):
        masks = [m for m in masks if m is not None]
        if len(masks) == 0:
            return None

        stacked = torch.stack([
            m.to(device=device, dtype=dtype or torch.float32).view(-1)
            for m in masks
        ], dim=0)
        state = {
            "score": stacked.mean(dim=0),
            "per_remote_valid": (stacked.transpose(0, 1) > 0).detach(),
        }
        if metric_states is None:
            return state
        if len(metric_states) != len(masks):
            raise ValueError("Remote metric states must align with prompt masks")

        metric_keys = (
            "score",
            "apce",
            "bbox_score",
            "motion_reliability",
        )
        for key in metric_keys:
            values = []
            any_available = False
            for slot, mask in zip(metric_states, masks):
                value = None if slot is None else slot.get(key, None)
                if value is None:
                    value = torch.full_like(
                        mask.to(device=device, dtype=dtype or torch.float32),
                        float("nan"),
                    )
                else:
                    any_available = True
                    value = value.detach().to(
                        device=device, dtype=dtype or torch.float32
                    ).reshape(-1)
                values.append(value)
            if any_available:
                state["per_remote_{}".format(key)] = torch.stack(
                    values, dim=1
                ).detach()
        return state

    def _predicted_remote_state_bank(self, pred_dict, num_views):
        """Build detached no-GT reliability from local warm predictions."""
        score_map = pred_dict.get("score_map", None)
        if not torch.is_tensor(score_map) or score_map.shape[0] % num_views != 0:
            return None
        response = score_map.detach().float().reshape(score_map.shape[0], -1)
        maximum = response.max(dim=1).values
        score = maximum.clamp(0.0, 1.0)
        minimum = response.min(dim=1).values
        denominator = ((response - minimum[:, None]) ** 2).mean(dim=1)
        apce = ((maximum - minimum) ** 2) / denominator.clamp_min(1e-8)
        apce_norm = max(float(self._get_cfg_value(
            "TEST.PCUM.MOTION_REDETECT_APCE_NORM", 200.0)), 1e-6)
        apce = (apce / apce_norm).clamp(0.0, 1.0)

        batch_size = score.shape[0] // num_views
        states = []
        for view_index in range(num_views):
            start = view_index * batch_size
            end = (view_index + 1) * batch_size
            states.append({
                "score": score[start:end].detach(),
                "apce": apce[start:end].detach(),
            })
        return states

    def _real_multiview_loss_weights(self, num_views, device):
        weights = self._get_cfg_value("TRAIN.PCUM.REAL_MULTIVIEW_LOSS_WEIGHTS", [])
        if not weights or len(weights) < num_views:
            return torch.ones(num_views, device=device) / float(num_views)

        weights = torch.as_tensor(weights[:num_views], device=device, dtype=torch.float32)
        weights = weights.clamp(min=0.0)
        if float(weights.sum().item()) <= 0:
            return torch.ones(num_views, device=device) / float(num_views)
        return weights / weights.sum()

    def _num_views(self, data):
        search_images = data.get("search_images", None)
        if isinstance(search_images, torch.Tensor) and search_images.dim() == 5:
            return int(search_images.shape[0])
        if isinstance(search_images, list):
            return len(search_images)
        return 1

    def _use_real_multiview_pcum(self, data):
        return (
            self._get_cfg_value("MODEL.PCUM.ENABLED", False)
            and self._get_cfg_value("TRAIN.PCUM.USE_REAL_MULTIVIEW", False)
            and self._num_views(data) >= 3
        )

    def _use_plain_collaboration(self, data):
        return (
            self._get_cfg_value("MODEL.PLAIN_COLLABORATION.ENABLED", False)
            and self._get_cfg_value(
                "TRAIN.PLAIN_COLLABORATION.ENABLED", False)
            and self._num_views(data) >= 3
        )

    def _use_target_prompt_collaboration(self, data):
        return (
            self._get_cfg_value(
                "MODEL.TARGET_PROMPT_COLLABORATION.ENABLED", False)
            and self._get_cfg_value(
                "TRAIN.TARGET_PROMPT_COLLABORATION.ENABLED", False)
            and self._num_views(data) >= 3
        )

    def _use_c3r(self, data):
        return (
            self._get_cfg_value("MODEL.C3R.ENABLED", False)
            and self._get_cfg_value("TRAIN.C3R.ENABLED", False)
            and self._num_views(data) >= 3
        )

    def _use_flat_multiview_baseline(self, data):
        return (
            (
                self._get_cfg_value("TRAIN.MULTIVIEW.FLAT_BASELINE", False)
                or (
                    self._get_cfg_value(
                        "TRAIN.PARTIAL_ADAPTATION.ENABLED", False)
                    and self._get_cfg_value(
                        "TRAIN.PARTIAL_ADAPTATION.FLAT_MULTIVIEW_BASELINE", False)
                )
            )
            and not self._get_cfg_value("MODEL.PCUM.ENABLED", False)
            and self._num_views(data) >= 3
        )

    def _plain_multiview_diagnostics(self, output):
        """Assert complete Plain ViT tokens and audit optional V1 collaboration."""
        if not self._get_cfg_value(
                "TRAIN.MULTIVIEW.DIAGNOSTICS_ENABLED", False):
            return {}
        if not output.get("pcum_flat_multiview", False):
            raise RuntimeError("Plain multiview diagnostics require the flat path")

        network = self._unwrap_network()
        backbone_type = str(self._get_cfg_value("MODEL.BACKBONE.TYPE", ""))
        if backbone_type != "vit_tiny_patch16_224_half":
            raise RuntimeError(
                "B0-ABC-Plain requires vit_tiny_patch16_224_half, got %s"
                % backbone_type)
        if getattr(network, "pcum", None) is not None:
            raise RuntimeError("B0-ABC-Plain unexpectedly constructed PCUM")
        if getattr(network, "c3r", None) is not None:
            raise RuntimeError("B0-ABC-Plain unexpectedly constructed C3R")
        if getattr(network, "search_prompt_gate", None) is not None:
            raise RuntimeError("B0-ABC-Plain unexpectedly constructed a prompt gate")

        backbone = getattr(network, "backbone", None)
        if backbone is None or len(getattr(backbone, "blocks", [])) != 6:
            raise RuntimeError("B0-ABC-Plain requires exactly six ViT blocks")
        if getattr(backbone, "embed_dim", None) != 192:
            raise RuntimeError("B0-ABC-Plain requires embedding dimension 192")
        if hasattr(backbone, "ce_loc") or hasattr(backbone, "atp"):
            raise RuntimeError("B0-ABC-Plain constructed an ARP/ATP backbone")

        expected_search = (
            int(self.cfg.DATA.SEARCH.SIZE)
            // int(self.cfg.MODEL.BACKBONE.STRIDE)
        ) ** 2
        expected_template = (
            int(self.cfg.DATA.TEMPLATE.SIZE)
            // int(self.cfg.MODEL.BACKBONE.STRIDE)
        ) ** 2
        feature = output.get("backbone_feat", None)
        if not torch.is_tensor(feature):
            raise RuntimeError("B0-ABC-Plain backbone feature is missing")
        if feature.shape[1] != expected_template + expected_search:
            raise RuntimeError(
                "Plain ViT token count mismatch: got %d, expected %d+%d"
                % (feature.shape[1], expected_template, expected_search))
        if int(getattr(network, "feat_len_s", -1)) != expected_search:
            raise RuntimeError("CENTER head search-token contract is not complete")
        if output.get("atp_masks", None) or output.get("removed_indexes_s", None):
            raise RuntimeError("B0-ABC-Plain emitted pruning/ATP diagnostics")
        score_map = output.get("score_map", None)
        pred_boxes = output.get("pred_boxes", None)
        expected_side = int(expected_search ** 0.5)
        if (not torch.is_tensor(score_map)
                or tuple(score_map.shape[-2:]) != (expected_side, expected_side)
                or score_map.shape[0] != feature.shape[0]):
            raise RuntimeError("B0-ABC-Plain CENTER score-map shape is invalid")
        if (not torch.is_tensor(pred_boxes)
                or pred_boxes.shape[0] != feature.shape[0]
                or pred_boxes.shape[-1] != 4):
            raise RuntimeError("B0-ABC-Plain CENTER bbox shape is invalid")

        collaboration = output.get("plain_collaboration", None)
        target_prompt = output.get("target_prompt_collaboration", None)
        status = {
            "Plain/search_tokens": float(expected_search),
            "Plain/template_tokens": float(expected_template),
            "Plain/total_tokens": float(feature.shape[1]),
            "Plain/transformer_blocks": float(len(backbone.blocks)),
            "Plain/flat_multiview": 1.0,
            "Plain/center_map_side": float(expected_side),
            "Plain/bbox_dim": float(pred_boxes.shape[-1]),
            "Plain/pcum_present": 0.0,
            "Plain/c3r_present": 0.0,
            "Plain/remote_state_present": float(
                collaboration is not None or target_prompt is not None),
            "Plain/pruning_present": 0.0,
        }
        if collaboration is not None:
            if getattr(network, "plain_collaboration", None) is None:
                raise RuntimeError("V1 output exists without a model adapter")
            weights = collaboration.get("remote_weights", None)
            if not torch.is_tensor(weights) or weights.shape[1] != 2:
                raise RuntimeError("V1 must provide two remote weights per receiver")
            status.update({
                "V1/used_remote": float(collaboration["used_remote"]),
                "V1/valid_remote_count": float(
                    collaboration["valid_remote_count"].float().mean().item()),
                "V1/residual_norm": float(
                    collaboration["residual_norm"].detach().item()),
                "V1/relative_residual_norm": float(
                    collaboration["relative_residual_norm"].detach().item()),
                "V1/residual_scale": float(
                    collaboration["residual_scale"].detach().item()),
            })
        if target_prompt is not None:
            if getattr(network, "target_prompt_collaboration", None) is None:
                raise RuntimeError("E3 output exists without the V2 adapter")
            weights = target_prompt.get("remote_weights", None)
            if not torch.is_tensor(weights) or weights.shape[1] != 2:
                raise RuntimeError("E3 must provide two remote weights per receiver")
            if int(output.get("target_prompt_k", -1)) != 8:
                raise RuntimeError("E3 requires fixed prompt K=8")
            status.update({
                "E3/prompt_k": 8.0,
                "E3/used_remote": float(target_prompt["used_remote"]),
                "E3/valid_remote_count": float(
                    target_prompt["valid_remote_count"].float().mean().item()),
                "E3/residual_norm": float(
                    target_prompt["residual_norm"].detach().item()),
                "E3/relative_residual_norm": float(
                    target_prompt["relative_residual_norm"].detach().item()),
                "E3/residual_scale": float(
                    target_prompt["residual_scale"].detach().item()),
            })
        return status

    def _multiview_backbone_diagnostics(self, output, data):
        self.last_arp_sample_diagnostics = []
        if not self._get_cfg_value(
                "TRAIN.MULTIVIEW.DIAGNOSTICS_ENABLED", False):
            return {}
        backbone_type = str(self._get_cfg_value("MODEL.BACKBONE.TYPE", ""))
        if backbone_type == "vit_tiny_patch16_224_half":
            return self._plain_multiview_diagnostics(output)
        if backbone_type == "vit_tiny_patch16_224_arp":
            return self._arp_multiview_diagnostics(output, data)
        raise RuntimeError(
            "Controlled multiview diagnostics do not recognize backbone %s"
            % backbone_type)

    def _arp_multiview_diagnostics(self, output, data):
        """Validate and summarize the frozen B1 ARP/ATP implementation.

        The returned values are detached logging statistics.  The hard ATP
        mask is the logical keep mask used by the existing STE training path;
        its complement is routed through the existing delta compensation.
        """
        if not output.get("pcum_flat_multiview", False):
            raise RuntimeError("B1-ABC-ARP diagnostics require the flat ABC path")
        if self._get_cfg_value(
                "TRAIN.MULTIVIEW.INDEPENDENT_VIEW_SAMPLING", False):
            raise RuntimeError("B1-ABC-ARP forbids independent-view sampling")
        if not self._get_cfg_value(
                "TRAIN.MULTIVIEW.REQUIRE_ALL_VIEWS_VISIBLE", False):
            raise RuntimeError("B1-ABC-ARP requires the common-visible sampler")

        required_switches = (
            "MODEL.BACKBONE.PRUNING_ENABLED",
            "MODEL.BACKBONE.DYNAMIC_THRESHOLD_ENABLED",
            "MODEL.BACKBONE.TOKEN_COMPENSATION_ENABLED",
        )
        disabled = [
            path for path in required_switches
            if not bool(self._get_cfg_value(path, False))
        ]
        if disabled:
            raise RuntimeError("B1 ARP audit switches are disabled: %s" % disabled)
        if list(self.cfg.MODEL.BACKBONE.CE_LOC) != [0]:
            raise RuntimeError("B1-ABC-ARP requires CE_LOC=[0]")
        if list(self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO) != [0.7]:
            raise RuntimeError("B1-ABC-ARP requires CE_KEEP_RATIO=[0.7]")

        network = self._unwrap_network()
        if getattr(network, "pcum", None) is not None:
            raise RuntimeError("B1-ABC-ARP unexpectedly constructed PCUM")
        if getattr(network, "c3r", None) is not None:
            raise RuntimeError("B1-ABC-ARP unexpectedly constructed C3R")
        if getattr(network, "search_prompt_gate", None) is not None:
            raise RuntimeError("B1-ABC-ARP unexpectedly constructed a prompt gate")
        backbone = getattr(network, "backbone", None)
        blocks = list(getattr(backbone, "blocks", []))
        if len(blocks) != 6 or getattr(backbone, "embed_dim", None) != 192:
            raise RuntimeError("B1-ABC-ARP requires ViT-Tiny dim=192 depth=6")
        atp = getattr(blocks[0], "atp", None)
        if atp is None or getattr(atp, "delta_estimator", None) is None:
            raise RuntimeError("B1-ABC-ARP ATP/compensation modules are missing")
        atp_parameters = [
            parameter for name, parameter in network.named_parameters()
            if ".atp." in name
        ]
        if not atp_parameters or not all(
                parameter.requires_grad for parameter in atp_parameters):
            raise RuntimeError("B1 ATP parameters are absent or frozen")

        expected_search = (
            int(self.cfg.DATA.SEARCH.SIZE)
            // int(self.cfg.MODEL.BACKBONE.STRIDE)
        ) ** 2
        expected_template = (
            int(self.cfg.DATA.TEMPLATE.SIZE)
            // int(self.cfg.MODEL.BACKBONE.STRIDE)
        ) ** 2
        feature = output.get("backbone_feat", None)
        if (not torch.is_tensor(feature)
                or feature.shape[1] != expected_template + expected_search):
            raise RuntimeError("B1 compensated feature did not restore 64+256 tokens")
        if int(output.get("arp_initial_search_tokens", -1)) != expected_search:
            raise RuntimeError("B1 initial search-token count is not 256")
        if int(output.get("arp_output_search_tokens", -1)) != expected_search:
            raise RuntimeError("B1 CENTER input did not restore 256 search tokens")
        score_map = output.get("score_map", None)
        pred_boxes = output.get("pred_boxes", None)
        if (not torch.is_tensor(score_map)
                or tuple(score_map.shape[-2:]) != (16, 16)
                or not torch.is_tensor(pred_boxes)
                or pred_boxes.shape[-1] != 4):
            raise RuntimeError("B1 CENTER head output contract is invalid")
        if any(key in output for key in ("remote_prompt", "remote_state")):
            raise RuntimeError("B1-ABC-ARP emitted remote collaboration state")

        keep_masks = output.get("atp_keep_masks", [])
        thresholds = output.get("atp_thresholds", [])
        compensation_masks = output.get("compensation_masks", [])
        delta_norms = output.get("compensation_delta_norms", [])
        if not keep_masks or not thresholds or not compensation_masks:
            raise RuntimeError("B1-ABC-ARP did not execute ATP pruning/compensation")
        if len(keep_masks) != len(thresholds) or len(keep_masks) != len(compensation_masks):
            raise RuntimeError("B1 ARP diagnostic layer counts do not match")

        num_views = int(output.get("num_views", 0))
        total_batch = int(keep_masks[0].shape[0])
        if num_views != 3 or total_batch % num_views != 0:
            raise RuntimeError("B1 flat output is not a [3*B] batch")
        group_batch = total_batch // num_views
        status = {
            "ARP/initial_search_tokens": float(expected_search),
            "ARP/restored_search_tokens": float(expected_search),
            "ARP/atp_parameter_count": float(sum(
                parameter.numel() for parameter in atp_parameters)),
            "ARP/flat_multiview": 1.0,
            "ARP/pcum_present": 0.0,
            "ARP/c3r_present": 0.0,
            "ARP/remote_state_present": 0.0,
        }

        layer_keep_ratios = []
        for layer_index, keep_mask in enumerate(keep_masks):
            hard_keep = keep_mask.detach().float()
            if hard_keep.shape != (total_batch, expected_search):
                raise RuntimeError("B1 ATP keep-mask shape is invalid")
            keep_ratio = hard_keep.mean(dim=1)
            layer_keep_ratios.append(keep_ratio)
            threshold = thresholds[layer_index].detach().float().reshape(total_batch, -1).mean(dim=1)
            compensation = compensation_masks[layer_index].detach().float()
            compensation_ratio = compensation.mean(dim=1)
            if not torch.equal(compensation, 1.0 - hard_keep):
                raise RuntimeError("B1 compensation mask is not the pruned-token complement")
            delta = delta_norms[layer_index]
            if delta is None:
                raise RuntimeError("B1 compensation delta was not executed")
            delta = delta.detach().float()
            active_delta_mean = delta.sum(dim=1) / compensation.sum(dim=1).clamp_min(1.0)

            prefix = "ARP/layer_%d" % int(self.cfg.MODEL.BACKBONE.CE_LOC[layer_index])
            status[prefix + "/mean_kept_search_tokens"] = float(
                keep_ratio.mean().item() * expected_search)
            status[prefix + "/mean_pruned_search_tokens"] = float(
                (1.0 - keep_ratio).mean().item() * expected_search)
            status[prefix + "/keep_ratio"] = float(keep_ratio.mean().item())
            status[prefix + "/threshold_mean"] = float(threshold.mean().item())
            status[prefix + "/threshold_std"] = float(
                threshold.std(unbiased=False).item())
            status[prefix + "/compensation_activation_ratio"] = float(
                compensation_ratio.mean().item())
            status[prefix + "/compensation_delta_norm"] = float(
                active_delta_mean.mean().item())

            for view_index, view_label in enumerate(("A", "B", "C")):
                start = view_index * group_batch
                end = (view_index + 1) * group_batch
                status[prefix + "/view_%s_keep_ratio" % view_label] = float(
                    keep_ratio[start:end].mean().item())

        primary_keep = layer_keep_ratios[0]
        primary_threshold = thresholds[0].detach().float().reshape(total_batch, -1).mean(dim=1)
        primary_compensation = compensation_masks[0].detach().float().mean(dim=1)
        search_anno = data.get("search_anno", None)
        target_ids = data.get("target_id", [])
        view_ids = data.get("view_ids", [])
        rows = []
        for view_index in range(num_views):
            for batch_index in range(group_batch):
                flat_index = view_index * group_batch + batch_index
                bbox = search_anno[view_index][batch_index].detach().float().reshape(-1, 4)[0]
                view_label = str(view_ids[view_index][batch_index])
                rows.append({
                    "target_id": str(target_ids[batch_index]),
                    "view_id": view_label,
                    "bbox_area": float((bbox[2] * bbox[3]).item()),
                    "kept_search_tokens": float(primary_keep[flat_index].item() * expected_search),
                    "pruned_search_tokens": float((1.0 - primary_keep[flat_index]).item() * expected_search),
                    "keep_ratio": float(primary_keep[flat_index].item()),
                    "atp_threshold": float(primary_threshold[flat_index].item()),
                    "compensation_activation_ratio": float(primary_compensation[flat_index].item()),
                })
        self.last_arp_sample_diagnostics = rows
        return status

    def _squeeze_if_needed(self, value):
        """
        去掉多余维度，保证图像是 [B, C, H, W]，bbox 是 [B, 4]。
        """
        if not isinstance(value, torch.Tensor):
            return value

        if value.dim() == 5 and value.shape[0] == 1:
            value = value[0]

        if value.dim() == 3 and value.shape[0] == 1 and value.shape[-1] == 4:
            value = value[0]

        if value.dim() == 3 and value.shape[1] == 1 and value.shape[-1] == 4:
            value = value[:, 0]

        if value.dim() == 1 and value.shape[0] == 4:
            value = value.unsqueeze(0)

        return value

    def _flatten_views(self, value, num_views):
        """
        Flatten [V, B, ...] view-major tensors into [V*B, ...].
        This lets DDP see one model forward per warm/final pass.
        """
        if isinstance(value, list):
            value = value[:num_views]
            if torch.is_tensor(value[0]):
                return torch.cat(value, dim=0)
            return value

        if not isinstance(value, torch.Tensor):
            return value

        if value.dim() >= 2 and value.shape[0] >= num_views:
            leading_shape = value.shape[:2]
            if leading_shape[0] == num_views:
                return value[:num_views].contiguous().view(
                    leading_shape[0] * leading_shape[1],
                    *value.shape[2:]
                )

        return value

    def _get_prompt_from_pred(self, pred_dict):
        for key in ("local_prompt", "pcum_local_prompt"):
            if key in pred_dict and pred_dict[key] is not None:
                return pred_dict[key]
        pcum_out = pred_dict.get("pcum", None)
        if isinstance(pcum_out, dict):
            for key in ("local_prompt", "prompt"):
                if key in pcum_out and pcum_out[key] is not None:
                    return pcum_out[key]
        return None

    def _get_remote_prompt_from_pred(self, pred_dict):
        for key in ("pseudo_remote_prompt", "remote_prompt", "pcum_remote_prompt"):
            if key in pred_dict and pred_dict[key] is not None:
                return pred_dict[key]
        pcum_out = pred_dict.get("pcum", None)
        if isinstance(pcum_out, dict):
            for key in ("pseudo_remote_prompt", "remote_prompt"):
                if key in pcum_out and pcum_out[key] is not None:
                    return pcum_out[key]
        return None

    def _make_gt_heatmap(self, gt_bbox):
        """
        gt_bbox: [B, 4], xywh, normalized.
        return: [B, 1, H, W]
        """
        gt_gaussian_maps = generate_heatmap(
            [gt_bbox],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE
        )

        gt_gaussian_maps = gt_gaussian_maps[0]

        if gt_gaussian_maps.dim() == 4 and gt_gaussian_maps.shape[1] == 1:
            # [B, 1, H, W]
            pass

        elif gt_gaussian_maps.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            gt_gaussian_maps = gt_gaussian_maps.unsqueeze(1)

        elif gt_gaussian_maps.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            gt_gaussian_maps = gt_gaussian_maps.unsqueeze(0).unsqueeze(0)

        else:
            raise ValueError(f"Unexpected heatmap shape: {gt_gaussian_maps.shape}")

        return gt_gaussian_maps

    def _make_prompt_inputs(self, gt_bbox):
        """
        Build synthetic low-bandwidth search-region prompts for training.

        The prompt map uses local search coordinates. This trains the gate and
        score-map bias without requiring cross-view calibration.
        """
        if not self._get_cfg_value("MODEL.USE_SEARCH_PROMPT", False):
            return None, None

        noisy_bbox = gt_bbox.clone()
        noise_std = float(self._get_cfg_value("TRAIN.PROMPT_NOISE_STD", 0.08))
        wrong_prob = float(self._get_cfg_value("TRAIN.PROMPT_WRONG_PROB", 0.1))
        drop_prob = float(self._get_cfg_value("TRAIN.PROMPT_DROP_PROB", 0.3))

        if noise_std > 0:
            noisy_bbox = noisy_bbox + torch.randn_like(noisy_bbox) * noise_std

        if wrong_prob > 0:
            wrong_mask = torch.rand(gt_bbox.shape[0], device=gt_bbox.device) < wrong_prob
            if wrong_mask.any():
                noisy_bbox[wrong_mask, :2] = torch.rand_like(noisy_bbox[wrong_mask, :2])
                noisy_bbox[wrong_mask, 2:] = gt_bbox[wrong_mask, 2:].clamp(0.05, 0.8)

        noisy_bbox = noisy_bbox.clamp(0.0, 1.0)
        noisy_bbox[:, 2:] = noisy_bbox[:, 2:].clamp(0.02, 1.0)

        prompt_map = self._make_gt_heatmap(noisy_bbox)

        if drop_prob > 0:
            keep = (torch.rand(gt_bbox.shape[0], 1, 1, 1, device=gt_bbox.device) >= drop_prob).float()
            prompt_map = prompt_map * keep
        else:
            keep = torch.ones(gt_bbox.shape[0], 1, 1, 1, device=gt_bbox.device)

        bbox_delta = (noisy_bbox - gt_bbox).abs().mean(dim=1, keepdim=True)
        gate_input = torch.cat([
            torch.full_like(bbox_delta, 0.2),  # simulated self score
            torch.full_like(bbox_delta, 80.0 / 200.0),  # simulated self APCE
            keep.view(gt_bbox.shape[0], 1),
            torch.full_like(bbox_delta, 150.0 / 200.0),  # simulated peer APCE
            bbox_delta,
            noisy_bbox[:, 2:3].clamp(0.0, 1.0),
        ], dim=1)

        return prompt_map, gate_input

    # ------------------------------------------------------------
    # FLOPs schedule
    # ------------------------------------------------------------
    def _update_flops_schedule(self, epoch):
        """
        保留原蒸馏 actor 里的 FLOPs 约束调度，但不再使用 teacher。

        默认：
        epoch < 20:
            不约束 FLOPs

        20 <= epoch < 100:
            FLOPs target 从 10e8 线性下降到 7e8
            FLOPs loss 权重从 0 线性增加到 5

        epoch >= 100:
            FLOPs target = 7e8
            FLOPs loss 权重 = 5
        """
        start_epoch = self._get_cfg_value("TRAIN.FLOPS_START_EPOCH", 0)
        end_epoch = self._get_cfg_value("TRAIN.FLOPS_END_EPOCH", 0)

        max_flops_target = self._get_cfg_value("TRAIN.MAX_FLOPS_TARGET", 7e8)
        initial_flops_target = self._get_cfg_value("TRAIN.INITIAL_FLOPS_TARGET", 10e8)

        max_flops_weight = self._get_cfg_value("TRAIN.FLOPS_WEIGHT", 5.0)

        if epoch < start_epoch:
            self.flops_weight = 0.0
            self.F_target = initial_flops_target

        elif epoch >= end_epoch:
            self.flops_weight = max_flops_weight
            self.F_target = max_flops_target

        else:
            frac = (epoch - start_epoch) / float(end_epoch - start_epoch)

            self.flops_weight = max_flops_weight * frac
            self.F_target = initial_flops_target - frac * (
                initial_flops_target - max_flops_target
            )

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward_pass(self, net, data):
        """
        单机 forward：只取第 0 个视角。
        """
        if self._use_target_prompt_collaboration(data):
            return forward_target_prompt_collaboration(self, net, data)
        if self._use_plain_collaboration(data):
            return forward_plain_collaboration(self, net, data)
        if self._use_c3r(data):
            return self.forward_pass_c3r(net, data)
        if self._use_real_multiview_pcum(data):
            return self.forward_pass_real_multiview_pcum(net, data)
        if self._use_flat_multiview_baseline(data):
            num_views = min(self._num_views(data), 3)
            return self._mark_flat_multiview(
                self._forward_flat_views(
                    net, data, num_views, remote_prompts=None),
                num_views,
            )

        template_img = self._select_first_view(data["template_images"])
        search_img = self._select_first_view(data["search_images"])
        template_anno = self._select_first_view(data["template_anno"])

        template_img = self._squeeze_if_needed(template_img)
        search_img = self._squeeze_if_needed(search_img)
        template_anno = self._squeeze_if_needed(template_anno)
        gt_bbox = self._select_first_view(data["search_anno"])
        gt_bbox = self._squeeze_if_needed(gt_bbox)

        box_mask_z = None
        ce_keep_rate = None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_img.shape[0],
                template_img.device,
                template_anno
            )

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH

            ce_keep_rate = adjust_keep_rate(
                data["epoch"],
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )

        temperature = 100.0
        if self._get_cfg_value("MODEL.BACKBONE.TEMPERATURE", False):
            temperature = adjust_temperature(
                data["epoch"],
                warmup_epochs=1,
                total_epochs=30,
                ITERS_PER_EPOCH=1
            )

        prompt_map, prompt_gate_input = self._make_prompt_inputs(gt_bbox)
        if prompt_map is not None:
            prompt_map = prompt_map.to(search_img.device)
            prompt_gate_input = prompt_gate_input.to(search_img.device)

        # 关键：
        # EnTeRTrack 训练时要传 training=True，
        # 否则 ATP 可能走推理分支，atp_masks / FLOPs 约束不稳定。
        out_dict = net(
            template=template_img,
            search=search_img,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            temperature=temperature,
            return_last_attn=False,
            return_atp=True,
            training=True,
            prompt_map=prompt_map,
            prompt_gate_input=prompt_gate_input
        )

        return out_dict

    def _forward_one_view(self, net, data, view_index=0, remote_prompts=None,
                          remote_states=None, remote_suppression_override=None):
        template_img = self._select_view(data["template_images"], view_index)
        search_img = self._select_view(data["search_images"], view_index)
        template_anno = self._select_view(data["template_anno"], view_index)
        gt_bbox = self._select_view(data["search_anno"], view_index)

        template_img = self._squeeze_if_needed(template_img)
        search_img = self._squeeze_if_needed(search_img)
        template_anno = self._squeeze_if_needed(template_anno)
        gt_bbox = self._squeeze_if_needed(gt_bbox)

        box_mask_z = None
        ce_keep_rate = None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_img.shape[0],
                template_img.device,
                template_anno
            )

            ce_keep_rate = adjust_keep_rate(
                data["epoch"],
                warmup_epochs=self.cfg.TRAIN.CE_START_EPOCH,
                total_epochs=self.cfg.TRAIN.CE_START_EPOCH + self.cfg.TRAIN.CE_WARM_EPOCH,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )

        temperature = 100.0
        if self._get_cfg_value("MODEL.BACKBONE.TEMPERATURE", False):
            temperature = adjust_temperature(
                data["epoch"],
                warmup_epochs=1,
                total_epochs=30,
                ITERS_PER_EPOCH=1
            )

        prompt_map, prompt_gate_input = self._make_prompt_inputs(gt_bbox)
        if prompt_map is not None:
            prompt_map = prompt_map.to(search_img.device)
            prompt_gate_input = prompt_gate_input.to(search_img.device)

        return net(
            template=template_img,
            search=search_img,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            temperature=temperature,
            return_last_attn=False,
            return_atp=True,
            training=True,
            prompt_map=prompt_map,
            prompt_gate_input=prompt_gate_input,
            remote_prompts=remote_prompts,
            remote_states=remote_states,
            remote_suppression_override=remote_suppression_override,
        )

    def _forward_flat_views(self, net, data, num_views, remote_prompts=None,
                            remote_states=None, remote_suppression_override=None,
                            model_training=True):
        template_img = self._flatten_views(data["template_images"], num_views)
        search_img = self._flatten_views(data["search_images"], num_views)
        template_anno = self._flatten_views(data["template_anno"], num_views)
        gt_bbox = self._flatten_views(data["search_anno"], num_views)

        template_img = self._squeeze_if_needed(template_img)
        search_img = self._squeeze_if_needed(search_img)
        template_anno = self._squeeze_if_needed(template_anno)
        gt_bbox = self._squeeze_if_needed(gt_bbox)

        box_mask_z = None
        ce_keep_rate = None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_img.shape[0],
                template_img.device,
                template_anno
            )

            ce_keep_rate = adjust_keep_rate(
                data["epoch"],
                warmup_epochs=self.cfg.TRAIN.CE_START_EPOCH,
                total_epochs=self.cfg.TRAIN.CE_START_EPOCH + self.cfg.TRAIN.CE_WARM_EPOCH,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )

        temperature = 100.0
        if self._get_cfg_value("MODEL.BACKBONE.TEMPERATURE", False):
            temperature = adjust_temperature(
                data["epoch"],
                warmup_epochs=1,
                total_epochs=30,
                ITERS_PER_EPOCH=1
            )

        prompt_map, prompt_gate_input = self._make_prompt_inputs(gt_bbox)
        if prompt_map is not None:
            prompt_map = prompt_map.to(search_img.device)
            prompt_gate_input = prompt_gate_input.to(search_img.device)

        return net(
            template=template_img,
            search=search_img,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            temperature=temperature,
            return_last_attn=False,
            return_atp=True,
            training=bool(model_training),
            prompt_map=prompt_map,
            prompt_gate_input=prompt_gate_input,
            remote_prompts=remote_prompts,
            remote_states=remote_states,
            remote_suppression_override=remote_suppression_override,
        )

    def forward_pass_c3r(self, net, data):
        """One frozen local forward, then C3R-only per-receiver head reruns."""
        target = net.module if hasattr(net, "module") else net
        if getattr(target, "c3r", None) is None:
            raise RuntimeError("C3R actor path requires model.c3r")
        num_views = min(self._num_views(data), 3)
        if num_views != 3:
            raise ValueError("C3R training requires exactly three synchronized views")
        with torch.no_grad():
            local_output = self._forward_flat_views(
                net, data, num_views, remote_prompts=None, model_training=False)
        feature = local_output["backbone_feat"].detach()
        search_tokens = feature[:, -target.feat_len_s:]
        total_batch = search_tokens.shape[0]
        if total_batch % num_views != 0:
            raise ValueError("Flattened C3R batch is not divisible by three views")
        batch_size = total_batch // num_views
        self._c3r_iteration += 1
        frame_id = self._c3r_iteration + 4
        frame_interval_ms = 33
        sender_ids = [index // batch_size for index in range(total_batch)]
        sequence_hashes = [
            ((self._c3r_iteration * 2654435761) + (index % batch_size)) & 0xFFFFFFFF
            for index in range(total_batch)
        ]
        frame_ids = [frame_id] * total_batch
        timestamps = [frame_id * frame_interval_ms] * total_batch
        messages = target.c3r.encoder(
            search_tokens=search_tokens,
            response=local_output["score_map"].detach(),
            bbox=local_output["pred_boxes"].detach(),
            previous_bbox=None,
            sender_ids=sender_ids,
            sequence_hashes=sequence_hashes,
            frame_ids=frame_ids,
            timestamp_ms=timestamps,
        )

        sample_outputs = []
        gate_logits = []
        gate_labels = []
        budget_losses = []
        accepted_count = 0
        for flat_index in range(total_batch):
            receiver_view = flat_index // batch_size
            sample_index = flat_index % batch_size
            remote_indices = [
                view * batch_size + sample_index
                for view in range(num_views) if view != receiver_view
            ]
            remotes = [messages[index] for index in remote_indices]
            wrong_pool = None
            if batch_size > 1:
                wrong_sample = (sample_index + 1) % batch_size
                wrong_pool = [
                    messages[view * batch_size + wrong_sample]
                    for view in range(num_views) if view != receiver_view
                ]
            remotes = self._c3r_perturbation.apply(
                remotes, frame_interval_ms=frame_interval_ms,
                wrong_pool=wrong_pool)
            collaboration = target.c3r.collaborate(
                local_tokens=search_tokens[flat_index:flat_index + 1],
                local_response=local_output["score_map"][flat_index:flat_index + 1],
                packets=remotes,
                receiver_id=receiver_view,
                sequence_hash=sequence_hashes[flat_index],
                local_frame_id=frame_id,
                local_timestamp_ms=frame_id * frame_interval_ms,
                frame_interval_ms=frame_interval_ms,
                last_frame_by_sender={},
            )
            if collaboration["used_remote"]:
                fused_feature = torch.cat((
                    feature[flat_index:flat_index + 1, :-target.feat_len_s],
                    collaboration["search_tokens"],
                ), dim=1)
                sample_output = target.forward_head(fused_feature, None)
            else:
                sample_output = {
                    key: value[flat_index:flat_index + 1]
                    for key, value in local_output.items()
                    if key in ("pred_boxes", "score_map", "size_map", "offset_map")
                }
            sample_outputs.append(sample_output)
            accepted_count += int(collaboration["accepted_count"])
            gate_logits.append(collaboration["gate_logits"])
            gate_labels.append(collaboration["gate_labels"])
            budget_losses.append(collaboration["residual_budget_loss"])

        collaborative_output = dict(local_output)
        for key in ("pred_boxes", "score_map", "size_map", "offset_map"):
            collaborative_output[key] = torch.cat(
                [sample[key] for sample in sample_outputs], dim=0)
        nonempty_logits = [value for value in gate_logits if value.numel()]
        nonempty_labels = [value for value in gate_labels if value.numel()]
        zero = search_tokens.sum() * 0.0
        collaborative_output["c3r"] = {
            "gate_logits": torch.cat(nonempty_logits) if nonempty_logits else zero.reshape(0),
            "gate_labels": torch.cat(nonempty_labels) if nonempty_labels else torch.empty(
                0, device=search_tokens.device, dtype=torch.long),
            "residual_budget_loss": torch.stack(budget_losses).mean() if budget_losses else zero,
            "accepted_count": accepted_count,
        }
        collaborative_output["pcum_flat_multiview"] = True
        collaborative_output["num_views"] = num_views
        return collaborative_output

    def _split_flat_prompts(self, prompt, num_views):
        if prompt is None:
            return None
        if prompt.shape[0] % num_views != 0:
            return None
        batch_size = prompt.shape[0] // num_views
        return [
            prompt[i * batch_size:(i + 1) * batch_size]
            for i in range(num_views)
        ]

    def _build_remote_inputs(self, data, remote_bank, target_view, num_views,
                             remote_state_bank=None, disable_dropout=False):
        drop_prob = float(self._get_cfg_value("TRAIN.PCUM.REMOTE_DROPOUT_PROB", 0.0))
        if (
            not disable_dropout
            and drop_prob > 0
            and torch.rand((), device=remote_bank[0].device).item() < drop_prob
        ):
            return None, None

        remote_prompts = []
        remote_masks = []
        remote_metric_states = []
        for remote_view in range(num_views):
            if remote_view == target_view:
                continue

            remote_prompt = remote_bank[remote_view]
            mask = self._view_prompt_valid_mask(
                data,
                remote_view,
                device=remote_prompt.device,
                dtype=remote_prompt.dtype,
            )
            remote_prompts.append(self._apply_prompt_mask(remote_prompt, mask))
            remote_masks.append(mask)
            remote_metric_states.append(
                None if remote_state_bank is None else remote_state_bank[remote_view]
            )

        remote_states = self._make_remote_state(
            remote_masks,
            device=remote_prompts[0].device if remote_prompts else None,
            dtype=remote_prompts[0].dtype if remote_prompts else None,
            metric_states=remote_metric_states,
        )
        return remote_prompts, remote_states

    def _build_flat_remote_inputs(self, data, remote_bank, num_views,
                                  remote_state_bank=None, disable_dropout=False):
        drop_prob = float(self._get_cfg_value("TRAIN.PCUM.REMOTE_DROPOUT_PROB", 0.0))
        if (
            not disable_dropout
            and drop_prob > 0
            and torch.rand((), device=remote_bank[0].device).item() < drop_prob
        ):
            return None, None

        num_remote = num_views - 1
        remote_slots = [[] for _ in range(num_remote)]
        mask_slots = [[] for _ in range(num_remote)]
        metric_slots = [dict() for _ in range(num_remote)]

        for target_view in range(num_views):
            remote_views = [view for view in range(num_views) if view != target_view]
            for slot, remote_view in enumerate(remote_views):
                remote_prompt = remote_bank[remote_view]
                mask = self._view_prompt_valid_mask(
                    data,
                    remote_view,
                    device=remote_prompt.device,
                    dtype=remote_prompt.dtype,
                )
                if mask is None:
                    mask = torch.ones(
                        remote_prompt.shape[0],
                        device=remote_prompt.device,
                        dtype=remote_prompt.dtype,
                    )
                remote_slots[slot].append(self._apply_prompt_mask(remote_prompt, mask))
                mask_slots[slot].append(mask)
                if remote_state_bank is not None:
                    for key, value in remote_state_bank[remote_view].items():
                        metric_slots[slot].setdefault(key, []).append(value)

        remote_prompts = [torch.cat(slot_prompts, dim=0) for slot_prompts in remote_slots]
        remote_masks = [torch.cat(slot_masks, dim=0) for slot_masks in mask_slots]
        remote_metric_states = None
        if remote_state_bank is not None:
            remote_metric_states = [
                {
                    key: torch.cat(values, dim=0).detach()
                    for key, values in slot.items()
                }
                for slot in metric_slots
            ]
        remote_states = self._make_remote_state(
            remote_masks,
            device=remote_prompts[0].device if remote_prompts else None,
            dtype=remote_prompts[0].dtype if remote_prompts else None,
            metric_states=remote_metric_states,
        )
        return remote_prompts, remote_states

    def forward_pass_real_multiview_pcum(self, net, data):
        """
        Use synchronized ThreeMDOT A/B/C views. First collect each view's local
        prompt, then run each view again with the other two prompts as remote.
        """
        num_views = min(self._num_views(data), 3)
        detach_remote = self._get_cfg_value("TRAIN.PCUM.DETACH_REAL_REMOTE", True)
        if detach_remote:
            with torch.no_grad():
                warm_output = self._forward_flat_views(
                    net, data, num_views, remote_prompts=None)
        else:
            warm_output = self._forward_flat_views(
                net, data, num_views, remote_prompts=None)

        local_prompt_flat = self._get_prompt_from_pred(warm_output)
        local_prompts = self._split_flat_prompts(local_prompt_flat, num_views)

        if local_prompts is None or any(prompt is None for prompt in local_prompts):
            warm_output["pcum_flat_multiview"] = True
            warm_output["num_views"] = num_views
            return warm_output

        remote_bank = [prompt.detach() if detach_remote else prompt for prompt in local_prompts]
        loss_mode = self._get_cfg_value("TRAIN.PCUM.REAL_MULTIVIEW_LOSS_MODE", "all_views")

        if loss_mode == "target_view":
            remote_prompts, remote_states = self._build_remote_inputs(
                data, remote_bank, target_view=0, num_views=num_views)
            target_out = self._forward_one_view(
                net, data, 0, remote_prompts=remote_prompts,
                remote_states=remote_states)
            target_out["pcum_real_multiview_target"] = True
            return target_out

        remote_prompts, remote_states = self._build_flat_remote_inputs(
            data, remote_bank, num_views=num_views)
        out = self._forward_flat_views(
            net, data, num_views, remote_prompts=remote_prompts,
            remote_states=remote_states)
        out["pcum_flat_multiview"] = True
        out["num_views"] = num_views
        return out

    def _mark_flat_multiview(self, output, num_views):
        output["pcum_flat_multiview"] = True
        output["num_views"] = int(num_views)
        return output

    def paired_local_stage(self, data):
        """Run the gradient-carrying local stage and cache detached references."""
        num_views = min(self._num_views(data), 3)
        if num_views < 3:
            raise ValueError("Paired PCUM supervision requires three UAV views")

        self._set_diagnostic_stage("local")
        local_output = self._mark_flat_multiview(
            self._forward_flat_views(self.net, data, num_views, remote_prompts=None),
            num_views,
        )
        _, local_status, local_components = self.compute_losses(
            local_output,
            data,
            include_auxiliary=False,
            return_components=True,
        )
        local_per_sample = self._flat_per_sample_tracking_loss(
            local_output, data, num_views)

        local_prompt_flat = self._get_prompt_from_pred(local_output)
        local_prompts = self._split_flat_prompts(local_prompt_flat, num_views)
        if local_prompts is None or any(prompt is None for prompt in local_prompts):
            raise RuntimeError("Local PCUM forward did not produce prompts")

        local_weight = float(self._get_cfg_value(
            "TRAIN.PCUM.LOCAL_LOSS_WEIGHT", 1.0))
        collab_weight = float(self._get_cfg_value(
            "TRAIN.PCUM.COLLAB_LOSS_WEIGHT", 1.0))
        pair_denominator = local_weight + collab_weight
        if pair_denominator <= 0:
            raise ValueError("LOCAL_LOSS_WEIGHT + COLLAB_LOSS_WEIGHT must be positive")

        if self._pcum_ranking_enabled():
            backward_loss = local_components["tracking"] * local_weight
        else:
            backward_loss = local_components["tracking"] * (local_weight / pair_denominator)
        cache = {
            "num_views": num_views,
            "remote_bank": [prompt.detach() for prompt in local_prompts],
            "remote_state_bank": self._predicted_remote_state_bank(
                local_output, num_views
            ),
            "local_per_sample": local_per_sample["total"].detach(),
            "local_tracking": local_components["tracking"].detach(),
            "local_status": local_status,
            "local_weight": local_weight,
            "collab_weight": collab_weight,
            "pair_denominator": pair_denominator,
        }
        return backward_loss, cache

    def _weighted_flat_tracking_mean(self, per_sample, num_views):
        num_views = int(num_views)
        if per_sample.numel() % num_views != 0:
            raise ValueError("Per-sample losses are not divisible by num_views")
        batch_size = per_sample.numel() // num_views
        view_weights = self._real_multiview_loss_weights(num_views, per_sample.device)
        loss = per_sample.sum() * 0.0
        for view_index, view_weight in enumerate(view_weights):
            start = view_index * batch_size
            end = (view_index + 1) * batch_size
            loss = loss + view_weight * per_sample[start:end].mean()
        return loss

    def _flat_visible_mask(self, data, num_views, device, total_count):
        visible = data.get("search_view_valid", None)
        if visible is None:
            return torch.ones(total_count, device=device, dtype=torch.bool)
        if isinstance(visible, list):
            visible = visible[:int(num_views)]
            if len(visible) == 0:
                return torch.ones(total_count, device=device, dtype=torch.bool)
            visible = torch.stack([
                torch.as_tensor(item).reshape(-1) for item in visible
            ], dim=0)
        elif not torch.is_tensor(visible):
            return torch.ones(total_count, device=device, dtype=torch.bool)

        visible = visible.to(device=device)
        if visible.dim() >= 3:
            visible = visible[..., 0]
        if visible.dim() == 1:
            visible = visible.view(1, -1).expand(int(num_views), -1)
        elif visible.dim() >= 2 and visible.shape[0] != int(num_views):
            if visible.shape[1] == int(num_views):
                visible = visible.transpose(0, 1)
        visible = visible[:int(num_views)].contiguous().view(-1).bool()
        if visible.numel() != int(total_count):
            return torch.ones(total_count, device=device, dtype=torch.bool)
        return visible

    def _make_zero_remote_prompts(self, remote_prompts):
        if remote_prompts is None:
            return None
        return [torch.zeros_like(prompt) for prompt in remote_prompts]

    def _make_delay_remote_bank(self, remote_bank):
        mode = str(self._get_cfg_value(
            "TRAIN.PCUM.DELAY_BRANCH_MODE", "batch_roll")).lower()
        if mode != "batch_roll":
            raise ValueError("Unsupported DELAY_BRANCH_MODE: %s" % mode)
        delayed = []
        for prompt in remote_bank:
            if prompt.shape[0] > 1:
                delayed.append(torch.roll(prompt, shifts=1, dims=0))
            else:
                delayed.append(torch.roll(prompt, shifts=1, dims=1))
        return delayed

    def _run_paired_remote_branch(self, data, num_views, remote_prompts,
                                  remote_states, stage_name,
                                  remote_suppression_override=None):
        self._set_diagnostic_stage(stage_name)
        output = self._mark_flat_multiview(
            self._forward_flat_views(
                self.net,
                data,
                num_views,
                remote_prompts=remote_prompts,
                remote_states=remote_states,
                remote_suppression_override=remote_suppression_override,
            ),
            num_views,
        )
        per_sample = self._flat_per_sample_tracking_loss(
            output, data, num_views)["total"]
        return output, per_sample

    def compute_ranking_loss(self, raw_per_sample, reference_per_sample,
                             num_views, margin=0.0, active_mask=None,
                             detach_reference=False):
        if raw_per_sample.shape != reference_per_sample.shape:
            raise ValueError("Ranking per-sample losses must match")
        if detach_reference:
            reference_per_sample = reference_per_sample.detach()
        delta = raw_per_sample - reference_per_sample
        if active_mask is None:
            active_mask = torch.ones_like(delta, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=delta.device, dtype=torch.bool)
        num_views = int(num_views)
        if delta.numel() % num_views != 0:
            raise ValueError("Per-sample losses are not divisible by num_views")
        batch_size = delta.numel() // num_views
        view_weights = self._real_multiview_loss_weights(num_views, delta.device)
        rank_loss = delta.sum() * 0.0
        for view_index, view_weight in enumerate(view_weights):
            start = view_index * batch_size
            end = (view_index + 1) * batch_size
            view_mask = active_mask[start:end]
            if bool(view_mask.any().item()):
                view_loss = torch.relu(
                    delta[start:end][view_mask] + float(margin)
                ).mean()
                rank_loss = rank_loss + view_weight * view_loss

        active_delta = delta[active_mask]
        if active_delta.numel() == 0:
            stats = {
                "raw_better_ratio": 0.0,
                "delta_mean": 0.0,
                "delta_std": 0.0,
                "delta_min": 0.0,
                "delta_max": 0.0,
                "active_count": 0.0,
            }
        else:
            detached_delta = active_delta.detach().float()
            stats = {
                "raw_better_ratio": float((detached_delta < 0).float().mean().item()),
                "delta_mean": float(detached_delta.mean().item()),
                "delta_std": float(detached_delta.std(unbiased=False).item()),
                "delta_min": float(detached_delta.min().item()),
                "delta_max": float(detached_delta.max().item()),
                "active_count": float(detached_delta.numel()),
            }
        return rank_loss, stats

    def compute_remote_suppression_label(self, a0_per_sample, local_per_sample,
                                         margin=0.0, active_mask=None):
        if a0_per_sample.shape != local_per_sample.shape:
            raise ValueError("A0 and local per-sample losses must match")
        label = (
            a0_per_sample.detach()
            > local_per_sample.detach() + float(margin)
        ).to(dtype=a0_per_sample.dtype)
        if active_mask is None:
            return label, torch.ones_like(label, dtype=torch.bool)
        mask = active_mask.to(device=label.device, dtype=torch.bool)
        return label, mask

    def _d2_suppression_loss(self, collaborative_output, a0_per_sample,
                             local_per_sample, active_mask):
        pcum_out = collaborative_output.get("pcum", {})
        suppress = pcum_out.get("remote_suppression", None)
        if not torch.is_tensor(suppress):
            raise RuntimeError("D2 remote suppression output is missing")
        suppress = suppress.reshape(-1)
        label, mask = self.compute_remote_suppression_label(
            a0_per_sample,
            local_per_sample,
            margin=self._pcum_suppress_label_margin(),
            active_mask=active_mask,
        )
        if suppress.numel() != label.numel():
            raise ValueError("Suppression output and label size mismatch")
        if bool(mask.any().item()):
            bce = F.binary_cross_entropy(
                suppress[mask].clamp(1e-6, 1.0 - 1e-6),
                label[mask],
            )
            suppress_mean_loss = suppress[mask].mean()
            label_ratio = label[mask].detach().float().mean()
        else:
            bce = suppress.sum() * 0.0
            suppress_mean_loss = suppress.sum() * 0.0
            label_ratio = torch.zeros((), device=suppress.device)

        detached = suppress.detach().float()
        if detached.numel():
            quantiles = torch.quantile(
                detached.reshape(-1),
                torch.tensor(
                    [0.1, 0.5, 0.9],
                    device=detached.device,
                    dtype=detached.dtype,
                ),
            )
            suppress_p10, suppress_p50, suppress_p90 = (
                float(value.item()) for value in quantiles
            )
        else:
            suppress_p10 = suppress_p50 = suppress_p90 = 0.0
        stats = {
            "suppress_mean": float(detached.mean().item()) if detached.numel() else 0.0,
            "suppress_std": float(detached.std(unbiased=False).item()) if detached.numel() else 0.0,
            "suppress_min": float(detached.min().item()) if detached.numel() else 0.0,
            "suppress_max": float(detached.max().item()) if detached.numel() else 0.0,
            "suppress_p10": suppress_p10,
            "suppress_p50": suppress_p50,
            "suppress_p90": suppress_p90,
            "effective_remote_retention": (
                float((1.0 - detached).mean().item())
                if detached.numel() else 0.0
            ),
            "suppress_label_ratio": float(label_ratio.detach().float().item()),
            "suppress_active_ratio": float(
                (detached > 0.5).float().mean().item()) if detached.numel() else 0.0,
        }
        for src_key, dst_key in (
            ("remote_delta_norm", "remote_delta_norm"),
            ("suppressed_delta_norm", "suppressed_delta_norm"),
            ("remote_suppression_active_ratio", "remote_suppression_active_ratio"),
        ):
            value = pcum_out.get(src_key, None)
            if torch.is_tensor(value):
                stats[dst_key] = float(value.detach().float().mean().item())
        return bce, suppress_mean_loss, stats

    def paired_collaborative_stage_d2(self, data, cache):
        """D2-G0: train only a suppression gate over frozen local/A0 features."""
        num_views = int(cache["num_views"])
        remote_prompts, remote_states = self._build_flat_remote_inputs(
            data,
            cache["remote_bank"],
            num_views=num_views,
            remote_state_bank=cache.get("remote_state_bank", None),
        )
        remote_active = remote_prompts is not None

        self._restore_paired_forward_rng(clear=False)
        with torch.no_grad():
            a0_output, a0_per_sample = self._run_paired_remote_branch(
                data,
                num_views,
                remote_prompts,
                remote_states,
                "a0_reference",
                remote_suppression_override=0.0,
            )
            a0_tracking = self._weighted_flat_tracking_mean(
                a0_per_sample, num_views)

        self._restore_paired_forward_rng(clear=False)
        self._set_diagnostic_stage("collaborative")
        collaborative_output = self._mark_flat_multiview(
            self._forward_flat_views(
                self.net,
                data,
                num_views,
                remote_prompts=remote_prompts,
                remote_states=remote_states,
            ),
            num_views,
        )
        _, collab_status, collab_components = self.compute_losses(
            collaborative_output,
            data,
            include_auxiliary=True,
            return_components=True,
        )
        collab_per_sample = self._flat_per_sample_tracking_loss(
            collaborative_output, data, num_views)["total"]

        remote_mask = torch.full_like(
            collab_per_sample,
            bool(remote_active),
            dtype=torch.bool,
        )
        visible_mask = self._flat_visible_mask(
            data, num_views, collab_per_sample.device, collab_per_sample.numel())
        loss_active_mask = remote_mask & visible_mask

        safe_loss, safe_stats = self.compute_safe_loss(
            collaborative_per_sample=collab_per_sample,
            local_per_sample=cache["local_per_sample"],
            num_views=num_views,
            margin=self._pcum_safe_margin(),
            hard_sample_quantile=0.0,
            active_mask=loss_active_mask,
        )

        rank_zero_loss = collab_per_sample.sum() * 0.0
        rank_zero_stats = None
        zero_tracking = None
        if remote_active:
            zero_prompts = self._make_zero_remote_prompts(remote_prompts)
            self._restore_paired_forward_rng(clear=False)
            with torch.no_grad():
                _, zero_per_sample = self._run_paired_remote_branch(
                    data,
                    num_views,
                    zero_prompts,
                    remote_states,
                    "zero",
                    remote_suppression_override=0.0,
                )
                zero_tracking = self._weighted_flat_tracking_mean(
                    zero_per_sample, num_views)
            rank_zero_loss, rank_zero_stats = self.compute_ranking_loss(
                collab_per_sample,
                zero_per_sample,
                num_views,
                margin=self._pcum_rank_zero_margin(),
                active_mask=loss_active_mask,
                detach_reference=True,
            )

        suppress_bce, suppress_mean_loss, suppress_stats = (
            self._d2_suppression_loss(
                collaborative_output,
                a0_per_sample,
                cache["local_per_sample"],
                loss_active_mask,
            )
        )

        self._paired_cpu_rng_state = None
        self._paired_cuda_rng_state = None
        self._paired_cuda_device = None

        safe_weight = self._pcum_safe_weight()
        rank_zero_weight = self._pcum_rank_zero_weight()
        suppress_bce_weight = self._pcum_suppress_bce_weight()
        suppress_mean_weight = self._pcum_suppress_mean_weight()
        backward_loss = (
            collab_components["tracking"]
            + safe_loss * safe_weight
            + rank_zero_loss * rank_zero_weight
            + suppress_bce * suppress_bce_weight
            + suppress_mean_loss * suppress_mean_weight
            + collab_components["auxiliary"]
        )
        pair_loss = (
            cache["local_weight"] * cache["local_tracking"]
            + cache["collab_weight"] * collab_components["tracking"].detach()
        )
        total_for_log = (
            pair_loss
            + safe_loss.detach() * safe_weight
            + rank_zero_loss.detach() * rank_zero_weight
            + suppress_bce.detach() * suppress_bce_weight
            + suppress_mean_loss.detach() * suppress_mean_weight
            + collab_components["auxiliary"].detach()
        )

        local_status = cache["local_status"]
        status = {
            "Loss/total": float(total_for_log.item()),
            "Loss/pair_tracking": float(pair_loss.item()),
            "Loss/local_tracking": float(cache["local_tracking"].item()),
            "Loss/a0_tracking": float(a0_tracking.detach().item()),
            "Loss/output_tracking": float(
                collab_components["tracking"].detach().item()),
            "Loss/collaborative_tracking": float(
                collab_components["tracking"].detach().item()),
            "Loss/zero_tracking": 0.0 if zero_tracking is None else float(
                zero_tracking.detach().item()),
            "Loss/safe": float(safe_loss.detach().item()),
            "Loss/rank_zero": float(rank_zero_loss.detach().item()),
            "Loss/remote_suppression_bce": float(suppress_bce.detach().item()),
            "Loss/remote_suppression_mean": float(suppress_mean_loss.detach().item()),
            "Loss/local_giou": float(local_status["Loss/giou"]),
            "Loss/local_l1": float(local_status["Loss/l1"]),
            "Loss/local_focal": float(local_status["Loss/location"]),
            "Loss/collaborative_giou": float(collab_status["Loss/giou"]),
            "Loss/collaborative_l1": float(collab_status["Loss/l1"]),
            "Loss/collaborative_focal": float(collab_status["Loss/location"]),
            "PCUM/d2_remote_suppression_only": 1.0,
            "PCUM/remote_active": float(remote_active),
            "PCUM/ranking_visible_only": 1.0,
            "PCUM/visible_ratio": float(visible_mask.float().mean().detach().item()),
            "PCUM/visible_ranking_sample_count": float(
                loss_active_mask.float().sum().detach().item()),
            "PCUM/invisible_sample_count": float(
                (~visible_mask).float().sum().detach().item()),
            "PCUM/raw_better_than_zero_visible_ratio": (
                0.0 if rank_zero_stats is None else rank_zero_stats["raw_better_ratio"]
            ),
            "PCUM/raw_better_than_local_visible_ratio": safe_stats[
                "collaborative_better_ratio"],
            "PCUM/safe_active_visible_ratio": safe_stats["safe_active_ratio"],
            "PCUM/suppress_mean": suppress_stats.get("suppress_mean", 0.0),
            "PCUM/suppress_std": suppress_stats.get("suppress_std", 0.0),
            "PCUM/suppress_min": suppress_stats.get("suppress_min", 0.0),
            "PCUM/suppress_max": suppress_stats.get("suppress_max", 0.0),
            "PCUM/suppress_p10": suppress_stats.get("suppress_p10", 0.0),
            "PCUM/suppress_p50": suppress_stats.get("suppress_p50", 0.0),
            "PCUM/suppress_p90": suppress_stats.get("suppress_p90", 0.0),
            "PCUM/effective_remote_retention": suppress_stats.get(
                "effective_remote_retention", 0.0),
            "PCUM/suppress_label_ratio": suppress_stats.get("suppress_label_ratio", 0.0),
            "PCUM/suppression_active_ratio": suppress_stats.get("suppress_active_ratio", 0.0),
            "PCUM/remote_delta_norm": suppress_stats.get("remote_delta_norm", 0.0),
            "PCUM/suppressed_delta_norm": suppress_stats.get("suppressed_delta_norm", 0.0),
            "PCUM/safe_margin": self._pcum_safe_margin(),
            "PCUM/rank_zero_margin": self._pcum_rank_zero_margin(),
            "PCUM/suppress_label_margin": self._pcum_suppress_label_margin(),
            "flops": float(collab_status.get("flops", 0.0)),
            "flops_actual": float(collab_status.get("flops_actual", 0.0)),
            "flops_target": float(collab_status.get("flops_target", self.F_target)),
            "flops_weight": float(collab_status.get("flops_weight", self.flops_weight)),
            "loss_prompt_align": float(collab_status.get("loss_prompt_align", 0.0)),
            "pcum_real_multiview": 1.0,
            "pcum_num_views": float(num_views),
        }
        view_weights = self._real_multiview_loss_weights(
            num_views, collab_per_sample.device)
        for view_index, view_weight in enumerate(view_weights):
            status["pcum_view_weight_%d" % view_index] = float(
                view_weight.detach().item())

        pcum_diag = collaborative_output.get("pcum", {}).get(
            "remote_aggregation_diagnostics", None)
        if isinstance(pcum_diag, dict):
            for src_key, dst_key in (
                ("remote_weight_entropy", "PCUM/remote_weight_entropy"),
                ("remote_weight_max", "PCUM/remote_weight_max"),
                ("remote_weight_mean", "PCUM/remote_weight_mean"),
                ("valid_remote_count", "PCUM/valid_remote_count"),
                ("remote_quality_mean", "PCUM/remote_quality_mean"),
                ("remote_quality_min", "PCUM/remote_quality_min"),
                ("remote_quality_max", "PCUM/remote_quality_max"),
            ):
                value = pcum_diag.get(src_key, None)
                if torch.is_tensor(value):
                    status[dst_key] = float(value.detach().float().mean().item())
                elif value is not None:
                    status[dst_key] = float(value)
        return backward_loss, status

    def paired_collaborative_stage(self, data, cache):
        """Run collaborative supervision using only detached remote prompts."""
        if self._pcum_remote_suppression_only():
            return self.paired_collaborative_stage_d2(data, cache)

        num_views = int(cache["num_views"])
        remote_prompts, remote_states = self._build_flat_remote_inputs(
            data,
            cache["remote_bank"],
            num_views=num_views,
            remote_state_bank=cache.get("remote_state_bank", None),
        )
        remote_active = remote_prompts is not None
        self._restore_paired_forward_rng(clear=False)

        self._set_diagnostic_stage("collaborative")
        collaborative_output = self._mark_flat_multiview(
            self._forward_flat_views(
                self.net,
                data,
                num_views,
                remote_prompts=remote_prompts,
                remote_states=remote_states,
            ),
            num_views,
        )
        _, collab_status, collab_components = self.compute_losses(
            collaborative_output,
            data,
            include_auxiliary=True,
            return_components=True,
        )
        collab_per_sample = self._flat_per_sample_tracking_loss(
            collaborative_output, data, num_views)["total"]

        ranking_enabled = self._pcum_ranking_enabled()
        remote_mask = torch.full_like(
            collab_per_sample,
            bool(remote_active),
            dtype=torch.bool,
        )
        visible_mask = self._flat_visible_mask(
            data, num_views, collab_per_sample.device, collab_per_sample.numel())
        visible_only = bool(ranking_enabled and self._pcum_visible_only_ranking())
        loss_active_mask = remote_mask & visible_mask if visible_only else remote_mask
        safe_loss, safe_stats = self.compute_safe_loss(
            collaborative_per_sample=collab_per_sample,
            local_per_sample=cache["local_per_sample"],
            num_views=num_views,
            margin=self._pcum_safe_margin(),
            hard_sample_quantile=float(self._get_cfg_value(
                "TRAIN.PCUM.SAFE_HARD_SAMPLE_QUANTILE", 0.0)),
            active_mask=loss_active_mask,
        )

        collab_factor = (
            cache["collab_weight"]
            if ranking_enabled
            else cache["collab_weight"] / cache["pair_denominator"]
        )
        rank_zero_loss = collab_per_sample.sum() * 0.0
        rank_delay_loss = collab_per_sample.sum() * 0.0
        rank_local_loss = collab_per_sample.sum() * 0.0
        rank_zero_stats = None
        rank_delay_stats = None
        rank_local_stats = None
        zero_tracking = None
        delay_tracking = None

        if ranking_enabled and remote_active:
            zero_prompts = self._make_zero_remote_prompts(remote_prompts)
            self._restore_paired_forward_rng(clear=False)
            _, zero_per_sample = self._run_paired_remote_branch(
                data, num_views, zero_prompts, remote_states, "zero")
            zero_tracking = self._weighted_flat_tracking_mean(
                zero_per_sample, num_views)
            delay_bank = self._make_delay_remote_bank(cache["remote_bank"])
            delay_prompts, delay_states = self._build_flat_remote_inputs(
                data,
                delay_bank,
                num_views=num_views,
                remote_state_bank=cache.get("remote_state_bank", None),
                disable_dropout=True,
            )
            self._restore_paired_forward_rng(clear=False)
            _, delay_per_sample = self._run_paired_remote_branch(
                data, num_views, delay_prompts, delay_states, "delay")
            delay_tracking = self._weighted_flat_tracking_mean(
                delay_per_sample, num_views)
            rank_zero_loss, rank_zero_stats = self.compute_ranking_loss(
                collab_per_sample,
                zero_per_sample,
                num_views,
                margin=self._pcum_rank_zero_margin(),
                active_mask=loss_active_mask,
            )
            rank_delay_loss, rank_delay_stats = self.compute_ranking_loss(
                collab_per_sample,
                delay_per_sample,
                num_views,
                margin=float(self._get_cfg_value(
                    "TRAIN.PCUM.RANK_DELAY_MARGIN", 0.02)),
                active_mask=loss_active_mask,
            )
            rank_local_loss, rank_local_stats = self.compute_ranking_loss(
                collab_per_sample,
                cache["local_per_sample"],
                num_views,
                margin=self._pcum_rank_local_margin(),
                active_mask=loss_active_mask,
            )
        elif ranking_enabled:
            rank_local_loss, rank_local_stats = self.compute_ranking_loss(
                collab_per_sample,
                cache["local_per_sample"],
                num_views,
                margin=self._pcum_rank_local_margin(),
                active_mask=loss_active_mask,
            )

        self._paired_cpu_rng_state = None
        self._paired_cuda_rng_state = None
        self._paired_cuda_device = None

        rank_zero_weight = self._pcum_rank_zero_weight()
        rank_delay_weight = self._pcum_rank_delay_weight()
        rank_local_weight = self._pcum_rank_local_weight()
        safe_weight = self._pcum_safe_weight()
        backward_loss = (
            collab_components["tracking"] * collab_factor
            + safe_loss * safe_weight
            + rank_zero_loss * rank_zero_weight
            + rank_delay_loss * rank_delay_weight
            + rank_local_loss * rank_local_weight
            + collab_components["auxiliary"]
        )
        if ranking_enabled:
            pair_loss = (
                cache["local_weight"] * cache["local_tracking"]
                + cache["collab_weight"] * collab_components["tracking"].detach()
            )
        else:
            pair_loss = (
                cache["local_weight"] * cache["local_tracking"]
                + cache["collab_weight"] * collab_components["tracking"].detach()
            ) / cache["pair_denominator"]
        total_for_log = (
            pair_loss
            + safe_loss.detach() * safe_weight
            + rank_zero_loss.detach() * rank_zero_weight
            + rank_delay_loss.detach() * rank_delay_weight
            + rank_local_loss.detach() * rank_local_weight
            + collab_components["auxiliary"].detach()
        )

        local_status = cache["local_status"]
        status = {
            "Loss/total": float(total_for_log.item()),
            "Loss/pair_tracking": float(pair_loss.item()),
            "Loss/local_tracking": float(cache["local_tracking"].item()),
            "Loss/collaborative_tracking": float(
                collab_components["tracking"].detach().item()),
            "Loss/zero_tracking": 0.0 if zero_tracking is None else float(
                zero_tracking.detach().item()),
            "Loss/delay_tracking": 0.0 if delay_tracking is None else float(
                delay_tracking.detach().item()),
            "Loss/safe": float(safe_loss.detach().item()),
            "Loss/rank_zero": float(rank_zero_loss.detach().item()),
            "Loss/rank_delay": float(rank_delay_loss.detach().item()),
            "Loss/rank_local": float(rank_local_loss.detach().item()),
            "Loss/local_giou": float(local_status["Loss/giou"]),
            "Loss/local_l1": float(local_status["Loss/l1"]),
            "Loss/local_focal": float(local_status["Loss/location"]),
            "Loss/collaborative_giou": float(collab_status["Loss/giou"]),
            "Loss/collaborative_l1": float(collab_status["Loss/l1"]),
            "Loss/collaborative_focal": float(collab_status["Loss/location"]),
            "PCUM/collaborative_better_ratio": safe_stats["collaborative_better_ratio"],
            "PCUM/raw_better_than_zero_ratio": (
                0.0 if rank_zero_stats is None else rank_zero_stats["raw_better_ratio"]
            ),
            "PCUM/raw_better_than_delay_ratio": (
                0.0 if rank_delay_stats is None else rank_delay_stats["raw_better_ratio"]
            ),
            "PCUM/raw_better_than_local_ratio": (
                0.0 if rank_local_stats is None else rank_local_stats["raw_better_ratio"]
            ),
            "PCUM/loss_delta_mean": safe_stats["delta_mean"],
            "PCUM/loss_delta_std": safe_stats["delta_std"],
            "PCUM/loss_delta_min": safe_stats["delta_min"],
            "PCUM/loss_delta_max": safe_stats["delta_max"],
            "PCUM/remote_active": float(remote_active),
            "PCUM/remote_dropout": float(not remote_active),
            "PCUM/safe_margin": self._pcum_safe_margin(),
            "PCUM/ranking_enabled": float(ranking_enabled),
            "PCUM/ranking_visible_only": float(visible_only),
            "PCUM/visible_ratio": float(visible_mask.float().mean().detach().item()),
            "PCUM/visible_ranking_sample_count": float(
                loss_active_mask.float().sum().detach().item()),
            "PCUM/invisible_sample_count": float(
                (~visible_mask).float().sum().detach().item()),
            "Loss/rank_zero_visible": float(rank_zero_loss.detach().item()) if visible_only else 0.0,
            "Loss/rank_local_visible": float(rank_local_loss.detach().item()) if visible_only else 0.0,
            "Loss/safe_visible": float(safe_loss.detach().item()) if visible_only else 0.0,
            "PCUM/raw_better_than_zero_visible_ratio": (
                0.0 if (not visible_only or rank_zero_stats is None) else rank_zero_stats["raw_better_ratio"]
            ),
            "PCUM/raw_better_than_local_visible_ratio": (
                0.0 if (not visible_only or rank_local_stats is None) else rank_local_stats["raw_better_ratio"]
            ),
            "PCUM/safe_active_visible_ratio": (
                0.0 if not visible_only else safe_stats["safe_active_ratio"]
            ),
            "PCUM/rank_zero_margin": self._pcum_rank_zero_margin(),
            "PCUM/rank_delay_margin": float(self._get_cfg_value(
                "TRAIN.PCUM.RANK_DELAY_MARGIN", 0.02)),
            "PCUM/rank_local_margin": self._pcum_rank_local_margin(),
            "flops": float(collab_status.get("flops", 0.0)),
            "flops_actual": float(collab_status.get("flops_actual", 0.0)),
            "flops_target": float(collab_status.get("flops_target", self.F_target)),
            "flops_weight": float(collab_status.get("flops_weight", self.flops_weight)),
            "loss_prompt_align": float(collab_status.get("loss_prompt_align", 0.0)),
            "pcum_real_multiview": 1.0,
            "pcum_num_views": float(num_views),
        }
        view_weights = self._real_multiview_loss_weights(
            num_views, collab_per_sample.device)
        for view_index, view_weight in enumerate(view_weights):
            status["pcum_view_weight_%d" % view_index] = float(
                view_weight.detach().item())
        pcum_diag = collaborative_output.get("pcum", {}).get(
            "remote_aggregation_diagnostics", None)
        if isinstance(pcum_diag, dict):
            for src_key, dst_key in (
                ("remote_weight_entropy", "PCUM/remote_weight_entropy"),
                ("remote_weight_max", "PCUM/remote_weight_max"),
                ("remote_weight_mean", "PCUM/remote_weight_mean"),
                ("valid_remote_count", "PCUM/valid_remote_count"),
                ("remote_quality_mean", "PCUM/remote_quality_mean"),
                ("remote_quality_min", "PCUM/remote_quality_min"),
                ("remote_quality_max", "PCUM/remote_quality_max"),
            ):
                value = pcum_diag.get(src_key, None)
                if torch.is_tensor(value):
                    status[dst_key] = float(value.detach().float().mean().item())
                elif value is not None:
                    status[dst_key] = float(value)
        return backward_loss, status

    def _single_view_per_sample_tracking_loss(self, pred_dict, gt_dict, view_index):
        gt_bbox = self._squeeze_if_needed(
            self._select_view(gt_dict["search_anno"], view_index))
        gt_maps = self._make_gt_heatmap(gt_bbox).to(pred_dict["score_map"].device)
        pred_boxes = pred_dict["pred_boxes"]
        batch_size, num_queries = pred_boxes.shape[:2]
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = (
            box_xywh_to_xyxy(gt_bbox)[:, None, :]
            .repeat((1, num_queries, 1))
            .view(-1, 4)
            .clamp(min=0.0, max=1.0)
            .to(pred_boxes_vec.device)
        )

        giou = giou_loss_details(pred_boxes_vec, gt_boxes_vec, batch_size)
        l1 = l1_loss_details(pred_boxes_vec, gt_boxes_vec, batch_size)
        focal_objective = self.objective["focal"]
        if not hasattr(focal_objective, "loss_details"):
            raise TypeError("Paired supervision requires FocalLoss.loss_details()")
        focal = focal_objective.loss_details(pred_dict["score_map"], gt_maps)
        total = (
            self.loss_weight["giou"] * giou["per_sample"]
            + self.loss_weight["l1"] * l1["per_sample"]
            + self.loss_weight["focal"] * focal["per_sample"]
        )
        return {
            "total": total,
            "giou": giou["per_sample"],
            "l1": l1["per_sample"],
            "focal": focal["per_sample"],
        }

    def _flat_per_sample_tracking_loss(self, pred_dict, gt_dict, num_views):
        total_batch = pred_dict["pred_boxes"].shape[0]
        batch_size = total_batch // int(num_views)
        per_view = []
        for view_index in range(int(num_views)):
            start = view_index * batch_size
            end = (view_index + 1) * batch_size
            per_view.append(self._single_view_per_sample_tracking_loss(
                self._slice_flat_pred(pred_dict, start, end),
                gt_dict,
                view_index,
            ))
        return {
            key: torch.cat([view[key] for view in per_view], dim=0)
            for key in ("total", "giou", "l1", "focal")
        }

    def compute_safe_loss(self, collaborative_per_sample, local_per_sample,
                          num_views, margin=0.0, hard_sample_quantile=0.0,
                          active_mask=None):
        if collaborative_per_sample.shape != local_per_sample.shape:
            raise ValueError("Local and collaborative per-sample losses must match")
        local_reference = local_per_sample.detach()
        delta = collaborative_per_sample - local_reference
        if active_mask is None:
            active_mask = torch.ones_like(delta, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=delta.device, dtype=torch.bool)

        num_views = int(num_views)
        if delta.numel() % num_views != 0:
            raise ValueError("Per-sample losses are not divisible by num_views")
        batch_size = delta.numel() // num_views
        view_weights = self._real_multiview_loss_weights(num_views, delta.device)
        safe_loss = delta.sum() * 0.0
        quantile = min(max(float(hard_sample_quantile), 0.0), 1.0)
        for view_index in range(num_views):
            start = view_index * batch_size
            end = (view_index + 1) * batch_size
            view_mask = active_mask[start:end]
            if quantile > 0.0 and bool(view_mask.any().item()):
                threshold = torch.quantile(local_reference[start:end][view_mask], quantile)
                view_mask = view_mask & (local_reference[start:end] >= threshold)
            if bool(view_mask.any().item()):
                view_safe = torch.relu(delta[start:end][view_mask] + float(margin)).mean()
                safe_loss = safe_loss + view_weights[view_index] * view_safe

        active_delta = delta[active_mask]
        if active_delta.numel() == 0:
            stats = {
                "collaborative_better_ratio": 0.0,
                "safe_active_ratio": 0.0,
                "delta_mean": 0.0,
                "delta_std": 0.0,
                "delta_min": 0.0,
                "delta_max": 0.0,
                "active_count": 0.0,
            }
        else:
            detached_delta = active_delta.detach().float()
            stats = {
                "collaborative_better_ratio": float((detached_delta < 0).float().mean().item()),
                "safe_active_ratio": float(
                    (torch.relu(detached_delta + float(margin)) > 0).float().mean().item()
                ),
                "delta_mean": float(detached_delta.mean().item()),
                "delta_std": float(detached_delta.std(unbiased=False).item()),
                "delta_min": float(detached_delta.min().item()),
                "delta_max": float(detached_delta.max().item()),
                "active_count": float(detached_delta.numel()),
            }
        return safe_loss, stats

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------
    def _slice_batch_value(self, value, start, end):
        if torch.is_tensor(value):
            if value.dim() > 0 and value.shape[0] >= end:
                return value[start:end]
            return value

        if isinstance(value, list):
            return [self._slice_batch_value(item, start, end) for item in value]

        if isinstance(value, tuple):
            return tuple(self._slice_batch_value(item, start, end) for item in value)

        if isinstance(value, dict):
            return {
                key: self._slice_batch_value(item, start, end)
                for key, item in value.items()
            }

        return value

    def _slice_flat_pred(self, pred_dict, start, end):
        sliced = {}
        skip_keys = {"pcum_flat_multiview", "num_views"}
        for key, value in pred_dict.items():
            if key in skip_keys:
                continue
            sliced[key] = self._slice_batch_value(value, start, end)
        return sliced

    def _compute_flat_multiview_losses(self, pred_dict, gt_dict, return_status=True,
                                       include_auxiliary=True,
                                       return_components=False):
        num_views = int(pred_dict.get("num_views", 1))
        if num_views <= 1:
            pred_dict = dict(pred_dict)
            pred_dict.pop("pcum_flat_multiview", None)
            pred_dict.pop("num_views", None)
            return self.compute_losses(
                pred_dict,
                gt_dict,
                return_status=return_status,
                include_auxiliary=include_auxiliary,
                return_components=return_components,
            )

        total_batch = pred_dict["pred_boxes"].shape[0]
        if total_batch % num_views != 0:
            raise ValueError(
                "Flattened multi-view batch size %d is not divisible by num_views %d"
                % (total_batch, num_views)
            )

        batch_size = total_batch // num_views
        view_weights = None
        total_loss = None
        component_totals = None
        status_acc = {}

        for view_index in range(num_views):
            start = view_index * batch_size
            end = (view_index + 1) * batch_size
            view_pred = self._slice_flat_pred(pred_dict, start, end)
            view_loss, view_status, view_components = self.compute_losses(
                view_pred,
                gt_dict,
                return_status=True,
                view_index=view_index,
                include_auxiliary=include_auxiliary,
                return_components=True,
            )

            if view_weights is None:
                view_weights = self._real_multiview_loss_weights(
                    num_views,
                    view_loss.device,
                )

            weighted_view_loss = view_loss * view_weights[view_index]
            total_loss = weighted_view_loss if total_loss is None else total_loss + weighted_view_loss
            if component_totals is None:
                component_totals = {
                    key: value * view_weights[view_index]
                    for key, value in view_components.items()
                }
            else:
                for key, value in view_components.items():
                    component_totals[key] = (
                        component_totals[key] + value * view_weights[view_index]
                    )

            for key, value in view_status.items():
                status_acc[key] = status_acc.get(key, 0.0) + float(value)

        if not return_status:
            if return_components:
                return total_loss, component_totals
            return total_loss

        status = {key: value / float(num_views) for key, value in status_acc.items()}
        status["pcum_real_multiview"] = 1.0
        status["pcum_num_views"] = float(num_views)
        if view_weights is not None:
            for i, weight in enumerate(view_weights):
                status["pcum_view_weight_%d" % i] = float(weight.detach().item())
        if return_components:
            return total_loss, status, component_totals
        return total_loss, status

    def compute_losses(self, pred_dict, gt_dict, return_status=True, view_index=0,
                       include_auxiliary=True, return_components=False):
        if isinstance(pred_dict, dict) and pred_dict.get("pcum_flat_multiview", False):
            return self._compute_flat_multiview_losses(
                pred_dict,
                gt_dict,
                return_status=return_status,
                include_auxiliary=include_auxiliary,
                return_components=return_components,
            )

        if isinstance(pred_dict, dict) and "multi_view" in pred_dict:
            total_loss = None
            status_acc = {}
            num_views = len(pred_dict["multi_view"])
            view_weights = None
            component_totals = None
            for i, view_pred in enumerate(pred_dict["multi_view"]):
                view_loss, view_status, view_components = self.compute_losses(
                    view_pred,
                    gt_dict,
                    return_status=True,
                    view_index=i,
                    include_auxiliary=include_auxiliary,
                    return_components=True,
                )
                if view_weights is None:
                    view_weights = self._real_multiview_loss_weights(
                        num_views,
                        view_loss.device,
                    )
                weighted_view_loss = view_loss * view_weights[i]
                total_loss = weighted_view_loss if total_loss is None else total_loss + weighted_view_loss
                if component_totals is None:
                    component_totals = {
                        key: value * view_weights[i]
                        for key, value in view_components.items()
                    }
                else:
                    for key, value in view_components.items():
                        component_totals[key] = component_totals[key] + value * view_weights[i]
                for key, value in view_status.items():
                    status_acc[key] = status_acc.get(key, 0.0) + float(value)

            status = {key: value / float(num_views) for key, value in status_acc.items()}
            status["pcum_real_multiview"] = 1.0
            status["pcum_num_views"] = float(num_views)
            if view_weights is not None:
                for i, weight in enumerate(view_weights):
                    status["pcum_view_weight_%d" % i] = float(weight.detach().item())
            if return_components:
                return total_loss, status, component_totals
            return total_loss, status

        gt_bbox = self._select_view(gt_dict["search_anno"], view_index)
        gt_bbox = self._squeeze_if_needed(gt_bbox)

        gt_gaussian_maps = self._make_gt_heatmap(gt_bbox).to(
            pred_dict["score_map"].device
        )

        pred_boxes = pred_dict["pred_boxes"]

        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        num_queries = pred_boxes.size(1)

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)

        gt_boxes_vec = (
            box_xywh_to_xyxy(gt_bbox)
            [:, None, :]
            .repeat((1, num_queries, 1))
            .view(-1, 4)
            .clamp(min=0.0, max=1.0)
            .to(pred_boxes_vec.device)
        )

        try:
            giou_loss, iou = self.objective["giou"](
                pred_boxes_vec,
                gt_boxes_vec
            )
        except Exception:
            device = pred_boxes.device
            giou_loss = torch.tensor(0.0, device=device)
            iou = torch.tensor(0.0, device=device)

        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)

        if "score_map" in pred_dict:
            location_loss = self.objective["focal"](
                pred_dict["score_map"],
                gt_gaussian_maps
            )
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        tracking_loss = (
            self.loss_weight["giou"] * giou_loss
            + self.loss_weight["l1"] * l1_loss
            + self.loss_weight["focal"] * location_loss
        )

        # Auxiliary losses are applied only to the collaborative stage.
        flops_constraint_loss = torch.tensor(0.0, device=l1_loss.device)
        layer_kept = torch.tensor(0.0, device=l1_loss.device)
        flops_actual = torch.tensor(0.0, device=l1_loss.device)

        if include_auxiliary and "atp_masks" in pred_dict and pred_dict["atp_masks"]:
            layer_kept = pred_dict["atp_masks"][0].sum(dim=-1).mean(dim=0)

            flops_actual, hidden_dim = self.compute_flops_loss(
                masks=pred_dict["atp_masks"],
                pred_dict=pred_dict
            )

            flops_constraint_loss = torch.relu(
                (flops_actual - self.F_target) / self.F_target
            )

        auxiliary_loss = self.flops_weight * flops_constraint_loss

        prompt_gate_loss = torch.tensor(0.0, device=tracking_loss.device)
        if include_auxiliary and "prompt_gate" in pred_dict:
            prompt_gate_weight = float(self._get_cfg_value("TRAIN.PROMPT_GATE_WEIGHT", 0.01))
            prompt_gate_loss = pred_dict["prompt_gate"].mean()
            auxiliary_loss = auxiliary_loss + prompt_gate_weight * prompt_gate_loss

        prompt_align_loss = torch.tensor(0.0, device=tracking_loss.device)
        if include_auxiliary and self._get_cfg_value("MODEL.PCUM.ENABLED", False):
            local_prompt = self._get_prompt_from_pred(pred_dict)
            if local_prompt is not None:
                remote_prompt = self._get_remote_prompt_from_pred(pred_dict)
                if remote_prompt is None and self._get_cfg_value("TRAIN.PCUM.USE_PSEUDO_REMOTE", True):
                    remote_prompt = build_pseudo_remote_prompts(local_prompt)

                if remote_prompt is not None:
                    prompt_align_loss = self.prompt_consistency_loss(local_prompt, remote_prompt)
                    align_weight = float(self._get_cfg_value(
                        "TRAIN.PCUM.ALIGN_LOSS_WEIGHT",
                        self._get_cfg_value("TRAIN.PCUM_CONSIST_WEIGHT", 0.0)
                    ))
                    auxiliary_loss = auxiliary_loss + align_weight * prompt_align_loss

        loss = tracking_loss + auxiliary_loss
        components = {
            "tracking": tracking_loss,
            "auxiliary": auxiliary_loss,
        }

        if return_status:
            mean_iou = iou.detach().mean()

            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),

                # 无 teacher，无 distillation
                "Loss/distill_feat": 0.0,
                "Loss/distill_box": 0.0,
                "Loss/distill_map": 0.0,

                # FLOPs
                "flops": flops_constraint_loss.item(),
                "flops_actual": flops_actual.detach().item()
                    if isinstance(flops_actual, torch.Tensor) else float(flops_actual),
                "flops_target": float(self.F_target),
                "flops_weight": float(self.flops_weight),
                "layer_kept_0": layer_kept.detach().item()
                    if isinstance(layer_kept, torch.Tensor) else float(layer_kept),
                "prompt_gate": prompt_gate_loss.detach().item()
                    if isinstance(prompt_gate_loss, torch.Tensor) else float(prompt_gate_loss),
                "loss_prompt_align": prompt_align_loss.detach().item()
                    if isinstance(prompt_align_loss, torch.Tensor) else float(prompt_align_loss),

                "IoU": mean_iou.item(),
            }

            if return_components:
                return loss, status, components
            return loss, status

        if return_components:
            return loss, components
        return loss

    # ------------------------------------------------------------
    # FLOPs computation
    # ------------------------------------------------------------
    def compute_flops_loss(self, masks, pred_dict):
        """
        计算 EnTeRTrack / ARP 的 FLOPs 约束。

        原蒸馏 actor 使用 pred_dict['tokens'][0] 来取 N 和 D。
        但你当前上传的 vit_arp.py 中 tokens 可能是空 list，
        所以这里做了兜底：
            1. 优先用 tokens[0]；
            2. 如果 tokens 为空，用 backbone_feat 和 cfg 推断 N、D。
        """
        device = masks[0].device

        tokens = pred_dict.get("tokens", None)

        if tokens is not None and len(tokens) > 0:
            B, N, D = tokens[0].shape

        else:
            backbone_feat = pred_dict.get("backbone_feat", None)

            if backbone_feat is not None:
                B = backbone_feat.shape[0]
                D = backbone_feat.shape[-1]
            else:
                B = masks[0].shape[0]
                # EnTeRTrack tiny 默认 hidden dim = 192
                D = self._get_cfg_value("MODEL.HIDDEN_DIM", 192)

            template_feat_sz = (
                self.cfg.DATA.TEMPLATE.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
            )
            search_feat_sz = (
                self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
            )

            lens_t = template_feat_sz * template_feat_sz
            lens_s = search_feat_sz * search_feat_sz
            N = lens_t + lens_s

        # EnTeRTrack tiny depth=6
        num_layers = self._get_cfg_value("MODEL.BACKBONE.DEPTH", 6)

        hidden_dim = int(D * 4.0)

        # 原代码中 mask_groups = [[1, 2, 3, 4, 5]]
        # 表示第 0 层 ATP mask 影响后续层。
        mask_groups = [[1, 2, 3, 4, 5]]

        per_layer_flops = []

        offset = len(mask_groups) - len(masks)

        template_feat_sz = (
            self.cfg.DATA.TEMPLATE.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        )
        lens_t = template_feat_sz * template_feat_sz

        for i in range(num_layers):
            used_mask = None

            for m_idx, group in enumerate(mask_groups):
                if i in group:
                    true_idx = m_idx - offset

                    if true_idx >= 0 and true_idx < len(masks):
                        used_mask = masks[true_idx]

                    break

            if used_mask is not None:
                # used_mask: [B, Ls]
                N_l = used_mask.sum(dim=-1)
                N_total = N_l + lens_t

            else:
                N_total = torch.full(
                    (B,),
                    float(N),
                    device=device,
                    dtype=torch.float32
                )

            flops_l = (
                4.0 * N_total * (D ** 2)
                + 2.0 * (N_total ** 2) * D
                + 2.0 * N_total * D * hidden_dim
            )

            per_layer_flops.append(flops_l)

        sample_flops = torch.stack(per_layer_flops, dim=1).sum(dim=1)

        return sample_flops.mean(), D
