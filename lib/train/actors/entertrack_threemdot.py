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

    def _restore_paired_forward_rng(self):
        if self._paired_cpu_rng_state is not None:
            torch.set_rng_state(self._paired_cpu_rng_state)
        if self._paired_cuda_rng_state is not None:
            torch.cuda.set_rng_state(
                self._paired_cuda_rng_state,
                device=self._paired_cuda_device,
            )
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
        })
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

    def _make_remote_state(self, masks, device=None, dtype=None):
        masks = [m for m in masks if m is not None]
        if len(masks) == 0:
            return None

        stacked = torch.stack([
            m.to(device=device, dtype=dtype or torch.float32).view(-1)
            for m in masks
        ], dim=0)
        return {"score": stacked.mean(dim=0)}

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
        if self._use_real_multiview_pcum(data):
            return self.forward_pass_real_multiview_pcum(net, data)

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
                          remote_states=None):
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
            remote_states=remote_states
        )

    def _forward_flat_views(self, net, data, num_views, remote_prompts=None,
                            remote_states=None):
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
            training=True,
            prompt_map=prompt_map,
            prompt_gate_input=prompt_gate_input,
            remote_prompts=remote_prompts,
            remote_states=remote_states
        )

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

    def _build_remote_inputs(self, data, remote_bank, target_view, num_views):
        drop_prob = float(self._get_cfg_value("TRAIN.PCUM.REMOTE_DROPOUT_PROB", 0.0))
        if drop_prob > 0 and torch.rand((), device=remote_bank[0].device).item() < drop_prob:
            return None, None

        remote_prompts = []
        remote_masks = []
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

        remote_states = self._make_remote_state(
            remote_masks,
            device=remote_prompts[0].device if remote_prompts else None,
            dtype=remote_prompts[0].dtype if remote_prompts else None,
        )
        return remote_prompts, remote_states

    def _build_flat_remote_inputs(self, data, remote_bank, num_views):
        drop_prob = float(self._get_cfg_value("TRAIN.PCUM.REMOTE_DROPOUT_PROB", 0.0))
        if drop_prob > 0 and torch.rand((), device=remote_bank[0].device).item() < drop_prob:
            return None, None

        num_remote = num_views - 1
        remote_slots = [[] for _ in range(num_remote)]
        mask_slots = [[] for _ in range(num_remote)]

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

        remote_prompts = [torch.cat(slot_prompts, dim=0) for slot_prompts in remote_slots]
        remote_masks = [torch.cat(slot_masks, dim=0) for slot_masks in mask_slots]
        remote_states = self._make_remote_state(
            remote_masks,
            device=remote_prompts[0].device if remote_prompts else None,
            dtype=remote_prompts[0].dtype if remote_prompts else None,
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

        backward_loss = local_components["tracking"] * (local_weight / pair_denominator)
        cache = {
            "num_views": num_views,
            "remote_bank": [prompt.detach() for prompt in local_prompts],
            "local_per_sample": local_per_sample["total"].detach(),
            "local_tracking": local_components["tracking"].detach(),
            "local_status": local_status,
            "local_weight": local_weight,
            "collab_weight": collab_weight,
            "pair_denominator": pair_denominator,
        }
        return backward_loss, cache

    def paired_collaborative_stage(self, data, cache):
        """Run collaborative supervision using only detached remote prompts."""
        num_views = int(cache["num_views"])
        remote_prompts, remote_states = self._build_flat_remote_inputs(
            data, cache["remote_bank"], num_views=num_views)
        remote_active = remote_prompts is not None
        self._restore_paired_forward_rng()

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
        safe_loss, safe_stats = self.compute_safe_loss(
            collaborative_per_sample=collab_per_sample,
            local_per_sample=cache["local_per_sample"],
            num_views=num_views,
            margin=float(self._get_cfg_value("TRAIN.PCUM.SAFE_MARGIN", 0.0)),
            hard_sample_quantile=float(self._get_cfg_value(
                "TRAIN.PCUM.SAFE_HARD_SAMPLE_QUANTILE", 0.0)),
            active_mask=remote_mask,
        )

        collab_factor = cache["collab_weight"] / cache["pair_denominator"]
        safe_weight = float(self._get_cfg_value("TRAIN.PCUM.SAFE_LOSS_WEIGHT", 0.0))
        backward_loss = (
            collab_components["tracking"] * collab_factor
            + safe_loss * safe_weight
            + collab_components["auxiliary"]
        )
        pair_loss = (
            cache["local_weight"] * cache["local_tracking"]
            + cache["collab_weight"] * collab_components["tracking"].detach()
        ) / cache["pair_denominator"]
        total_for_log = (
            pair_loss
            + safe_loss.detach() * safe_weight
            + collab_components["auxiliary"].detach()
        )

        local_status = cache["local_status"]
        status = {
            "Loss/total": float(total_for_log.item()),
            "Loss/pair_tracking": float(pair_loss.item()),
            "Loss/local_tracking": float(cache["local_tracking"].item()),
            "Loss/collaborative_tracking": float(
                collab_components["tracking"].detach().item()),
            "Loss/safe": float(safe_loss.detach().item()),
            "Loss/local_giou": float(local_status["Loss/giou"]),
            "Loss/local_l1": float(local_status["Loss/l1"]),
            "Loss/local_focal": float(local_status["Loss/location"]),
            "Loss/collaborative_giou": float(collab_status["Loss/giou"]),
            "Loss/collaborative_l1": float(collab_status["Loss/l1"]),
            "Loss/collaborative_focal": float(collab_status["Loss/location"]),
            "PCUM/collaborative_better_ratio": safe_stats["collaborative_better_ratio"],
            "PCUM/loss_delta_mean": safe_stats["delta_mean"],
            "PCUM/loss_delta_std": safe_stats["delta_std"],
            "PCUM/loss_delta_min": safe_stats["delta_min"],
            "PCUM/loss_delta_max": safe_stats["delta_max"],
            "PCUM/remote_active": float(remote_active),
            "PCUM/remote_dropout": float(not remote_active),
            "PCUM/safe_margin": float(self._get_cfg_value("TRAIN.PCUM.SAFE_MARGIN", 0.0)),
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
                "delta_mean": 0.0,
                "delta_std": 0.0,
                "delta_min": 0.0,
                "delta_max": 0.0,
            }
        else:
            detached_delta = active_delta.detach().float()
            stats = {
                "collaborative_better_ratio": float((detached_delta < 0).float().mean().item()),
                "delta_mean": float(detached_delta.mean().item()),
                "delta_std": float(detached_delta.std(unbiased=False).item()),
                "delta_min": float(detached_delta.min().item()),
                "delta_max": float(detached_delta.max().item()),
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
