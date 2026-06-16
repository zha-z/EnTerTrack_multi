from . import BaseActor

import torch
import torch.nn.functional as F

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate, adjust_temperature


class EnTeRTrackActorThreeMDOT(BaseActor):
    """
    三模板 + 单搜索区域 版本
    1. 不使用 prompt
    2. 不使用 teacher
    3. 不使用 distillation
    4. 只监督一个 search 分支
    5. 保留 FLOPs 约束
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)

        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

        self.flops_weight = 0.0
        self.F_target = 1.0

    def __call__(self, data):
        self._update_flops_schedule(data["epoch"])
        out_dict = self.forward_pass(self.net, data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _get_cfg_value(self, path, default):
        node = self.cfg
        for key in path.split("."):
            if not hasattr(node, key):
                return default
            node = getattr(node, key)
        return node

    def _squeeze_if_needed(self, value):
        """
        图像:
            [1, B, C, H, W] -> [B, C, H, W]
        标注:
            [1, B, 4] -> [B, 4]
            [B, 1, 4] -> [B, 4]
            [4] -> [1, 4]
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

    def _select_template(self, value, idx):
        """
        从 template_images/template_anno 中取第 idx 个模板视角。
        支持:
        - list
        - tensor [T, B, ...]
        """
        if isinstance(value, list):
            return value[idx]

        if isinstance(value, torch.Tensor):
            if value.dim() >= 3:
                return value[idx]

        raise TypeError(f"Unsupported template container type: {type(value)}")

    def _select_search(self, value):
        """
        search 只取第 0 个，因为现在就是单搜索区域。
        支持:
        - list
        - tensor [1, B, ...]
        - tensor [B, ...]
        """
        if isinstance(value, list):
            value = value[0]

        elif isinstance(value, torch.Tensor):
            if value.dim() == 5:
                value = value[0]
            elif value.dim() == 3 and value.shape[-1] == 4:
                value = value[0]

        return value

    def _make_gt_heatmap(self, gt_bbox):
        """
        gt_bbox: [B, 4], xywh, normalized
        return : [B, 1, H, W]
        """
        gt_gaussian_maps = generate_heatmap(
            [gt_bbox],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE
        )

        gt_gaussian_maps = gt_gaussian_maps[0]

        if gt_gaussian_maps.dim() == 4 and gt_gaussian_maps.shape[1] == 1:
            pass
        elif gt_gaussian_maps.dim() == 3:
            gt_gaussian_maps = gt_gaussian_maps.unsqueeze(1)
        elif gt_gaussian_maps.dim() == 2:
            gt_gaussian_maps = gt_gaussian_maps.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected heatmap shape: {gt_gaussian_maps.shape}")

        return gt_gaussian_maps

    # ------------------------------------------------------------
    # FLOPs schedule
    # ------------------------------------------------------------
    def _update_flops_schedule(self, epoch):
        start_epoch = self._get_cfg_value("TRAIN.FLOPS_START_EPOCH", 0)
        end_epoch = self._get_cfg_value("TRAIN.FLOPS_END_EPOCH", 0)

        max_flops_target = self._get_cfg_value("TRAIN.MAX_FLOPS_TARGET", 12e8)
        initial_flops_target = self._get_cfg_value("TRAIN.INITIAL_FLOPS_TARGET", 15e8)

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
    # forward
    # ------------------------------------------------------------
    def forward_pass(self, net, data):
        # 3 个模板
        template_img_1 = self._squeeze_if_needed(self._select_template(data["template_images"], 0))
        template_img_2 = self._squeeze_if_needed(self._select_template(data["template_images"], 1))
        template_img_3 = self._squeeze_if_needed(self._select_template(data["template_images"], 2))

        template_anno_1 = self._squeeze_if_needed(self._select_template(data["template_anno"], 0))
        template_anno_2 = self._squeeze_if_needed(self._select_template(data["template_anno"], 1))
        template_anno_3 = self._squeeze_if_needed(self._select_template(data["template_anno"], 2))

        # 1 个 search
        search_img = self._squeeze_if_needed(self._select_search(data["search_images"]))

        ce_keep_rate = None
        box_mask_z1, box_mask_z2, box_mask_z3 = None, None, None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z1 = generate_mask_cond(
                self.cfg, template_img_1.shape[0], template_img_1.device, template_anno_1
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

        # 关键：这里已经改成三模板 + 单搜索区域
        out_dict = net(
            template=template_img_1,
            template2=template_img_2,
            template3=template_img_3,
            search=search_img,
            ce_template_mask=box_mask_z1,
            ce_keep_rate=ce_keep_rate,
            temperature=temperature,
            return_last_attn=False,
            return_atp=True,
            training=True
        )

        return out_dict

    # ------------------------------------------------------------
    # loss
    # ------------------------------------------------------------
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = self._select_search(gt_dict["search_anno"])
        gt_bbox = self._squeeze_if_needed(gt_bbox)

        gt_gaussian_maps = self._make_gt_heatmap(gt_bbox).to(
            pred_dict["score_map"].device
        )

        pred_boxes = pred_dict["pred_boxes"]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs NAN! Stop training.")

        num_queries = pred_boxes.size(1)

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat(1, num_queries, 1).view(-1, 4).clamp(0.0, 1.0)

        giou_loss, iou = self.objective["giou"](pred_boxes_vec, gt_boxes_vec)
        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)

        location_loss = self.objective["focal"](
            pred_dict["score_map"],
            gt_gaussian_maps
        )

        # --------------------------------------------------------
        # FLOPs loss
        # --------------------------------------------------------
        flops_loss = torch.tensor(0.0, device=pred_boxes.device)
        flops_actual = torch.tensor(0.0, device=pred_boxes.device)
        layer_kept_0 = torch.tensor(0.0, device=pred_boxes.device)

        if "atp_masks" in pred_dict and pred_dict["atp_masks"]:
            # atp_masks[0]: [B, Ls]，表示 search token 保留情况
            layer_kept_0 = pred_dict["atp_masks"][0].sum(dim=-1).mean(dim=0)

            flops_actual, hidden_dim = self.compute_flops_loss(
                masks=pred_dict["atp_masks"],
                pred_dict=pred_dict
            )

            flops_loss = F.relu(
                (flops_actual - self.F_target) / self.F_target
            )

        loss = (
            self.loss_weight["giou"] * giou_loss
            + self.loss_weight["l1"] * l1_loss
            + self.loss_weight["focal"] * location_loss
            + self.flops_weight * flops_loss
        )

        if return_status:
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/distill_feat": 0.0,
                "Loss/distill_box": 0.0,
                "Loss/distill_map": 0.0,
                "flops": flops_loss.item(),
                "flops_actual": flops_actual.item() if torch.is_tensor(flops_actual) else float(flops_actual),
                "flops_target": float(self.F_target),
                "flops_weight": float(self.flops_weight),
                "layer_kept_0": layer_kept_0.item() if torch.is_tensor(layer_kept_0) else float(layer_kept_0),
                "IoU": iou.mean().item()
            }
            return loss, status

        return loss
    def compute_flops_loss(self, masks, pred_dict):
        """
        计算 EnTeRTrack / ARP 的 FLOPs 约束。

        三模板版本：
            N = 3 * template_tokens + search_tokens

        masks:
            list of atp masks, each mask shape is [B, Ls]
        """
        device = masks[0].device

        # --------------------------------------------------------
        # 1. 获取 B 和 D
        # --------------------------------------------------------
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
                D = self._get_cfg_value("MODEL.HIDDEN_DIM", 192)

        # --------------------------------------------------------
        # 2. token 数量
        # --------------------------------------------------------
        template_feat_sz = (
            self.cfg.DATA.TEMPLATE.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        )
        search_feat_sz = (
            self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        )

        num_templates_used = int(pred_dict.get("num_templates_used", 1))

        lens_t_single = template_feat_sz * template_feat_sz
        lens_t = lens_t_single * num_templates_used
        lens_s = search_feat_sz * search_feat_sz

        N = lens_t + lens_s

        # --------------------------------------------------------
        # 3. Transformer 层数
        # --------------------------------------------------------
        num_layers = self._get_cfg_value("MODEL.BACKBONE.DEPTH", 6)
        hidden_dim = int(D * 4.0)

        # EnTeRTrack tiny 当前一般只有一个 ATP mask，作用于后续层
        mask_groups = [[1, 2, 3, 4, 5]]
        offset = len(mask_groups) - len(masks)

        per_layer_flops = []

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
                N_s = used_mask.sum(dim=-1)
                N_total = N_s + lens_t
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