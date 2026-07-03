import copy
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from lib.config.entertrack.config import cfg, update_config_from_file
from lib.models.entertrack.pcum import PCUM
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT
from lib.train.optimizer_groups import build_optimizer_param_groups
from lib.utils.box_ops import (
    giou_loss,
    giou_loss_details,
    l1_loss_details,
)
from lib.utils.focal_loss import FocalLoss


class TinyPairedNetwork(nn.Module):
    def __init__(self, dim=16, diagnostics_film=True):
        super().__init__()
        self.backbone = nn.Linear(3, dim)
        self.box_head = nn.Linear(dim, 4)
        self.score_head = nn.Linear(dim, 1)
        self.pcum = PCUM(
            token_dim=dim,
            prompt_dim=dim,
            num_prompts=2,
            topk=2,
            fusion_mode="film" if diagnostics_film else "gated_add",
            align_gate="cosine_confidence",
            enabled=True,
            fusion_init_scale=0.005,
            fusion_scale_max=0.03,
        )

    def forward(self, template, search, remote_prompts=None, remote_states=None,
                **_kwargs):
        search_base = self.backbone(search.mean(dim=(-1, -2)))
        template_base = self.backbone(template.mean(dim=(-1, -2)))
        search_tokens = search_base.unsqueeze(1).repeat(1, 4, 1)
        template_tokens = template_base.unsqueeze(1).repeat(1, 2, 1)
        pcum_out = self.pcum(
            {"search": search_tokens, "template": template_tokens},
            remote_prompts=remote_prompts,
            remote_states=remote_states,
        )
        pooled = pcum_out["search_tokens"].mean(dim=1)
        pred_boxes = torch.sigmoid(self.box_head(pooled)).unsqueeze(1)
        score = torch.sigmoid(self.score_head(pooled)).view(-1, 1, 1, 1)
        return {
            "pred_boxes": pred_boxes,
            "score_map": score.expand(-1, 1, 4, 4),
            "local_prompt": pcum_out["local_prompt"],
            "aligned_prompt": pcum_out["aligned_prompt"],
            "remote_prompt": remote_prompts,
            "pcum": pcum_out,
            "atp_masks": [],
            "backbone_feat": search_tokens,
        }


def paired_cfg(pcum_lr=8e-6, safe_weight=0.5, diagnostics=False):
    local_cfg = copy.deepcopy(cfg)
    local_cfg.MODEL.PCUM.ENABLED = True
    local_cfg.MODEL.PCUM.FUSION = "film"
    local_cfg.MODEL.PCUM.ALIGN_GATE = "cosine_confidence"
    local_cfg.MODEL.PCUM.FUSION_INIT_SCALE = 0.005
    local_cfg.MODEL.PCUM.FUSION_SCALE_MAX = 0.03
    local_cfg.MODEL.BACKBONE.CE_LOC = []
    local_cfg.MODEL.BACKBONE.STRIDE = 16
    local_cfg.MODEL.USE_SEARCH_PROMPT = False
    local_cfg.DATA.SEARCH.SIZE = 64
    local_cfg.TRAIN.BATCH_SIZE = 2
    local_cfg.TRAIN.LR = 8e-5
    local_cfg.TRAIN.PCUM_LR = pcum_lr
    local_cfg.TRAIN.BACKBONE_MULTIPLIER = 0.03
    local_cfg.TRAIN.FLOPS_WEIGHT = 0.0
    local_cfg.TRAIN.PCUM.USE_REAL_MULTIVIEW = True
    local_cfg.TRAIN.PCUM.USE_PSEUDO_REMOTE = False
    local_cfg.TRAIN.PCUM.DETACH_REAL_REMOTE = True
    local_cfg.TRAIN.PCUM.USE_REMOTE_VISIBLE_MASK = True
    local_cfg.TRAIN.PCUM.REAL_MULTIVIEW_LOSS_WEIGHTS = [1.6, 1.0, 1.0]
    local_cfg.TRAIN.PCUM.REMOTE_DROPOUT_PROB = 0.0
    local_cfg.TRAIN.PCUM.PAIRED_SUPERVISION = True
    local_cfg.TRAIN.PCUM.LOCAL_LOSS_WEIGHT = 1.0
    local_cfg.TRAIN.PCUM.COLLAB_LOSS_WEIGHT = 1.0
    local_cfg.TRAIN.PCUM.SAFE_LOSS_WEIGHT = safe_weight
    local_cfg.TRAIN.PCUM.SAFE_MARGIN = 0.0
    local_cfg.TRAIN.PCUM.SAFE_HARD_SAMPLE_QUANTILE = 0.0
    local_cfg.TRAIN.PCUM.DIAGNOSTICS_ENABLED = diagnostics
    local_cfg.TRAIN.PCUM.DIAGNOSTICS_INTERVAL = 1
    return local_cfg


def make_actor(local_cfg, batch_size=2):
    settings = type("Settings", (), {"batchsize": batch_size})()
    return EnTeRTrackActorThreeMDOT(
        net=TinyPairedNetwork(),
        objective={"giou": giou_loss, "l1": F.l1_loss, "focal": FocalLoss()},
        loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
        settings=settings,
        cfg=local_cfg,
    )


def make_data(batch_size=2):
    torch.manual_seed(11)
    bbox = torch.tensor([0.3, 0.3, 0.2, 0.2]).view(1, 1, 4)
    return {
        "template_images": torch.randn(3, batch_size, 3, 16, 16),
        "search_images": torch.randn(3, batch_size, 3, 16, 16),
        "template_anno": bbox.repeat(3, batch_size, 1),
        "search_anno": bbox.repeat(3, batch_size, 1),
        "template_view_valid": torch.ones(3, batch_size, dtype=torch.bool),
        "search_view_valid": torch.ones(3, batch_size, dtype=torch.bool),
        "epoch": 1,
    }


class LossCompatibilityTest(unittest.TestCase):
    def test_giou_and_l1_details_restore_legacy_scalar(self):
        boxes1 = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.2, 0.2, 0.6, 0.7]])
        boxes2 = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.2, 0.1, 0.7, 0.8]])
        legacy_giou, _ = giou_loss(boxes1, boxes2)
        giou = giou_loss_details(boxes1, boxes2, batch_size=2)
        self.assertTrue(torch.allclose(
            legacy_giou, giou["numerator"] / giou["denominator"]))

        legacy_l1 = F.l1_loss(boxes1, boxes2)
        l1 = l1_loss_details(boxes1, boxes2, batch_size=2)
        self.assertTrue(torch.allclose(
            legacy_l1, l1["numerator"] / l1["denominator"]))

    def test_focal_details_restore_batch_positive_reduction(self):
        prediction = torch.full((2, 1, 3, 3), 0.25)
        target = torch.zeros_like(prediction)
        target[0, 0, 0, 0] = 1.0
        target[0, 0, 1, 1] = 1.0
        target[1, 0, 2, 2] = 1.0
        objective = FocalLoss()
        legacy = objective(prediction, target)
        details = objective.loss_details(prediction, target)
        restored = details["numerator"] / details["denominator"]
        self.assertTrue(torch.allclose(legacy, restored))
        self.assertFalse(torch.allclose(legacy, details["per_sample"].mean()))


class PairedSupervisionTest(unittest.TestCase):
    def test_safe_loss_formula_and_local_stop_gradient(self):
        actor = make_actor(paired_cfg())
        local = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        collaborative = torch.tensor([1.5, 1.0, 4.0], requires_grad=True)
        safe, _ = actor.compute_safe_loss(
            collaborative, local, num_views=3, margin=0.2)
        weights = torch.tensor([1.6, 1.0, 1.0]) / 3.6
        expected = (
            torch.relu(collaborative - local.detach() + 0.2) * weights
        ).sum()
        self.assertTrue(torch.allclose(safe, expected))
        safe.backward()
        self.assertIsNone(local.grad)
        self.assertIsNotNone(collaborative.grad)

    def test_remote_dropout_masks_safe_loss(self):
        actor = make_actor(paired_cfg())
        local = torch.tensor([1.0, 1.0, 1.0])
        collaborative = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
        safe, stats = actor.compute_safe_loss(
            collaborative,
            local,
            num_views=3,
            margin=0.1,
            active_mask=torch.zeros(3, dtype=torch.bool),
        )
        self.assertEqual(float(safe.item()), 0.0)
        self.assertEqual(stats["delta_mean"], 0.0)

    def test_two_stage_forward_backward_and_detached_remote_bank(self):
        local_cfg = paired_cfg(diagnostics=True)
        actor = make_actor(local_cfg)
        data = make_data()
        optimizer = torch.optim.AdamW(
            build_optimizer_param_groups(actor.net, local_cfg),
            lr=local_cfg.TRAIN.LR,
        )
        optimizer.zero_grad()
        actor.begin_paired_iteration(data, diagnostics_active=True)

        local_loss, cache = actor.paired_local_stage(data)
        self.assertTrue(all(not prompt.requires_grad for prompt in cache["remote_bank"]))
        local_loss.backward()
        self.assertIsNotNone(actor.net.box_head.weight.grad)
        self.assertGreater(
            actor.net.pcum.encoder.input_proj.weight.grad.norm().item(), 0.0)

        collaborative_loss, status = actor.paired_collaborative_stage(data, cache)
        collaborative_loss.backward()
        diagnostics = actor.collect_gradient_diagnostics()
        optimizer.step()

        self.assertIn("Loss/local_tracking", status)
        self.assertIn("Loss/collaborative_tracking", status)
        self.assertIn("local_feature_relative_change", diagnostics)
        self.assertIn("collaborative_feature_relative_change", diagnostics)
        self.assertIn("remote_incremental_feature_change", diagnostics)
        self.assertGreater(diagnostics["Grad/pcum_encoder"], 0.0)
        actor.close_diagnostics()

    def test_diagnostics_disabled_registers_no_hook(self):
        actor = make_actor(paired_cfg(diagnostics=False))
        self.assertIsNone(actor._fusion_hook_handle)

    def test_safe_weight_zero_keeps_normalized_pair(self):
        local_cfg = paired_cfg(safe_weight=0.0)
        actor = make_actor(local_cfg)
        data = make_data()
        actor.begin_paired_iteration(data)
        local_backward, cache = actor.paired_local_stage(data)
        collaborative_backward, status = actor.paired_collaborative_stage(data, cache)
        expected = 0.5 * (
            status["Loss/local_tracking"] + status["Loss/collaborative_tracking"])
        actual = float(local_backward.detach().item() + collaborative_backward.detach().item())
        self.assertAlmostEqual(actual, expected, places=5)

    def test_remote_dropout_keeps_pair_and_disables_safe_term(self):
        local_cfg = paired_cfg()
        local_cfg.TRAIN.PCUM.REMOTE_DROPOUT_PROB = 1.0
        actor = make_actor(local_cfg)
        data = make_data()
        actor.begin_paired_iteration(data)
        _, cache = actor.paired_local_stage(data)
        _, status = actor.paired_collaborative_stage(data, cache)
        self.assertEqual(status["PCUM/remote_active"], 0.0)
        self.assertEqual(status["PCUM/remote_dropout"], 1.0)
        self.assertEqual(status["Loss/safe"], 0.0)


class OptimizerAndConfigTest(unittest.TestCase):
    def test_pcum_is_in_exactly_one_optimizer_group(self):
        local_cfg = paired_cfg(pcum_lr=4e-5)
        network = TinyPairedNetwork()
        groups = build_optimizer_param_groups(network, local_cfg)
        all_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        pcum_group = next(group for group in groups
                          if group.get("group_name") == "pcum")
        self.assertAlmostEqual(pcum_group["lr"], 4e-5)
        pcum_ids = {id(parameter) for parameter in network.pcum.parameters()}
        self.assertEqual(pcum_ids, {id(parameter) for parameter in pcum_group["params"]})

    def test_experiment_pcum_learning_rates(self):
        expected = {
            "pcum_supervision_e1_paired_lr8e6.yaml": 8e-6,
            "pcum_supervision_e2a_safe_m0_lr8e6.yaml": 8e-6,
            "pcum_supervision_e2b_safe_m0001_lr8e6.yaml": 8e-6,
            "pcum_supervision_e3_safe_m0_lr4e5.yaml": 4e-5,
            "pcum_supervision_e4_safe_m0_lr8e5.yaml": 8e-5,
        }
        for filename, learning_rate in expected.items():
            local_cfg = copy.deepcopy(cfg)
            update_config_from_file(
                "experiments/entertrack/" + filename,
                base_cfg=local_cfg,
            )
            self.assertTrue(local_cfg.TRAIN.PCUM.PAIRED_SUPERVISION)
            self.assertEqual(local_cfg.MODEL.PCUM.FUSION, "film")
            self.assertAlmostEqual(local_cfg.TRAIN.PCUM_LR, learning_rate)

        baseline_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/pcum_ablation_current_full.yaml",
            base_cfg=baseline_cfg,
        )
        self.assertFalse(baseline_cfg.TRAIN.PCUM.PAIRED_SUPERVISION)


if __name__ == "__main__":
    unittest.main()
