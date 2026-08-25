import copy
import os
import sys
import tempfile
import unittest

import torch
from torch import nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack.entertrack import EnTeRTrack  # noqa: E402
from lib.models.entertrack.plain_collaboration import (  # noqa: E402
    PlainCollaborationV1,
)
from lib.models.entertrack.plain_collaboration_checkpoint import (  # noqa: E402
    load_plain_collaboration_initialization,
)
from lib.train.actors.plain_collaboration import (  # noqa: E402
    build_flat_remote_tokens,
    validate_synchronized_abc_metadata,
)
from lib.train.optimizer_groups import build_optimizer_param_groups  # noqa: E402
from lib.train.plain_collaboration_freeze import (  # noqa: E402
    apply_plain_collaboration_freeze,
    assert_plain_collaboration_optimizer_membership,
)


class FakeCenterHead(nn.Module):
    feat_sz = 16

    def forward(self, feature, gt_score_map=None):
        batch_size = feature.shape[0]
        score = feature.mean(dim=1, keepdim=True)
        bbox = feature.new_zeros(batch_size, 1, 4)
        size = feature.new_zeros(batch_size, 2, 16, 16)
        offset = feature.new_zeros(batch_size, 2, 16, 16)
        return score, bbox, size, offset


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class DummyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.box_head = nn.Linear(4, 4)
        self.plain_collaboration = PlainCollaborationV1(
            token_dim=4, num_heads=2, enabled=True)
        self.plain_collaboration_freeze_local = False


class PlainCollaborationModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.local = torch.randn(2, 8, 12)
        self.remote = torch.randn(2, 2, 8, 12)

    def _module(self, **kwargs):
        defaults = dict(
            token_dim=12,
            num_heads=3,
            enabled=True,
            residual_init_scale=0.01,
            residual_scale_max=0.25,
            relative_norm_cap=0.25,
        )
        defaults.update(kwargs)
        return PlainCollaborationV1(**defaults)

    def test_default_config_is_disabled(self):
        self.assertFalse(cfg.MODEL.PLAIN_COLLABORATION.ENABLED)
        self.assertFalse(cfg.TRAIN.PLAIN_COLLABORATION.ENABLED)

    def test_disabled_returns_exact_input_object(self):
        module = self._module(enabled=False)
        result = module(self.local, self.remote)
        self.assertIs(result["search_tokens"], self.local)
        self.assertFalse(result["used_remote"])

    def test_missing_remote_returns_exact_input_object(self):
        result = self._module()(self.local, None)
        self.assertIs(result["search_tokens"], self.local)
        self.assertFalse(result["used_remote"])

    def test_equal_weight_two_sender_forward_is_finite(self):
        result = self._module()(self.local, self.remote)
        self.assertEqual(result["search_tokens"].shape, self.local.shape)
        self.assertTrue(torch.isfinite(result["search_tokens"]).all())
        self.assertTrue(result["used_remote"])
        torch.testing.assert_close(
            result["remote_weights"],
            torch.full((2, 2), 0.5),
        )

    def test_residual_is_bounded_relative_to_local(self):
        module = self._module(
            residual_init_scale=0.20,
            relative_norm_cap=0.05)
        result = module(self.local, self.remote)
        delta = result["search_tokens"] - self.local
        ratio = delta.flatten(1).norm(dim=1) / self.local.flatten(1).norm(dim=1)
        self.assertTrue(torch.all(ratio <= 0.050001))

    def test_nonfinite_sender_is_dropped(self):
        remote = self.remote.clone()
        remote[0, 0, 0, 0] = float("nan")
        result = self._module()(self.local, remote)
        self.assertEqual(result["valid_remote_count"].tolist(), [1, 2])
        torch.testing.assert_close(
            result["remote_weights"][0], torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.isfinite(result["search_tokens"]).all())

    def test_backward_reaches_adapter_only(self):
        module = self._module()
        output = module(self.local, self.remote)["search_tokens"]
        loss = output.square().mean()
        loss.backward()
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients))


class PlainCollaborationMappingTests(unittest.TestCase):
    def test_view_major_sender_mapping(self):
        # B=2, token payload encodes flat index: A0,A1,B0,B1,C0,C1.
        search = torch.arange(6, dtype=torch.float32).view(6, 1, 1)
        remote = build_flat_remote_tokens(search, num_views=3)
        self.assertEqual(remote.shape, (6, 2, 1, 1))
        self.assertEqual(remote[:, :, 0, 0].tolist(), [
            [2.0, 4.0], [3.0, 5.0],
            [0.0, 4.0], [1.0, 5.0],
            [0.0, 2.0], [1.0, 3.0],
        ])

    def test_metadata_accepts_synchronized_canonical_abc(self):
        data = {
            "target_id": ["T0", "T1"],
            "view_ids": [["A", "A"], ["B", "B"], ["C", "C"]],
            "search_frame_ids": [[7, 9], [7, 9], [7, 9]],
        }
        validate_synchronized_abc_metadata(data, 3, 2)

    def test_metadata_rejects_misaligned_frame(self):
        data = {
            "target_id": ["T0"],
            "view_ids": [["A"], ["B"], ["C"]],
            "search_frame_ids": [[7], [8], [7]],
        }
        with self.assertRaisesRegex(ValueError, "different search_frame_ids"):
            validate_synchronized_abc_metadata(data, 3, 1)


class PlainCollaborationIntegrationTests(unittest.TestCase):
    def test_head_only_center_contract(self):
        adapter = PlainCollaborationV1(
            token_dim=192, num_heads=3, enabled=True)
        model = EnTeRTrack(
            transformer=DummyBackbone(),
            box_head=FakeCenterHead(),
            head_type="CENTER",
            plain_collaboration=adapter,
            plain_collaboration_freeze_local=True,
        )
        feature = torch.randn(2, 320, 192)
        remote = torch.randn(2, 2, 256, 192)
        output = model(
            template=None,
            search=None,
            collaboration_feature=feature,
            plain_remote_tokens=remote,
            plain_remote_valid=torch.ones(2, 2, dtype=torch.bool),
        )
        self.assertEqual(output["score_map"].shape, (2, 1, 16, 16))
        self.assertEqual(output["pred_boxes"].shape, (2, 1, 4))
        self.assertEqual(output["local_search_tokens"].shape, (2, 256, 192))

    def test_strict_freeze_and_optimizer_membership(self):
        model = DummyTrainModel()
        local_cfg = copy.deepcopy(cfg)
        local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED = True
        local_cfg.TRAIN.PLAIN_COLLABORATION.ENABLED = True
        local_cfg.TRAIN.PLAIN_COLLABORATION.FREEZE_LOCAL = True
        report = apply_plain_collaboration_freeze(model, local_cfg)
        self.assertTrue(report["trainable_names"])
        self.assertTrue(all(
            name.startswith("plain_collaboration.")
            for name in report["trainable_names"]))
        groups = build_optimizer_param_groups(model, local_cfg)
        optimizer = torch.optim.AdamW(groups)
        names = assert_plain_collaboration_optimizer_membership(model, optimizer)
        self.assertTrue(all(
            name.startswith("plain_collaboration.") for name in names))

    def test_v1_config_is_single_factor_and_loadable(self):
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/plain_collaboration_v1.yaml",
            base_cfg=local_cfg,
        )
        self.assertTrue(local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED)
        self.assertTrue(local_cfg.TRAIN.PLAIN_COLLABORATION.ENABLED)
        self.assertTrue(local_cfg.TRAIN.PLAIN_COLLABORATION.FREEZE_LOCAL)
        self.assertTrue(local_cfg.TRAIN.MULTIVIEW.REQUIRE_ALL_VIEWS_VISIBLE)
        self.assertTrue(local_cfg.TRAIN.MULTIVIEW.CANONICAL_VIEW_ORDER)
        self.assertFalse(local_cfg.MODEL.PCUM.ENABLED)
        self.assertFalse(local_cfg.MODEL.C3R.ENABLED)
        self.assertFalse(local_cfg.MODEL.FCVC.ENABLED)
        self.assertEqual(local_cfg.MODEL.BACKBONE.CE_LOC, [])

    def test_b0_checkpoint_loads_only_local_core(self):
        local_model = DummyTrainModel()
        local_state = {
            key: value
            for key, value in local_model.state_dict().items()
            if not key.startswith("plain_collaboration.")
        }
        for value in local_state.values():
            value.fill_(0.75)
        with tempfile.NamedTemporaryFile(suffix=".pth.tar") as checkpoint_file:
            torch.save({"net": local_state, "epoch": 25}, checkpoint_file.name)
            report = load_plain_collaboration_initialization(
                local_model, checkpoint_file.name)
        self.assertTrue(report["strict_full_load"])
        self.assertGreater(report["fresh_adapter_key_count"], 0)
        for name, value in local_model.state_dict().items():
            if not name.startswith("plain_collaboration."):
                torch.testing.assert_close(value, torch.full_like(value, 0.75))


if __name__ == "__main__":
    unittest.main()
