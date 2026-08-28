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
from lib.models.entertrack.target_prompt_collaboration import (  # noqa: E402
    TargetPromptCollaboration,
    TargetPromptExtractor,
)
from lib.models.entertrack.target_prompt_collaboration_checkpoint import (  # noqa: E402
    load_target_prompt_collaboration_initialization,
)
from lib.train.actors.target_prompt_collaboration import (  # noqa: E402
    build_flat_remote_prompts,
)
from lib.train.optimizer_groups import build_optimizer_param_groups  # noqa: E402
from lib.train.target_prompt_collaboration_freeze import (  # noqa: E402
    apply_target_prompt_collaboration_freeze,
    assert_target_prompt_optimizer_membership,
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


class DummyE3Model(nn.Module):
    def __init__(self, dim=12, heads=3):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.box_head = nn.Linear(4, 4)
        self.pcum = None
        self.c3r = None
        self.plain_collaboration = None
        self.search_prompt_gate = None
        self.target_prompt_extractor = TargetPromptExtractor(8, dim)
        self.target_prompt_collaboration = TargetPromptCollaboration(
            token_dim=dim, num_heads=heads, enabled=True)
        self.target_prompt_collaboration_freeze_local = False


class TargetPromptExtractorTests(unittest.TestCase):
    def test_shape_parameter_count_and_determinism(self):
        extractor = TargetPromptExtractor(prompt_k=8, token_dim=192)
        search = torch.randn(2, 256, 192)
        score = torch.randn(2, 1, 16, 16)
        first = extractor(search, score)
        second = extractor(search, score)
        self.assertEqual(first.shape, (2, 8, 192))
        self.assertEqual(sum(p.numel() for p in extractor.parameters()), 0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_topk_indices_gather_exact_tokens(self):
        extractor = TargetPromptExtractor(prompt_k=8, token_dim=2)
        search = torch.arange(512, dtype=torch.float32).view(1, 256, 2)
        score = torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16)
        result = extractor.extract_with_metadata(search, score)
        expected_indices = torch.arange(255, 247, -1).view(1, 8)
        torch.testing.assert_close(result["topk_indices"], expected_indices)
        torch.testing.assert_close(
            result["prompt"], search[:, expected_indices[0]])

    def test_prediction_only_api_has_no_gt_dependency(self):
        extractor = TargetPromptExtractor(prompt_k=8, token_dim=4)
        search = torch.randn(1, 256, 4)
        score = torch.randn(1, 1, 16, 16)
        before = extractor.extract_with_metadata(search, score)
        after = extractor.extract_with_metadata(search, score)
        torch.testing.assert_close(before["prompt"], after["prompt"])
        self.assertNotIn("gt", TargetPromptExtractor.forward.__code__.co_varnames)
        self.assertNotIn("bbox", TargetPromptExtractor.forward.__code__.co_varnames)


class TargetPromptCollaborationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.local = torch.randn(2, 256, 12)
        self.remote = torch.randn(2, 2, 8, 12)

    def module(self, **kwargs):
        options = dict(token_dim=12, num_heads=3, enabled=True)
        options.update(kwargs)
        return TargetPromptCollaboration(**options)

    def test_remote_shape_and_k_not_equal_local_is_legal(self):
        output = self.module()(self.local, self.remote)
        self.assertEqual(output["search_tokens"].shape, (2, 256, 12))
        self.assertEqual(output["remote_weights"].shape, (2, 2))
        self.assertTrue(output["used_remote"])

    def test_disabled_and_no_remote_are_exact_bypass(self):
        disabled = self.module(enabled=False)(self.local, self.remote)
        no_remote = self.module()(self.local, None)
        self.assertIs(disabled["search_tokens"], self.local)
        self.assertIs(no_remote["search_tokens"], self.local)
        self.assertFalse(disabled["used_remote"])
        self.assertFalse(no_remote["used_remote"])

    def test_invalid_sender_is_masked_and_all_invalid_bypasses(self):
        remote = self.remote.clone()
        remote[0, 0, 0, 0] = float("nan")
        output = self.module()(self.local, remote)
        self.assertEqual(output["valid_remote_count"].tolist(), [1, 2])
        self.assertTrue(torch.isfinite(output["search_tokens"]).all())
        all_invalid = self.module()(
            self.local, torch.full_like(self.remote, float("nan")))
        torch.testing.assert_close(
            all_invalid["search_tokens"], self.local, rtol=0, atol=0)
        self.assertFalse(all_invalid["used_remote"])

    def test_explicit_invalid_mask_bypasses_exactly(self):
        output = self.module()(
            self.local, self.remote,
            remote_valid=torch.zeros(2, 2, dtype=torch.bool))
        torch.testing.assert_close(output["search_tokens"], self.local,
                                   rtol=0, atol=0)
        self.assertFalse(output["used_remote"])

    def test_relative_residual_norm_is_capped(self):
        output = self.module(
            residual_init_scale=0.20,
            relative_norm_cap=0.05)(self.local, self.remote)
        ratio = ((output["search_tokens"] - self.local).flatten(1).norm(dim=1)
                 / self.local.flatten(1).norm(dim=1))
        self.assertTrue(torch.all(ratio <= 0.050001))

    def test_backward_is_finite(self):
        module = self.module()
        loss = module(self.local, self.remote)["search_tokens"].square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in module.parameters()]
        self.assertTrue(any(value is not None for value in gradients))
        self.assertTrue(all(value is None or torch.isfinite(value).all()
                            for value in gradients))


class TargetPromptIntegrationTests(unittest.TestCase):
    def test_view_major_remote_prompt_mapping(self):
        prompt = torch.arange(6, dtype=torch.float32).view(6, 1, 1)
        remote = build_flat_remote_prompts(prompt, num_views=3)
        self.assertEqual(remote[:, :, 0, 0].tolist(), [
            [2.0, 4.0], [3.0, 5.0],
            [0.0, 4.0], [1.0, 5.0],
            [0.0, 2.0], [1.0, 3.0]])

    def test_head_only_center_contract(self):
        adapter = TargetPromptCollaboration(
            token_dim=192, num_heads=3, enabled=True)
        model = EnTeRTrack(
            transformer=DummyBackbone(),
            box_head=FakeCenterHead(),
            head_type="CENTER",
            target_prompt_collaboration=adapter,
            target_prompt_extractor=TargetPromptExtractor(8, 192),
            target_prompt_collaboration_freeze_local=True)
        feature = torch.randn(2, 320, 192)
        output = model(
            template=None,
            search=None,
            collaboration_feature=feature,
            target_prompt_remote_tokens=torch.randn(2, 2, 8, 192),
            target_prompt_remote_valid=torch.ones(2, 2, dtype=torch.bool))
        self.assertEqual(output["score_map"].shape, (2, 1, 16, 16))
        self.assertEqual(output["pred_boxes"].shape, (2, 1, 4))
        self.assertEqual(output["local_search_tokens"].shape, (2, 256, 192))

    def test_default_disabled_and_e3_config_single_factor(self):
        self.assertFalse(cfg.MODEL.TARGET_PROMPT_COLLABORATION.ENABLED)
        self.assertFalse(cfg.TRAIN.TARGET_PROMPT_COLLABORATION.ENABLED)
        self.assertFalse(cfg.TEST.TARGET_PROMPT_COLLABORATION.ENABLED)
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/target_prompt_collaboration_e3.yaml",
            base_cfg=local_cfg)
        self.assertTrue(local_cfg.MODEL.TARGET_PROMPT_COLLABORATION.ENABLED)
        self.assertTrue(local_cfg.TRAIN.TARGET_PROMPT_COLLABORATION.ENABLED)
        self.assertTrue(local_cfg.TEST.TARGET_PROMPT_COLLABORATION.ENABLED)
        self.assertTrue(local_cfg.TEST.TARGET_PROMPT_COLLABORATION.SAFE_COMMIT)
        self.assertEqual(local_cfg.MODEL.TARGET_PROMPT_COLLABORATION.PROMPT_K, 8)
        self.assertFalse(local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED)
        self.assertFalse(local_cfg.MODEL.PCUM.ENABLED)
        self.assertFalse(local_cfg.MODEL.C3R.ENABLED)
        self.assertFalse(local_cfg.MODEL.FCVC.ENABLED)

    def test_strict_freeze_and_optimizer_membership(self):
        model = DummyE3Model()
        local_cfg = copy.deepcopy(cfg)
        local_cfg.MODEL.TARGET_PROMPT_COLLABORATION.ENABLED = True
        local_cfg.TRAIN.TARGET_PROMPT_COLLABORATION.ENABLED = True
        report = apply_target_prompt_collaboration_freeze(model, local_cfg)
        self.assertEqual(report["extractor_parameters"], 0)
        self.assertTrue(report["trainable_names"])
        self.assertTrue(all(name.startswith("target_prompt_collaboration.")
                            for name in report["trainable_names"]))
        optimizer = torch.optim.AdamW(
            build_optimizer_param_groups(model, local_cfg))
        names = assert_target_prompt_optimizer_membership(model, optimizer)
        self.assertTrue(all(name.startswith("target_prompt_collaboration.")
                            for name in names))

    def test_b0_initialization_is_strict_and_adapter_is_fresh(self):
        model = DummyE3Model()
        initial_adapter = {
            key: value.clone() for key, value in model.state_dict().items()
            if key.startswith("target_prompt_collaboration.")}
        source = {
            key: torch.full_like(value, 3.0)
            for key, value in model.state_dict().items()
            if not key.startswith(("target_prompt_collaboration.",
                                   "target_prompt_extractor."))}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b0.pth.tar")
            torch.save({"net": source}, path)
            report = load_target_prompt_collaboration_initialization(model, path)
        self.assertTrue(report["strict_full_load"])
        for key, value in model.state_dict().items():
            if key.startswith("target_prompt_collaboration."):
                torch.testing.assert_close(value, initial_adapter[key])
            elif not key.startswith("target_prompt_extractor."):
                torch.testing.assert_close(value, torch.full_like(value, 3.0))

    def test_theoretical_payload_is_exactly_32x_smaller(self):
        self.assertEqual(256 * 192 * 4, 196608)
        self.assertEqual(8 * 192 * 4, 6144)
        self.assertEqual(256 * 192 * 2, 98304)
        self.assertEqual(8 * 192 * 2, 3072)
        self.assertEqual(196608 // 6144, 32)


if __name__ == "__main__":
    unittest.main()
