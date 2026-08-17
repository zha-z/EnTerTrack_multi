import copy
import inspect
import os
import sys
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.models.entertrack.pcum import RemotePromptAggregator, build_pcum  # noqa: E402
from lib.test.utils.pcum_remote_state import build_remote_state  # noqa: E402
from tracking.audit_b0_pcum_config import (  # noqa: E402
    DEFAULT_A0,
    DEFAULT_B0,
    DEFAULT_INIT_SEED,
    DEFAULT_TRAINING_CANDIDATE,
    FROZEN_B0_SHA256,
    config_diff,
    load_config,
    make_training_model,
    one_step_training_smoke,
    parameter_audit,
    pcum_named_parameters,
    state_dict_sha256,
    training_ready_audit,
)


class B0PCUMDependencyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b0_cfg = load_config(DEFAULT_B0)
        cls.a0_cfg = load_config(DEFAULT_A0)
        cls.candidate_cfg = load_config(DEFAULT_TRAINING_CANDIDATE)
        cls.audit = parameter_audit(cls.b0_cfg, cls.a0_cfg)
        cls.training_ready = training_ready_audit(cls.candidate_cfg)

    def test_confidence_softmax_aggregator_is_parameter_free(self):
        aggregator = RemotePromptAggregator(
            mode="confidence_softmax", temperature=0.10
        )
        self.assertEqual(sum(p.numel() for p in aggregator.parameters()), 0)
        self.assertEqual(sum(b.numel() for b in aggregator.buffers()), 0)

    def test_complete_pcum_is_trainable_and_has_no_persistent_buffers(self):
        self.assertEqual(self.audit["pcum_parameter_tensors"], 34)
        self.assertEqual(self.audit["pcum_parameter_count"], 682753)
        self.assertEqual(self.audit["pcum_trainable_count"], 682753)
        self.assertEqual(self.audit["pcum_buffer_count"], 0)

    def test_b0_checkpoint_is_unchanged_and_has_no_pcum_state(self):
        self.assertEqual(self.audit["checkpoint_sha256"], FROZEN_B0_SHA256)
        self.assertEqual(self.audit["b0_checkpoint_pcum_keys"], [])
        self.assertEqual(len(self.audit["hypothetical_missing_pcum_keys"]), 34)

    def test_a0_checkpoint_contains_trained_pcum_state(self):
        self.assertEqual(self.audit["a0_checkpoint_epoch"], 15)
        self.assertEqual(len(self.audit["a0_checkpoint_pcum_keys"]), 34)
        self.assertEqual(self.audit["a0_checkpoint_pcum_values"], 682753)

    def test_b0_and_a0_token_dimensions_are_compatible(self):
        self.assertEqual(self.b0_cfg.MODEL.BACKBONE.STRIDE, 16)
        self.assertEqual(self.b0_cfg.TEST.SEARCH_SIZE, 256)
        self.assertEqual(self.b0_cfg.TEST.TEMPLATE_SIZE, 128)
        self.assertEqual(self.a0_cfg.MODEL.PCUM.TOKEN_DIM, 192)
        self.assertEqual(self.a0_cfg.MODEL.PCUM.PROMPT_DIM, 192)
        pcum = build_pcum(self.a0_cfg, token_dim=192).eval()
        with torch.inference_mode():
            output = pcum({
                "search": torch.randn(1, 256, 192),
                "template": torch.randn(1, 64, 192),
            })
        self.assertEqual(tuple(output["search_tokens"].shape), (1, 256, 192))

    def test_pcum_has_no_arp_atp_or_pruning_index_argument(self):
        parameters = set(inspect.signature(
            type(self.audit["pcum"]).forward
        ).parameters)
        forbidden = {
            "atp", "arp", "ce_index", "pruning_index",
            "compensation_token", "intermediate_hook",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))

    def test_no_gt_remote_state(self):
        state = build_remote_state(
            scores=[0.8, 0.6],
            motion_reliabilities=[0.9, 0.7],
            source="tracker",
            device=torch.device("cpu"),
            gt_visibility=None,
            apces=[0.8, 0.6],
            bbox_scores=[0.7, 0.5],
        )
        self.assertNotIn("visible", state)
        self.assertEqual(self.a0_cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker")
        self.assertFalse(self.a0_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)

    def test_temperature_and_aggregation_match_a0(self):
        self.assertEqual(
            self.a0_cfg.MODEL.PCUM.REMOTE_AGGREGATION,
            "confidence_softmax",
        )
        self.assertAlmostEqual(
            self.a0_cfg.MODEL.PCUM.REMOTE_WEIGHT_TEMPERATURE,
            0.10,
        )

    def test_disabled_pcum_is_identity(self):
        disabled_cfg = copy.deepcopy(self.a0_cfg)
        disabled_cfg.MODEL.PCUM.ENABLED = False
        pcum = build_pcum(disabled_cfg, token_dim=192).eval()
        search = torch.randn(1, 256, 192)
        with torch.inference_mode():
            output = pcum({"search": search})
        self.assertTrue(torch.equal(output["search_tokens"], search))

    def test_candidate_diff_rejects_backbone_change(self):
        candidate = copy.deepcopy(self.b0_cfg)
        candidate.MODEL.PCUM.ENABLED = True
        candidate.TEST.PCUM.REMOTE_STATE_SOURCE = "tracker"
        candidate.MODEL.BACKBONE.TYPE = "vit_tiny_patch16_224_arp"
        rows = config_diff(self.b0_cfg, candidate)
        status = {row[0]: row[3] for row in rows}
        self.assertTrue(status["MODEL.PCUM.ENABLED"])
        self.assertTrue(status["TEST.PCUM.REMOTE_STATE_SOURCE"])
        self.assertFalse(status["MODEL.BACKBONE.TYPE"])

    def test_frozen_b0_pcum_candidate_is_training_ready(self):
        ready = self.training_ready
        self.assertEqual(ready["failures"], [])
        self.assertEqual(ready["trainable_tensor_count"], 34)
        self.assertEqual(ready["trainable_parameter_count"], 682753)
        self.assertTrue(all(
            name.startswith("pcum.") for name in ready["trainable_names"]
        ))
        self.assertEqual(
            ready["initialization_audit"]["inherited_a0_pcum_parameters"],
            0,
        )

    def test_frozen_b0_pcum_optimizer_contains_only_pcum(self):
        ready = self.training_ready
        self.assertEqual(
            {group["group_name"] for group in ready["optimizer_groups"]},
            {"pcum"},
        )
        self.assertEqual(len(ready["optimizer_groups"]), 1)
        self.assertAlmostEqual(
            ready["optimizer_groups"][0]["lr"],
            self.candidate_cfg.TRAIN.PCUM_LR,
        )

    def test_frozen_b0_pcum_core_and_inference_switches(self):
        cfg = self.candidate_cfg
        self.assertEqual(cfg.MODEL_ROLE, "posthoc_b0_pcum_frozen")
        self.assertTrue(cfg.TRAIN_PCUM_ONLY)
        self.assertEqual(cfg.FIXED_FINAL_EPOCH, 15)
        self.assertEqual(cfg.TRAIN.EPOCH, 15)
        self.assertEqual(cfg.MODEL.BACKBONE.TYPE, "vit_tiny_patch16_224_half")
        self.assertFalse(cfg.MODEL.USE_SEARCH_PROMPT)
        self.assertFalse(cfg.MODEL.BACKBONE.PRUNING_ENABLED)
        self.assertFalse(cfg.MODEL.BACKBONE.TOKEN_COMPENSATION_ENABLED)
        self.assertFalse(cfg.TEST.MCR.ENABLED)
        self.assertEqual(cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker")
        self.assertFalse(cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)
        self.assertEqual(cfg.MODEL.PCUM.REMOTE_AGGREGATION, "confidence_softmax")
        self.assertAlmostEqual(cfg.MODEL.PCUM.REMOTE_WEIGHT_TEMPERATURE, 0.10)

    def test_candidate_uses_frozen_b0_checkpoint_hash(self):
        self.assertEqual(self.candidate_cfg.B0_CHECKPOINT, self.candidate_cfg.MODEL.PRETRAIN_FILE)
        self.assertEqual(self.training_ready["initialization_audit"]["checkpoint_epoch"], 25)
        self.assertEqual(self.audit["checkpoint_sha256"], FROZEN_B0_SHA256)

    def test_pcum_initialization_is_deterministic_by_seed(self):
        model_a = make_training_model(self.candidate_cfg, seed=DEFAULT_INIT_SEED)
        model_b = make_training_model(self.candidate_cfg, seed=DEFAULT_INIT_SEED)
        hash_a = state_dict_sha256([
            (name, parameter.detach())
            for name, parameter in pcum_named_parameters(model_a)
        ])
        hash_b = state_dict_sha256([
            (name, parameter.detach())
            for name, parameter in pcum_named_parameters(model_b)
        ])
        self.assertEqual(hash_a, hash_b)

    def test_pcum_initialization_changes_with_seed(self):
        model_a = make_training_model(self.candidate_cfg, seed=DEFAULT_INIT_SEED)
        model_b = make_training_model(self.candidate_cfg, seed=DEFAULT_INIT_SEED + 1)
        hash_a = state_dict_sha256([
            (name, parameter.detach())
            for name, parameter in pcum_named_parameters(model_a)
        ])
        hash_b = state_dict_sha256([
            (name, parameter.detach())
            for name, parameter in pcum_named_parameters(model_b)
        ])
        self.assertNotEqual(hash_a, hash_b)

    def test_checkpoint_resume_preserves_pcum_state(self):
        model = make_training_model(self.candidate_cfg, seed=DEFAULT_INIT_SEED)
        saved_state = copy.deepcopy(model.state_dict())
        resumed = make_training_model(self.candidate_cfg, seed=DEFAULT_INIT_SEED + 9)
        resumed.load_state_dict(saved_state, strict=True)
        expected = state_dict_sha256([
            (name, value)
            for name, value in saved_state.items()
            if name.startswith("pcum.")
        ])
        actual = state_dict_sha256([
            (name, parameter.detach())
            for name, parameter in pcum_named_parameters(resumed)
        ])
        self.assertEqual(expected, actual)

    def test_one_step_smoke_updates_only_pcum(self):
        smoke = one_step_training_smoke(self.candidate_cfg)
        self.assertTrue(smoke["ready"])
        self.assertTrue(smoke["loss_finite"])
        self.assertTrue(smoke["stats_finite"])
        self.assertTrue(smoke["grad_finite"])
        self.assertTrue(smoke["grad_nonzero"])
        self.assertTrue(smoke["pcum_changed"])
        self.assertTrue(smoke["core_byte_identical"])
        self.assertEqual(smoke["remote_source"], "tracker")
        self.assertFalse(smoke["inference_visible_mask"])
        self.assertEqual(smoke["gt_inference_fields"], [])

    def test_training_ready_fails_when_core_is_trainable(self):
        bad_cfg = copy.deepcopy(self.candidate_cfg)
        bad_cfg.TRAIN.PCUM_RANKING.FREEZE_HEAD = False
        bad_cfg.TRAIN.FREEZE_HEAD = False
        ready = training_ready_audit(bad_cfg)
        self.assertTrue(any("non-PCUM" in failure for failure in ready["failures"]))


if __name__ == "__main__":
    unittest.main()
