import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from lib.config.entertrack.config import cfg, update_config_from_file
from lib.models.entertrack import build_entertrack
from lib.test.tracker.entertrack import EnTeRTrack
from lib.test.tracker.mcr_redetection import (
    EMAMotionPredictor,
    FixedCandidateEvaluator,
    MCRRedetectionManager,
    MultiAnchorCandidateGenerator,
    MultiFrameCandidateVerifier,
    PeriodicSearchScheduler,
    RedetectionCandidate,
    SafeBBoxSwitcher,
    save_mcr_diagnostics,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_config(**overrides):
    values = {
        "ENABLED": True,
        "SHADOW_ONLY": True,
        "MOTION_ENABLED": True,
        "LOCAL_ENABLED": True,
        "GLOBAL_ENABLED": False,
        "REMOTE_VERIFY_ENABLED": True,
        "MULTIFRAME_CONFIRM_ENABLED": True,
        "VERIFY_EVERY_FRAME_WHEN_PENDING": True,
        "LOCAL_INTERVAL": 2,
        "LOCAL_SCALES": [1.5],
        "ANCHORS": ["current"],
        "MAX_REGIONS_PER_TRIGGER": 6,
        "ANCHOR_DEDUP_NORMALIZED_DISTANCE": 0.25,
        "VELOCITY_EMA": 0.8,
        "MAX_HISTORY": 10,
        "CONFIRM_FRAMES": 2,
        "VERIFY_WINDOW": 3,
        "VERIFY_SEARCH_SCALE": 1.5,
        "MIN_CONFIRM_IOU": 0.3,
        "MAX_CONFIRM_CENTER_DISTANCE": 1.0,
        "COOLDOWN_AFTER_SWITCH": 2,
        "UPDATE_FREEZE_FRAMES_AFTER_SWITCH": 3,
        "VISUAL_WEIGHT": 0.55,
        "REMOTE_WEIGHT": 0.20,
        "MOTION_WEIGHT": 0.15,
        "GEOMETRY_WEIGHT": 0.10,
        "MIN_VISUAL_SCORE": 0.3,
        "MIN_CANDIDATE_SCORE": 0.5,
        "SWITCH_MARGIN": 0.05,
        "MAX_AREA_RATIO_CHANGE": 4.0,
        "MAX_ASPECT_RATIO_CHANGE": 3.0,
        "RELIABLE_SCORE_THRESHOLD": 0.3,
        "RELIABLE_APCE_THRESHOLD": 0.0,
        "CURRENT_LARGE_SCALE_GEOMETRY_GUARD": SimpleNamespace(
            ENABLED=False, MIN_SCALE=2.0, MIN_GEOMETRY=0.4),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def candidate_callback(score=0.95, offset=(0.0, 0.0), remote_score=0.9, calls=None):
    def callback(center, scale, anchor_type, reference_bbox):
        if calls is not None:
            calls.append((center, scale, anchor_type, list(reference_bbox)))
        return RedetectionCandidate(
            bbox=[center[0] - 5.0 + offset[0], center[1] - 5.0 + offset[1], 10.0, 10.0],
            visual_score=score,
            remote_score=remote_score,
            anchor_type=anchor_type,
            scale=scale,
            remote_diagnostics={"available": True} if remote_score is not None else None,
        )
    return callback


class TestMCRRedetection(unittest.TestCase):
    @staticmethod
    def guard_candidate(anchor="current", scale=2.0, geometry=0.39):
        return RedetectionCandidate(
            bbox=[10.0, 10.0, 10.0, 10.0],
            visual_score=0.95,
            anchor_type=anchor,
            scale=scale,
            geometry_consistency=geometry,
            total_score=0.95,
        )

    @staticmethod
    def guard_config(enabled=True, **overrides):
        guard = SimpleNamespace(
            ENABLED=enabled, MIN_SCALE=2.0, MIN_GEOMETRY=0.4)
        return make_config(CURRENT_LARGE_SCALE_GEOMETRY_GUARD=guard, **overrides)

    def guard_eligible(self, candidate, enabled=True):
        manager = MCRRedetectionManager(self.guard_config(enabled=enabled))
        return manager._eligible(
            candidate, reference_score=0.2,
            reference_bbox=[10.0, 10.0, 10.0, 10.0],
            image_size=(100, 100),
        )

    def test_full_local_model_shape_and_strict_e4_checkpoint_load(self):
        config_path = os.path.join(
            ROOT, "experiments", "entertrack", "pcum_v2a_mcr_v0_full_local.yaml")
        update_config_from_file(config_path)
        self.assertEqual(cfg.DATA.SEARCH.SIZE, 256)
        self.assertEqual(cfg.DATA.TEMPLATE.SIZE, 128)
        self.assertEqual(cfg.TEST.SEARCH_SIZE, 256)
        self.assertEqual(cfg.TEST.TEMPLATE_SIZE, 128)
        self.assertEqual(cfg.MODEL.BACKBONE.STRIDE, 16)

        model = build_entertrack(cfg, training=False)
        self.assertEqual(tuple(model.backbone.pos_embed_x.shape), (1, 256, 192))
        checkpoint_path = os.path.join(
            ROOT,
            "output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/"
            "train/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/"
            "EnTeRTrack_ep0015.pth.tar",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
        incompatible = model.load_state_dict(state_dict, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_candidate_scale_changes_crop_factor_not_network_output_size(self):
        tracker = EnTeRTrack.__new__(EnTeRTrack)
        tracker.params = SimpleNamespace(search_factor=4.0, search_size=256)
        tracker.state = [40.0, 40.0, 20.0, 20.0]
        tracker.save_all_boxes = False
        tracker.pcum_diagnostics_enabled = False
        tracker.last_pcum_diagnostic = None
        tracker.output_window = torch.zeros(1)
        tracker._pcum_diagnostic_hooks = SimpleNamespace(
            alignment={}, fusion={})
        tracker.preprocessor = SimpleNamespace(
            process=lambda patch_array, mask_array: SimpleNamespace(
                tensors=torch.zeros(1, 3, 256, 256)))
        tracker._network_forward = lambda *args, **kwargs: {}
        tracker._decode_prediction = lambda *args, **kwargs: (
            [128.0, 128.0, 20.0, 20.0],
            torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            torch.tensor([0.9]),
            torch.ones(1, 1, 16, 16),
        )

        sample_calls = []

        def fake_sample_target(image, bbox, search_factor, output_sz):
            sample_calls.append((float(search_factor), int(output_sz), list(bbox)))
            return (
                np.zeros((output_sz, output_sz, 3), dtype=np.uint8),
                1.0,
                np.zeros((output_sz, output_sz), dtype=np.uint8),
            )

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch("lib.test.tracker.entertrack.sample_target", side_effect=fake_sample_target), \
                patch("torch.cuda.is_available", return_value=False):
            tracker.run_redetection_candidate(
                image, center=[50.0, 50.0], scale=1.5,
                anchor_type="current", reference_bbox=tracker.state)
            tracker.run_redetection_candidate(
                image, center=[50.0, 50.0], scale=3.0,
                anchor_type="current", reference_bbox=tracker.state)

        self.assertEqual([item[0] for item in sample_calls], [6.0, 12.0])
        self.assertEqual([item[1] for item in sample_calls], [256, 256])

    def test_disabled_preserves_old_behavior_without_forward(self):
        manager = MCRRedetectionManager(make_config(ENABLED=False))
        calls = []
        bbox = [10.0, 20.0, 30.0, 40.0]
        result = manager.process(10, bbox, 0.1, 1.0, (100, 100),
                                 candidate_callback(calls=calls))
        self.assertEqual(result["bbox"], bbox)
        self.assertFalse(result["switched"])
        self.assertEqual(calls, [])

    def test_geometry_guard_disabled_preserves_old_eligibility(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(geometry=0.10), enabled=False)
        self.assertTrue(eligible)
        self.assertIsNone(reason)

    def test_geometry_guard_rejects_current_at_min_scale_below_threshold(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(scale=2.0, geometry=0.39))
        self.assertFalse(eligible)
        self.assertEqual(reason, "current_large_scale_low_geometry")

    def test_geometry_guard_rejects_current_larger_scale(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(scale=3.0, geometry=0.10))
        self.assertFalse(eligible)
        self.assertEqual(reason, "current_large_scale_low_geometry")

    def test_geometry_guard_does_not_reject_small_current_scale(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(scale=1.5, geometry=0.10))
        self.assertTrue(eligible)
        self.assertIsNone(reason)

    def test_geometry_guard_threshold_is_strict(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(scale=2.0, geometry=0.40))
        self.assertTrue(eligible)
        self.assertIsNone(reason)

    def test_geometry_guard_does_not_reject_motion_anchor(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(anchor="motion", scale=3.0, geometry=0.10))
        self.assertTrue(eligible)
        self.assertIsNone(reason)

    def test_geometry_guard_does_not_reject_last_reliable_anchor(self):
        eligible, reason = self.guard_eligible(
            self.guard_candidate(anchor="last_reliable", scale=3.0, geometry=0.10))
        self.assertTrue(eligible)
        self.assertIsNone(reason)

    def test_geometry_guard_rejection_is_logged_and_cannot_enter_pending(self):
        manager = MCRRedetectionManager(self.guard_config(
            SHADOW_ONLY=False, LOCAL_INTERVAL=1, LOCAL_SCALES=[2.0],
            ANCHORS=["current"], MAX_REGIONS_PER_TRIGGER=1))
        manager.reset([10.0, 10.0, 10.0, 10.0])

        def fixed_scores(candidate, predicted_center, reliable_bbox, reference_bbox):
            if candidate.anchor_type == "main":
                candidate.geometry_consistency = 1.0
                candidate.total_score = 0.2
            else:
                candidate.geometry_consistency = 0.39
                candidate.total_score = 0.95
            return candidate.total_score

        manager.evaluator.evaluate = fixed_scores
        result = manager.process(
            1, [10.0, 10.0, 10.0, 10.0], 0.2, 1.0, (100, 100),
            candidate_callback())
        diagnostic = result["diagnostic"]
        self.assertEqual(
            diagnostic["reject_reason"], "current_large_scale_low_geometry")
        self.assertIsNone(diagnostic["pending_candidate"])
        self.assertIsNone(manager.verifier.pending)
        self.assertEqual(
            diagnostic["candidates"][0]["rejection_reason"],
            "current_large_scale_low_geometry")
        self.assertEqual(diagnostic["safety_rejections"], [{
            "reason": "current_large_scale_low_geometry",
            "anchor": "current",
            "scale": 2.0,
            "geometry_score": 0.39,
            "min_geometry": 0.4,
            "min_scale": 2.0,
            "candidate_total_score": 0.95,
            "current_reference_score": 0.2,
            "frame_id": 1,
        }])
        self.assertEqual(
            manager.summary()["current_large_scale_geometry_reject_count"], 1)

    def test_shadow_mode_does_not_change_bbox_or_main_score(self):
        manager = MCRRedetectionManager(make_config(SHADOW_ONLY=True))
        manager.reset([45.0, 45.0, 10.0, 10.0])
        main_score = 0.2
        first = manager.process(2, [45, 45, 10, 10], main_score, 1.0, (100, 100),
                                candidate_callback())
        second = manager.process(3, [46, 45, 10, 10], main_score, 1.0, (100, 100),
                                 candidate_callback())
        self.assertEqual(second["bbox"], [46.0, 45.0, 10.0, 10.0])
        self.assertFalse(second["switched"])
        self.assertEqual(main_score, 0.2)
        self.assertIsNotNone(first["diagnostic"]["pending_candidate"])

    def test_fixed_interval_scheduler(self):
        scheduler = PeriodicSearchScheduler(local_interval=3)
        self.assertIsNone(scheduler.action(1))
        self.assertIsNone(scheduler.action(2))
        self.assertEqual(scheduler.action(3), "periodic")
        self.assertEqual(scheduler.action(6), "periodic")

    def test_pending_is_verified_every_frame(self):
        scheduler = PeriodicSearchScheduler(local_interval=10)
        self.assertEqual(scheduler.action(3, pending=True), "pending_verify")

    def test_cooldown(self):
        scheduler = PeriodicSearchScheduler(local_interval=1, cooldown_after_switch=2)
        scheduler.switched()
        self.assertIsNone(scheduler.action(1))
        self.assertIsNone(scheduler.action(2))
        self.assertEqual(scheduler.action(3), "periodic")

    def test_ema_motion_prediction(self):
        motion = EMAMotionPredictor(velocity_ema=0.8)
        motion.reset([0, 0, 10, 10])
        motion.observe([10, 0, 10, 10], reliable=True)
        self.assertAlmostEqual(motion.velocity[0], 2.0)
        self.assertEqual(motion.predicted_center.tolist(), [17.0, 5.0])

    def test_three_anchors(self):
        generator = MultiAnchorCandidateGenerator(
            ["current", "motion", "last_reliable"], [1.5], 6, 0.1)
        regions = generator.generate([10, 10, 10, 10], [30, 15], [50, 15], (100, 100))
        self.assertEqual([item["anchor_type"] for item in regions],
                         ["current", "motion", "last_reliable"])

    def test_anchor_deduplication(self):
        generator = MultiAnchorCandidateGenerator(
            ["current", "motion", "last_reliable"], [1.5, 2.0], 6, 0.25)
        regions = generator.generate([10, 10, 10, 10], [15.1, 15.1], [15.2, 15.2], (100, 100))
        self.assertEqual(len(regions), 2)
        self.assertTrue(all(item["anchor_type"] == "current" for item in regions))

    def test_candidate_forward_contract_is_non_committing(self):
        source = inspect.getsource(EnTeRTrack.run_redetection_candidate)
        self.assertNotIn("self.state =", source)
        self.assertIn("torch.set_rng_state", source)
        run_source = inspect.getsource(EnTeRTrack._run_candidate)
        self.assertNotIn("self.state =", run_source)

    def test_missing_remote_renormalizes_weights(self):
        evaluator = FixedCandidateEvaluator(0.55, 0.20, 0.15, 0.10)
        item = RedetectionCandidate([0, 0, 10, 10], visual_score=1.0, remote_score=None)
        score = evaluator.evaluate(item, [5, 5], [0, 0, 10, 10], [0, 0, 10, 10])
        self.assertAlmostEqual(sum(item.available_weights.values()), 1.0)
        self.assertNotIn("remote", item.available_weights)
        self.assertAlmostEqual(score, 1.0)

    def test_score_component_directions(self):
        evaluator = FixedCandidateEvaluator()
        good = RedetectionCandidate([0, 0, 10, 10], 0.9, remote_score=0.9)
        bad = RedetectionCandidate([50, 50, 30, 5], 0.2, remote_score=0.1)
        good_score = evaluator.evaluate(good, [5, 5], [0, 0, 10, 10], [0, 0, 10, 10])
        bad_score = evaluator.evaluate(bad, [5, 5], [0, 0, 10, 10], [0, 0, 10, 10])
        self.assertGreater(good_score, bad_score)
        self.assertGreater(good.motion_consistency, bad.motion_consistency)
        self.assertGreater(good.geometry_consistency, bad.geometry_consistency)

    def test_single_frame_cannot_switch(self):
        verifier = MultiFrameCandidateVerifier(confirm_frames=2)
        self.assertFalse(verifier.start(RedetectionCandidate([0, 0, 10, 10], 0.9)))
        self.assertEqual(verifier.confirm_count, 1)

    def test_consistent_multiframe_candidate_confirms(self):
        verifier = MultiFrameCandidateVerifier(confirm_frames=2)
        verifier.start(RedetectionCandidate([0, 0, 10, 10], 0.9))
        status, reason = verifier.verify(RedetectionCandidate([1, 0, 10, 10], 0.9))
        self.assertEqual(status, "confirmed")
        self.assertIsNone(reason)

    def test_inconsistent_candidate_rejected(self):
        verifier = MultiFrameCandidateVerifier(confirm_frames=2)
        verifier.start(RedetectionCandidate([0, 0, 10, 10], 0.9))
        status, reason = verifier.verify(RedetectionCandidate([80, 80, 10, 10], 0.9))
        self.assertEqual(status, "rejected")
        self.assertIn("confirmation_inconsistent", reason)

    def test_updates_frozen_only_after_active_switch(self):
        shadow = SafeBBoxSwitcher(shadow_only=True, freeze_frames=3)
        shadow.switch([0, 0, 10, 10], [1, 1, 10, 10])
        self.assertTrue(shadow.updates_allowed)
        active = SafeBBoxSwitcher(shadow_only=False, freeze_frames=3)
        active.switch([0, 0, 10, 10], [1, 1, 10, 10])
        self.assertFalse(active.updates_allowed)
        self.assertEqual(active.update_freeze_remaining, 3)

    def test_active_switch_and_protection_period(self):
        manager = MCRRedetectionManager(make_config(SHADOW_ONLY=False))
        manager.reset([45, 45, 10, 10])
        manager.process(2, [45, 45, 10, 10], 0.2, 1.0, (100, 100), candidate_callback())
        result = manager.process(3, [45, 45, 10, 10], 0.2, 1.0, (100, 100), candidate_callback())
        self.assertTrue(result["switched"])
        self.assertFalse(result["updates_allowed"])
        self.assertEqual(result["diagnostic"]["switch_event"], "confirmed_switch")

    def test_sequence_reset_clears_state(self):
        manager = MCRRedetectionManager(make_config())
        manager.reset([0, 0, 10, 10])
        manager.verifier.start(RedetectionCandidate([1, 1, 10, 10], 0.9))
        manager.scheduler.switched()
        manager.reset([20, 20, 10, 10])
        self.assertIsNone(manager.verifier.pending)
        self.assertEqual(manager.scheduler.cooldown_remaining, 0)
        self.assertEqual(manager.motion.current_center.tolist(), [25.0, 25.0])

    def test_global_disabled_has_no_global_behavior(self):
        manager = MCRRedetectionManager(make_config(GLOBAL_ENABLED=False, LOCAL_INTERVAL=100))
        calls = []
        manager.reset([0, 0, 10, 10])
        manager.process(1, [0, 0, 10, 10], 0.9, 1.0, (100, 100),
                        candidate_callback(calls=calls))
        self.assertEqual(calls, [])

    def test_global_enabled_is_explicitly_unimplemented(self):
        manager = MCRRedetectionManager(make_config(GLOBAL_ENABLED=True))
        with self.assertRaisesRegex(NotImplementedError, "global tiled search"):
            manager.process(1, [0, 0, 10, 10], 0.9, 1.0, (100, 100), candidate_callback())

    def test_no_gt_public_process_interface(self):
        parameters = inspect.signature(MCRRedetectionManager.process).parameters
        forbidden = {"gt_bbox", "target_visible", "visibility", "iou_with_gt"}
        self.assertTrue(forbidden.isdisjoint(parameters))

    def test_old_yaml_loads_with_mcr_disabled_default(self):
        path = os.path.join(ROOT, "experiments", "entertrack", "pcum_v2_a0_softmax_t010_ep0015.yaml")
        update_config_from_file(path)
        self.assertFalse(cfg.TEST.MCR.ENABLED)

    def test_all_ablation_configs_load_independently(self):
        expected = {
            "pcum_v2a_mcr_v0_shadow_val.yaml": (True, True, True, 3),
            "pcum_v2a_mcr_v0_active_smoke.yaml": (False, True, True, 3),
            "pcum_v2a_mcr_v0_current_anchor_only.yaml": (False, True, True, 1),
            "pcum_v2a_mcr_v0_current_motion_anchors.yaml": (False, True, True, 2),
            "pcum_v2a_mcr_v0_no_remote_verify.yaml": (False, False, True, 3),
            "pcum_v2a_mcr_v0_no_multiframe_confirm.yaml": (False, True, False, 3),
            "pcum_v2a_mcr_v0_full_local.yaml": (False, True, True, 3),
            "pcum_v2a_mcr_v0_full_local_safegeom.yaml": (False, True, True, 3),
        }
        for name, values in expected.items():
            path = os.path.join(ROOT, "experiments", "entertrack", name)
            update_config_from_file(path)
            self.assertTrue(cfg.TEST.MCR.ENABLED, name)
            self.assertEqual(cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker", name)
            self.assertFalse(cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK, name)
            self.assertEqual(cfg.TEST.MCR.SHADOW_ONLY, values[0], name)
            self.assertEqual(cfg.TEST.MCR.REMOTE_VERIFY_ENABLED, values[1], name)
            self.assertEqual(cfg.TEST.MCR.MULTIFRAME_CONFIRM_ENABLED, values[2], name)
            self.assertEqual(len(cfg.TEST.MCR.ANCHORS), values[3], name)
            self.assertEqual(cfg.DATA.SEARCH.SIZE, 256, name)
            self.assertEqual(cfg.DATA.TEMPLATE.SIZE, 128, name)
            self.assertEqual(cfg.TEST.SEARCH_SIZE, 256, name)
            self.assertEqual(cfg.TEST.TEMPLATE_SIZE, 128, name)
            self.assertEqual(cfg.MODEL.BACKBONE.STRIDE, 16, name)

    def test_diagnostic_writer(self):
        record = {"frame_id": 1, "trigger_reason": None, "candidates": [{
                      "total_score": 0.8,
                      "rejection_reason": "current_large_scale_low_geometry",
                  }],
                  "pending_candidate": None, "selected_candidate": None,
                  "additional_forward_count": 0, "switch_event": None,
                  "reject_reason": None}
        with tempfile.TemporaryDirectory() as directory:
            jsonl, summary = save_mcr_diagnostics(directory, "sequence", [record])
            self.assertTrue(os.path.isfile(jsonl))
            self.assertTrue(os.path.isfile(summary))
            with open(summary) as handle:
                summary_data = json.load(handle)
            self.assertEqual(
                summary_data["current_large_scale_geometry_reject_count"], 1)


if __name__ == "__main__":
    unittest.main()
