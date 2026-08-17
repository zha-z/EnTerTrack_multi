import os
import sys
import unittest
import copy
import importlib.util
import tempfile
import contextlib
import io
import csv
import json
import math

import torch
from torch import nn
import torch.nn.functional as F
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from easydict import EasyDict as edict
from lib.models.entertrack.pcum import (  # noqa: E402
    MultiLayerPromptEncoder,
    PCUM,
    PromptAligner,
    PromptConsistencyLoss,
    PromptFusion,
    RemoteSuppressionGate,
    RemotePromptAggregator,
    SaliencyTokenSelector,
    build_pcum,
    build_pseudo_remote_prompts,
    validate_remote_aggregation,
)
from lib.config.entertrack.config import cfg  # noqa: E402
from lib.config.entertrack.config import update_config_from_file  # noqa: E402
from lib.test.evaluation.tracker import _pcum_motion_reliability  # noqa: E402
from lib.test.evaluation.tracker import Tracker as EvaluationTracker  # noqa: E402
from lib.test.tracker.entertrack import (  # noqa: E402
    EnTeRTrack,
    deterministic_reliability_selector_decision,
    validate_reliability_selector,
)
from lib.test.tracker.motion_state import (  # noqa: E402
    MOTION_DIAGNOSTIC_FIELDS,
    MotionState,
    MotionStateManager,
    save_motion_diagnostics,
)
from lib.test.evaluation.running import run_three_multi_sequence  # noqa: E402
from lib.test.evaluation.running import _save_tracker_output  # noqa: E402
from lib.test.utils.pcum_diagnostics import (  # noqa: E402
    PCUMDiagnosticHooks,
    build_frame_diagnostic_row,
    diagnostic_filename,
    normalized_response_entropy,
    visibility_for_remote_selection,
)
from lib.test.utils.pcum_remote_ablation import RemotePromptAblator  # noqa: E402
from lib.test.utils.pcum_remote_state import (  # noqa: E402
    build_remote_state,
    read_gt_visibility,
    validate_remote_state_source,
)
from lib.train.data.sampler_threemdot import TrackingSamplerThreeMDOT  # noqa: E402
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT  # noqa: E402
from lib.train.optimizer_groups import build_optimizer_param_groups  # noqa: E402
from lib.train.pcum_freeze import (  # noqa: E402
    apply_pcum_ranking_freeze,
    assert_pcum_frozen_batchnorm_eval,
    assert_optimizer_has_only_pcum_params,
    set_pcum_frozen_modules_eval,
)


class MotionStateShadowTests(unittest.TestCase):
    def _manager(self, **kwargs):
        defaults = {
            "score_low": 0.5,
            "score_recover": 0.6,
            "apce_low": 10.0,
            "apce_recover": 12.0,
            "motion_residual_high": 2.0,
            "border_margin": 0.0,
            "k_lost": 3,
            "k_normal": 2,
            "k_recover": 2,
        }
        defaults.update(kwargs)
        manager = MotionStateManager(**defaults)
        manager.reset([10, 10, 20, 20], image_size=(100, 100))
        return manager

    def _update(self, manager, frame_id, bbox=None, score=0.8, apce=20.0,
                response=None):
        return manager.update_prediction_only(
            frame_id=frame_id,
            predicted_bbox=bbox or [10 + frame_id, 10, 20, 20],
            max_score=score,
            apce=apce,
            response=response,
            image_size=(100, 100),
        )

    def test_motion_state_defaults_are_disabled(self):
        self.assertFalse(cfg.TEST.MOTION_STATE.ENABLED)
        self.assertTrue(cfg.TEST.MOTION_STATE.SHADOW_ONLY)
        self.assertFalse(cfg.TEST.MOTION_STATE.LOG_ENABLED)

    def test_motion_shadow_validation_config_is_no_gt(self):
        shadow_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/pcum_v2a_motion_state_shadow_val.yaml",
            base_cfg=shadow_cfg,
        )
        self.assertTrue(shadow_cfg.TEST.MOTION_STATE.ENABLED)
        self.assertTrue(shadow_cfg.TEST.MOTION_STATE.SHADOW_ONLY)
        self.assertTrue(shadow_cfg.TEST.MOTION_STATE.LOG_ENABLED)
        self.assertEqual(shadow_cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker")
        self.assertFalse(shadow_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)
        self.assertEqual(
            shadow_cfg.MODEL.PCUM.REMOTE_AGGREGATION,
            "confidence_softmax",
        )
        self.assertAlmostEqual(
            shadow_cfg.MODEL.PCUM.REMOTE_WEIGHT_TEMPERATURE,
            0.10,
        )

    def test_shadow_disabled_returns_original_output_object(self):
        tracker = EnTeRTrack.__new__(EnTeRTrack)
        tracker.motion_state_manager = None
        output = {"target_bbox": [1, 2, 3, 4], "score": 0.7}
        result = tracker._attach_motion_shadow_diagnostics({}, output)
        self.assertIs(result, output)
        self.assertEqual(result, {"target_bbox": [1, 2, 3, 4], "score": 0.7})

    def test_shadow_enabled_does_not_change_bbox_or_score(self):
        tracker = EnTeRTrack.__new__(EnTeRTrack)
        tracker.motion_state_manager = self._manager()
        tracker.motion_state_log_enabled = True
        tracker.frame_id = 1
        output = {"target_bbox": [11, 10, 20, 20], "score": 0.7}
        original = copy.deepcopy(output)
        candidate = {
            "target_bbox": list(output["target_bbox"]),
            "max_score": torch.tensor([0.8]),
            "apce": torch.tensor([20.0]),
            "response": torch.ones(1, 1, 4, 4),
            "image": torch.zeros(100, 100, 3).numpy(),
        }
        result = tracker._attach_motion_shadow_diagnostics(candidate, output)
        self.assertEqual(result["target_bbox"], original["target_bbox"])
        self.assertEqual(result["score"], original["score"])
        self.assertIn("motion_state_diagnostics", result)

    def test_reset_clears_sequence_state(self):
        manager = self._manager()
        for frame_id in range(1, 4):
            self._update(manager, frame_id, score=0.1, apce=1.0)
        self.assertEqual(manager.state, MotionState.LOST)
        manager.reset([30, 30, 10, 10], image_size=(100, 100))
        self.assertEqual(manager.state, MotionState.NORMAL)
        self.assertEqual(manager.low_quality_count, 0)
        self.assertEqual(len(manager.reliable_centers), 1)

    def test_constant_velocity_ema_prediction(self):
        manager = self._manager(velocity_ema=0.5)
        self._update(manager, 1, bbox=[12, 10, 20, 20])
        record = self._update(manager, 2, bbox=[14, 10, 20, 20])
        self.assertEqual(record["velocity"], [1.0, 0.0])
        self.assertEqual(record["predicted_motion_center"], [23.0, 20.0])

    def test_k_low_quality_frames_enter_lost(self):
        manager = self._manager(k_lost=3)
        for frame_id in range(1, 4):
            record = self._update(manager, frame_id, score=0.1, apce=1.0)
        self.assertEqual(record["state"], MotionState.LOST.value)

    def test_recovery_reaches_normal_after_k_frames(self):
        manager = self._manager(k_lost=2, k_recover=2)
        self._update(manager, 1, score=0.1, apce=1.0)
        self._update(manager, 2, score=0.1, apce=1.0)
        self.assertEqual(manager.state, MotionState.LOST)
        self._update(manager, 3, score=0.8, apce=20.0)
        record = self._update(manager, 4, score=0.8, apce=20.0)
        self.assertEqual(record["state"], MotionState.NORMAL.value)

    def test_single_noise_frame_does_not_oscillate(self):
        manager = self._manager(k_normal=2)
        first = self._update(manager, 1, score=0.1, apce=1.0)
        second = self._update(manager, 2, score=0.8, apce=20.0)
        third = self._update(manager, 3, score=0.8, apce=20.0)
        self.assertEqual(first["state"], MotionState.UNCERTAIN.value)
        self.assertEqual(second["state"], MotionState.UNCERTAIN.value)
        self.assertEqual(third["state"], MotionState.NORMAL.value)

    def test_missing_response_is_explicitly_unavailable(self):
        record = self._update(self._manager(), 1, response=None)
        self.assertIsNone(record["response_entropy"])
        self.assertIn("response_entropy", record["missing_fields"])

    def test_prediction_only_api_has_no_gt_arguments(self):
        import inspect
        parameters = inspect.signature(
            MotionStateManager.update_prediction_only
        ).parameters
        forbidden = {"gt_bbox", "gt_visibility", "target_visible", "oracle_mask"}
        self.assertTrue(forbidden.isdisjoint(parameters))

    def test_log_fields_and_summary_are_complete(self):
        manager = self._manager()
        records = [manager.get_diagnostics(), self._update(manager, 1)]
        for field in MOTION_DIAGNOSTIC_FIELDS:
            self.assertIn(field, records[-1])
        with tempfile.TemporaryDirectory() as results_dir:
            jsonl_path, summary_path = save_motion_diagnostics(
                results_dir, "md0001-1", records
            )
            self.assertTrue(os.path.isfile(jsonl_path))
            with open(summary_path) as file_handle:
                summary = json.load(file_handle)
            self.assertEqual(summary["sequences"]["md0001-1"]["frame_count"], 2)

    def test_managers_do_not_share_sequence_state(self):
        first = self._manager(k_lost=2)
        second = self._manager(k_lost=2)
        self._update(first, 1, score=0.1, apce=1.0)
        self._update(first, 2, score=0.1, apce=1.0)
        self.assertEqual(first.state, MotionState.LOST)
        self.assertEqual(second.state, MotionState.NORMAL)


class FakeThreeMDOTDataset:
    def __init__(self):
        self.sequence_list = ["md0001-1", "md0001-2", "md0001-3"]
        self.seq_per_class = {"md0001": [0, 1, 2]}
        self.visible = [
            torch.tensor([False, True, True, True, False]),
            torch.tensor([False, True, True, True, False]),
            torch.tensor([False, True, True, True, False]),
        ]

    def __len__(self):
        return 3

    def get_name(self):
        return "THREEMDOT"

    def is_video_sequence(self):
        return True

    def get_num_sequences(self):
        return 3

    def get_sequence_info(self, seq_id):
        visible = self.visible[seq_id]
        return {
            "visible": visible,
            "valid": torch.ones_like(visible).bool(),
        }

    def get_frames(self, seq_id, frame_ids, seq_info_dict):
        frames = [
            torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
            for _ in frame_ids
        ]
        anno = {
            "bbox": [
                torch.tensor([2.0, 2.0, 3.0, 3.0])
                for _ in frame_ids
            ]
        }
        return frames, anno, {"object_class_name": "target"}


class FakeSequence:
    def __init__(self, name):
        self.name = name
        self.dataset = "threemdot"
        self.object_ids = None


class FakeTrackerInfo:
    def __init__(self, results_dir, save_decision_log=True):
        self.name = "entertrack"
        self.parameter_name = "fake"
        self.run_id = 0
        self.results_dir = results_dir
        self.called = False
        self.save_decision_log = save_decision_log

    def get_parameters(self):
        pcum = edict({"SAVE_DECISION_LOG": self.save_decision_log})
        return edict({"cfg": edict({"TEST": edict({"PCUM": pcum})})})

    def Fuse_three_multi_run_sequence(self, seq_a, seq_b, seq_c, debug=False):
        self.called = True
        output = {
            "target_bbox": [[1, 2, 3, 4]],
            "time": [1.0],
            "max_score": [0.5],
            "APCE": [120.0],
            "pcum_decision": [[0.0] * 8],
        }
        return copy.deepcopy(output), copy.deepcopy(output), copy.deepcopy(output)


def mark_valid(data):
    data["valid"] = True
    return data


class RemotePromptAblationTest(unittest.TestCase):
    def test_default_mode_is_normal(self):
        self.assertEqual(cfg.TEST.PCUM.REMOTE_ABLATION, "normal")
        self.assertEqual(cfg.TEST.PCUM.REMOTE_ABLATION_OFFSET, 10)

    def test_normal_does_not_change_prompt(self):
        prompt = torch.randn(1, 4, 8, dtype=torch.float32)
        ablator = RemotePromptAblator(mode="normal", offset=10)
        output = ablator.apply(0, prompt, target_device=prompt.device)
        self.assertTrue(torch.equal(output, prompt))
        self.assertEqual(output.shape, prompt.shape)
        self.assertEqual(output.dtype, prompt.dtype)
        self.assertEqual(output.device, prompt.device)

    def test_zero_preserves_prompt_metadata_and_remote_count(self):
        prompts = [
            torch.randn(1, 4, 8, dtype=torch.float32),
            torch.randn(1, 4, 8, dtype=torch.float64),
        ]
        ablator = RemotePromptAblator(mode="zero", offset=10)
        outputs = [
            ablator.apply(index, prompt, target_device=prompt.device)
            for index, prompt in enumerate(prompts)
        ]
        self.assertEqual(len(outputs), len(prompts))
        for output, prompt in zip(outputs, prompts):
            self.assertTrue(torch.equal(output, torch.zeros_like(prompt)))
            self.assertEqual(output.shape, prompt.shape)
            self.assertEqual(output.dtype, prompt.dtype)
            self.assertEqual(output.device, prompt.device)

    def test_temporal_shuffle_is_causal_and_per_uav(self):
        ablator = RemotePromptAblator(mode="temporal_shuffle", offset=10)
        outputs = []
        for frame_index in range(1, 13):
            prompts = [
                torch.full((1, 2, 3), float(source_index * 100 + frame_index))
                for source_index in range(3)
            ]
            ablator.record(prompts)
            outputs.append([
                ablator.apply(source_index, prompts[source_index])
                for source_index in range(3)
            ])

        for frame_index in range(10):
            for source_index in range(3):
                expected = float(source_index * 100 + 1)
                self.assertTrue(torch.all(
                    outputs[frame_index][source_index] == expected
                ))

        # Frame 11 uses frame 1; frame 12 uses frame 2.
        for source_index in range(3):
            self.assertTrue(torch.all(outputs[10][source_index] == source_index * 100 + 1))
            self.assertTrue(torch.all(outputs[11][source_index] == source_index * 100 + 2))

    def test_temporal_shuffle_reset_clears_history(self):
        ablator = RemotePromptAblator(mode="temporal_shuffle", offset=10)
        prompt = torch.ones(1, 2, 3)
        ablator.record([prompt])
        self.assertTrue(torch.equal(ablator.apply(0, prompt), prompt))

        ablator.reset()
        with self.assertRaises(RuntimeError):
            ablator.apply(0, prompt)

        new_prompt = torch.full((1, 2, 3), 9.0)
        ablator.record([new_prompt])
        self.assertTrue(torch.equal(ablator.apply(0, new_prompt), new_prompt))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            RemotePromptAblator(mode="invalid")


class RemoteStateSourceTest(unittest.TestCase):
    class VisibilityProbe:
        def __init__(self, values):
            self.values = values
            self.access_count = 0

        @property
        def target_visible(self):
            self.access_count += 1
            return self.values

    def test_default_source_is_tracker(self):
        self.assertEqual(cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker")

    def test_yaml_without_source_keeps_tracker_default(self):
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/"
            "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_raw.yaml",
            base_cfg=local_cfg,
        )
        self.assertEqual(local_cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker")

    def test_gt_legacy_yaml_is_explicit(self):
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/"
            "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_gt_legacy.yaml",
            base_cfg=local_cfg,
        )
        self.assertEqual(local_cfg.TEST.PCUM.REMOTE_STATE_SOURCE, "gt_legacy")

    def test_tracker_without_visible_mask_does_not_read_gt(self):
        sequences = [self.VisibilityProbe([True, False]) for _ in range(3)]
        visibility = read_gt_visibility("tracker", False, sequences, 1)
        self.assertIsNone(visibility)
        self.assertEqual([seq.access_count for seq in sequences], [0, 0, 0])

    def test_gt_legacy_explicitly_reads_visibility(self):
        sequences = [
            self.VisibilityProbe([True, False]),
            self.VisibilityProbe([True, True]),
            self.VisibilityProbe([True, False]),
        ]
        visibility = read_gt_visibility("gt_legacy", False, sequences, 1)
        self.assertEqual(visibility, [False, True, False])
        self.assertEqual([seq.access_count for seq in sequences], [1, 1, 1])

    def test_oracle_mask_explicitly_reads_visibility(self):
        sequences = [self.VisibilityProbe([True, False]) for _ in range(3)]
        visibility = read_gt_visibility("tracker", True, sequences, 1)
        self.assertEqual(visibility, [False, False, False])
        self.assertEqual([seq.access_count for seq in sequences], [1, 1, 1])

    def test_tracker_state_uses_prediction_only(self):
        state = build_remote_state(
            scores=[0.8, 0.4],
            motion_reliabilities=[0.5, 1.0],
            source="tracker",
            device=torch.device("cpu"),
            use_motion_confidence=True,
        )
        self.assertNotIn("visible", state)
        self.assertAlmostEqual(state["score"].item(), 0.6, places=6)
        self.assertAlmostEqual(state["confidence"].item(), 0.5, places=6)

        lower_state = build_remote_state(
            scores=[0.2, 0.1],
            motion_reliabilities=[0.1, 0.2],
            source="tracker",
            device=torch.device("cpu"),
            use_motion_confidence=True,
        )
        self.assertLess(lower_state["confidence"], state["confidence"])

    def test_none_source_returns_none(self):
        state = build_remote_state(
            scores=[0.8, 0.4],
            motion_reliabilities=[0.5, 1.0],
            source="none",
            device=torch.device("cpu"),
        )
        self.assertIsNone(state)

    def test_invalid_source_raises(self):
        with self.assertRaises(ValueError):
            validate_remote_state_source("annotation")

    def test_gt_legacy_matches_previous_state_formula(self):
        state = build_remote_state(
            scores=[0.8, 0.4],
            motion_reliabilities=[0.5, 1.0],
            source="gt_legacy",
            device=torch.device("cpu"),
            use_motion_confidence=False,
            gt_visibility=[True, False],
        )
        self.assertAlmostEqual(state["score"].item(), 0.6, places=6)
        self.assertAlmostEqual(state["visible"].item(), 0.5, places=6)
        self.assertAlmostEqual(state["confidence"].item(), 0.4, places=6)
        self.assertAlmostEqual(state["motion_reliability"].item(), 0.25, places=6)

        prompt = torch.randn(1, 4, 8)
        output = RemotePromptAblator("normal").apply(0, prompt)
        self.assertTrue(torch.equal(output, prompt))


class RemoteAggregationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.prompts = torch.randn(2, 3, 4, 8)

    def test_default_mode_is_mean(self):
        self.assertEqual(cfg.MODEL.PCUM.REMOTE_AGGREGATION, "mean")

    def test_mean_matches_legacy_mean_exactly(self):
        aggregator = RemotePromptAggregator(mode="mean")
        output = aggregator(self.prompts)
        legacy = self.prompts.mean(dim=1)
        self.assertTrue(torch.equal(output["prompt"], legacy))

    def test_confidence_softmax_weights_sum_to_one(self):
        state = {
            "per_remote_score": torch.tensor([
                [0.9, 0.4, 0.2],
                [0.2, 0.6, 0.8],
            ]),
            "per_remote_apce": torch.tensor([
                [0.8, 0.5, 0.3],
                [0.3, 0.7, 0.9],
            ]),
            "per_remote_valid": torch.ones(2, 3, dtype=torch.bool),
        }
        output = RemotePromptAggregator(
            mode="confidence_softmax", temperature=0.25
        )(self.prompts, state)
        self.assertTrue(torch.allclose(
            output["weights"].sum(dim=1), torch.ones(2)
        ))
        self.assertTrue(torch.isfinite(output["prompt"]).all())

    def test_invalid_remote_has_zero_weight(self):
        state = {
            "per_remote_score": torch.tensor([[0.9, 0.8, 0.7]]).expand(2, -1),
            "per_remote_valid": torch.tensor([
                [True, False, True],
                [False, True, True],
            ]),
        }
        output = RemotePromptAggregator("confidence_softmax")(
            self.prompts, state
        )
        self.assertEqual(float(output["weights"][0, 1]), 0.0)
        self.assertEqual(float(output["weights"][1, 0]), 0.0)

    def test_all_invalid_falls_back_stably(self):
        state = {
            "per_remote_score": torch.tensor([[0.9, 0.8, 0.7]]).expand(2, -1),
            "per_remote_valid": torch.zeros(2, 3, dtype=torch.bool),
        }
        output = RemotePromptAggregator("confidence_softmax")(
            self.prompts, state
        )
        expected = torch.full((2, 3), 1.0 / 3.0)
        self.assertTrue(torch.allclose(output["weights"], expected))
        self.assertTrue(bool(output["fallback"].all()))
        self.assertTrue(torch.isfinite(output["prompt"]).all())

    def test_missing_metrics_use_uniform_fallback(self):
        state = {"per_remote_valid": torch.ones(2, 3, dtype=torch.bool)}
        output = RemotePromptAggregator("confidence_sigmoid")(
            self.prompts, state
        )
        self.assertTrue(torch.allclose(
            output["weights"], torch.full((2, 3), 1.0 / 3.0)
        ))
        self.assertTrue(torch.equal(
            output["quality"], torch.zeros_like(output["quality"])
        ))
        self.assertTrue(bool(output["fallback"].all()))

    def test_remote_state_metrics_are_detached(self):
        score = torch.tensor([[0.8, 0.4, 0.2]], requires_grad=True)
        state = {
            "per_remote_score": score.expand(2, -1),
            "per_remote_valid": torch.ones(2, 3, dtype=torch.bool),
        }
        prompts = self.prompts.clone().requires_grad_(True)
        output = RemotePromptAggregator("confidence_softmax")(prompts, state)
        output["prompt"].sum().backward()
        self.assertIsNone(score.grad)
        self.assertIsNotNone(prompts.grad)

    def test_illegal_mode_raises(self):
        with self.assertRaises(ValueError):
            validate_remote_aggregation("oracle")
        with self.assertRaises(ValueError):
            RemotePromptAggregator("oracle")


class ReliabilitySelectorTest(unittest.TestCase):
    def test_default_selector_is_none(self):
        self.assertEqual(cfg.TEST.PCUM.RELIABILITY_SELECTOR, "none")
        self.assertEqual(cfg.TEST.PCUM.SELECTOR_MARGIN, 0.0)
        self.assertEqual(cfg.TEST.PCUM.SELECTOR_MOTION_THRESHOLD, 0.0)
        self.assertTrue(cfg.TEST.PCUM.SELECTOR_DIAGNOSTICS)

    def test_validate_selector_modes(self):
        self.assertEqual(validate_reliability_selector("none"), "none")
        self.assertEqual(
            validate_reliability_selector("deterministic"),
            "deterministic",
        )
        with self.assertRaises(ValueError):
            validate_reliability_selector("oracle")

    def test_deterministic_selector_accepts_confident_collab(self):
        decision = deterministic_reliability_selector_decision(
            local_confidence=0.20,
            collaborative_confidence=0.27,
            collaborative_motion_reliability=0.5,
            margin=0.05,
            motion_threshold=0.2,
        )
        self.assertTrue(decision["use_collaborative"])
        self.assertAlmostEqual(decision["confidence_delta"], 0.07)

    def test_deterministic_selector_rejects_low_margin(self):
        decision = deterministic_reliability_selector_decision(
            local_confidence=0.20,
            collaborative_confidence=0.23,
            collaborative_motion_reliability=0.5,
            margin=0.05,
            motion_threshold=0.2,
        )
        self.assertFalse(decision["use_collaborative"])

    def test_deterministic_selector_rejects_low_motion(self):
        decision = deterministic_reliability_selector_decision(
            local_confidence=0.20,
            collaborative_confidence=0.40,
            collaborative_motion_reliability=0.1,
            margin=0.05,
            motion_threshold=0.2,
        )
        self.assertFalse(decision["use_collaborative"])


class PCUMShapeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = 2
        self.search_len = 25
        self.template_len = 9
        self.dim = 32
        self.search = torch.randn(self.batch, self.search_len, self.dim)
        self.template = torch.randn(self.batch, self.template_len, self.dim)

    def test_selector_sources(self):
        total_len = self.search_len + self.template_len
        attention = torch.randn(self.batch, 4, total_len)
        confidence = torch.rand(self.batch, total_len)

        for source in ("attention_score", "feature_norm", "confidence_score"):
            selector = SaliencyTokenSelector(topk=6, source=source)
            out = selector(
                self.search,
                template_feature=self.template,
                attention_score=attention,
                confidence_score=confidence,
            )
            self.assertEqual(out["tokens"].shape, (self.batch, 6, self.dim))
            self.assertEqual(out["indices"].shape, (self.batch, 6))
            self.assertEqual(out["scores"].shape, (self.batch, 6))
            self.assertEqual(out["is_search"].shape, (self.batch, 6))

    def test_prompt_encoder_single_and_multi_layer(self):
        encoder = MultiLayerPromptEncoder(
            input_dim=self.dim,
            prompt_dim=self.dim,
            num_prompts=4,
            num_heads=4,
            depth=1,
        )

        single = encoder(self.search)
        multi = encoder([self.search, self.search + 0.1])
        self.assertEqual(single.shape, (self.batch, 4, self.dim))
        self.assertEqual(multi.shape, (self.batch, 4, self.dim))

    def test_aligner_local_and_remote(self):
        local = torch.randn(self.batch, 4, self.dim)
        remote = torch.randn(self.batch, 3, 4, self.dim)
        aligner = PromptAligner(prompt_dim=self.dim, gate="cosine_confidence")

        local_out = aligner(local)
        self.assertEqual(local_out["prompt"].shape, local.shape)
        self.assertIsNone(local_out["gate"])

        remote_out = aligner(
            local,
            remote_prompt=remote,
            local_state={"score": torch.tensor([0.4, 0.8])},
            remote_state={"score": torch.tensor([0.9, 0.5])},
        )
        self.assertEqual(remote_out["prompt"].shape, local.shape)
        self.assertEqual(remote_out["gate"].shape, (self.batch, 1, 1))

        invalid_out = aligner(
            local,
            remote_prompt=remote[:, 0],
            remote_state={"score": torch.zeros(self.batch)},
        )
        self.assertTrue(torch.equal(invalid_out["gate"], torch.zeros_like(invalid_out["gate"])))

    def test_fusion_modes(self):
        prompt = torch.randn(self.batch, 4, self.dim)
        for mode in ("gated_add", "film"):
            fusion = PromptFusion(token_dim=self.dim, prompt_dim=self.dim, mode=mode)
            fused = fusion(self.search, prompt)
            self.assertEqual(fused.shape, self.search.shape)

    def test_fusion_scale_soft_cap(self):
        fusion = PromptFusion(
            token_dim=self.dim,
            prompt_dim=self.dim,
            mode="film",
            init_scale=10.0,
            max_scale=0.03,
        )
        self.assertLessEqual(abs(fusion._residual_scale().item()), 0.030001)

    def test_pcum_disabled_identity_and_enabled_shapes(self):
        disabled = PCUM(token_dim=self.dim, prompt_dim=self.dim, enabled=False)
        disabled_out = disabled({"search": self.search, "template": self.template})
        self.assertIs(disabled_out["search_tokens"], self.search)
        self.assertIsNone(disabled_out["local_prompt"])

        enabled = PCUM(
            token_dim=self.dim,
            prompt_dim=self.dim,
            num_prompts=4,
            topk=8,
            enabled=True,
        )
        out = enabled(
            {
                "search": self.search,
                "template": self.template,
                "layers": [self.search, self.search + 0.2],
                "confidence_score": torch.rand(self.batch, self.search_len),
            },
            remote_prompts=torch.randn(self.batch, 4, self.dim),
            remote_states={"score": torch.tensor([0.7, 0.6])},
        )
        self.assertEqual(out["search_tokens"].shape, self.search.shape)
        self.assertEqual(out["local_prompt"].shape, (self.batch, 4, self.dim))
        self.assertEqual(out["aligned_prompt"].shape, (self.batch, 4, self.dim))
        self.assertEqual(out["selected_indices"].shape, (self.batch, 8))

    def test_build_pcum_from_config_defaults_disabled(self):
        module = build_pcum(cfg, token_dim=self.dim)
        out = module({"search": self.search})
        self.assertIs(out["search_tokens"], self.search)
        self.assertIsNone(out["local_prompt"])

    def test_pseudo_remote_prompts_batch_and_single_fallback(self):
        prompt = torch.randn(self.batch, 4, self.dim)
        remote = build_pseudo_remote_prompts(prompt, noise_std=0.0)
        self.assertTrue(torch.equal(remote[0], prompt[1]))
        self.assertTrue(torch.equal(remote[1], prompt[0]))

        single = torch.randn(1, 4, self.dim)
        fallback = build_pseudo_remote_prompts(single, noise_std=0.01)
        self.assertEqual(fallback.shape, single.shape)
        self.assertFalse(torch.equal(fallback, single))

        multi_view = torch.randn(self.batch, 2, 4, self.dim)
        remote_view = build_pseudo_remote_prompts(multi_view, noise_std=0.0)
        self.assertTrue(torch.equal(remote_view, multi_view[:, 1]))

    def test_prompt_consistency_loss_stop_gradient(self):
        local = torch.randn(self.batch, 4, self.dim, requires_grad=True)
        remote = torch.randn(self.batch, 4, self.dim, requires_grad=True)
        loss_fn = PromptConsistencyLoss(stop_gradient_teacher=True)
        loss = loss_fn(local, remote)
        loss.backward()
        self.assertIsNotNone(local.grad)
        self.assertIsNone(remote.grad)

    def test_actor_prompt_alignment_loss_forward(self):
        local_cfg = edict(cfg.copy())
        local_cfg.MODEL = edict(cfg.MODEL.copy())
        local_cfg.MODEL.PCUM = edict(cfg.MODEL.PCUM.copy())
        local_cfg.MODEL.PCUM.ENABLED = True
        local_cfg.TRAIN = edict(cfg.TRAIN.copy())
        local_cfg.TRAIN.PCUM = edict(cfg.TRAIN.PCUM.copy())
        local_cfg.TRAIN.PCUM.ALIGN_LOSS_WEIGHT = 0.5
        local_cfg.TRAIN.PCUM.USE_PSEUDO_REMOTE = True
        local_cfg.TRAIN.PCUM.STOP_GRAD_TEACHER = True
        local_cfg.DATA = edict(cfg.DATA.copy())
        local_cfg.DATA.SEARCH = edict(cfg.DATA.SEARCH.copy())
        local_cfg.DATA.SEARCH.SIZE = 320
        local_cfg.MODEL.BACKBONE = edict(cfg.MODEL.BACKBONE.copy())
        local_cfg.MODEL.BACKBONE.STRIDE = 16

        settings = edict({"batchsize": self.batch})
        objective = {
            "giou": lambda pred, target: (pred.sum() * 0.0, torch.ones(pred.shape[0], device=pred.device)),
            "l1": F.l1_loss,
            "focal": lambda pred, target: ((pred - target) ** 2).mean(),
        }
        loss_weight = {"giou": 2.0, "l1": 5.0, "focal": 1.0}
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=local_cfg,
        )

        pred_dict = {
            "pred_boxes": torch.full((self.batch, 1, 4), 0.5),
            "score_map": torch.zeros(self.batch, 1, 20, 20),
            "local_prompt": torch.randn(self.batch, 4, self.dim),
        }
        gt_dict = {
            "search_anno": torch.tensor([
                [0.4, 0.4, 0.2, 0.2],
                [0.5, 0.5, 0.2, 0.2],
            ], dtype=torch.float32)
        }

        loss, status = actor.compute_losses(pred_dict, gt_dict)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_prompt_align", status)
        self.assertGreater(status["loss_prompt_align"], 0.0)

    def test_actor_flat_multiview_loss_uses_view_weights(self):
        local_cfg = edict(cfg.copy())
        local_cfg.MODEL = edict(cfg.MODEL.copy())
        local_cfg.MODEL.PCUM = edict(cfg.MODEL.PCUM.copy())
        local_cfg.MODEL.PCUM.ENABLED = True
        local_cfg.TRAIN = edict(cfg.TRAIN.copy())
        local_cfg.TRAIN.PCUM = edict(cfg.TRAIN.PCUM.copy())
        local_cfg.TRAIN.PCUM.REAL_MULTIVIEW_LOSS_WEIGHTS = [1.6, 1.0, 1.0]
        local_cfg.DATA = edict(cfg.DATA.copy())
        local_cfg.DATA.SEARCH = edict(cfg.DATA.SEARCH.copy())
        local_cfg.DATA.SEARCH.SIZE = 320
        local_cfg.MODEL.BACKBONE = edict(cfg.MODEL.BACKBONE.copy())
        local_cfg.MODEL.BACKBONE.STRIDE = 16

        settings = edict({"batchsize": self.batch})
        objective = {
            "giou": lambda pred, target: (pred.sum() * 0.0, torch.ones(pred.shape[0], device=pred.device)),
            "l1": F.l1_loss,
            "focal": lambda pred, target: ((pred - target) ** 2).mean(),
        }
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective=objective,
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=settings,
            cfg=local_cfg,
        )

        num_views = 3
        pred_dict = {
            "pred_boxes": torch.full((self.batch * num_views, 1, 4), 0.5),
            "score_map": torch.zeros(self.batch * num_views, 1, 20, 20),
            "pcum_flat_multiview": True,
            "num_views": num_views,
        }
        gt_dict = {
            "search_anno": torch.tensor([
                [[0.4, 0.4, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
                [[0.3, 0.3, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]],
                [[0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
            ], dtype=torch.float32)
        }

        loss, status = actor.compute_losses(pred_dict, gt_dict)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(status["pcum_real_multiview"], 1.0)
        self.assertEqual(status["pcum_num_views"], 3.0)
        self.assertAlmostEqual(status["pcum_view_weight_0"], 1.6 / 3.6, places=6)
        self.assertAlmostEqual(status["pcum_view_weight_1"], 1.0 / 3.6, places=6)
        self.assertAlmostEqual(status["pcum_view_weight_2"], 1.0 / 3.6, places=6)

    def test_actor_remote_prompt_visibility_mask(self):
        local_cfg = edict(cfg.copy())
        local_cfg.MODEL = edict(cfg.MODEL.copy())
        local_cfg.MODEL.PCUM = edict(cfg.MODEL.PCUM.copy())
        local_cfg.MODEL.PCUM.ENABLED = True
        local_cfg.TRAIN = edict(cfg.TRAIN.copy())
        local_cfg.TRAIN.PCUM = edict(cfg.TRAIN.PCUM.copy())
        local_cfg.TRAIN.PCUM.USE_REAL_MULTIVIEW = True
        local_cfg.TRAIN.PCUM.USE_REMOTE_VISIBLE_MASK = True

        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=local_cfg,
        )

        data = {
            "template_view_valid": torch.tensor([
                [True, True],
                [False, True],
                [True, False],
            ]),
            "search_view_valid": torch.tensor([
                [True, True],
                [False, True],
                [True, False],
            ]),
        }
        remote_bank = [
            torch.ones(self.batch, 4, self.dim),
            torch.ones(self.batch, 4, self.dim) * 2,
            torch.ones(self.batch, 4, self.dim) * 3,
        ]

        remote_prompts, remote_states = actor._build_remote_inputs(
            data,
            remote_bank,
            target_view=0,
            num_views=3,
        )

        self.assertTrue(torch.equal(remote_prompts[0][0], torch.zeros_like(remote_prompts[0][0])))
        self.assertTrue(torch.equal(remote_prompts[1][1], torch.zeros_like(remote_prompts[1][1])))
        self.assertTrue(torch.equal(remote_states["score"], torch.tensor([0.5, 0.5])))

    def test_d0_ranking_defaults_are_disabled(self):
        self.assertFalse(cfg.TRAIN.PCUM.RANKING_ENABLED)
        self.assertFalse(cfg.TRAIN.PCUM_RANKING.ENABLED)
        self.assertFalse(cfg.TRAIN.PCUM_RANKING.FREEZE_BACKBONE)
        self.assertFalse(cfg.TRAIN.PCUM_RANKING.FREEZE_HEAD)
        self.assertFalse(cfg.TRAIN.PCUM_RANKING.VISIBLE_ONLY)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.LAMBDA_DELAY, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.MARGIN_ZERO, 0.02)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.MARGIN_LOCAL, 0.0)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.SAFE_MARGIN, 0.0)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.LAMBDA_ZERO, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.LAMBDA_LOCAL, 0.05)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.LAMBDA_SAFE, 0.0)
        self.assertAlmostEqual(cfg.TRAIN.PCUM.RANK_ZERO_MARGIN, 0.02)
        self.assertAlmostEqual(cfg.TRAIN.PCUM.RANK_DELAY_MARGIN, 0.02)
        self.assertAlmostEqual(cfg.TRAIN.PCUM.RANK_LOCAL_MARGIN, 0.0)
        self.assertAlmostEqual(cfg.TRAIN.PCUM.RANK_ZERO_WEIGHT, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.PCUM.RANK_DELAY_WEIGHT, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.PCUM.RANK_LOCAL_WEIGHT, 0.05)
        self.assertEqual(cfg.TRAIN.PCUM.DELAY_BRANCH_MODE, "batch_roll")

    def test_d0_ranking_alias_overrides_legacy_fields(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.TRAIN.PCUM.RANKING_ENABLED = False
        local_cfg.TRAIN.PCUM.RANK_DELAY_WEIGHT = 0.3
        local_cfg.TRAIN.PCUM_RANKING.ENABLED = True
        local_cfg.TRAIN.PCUM_RANKING.LAMBDA_DELAY = 0.0
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=local_cfg,
        )
        self.assertTrue(actor._pcum_ranking_enabled())
        self.assertAlmostEqual(actor._pcum_rank_delay_weight(), 0.0)

    def test_d1_ranking_aliases_override_legacy_fields(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.TRAIN.PCUM.SAFE_LOSS_WEIGHT = 0.1
        local_cfg.TRAIN.PCUM.SAFE_MARGIN = 0.0
        local_cfg.TRAIN.PCUM.RANK_ZERO_WEIGHT = 0.2
        local_cfg.TRAIN.PCUM.RANK_LOCAL_WEIGHT = 0.05
        local_cfg.TRAIN.PCUM.RANK_ZERO_MARGIN = 0.02
        local_cfg.TRAIN.PCUM.RANK_LOCAL_MARGIN = 0.0
        local_cfg.TRAIN.PCUM_RANKING.LAMBDA_SAFE = 0.3
        local_cfg.TRAIN.PCUM_RANKING.SAFE_MARGIN = 0.0
        local_cfg.TRAIN.PCUM_RANKING.LAMBDA_ZERO = 0.05
        local_cfg.TRAIN.PCUM_RANKING.LAMBDA_LOCAL = 0.05
        local_cfg.TRAIN.PCUM_RANKING.MARGIN_ZERO = 0.02
        local_cfg.TRAIN.PCUM_RANKING.MARGIN_LOCAL = 0.0
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=local_cfg,
        )
        self.assertAlmostEqual(actor._pcum_safe_weight(), 0.3)
        self.assertAlmostEqual(actor._pcum_safe_margin(), 0.0)
        self.assertAlmostEqual(actor._pcum_rank_zero_weight(), 0.05)
        self.assertAlmostEqual(actor._pcum_rank_local_weight(), 0.05)
        self.assertAlmostEqual(actor._pcum_rank_zero_margin(), 0.02)
        self.assertAlmostEqual(actor._pcum_rank_local_margin(), 0.0)

    def test_d0_ranking_loss_rewards_raw_when_better(self):
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=copy.deepcopy(cfg),
        )
        raw = torch.tensor([0.4, 0.6, 0.5, 0.7, 0.6, 0.8])
        worse = raw + 0.10
        better = raw - 0.10
        loss_good, stats_good = actor.compute_ranking_loss(
            raw, worse, num_views=3, margin=0.02)
        loss_bad, stats_bad = actor.compute_ranking_loss(
            raw, better, num_views=3, margin=0.02)
        self.assertAlmostEqual(float(loss_good.item()), 0.0, places=6)
        self.assertGreater(float(loss_bad.item()), 0.0)
        self.assertEqual(stats_good["raw_better_ratio"], 1.0)
        self.assertEqual(stats_bad["raw_better_ratio"], 0.0)

    def test_d1_visible_only_mask_flattens_view_major_flags(self):
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=copy.deepcopy(cfg),
        )
        data = {
            "search_view_valid": torch.tensor([
                [True, False],
                [False, True],
                [True, True],
            ])
        }
        mask = actor._flat_visible_mask(data, num_views=3, device=torch.device("cpu"), total_count=6)
        expected = torch.tensor([True, False, False, True, True, True])
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_d1_visible_only_ranking_ignores_invisible_samples(self):
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=copy.deepcopy(cfg),
        )
        raw = torch.tensor([0.40, 2.00, 0.40, 2.00, 0.40, 2.00])
        reference = torch.tensor([0.60, 0.10, 0.60, 0.10, 0.60, 0.10])
        visible = torch.tensor([True, False, True, False, True, False])
        visible_loss, visible_stats = actor.compute_ranking_loss(
            raw, reference, num_views=3, margin=0.02, active_mask=visible)
        all_loss, all_stats = actor.compute_ranking_loss(
            raw, reference, num_views=3, margin=0.02)
        self.assertAlmostEqual(float(visible_loss.item()), 0.0, places=6)
        self.assertGreater(float(all_loss.item()), 0.0)
        self.assertEqual(visible_stats["raw_better_ratio"], 1.0)
        self.assertLess(all_stats["raw_better_ratio"], 1.0)

    def test_d1_safe_loss_reports_visible_active_ratio(self):
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=copy.deepcopy(cfg),
        )
        collab = torch.tensor([0.20, 0.80, 0.20, 0.80, 0.20, 0.80])
        local = torch.tensor([0.50, 0.10, 0.50, 0.10, 0.50, 0.10])
        visible = torch.tensor([True, False, True, False, True, False])
        loss, stats = actor.compute_safe_loss(
            collab, local, num_views=3, margin=0.0, active_mask=visible)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(stats["safe_active_ratio"], 0.0)
        self.assertEqual(stats["active_count"], 3.0)

    def _make_d2_pcum_pair(self):
        torch.manual_seed(123)
        standard = PCUM(
            token_dim=8,
            prompt_dim=8,
            num_prompts=2,
            topk=4,
            fusion_mode="gated_add",
            align_gate="cosine_confidence",
            enabled=True,
            fusion_init_scale=0.1,
            remote_aggregation="confidence_softmax",
            remote_weight_temperature=0.10,
            remote_suppression_enabled=False,
        )
        d2 = PCUM(
            token_dim=8,
            prompt_dim=8,
            num_prompts=2,
            topk=4,
            fusion_mode="gated_add",
            align_gate="cosine_confidence",
            enabled=True,
            fusion_init_scale=0.1,
            remote_aggregation="confidence_softmax",
            remote_weight_temperature=0.10,
            remote_suppression_enabled=True,
            remote_suppression_gate_init_bias=-4.0,
        )
        d2.load_state_dict(standard.state_dict(), strict=False)
        return standard, d2

    def test_d2_defaults_are_disabled(self):
        self.assertFalse(cfg.MODEL.PCUM.REMOTE_SUPPRESSION_ENABLED)
        self.assertAlmostEqual(cfg.MODEL.PCUM.REMOTE_SUPPRESSION_GATE_INIT_BIAS, -4.0)
        self.assertFalse(cfg.TRAIN.PCUM_RANKING.REMOTE_SUPPRESSION_ONLY)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.LAMBDA_SUPPRESS_BCE, 0.10)
        self.assertAlmostEqual(cfg.TRAIN.PCUM_RANKING.LAMBDA_SUPPRESS_MEAN, 0.001)

        pcum = PCUM(token_dim=8, enabled=True, remote_suppression_enabled=False)
        self.assertIsNone(pcum.remote_suppression_gate)

    def test_d2_use_remote_false_matches_original_local_path(self):
        standard, d2 = self._make_d2_pcum_pair()
        features = {
            "search": torch.randn(2, 6, 8),
            "template": torch.randn(2, 3, 8),
        }
        standard_out = standard(features, remote_prompts=None)
        d2_out = d2(features, remote_prompts=None)
        self.assertTrue(torch.allclose(
            standard_out["search_tokens"],
            d2_out["search_tokens"],
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertAlmostEqual(float(d2_out["remote_delta_norm"].item()), 0.0, places=7)

    def test_d2_no_remote_prompt_has_zero_remote_delta(self):
        _, d2 = self._make_d2_pcum_pair()
        features = torch.randn(2, 6, 8)
        out = d2(features, remote_prompts=None)
        self.assertAlmostEqual(float(out["remote_delta_norm"].item()), 0.0, places=7)
        self.assertAlmostEqual(float(out["suppressed_delta_norm"].item()), 0.0, places=7)

    def test_d2_initialization_is_close_to_a0_raw(self):
        _, d2 = self._make_d2_pcum_pair()
        features = {
            "search": torch.randn(2, 6, 8),
            "template": torch.randn(2, 3, 8),
        }
        remote_prompts = [
            torch.randn(2, 2, 8),
            torch.randn(2, 2, 8),
        ]
        remote_states = {
            "per_remote_valid": torch.ones(2, 2, dtype=torch.bool),
            "per_remote_score": torch.tensor([[0.9, 0.7], [0.8, 0.6]]),
        }
        d2_out = d2(features, remote_prompts=remote_prompts, remote_states=remote_states)
        a0_out = d2(
            features,
            remote_prompts=remote_prompts,
            remote_states=remote_states,
            remote_suppression_override=0.0,
        )
        suppress = d2_out["remote_suppression"].detach().reshape(-1)
        self.assertAlmostEqual(float(suppress.mean().item()), float(torch.sigmoid(torch.tensor(-4.0)).item()), places=6)
        delta_norm = float(d2_out["remote_delta_norm"].item())
        suppressed_norm = float(d2_out["suppressed_delta_norm"].item())
        diff_norm = float((d2_out["search_tokens"] - a0_out["search_tokens"]).abs().max().item())
        self.assertGreater(delta_norm, 0.0)
        self.assertLess(suppressed_norm, delta_norm * 0.03)
        self.assertLess(diff_norm, 1e-3)

    def test_d2_optimizer_whitelist_only_remote_suppression_gate(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.TRAIN.LR = 1e-4
        local_cfg.TRAIN.PCUM_LR = 8e-5
        local_cfg.TRAIN.PCUM_RANKING.FREEZE_BACKBONE = True
        local_cfg.TRAIN.PCUM_RANKING.FREEZE_HEAD = True
        local_cfg.TRAIN.PCUM_RANKING.REMOTE_SUPPRESSION_ONLY = True

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.box_head = nn.Linear(2, 2)
                self.pcum = nn.Module()
                self.pcum.encoder = nn.Linear(2, 2)
                self.pcum.aligner = nn.Linear(2, 2)
                self.pcum.fusion = nn.Linear(2, 2)
                self.pcum.remote_suppression_gate = RemoteSuppressionGate(2)

        net = TinyNet()
        summary = apply_pcum_ranking_freeze(net, local_cfg)
        self.assertGreater(summary["counts"]["pcum_trainable"], 0)
        self.assertTrue(all(
            name.startswith("pcum.remote_suppression_gate.")
            for name in summary["trainable_names"]
        ))
        self.assertFalse(net.pcum.encoder.weight.requires_grad)
        self.assertFalse(net.pcum.aligner.weight.requires_grad)
        self.assertFalse(net.pcum.fusion.weight.requires_grad)
        self.assertTrue(net.pcum.remote_suppression_gate.bias.requires_grad)
        groups = build_optimizer_param_groups(net, local_cfg)
        trainable_names = [
            name for name, parameter in net.named_parameters()
            if parameter.requires_grad
        ]
        self.assertEqual({group["group_name"] for group in groups}, {"pcum"})
        self.assertTrue(all(
            name.startswith("pcum.remote_suppression_gate.")
            for name in trainable_names
        ))
        assert_optimizer_has_only_pcum_params(net, local_cfg)

    def test_d2_rank_zero_reference_is_detached(self):
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=copy.deepcopy(cfg),
        )
        raw = torch.tensor([0.6, 0.7, 0.6, 0.7, 0.6, 0.7], requires_grad=True)
        zero = torch.tensor([0.5, 0.6, 0.5, 0.6, 0.5, 0.6], requires_grad=True)
        loss, _ = actor.compute_ranking_loss(
            raw,
            zero,
            num_views=3,
            margin=0.02,
            detach_reference=True,
        )
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertIsNone(zero.grad)

    def test_d2_frozen_branch_has_no_gradients(self):
        _, d2 = self._make_d2_pcum_pair()
        for name, parameter in d2.named_parameters():
            parameter.requires_grad = name.startswith("remote_suppression_gate.")
        features = {
            "search": torch.randn(2, 6, 8),
            "template": torch.randn(2, 3, 8),
        }
        remote_prompts = [torch.randn(2, 2, 8), torch.randn(2, 2, 8)]
        out = d2(features, remote_prompts=remote_prompts)
        out["search_tokens"].sum().backward()
        gate_grads = []
        frozen_grad_present = False
        for name, parameter in d2.named_parameters():
            if name.startswith("remote_suppression_gate."):
                gate_grads.append(parameter.grad)
            elif parameter.grad is not None:
                frozen_grad_present = True
        self.assertTrue(any(grad is not None and grad.abs().sum() > 0 for grad in gate_grads))
        self.assertFalse(frozen_grad_present)

    def test_d2_smoke_yaml_uses_no_gt_inference(self):
        path = os.path.join(
            ROOT,
            "experiments/entertrack/pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_smoke.yaml",
        )
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["MODEL"]["PCUM"]["REMOTE_SUPPRESSION_ENABLED"])
        self.assertTrue(config["TRAIN"]["PCUM_RANKING"]["REMOTE_SUPPRESSION_ONLY"])
        self.assertTrue(config["TRAIN"]["PCUM_RANKING"]["VISIBLE_ONLY"])
        self.assertFalse(config["TEST"]["PCUM"]["USE_REMOTE_VISIBLE_MASK"])
        self.assertEqual(config["TEST"]["PCUM"]["REMOTE_STATE_SOURCE"], "tracker")

    def test_d0_zero_and_delay_prompts_preserve_shape(self):
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=edict({"batchsize": self.batch}),
            cfg=copy.deepcopy(cfg),
        )
        remote_prompts = [
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
        ]
        zero_prompts = actor._make_zero_remote_prompts(remote_prompts)
        self.assertEqual(len(zero_prompts), len(remote_prompts))
        for zero, prompt in zip(zero_prompts, remote_prompts):
            self.assertTrue(torch.equal(zero, torch.zeros_like(prompt)))
            self.assertEqual(zero.shape, prompt.shape)
            self.assertEqual(zero.dtype, prompt.dtype)
            self.assertEqual(zero.device, prompt.device)

        delayed = actor._make_delay_remote_bank(remote_prompts)
        for delayed_prompt, prompt in zip(delayed, remote_prompts):
            self.assertEqual(delayed_prompt.shape, prompt.shape)
            self.assertTrue(torch.equal(delayed_prompt[0], prompt[1]))
            self.assertTrue(torch.equal(delayed_prompt[1], prompt[0]))

    def test_optimizer_groups_keep_pcum_separate_after_freeze(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.TRAIN.LR = 1e-4
        local_cfg.TRAIN.BACKBONE_MULTIPLIER = 0.0
        local_cfg.TRAIN.PCUM_LR = 8e-5

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.box_head = nn.Linear(2, 2)
                self.pcum = nn.Linear(2, 2)

        net = TinyNet()
        for name, parameter in net.named_parameters():
            if name.startswith("backbone."):
                parameter.requires_grad = False
        groups = build_optimizer_param_groups(net, local_cfg)
        group_names = {group["group_name"] for group in groups}
        self.assertIn("pcum", group_names)
        self.assertIn("head_and_other", group_names)
        self.assertNotIn("backbone", group_names)

    def test_strict_pcum_ranking_freeze_excludes_head_and_backbone(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.TRAIN.LR = 1e-4
        local_cfg.TRAIN.PCUM_LR = 8e-5
        local_cfg.TRAIN.PCUM_RANKING.FREEZE_BACKBONE = True
        local_cfg.TRAIN.PCUM_RANKING.FREEZE_HEAD = True

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.box_head = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.pcum = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))

        net = TinyNet()
        summary = apply_pcum_ranking_freeze(net, local_cfg)
        self.assertGreater(summary["counts"]["pcum_trainable"], 0)
        self.assertTrue(all(name.startswith("pcum.") for name in summary["trainable_names"]))
        self.assertFalse(net.backbone[0].weight.requires_grad)
        self.assertFalse(net.box_head[0].weight.requires_grad)
        self.assertTrue(net.pcum[0].weight.requires_grad)
        self.assertFalse(net.backbone[1].training)
        self.assertFalse(net.box_head[1].training)
        self.assertTrue(net.pcum[1].training)

        groups = build_optimizer_param_groups(net, local_cfg)
        self.assertEqual({group["group_name"] for group in groups}, {"pcum"})
        assert_optimizer_has_only_pcum_params(net, local_cfg)

    def test_strict_pcum_ranking_freeze_reapplies_eval_after_train(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.TRAIN.PCUM_RANKING.FREEZE_BACKBONE = True
        local_cfg.TRAIN.PCUM_RANKING.FREEZE_HEAD = True

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.box_head = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.pcum = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))

        net = TinyNet()
        apply_pcum_ranking_freeze(net, local_cfg)
        net.train()
        with self.assertRaises(RuntimeError):
            assert_pcum_frozen_batchnorm_eval(net, local_cfg)
        set_pcum_frozen_modules_eval(net, local_cfg)
        assert_pcum_frozen_batchnorm_eval(net, local_cfg)
        self.assertFalse(net.backbone[1].training)
        self.assertFalse(net.box_head[1].training)
        self.assertTrue(net.pcum[1].training)

    def test_threemdot_sampler_requires_all_views_visible(self):
        torch.manual_seed(7)
        sampler = TrackingSamplerThreeMDOT(
            datasets=[FakeThreeMDOTDataset()],
            p_datasets=None,
            samples_per_epoch=1,
            max_gap=2,
            num_search_frames=1,
            num_template_frames=1,
            processing=mark_valid,
            frame_sample_mode="causal",
            require_all_views_visible=True,
            max_retry=50,
        )

        data = sampler[0]
        self.assertEqual(len(data["template_images"]), 3)
        self.assertEqual(len(data["search_images"]), 3)
        self.assertTrue(bool(data["template_view_valid"].all()))
        self.assertTrue(bool(data["search_view_valid"].all()))

    def test_pcum_motion_reliability_ignores_visibility_annotation(self):
        stable = {
            "prev_bbox": [100.0, 100.0, 40.0, 40.0],
            "target_bbox": [106.0, 104.0, 42.0, 41.0],
            "max_score": 0.8,
            "apce": 180.0,
            "visible": True,
        }
        annotated_invisible = dict(stable, visible=False)
        large_jump = dict(stable, target_bbox=[260.0, 260.0, 42.0, 41.0])

        stable_score = _pcum_motion_reliability(stable)
        invisible_score = _pcum_motion_reliability(annotated_invisible)
        jump_score = _pcum_motion_reliability(large_jump)

        self.assertGreater(stable_score, 0.1)
        self.assertEqual(invisible_score, stable_score)
        self.assertLess(jump_score, stable_score)

    def test_pcum_motion_redetect_config_loads(self):
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/pcum_ablation_current_full_remote_motion_redetect.yaml",
            base_cfg=local_cfg,
        )

        self.assertTrue(local_cfg.TEST.PCUM.USE_MOTION_REDETECT)
        self.assertEqual(local_cfg.TEST.PCUM.LOCAL_LOW_MODE, "apce")
        self.assertTrue(local_cfg.TEST.PCUM.MOTION_REDETECT_USE_LOCAL_CANDIDATE)
        self.assertAlmostEqual(local_cfg.TEST.PCUM.MOTION_REDETECT_LOCAL_MIN_GAIN, 0.02)
        self.assertTrue(local_cfg.TEST.PCUM.KEEP_LOCAL_IF_REMOTE_CONFIDENCE_WORSE)
        self.assertAlmostEqual(local_cfg.TEST.PCUM.REMOTE_CONFIDENCE_MAX_DROP, 0.02)
        self.assertTrue(local_cfg.TEST.PCUM.SAVE_DECISION_LOG)

    def test_three_mdot_reruns_when_decision_log_missing(self):
        seq_a = FakeSequence("case-1")
        seq_b = FakeSequence("case-2")
        seq_c = FakeSequence("case-3")

        with tempfile.TemporaryDirectory(dir="/tmp") as results_dir:
            for seq in (seq_a, seq_b, seq_c):
                with open(os.path.join(results_dir, seq.name + ".txt"), "w") as fh:
                    fh.write("1\t2\t3\t4\n")

            tracker = FakeTrackerInfo(results_dir, save_decision_log=True)
            with contextlib.redirect_stdout(io.StringIO()):
                run_three_multi_sequence(seq_a, seq_b, seq_c, tracker, debug=False, num_gpu=1)
            self.assertTrue(tracker.called)
            for seq in (seq_a, seq_b, seq_c):
                self.assertTrue(os.path.isfile(
                    os.path.join(results_dir, seq.name + "_pcum_decision.txt")
                ))

            tracker2 = FakeTrackerInfo(results_dir, save_decision_log=True)
            with contextlib.redirect_stdout(io.StringIO()):
                run_three_multi_sequence(seq_a, seq_b, seq_c, tracker2, debug=False, num_gpu=1)
            self.assertFalse(tracker2.called)

    def test_entertrack_pcum_is_wired_by_config(self):
        if importlib.util.find_spec("timm") is None:
            self.skipTest("timm is required for full EnTeRTrack model construction")

        from lib.models.entertrack import build_entertrack  # noqa: E402

        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/pcum_ablation_local_gated.yaml",
            base_cfg=local_cfg,
        )
        model = build_entertrack(local_cfg, training=False)
        self.assertIsNotNone(model.pcum)
        pcum_keys = [name for name, _ in model.named_parameters() if name.startswith("pcum.")]
        self.assertGreater(len(pcum_keys), 0)

        template = torch.randn(1, 3, local_cfg.TEST.TEMPLATE_SIZE, local_cfg.TEST.TEMPLATE_SIZE)
        search = torch.randn(1, 3, local_cfg.TEST.SEARCH_SIZE, local_cfg.TEST.SEARCH_SIZE)
        with torch.no_grad():
            out = model(template, search, training=False)
        self.assertIn("local_prompt", out)
        self.assertIsNotNone(out["local_prompt"])
        self.assertEqual(out["local_prompt"].shape[1], local_cfg.MODEL.PCUM.NUM_PROMPTS)

        base_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/pcum_ablation_baseline.yaml",
            base_cfg=base_cfg,
        )
        baseline = build_entertrack(base_cfg, training=False)
        self.assertIsNone(baseline.pcum)

    def test_diagnostics_disabled_is_elementwise_transparent(self):
        network = nn.Module()
        network.pcum = PCUM(
            token_dim=self.dim,
            prompt_dim=self.dim,
            num_prompts=4,
            topk=8,
            fusion_mode="gated_add",
            enabled=True,
        )
        remote = torch.randn(self.batch, 4, self.dim)
        before = network.pcum(
            {"search": self.search, "template": self.template},
            remote_prompts=remote,
        )["search_tokens"]

        hooks = PCUMDiagnosticHooks(network, enabled=False)
        after = network.pcum(
            {"search": self.search, "template": self.template},
            remote_prompts=remote,
        )["search_tokens"]

        self.assertEqual(hooks.handle_count, 0)
        self.assertTrue(torch.equal(before, after))

    def test_diagnostics_enabled_does_not_change_prediction(self):
        network = nn.Module()
        network.pcum = PCUM(
            token_dim=self.dim,
            prompt_dim=self.dim,
            num_prompts=4,
            topk=8,
            fusion_mode="gated_add",
            enabled=True,
        )
        remote = torch.randn(self.batch, 4, self.dim)
        before = network.pcum(
            {"search": self.search, "template": self.template},
            remote_prompts=remote,
        )["search_tokens"]

        hooks = PCUMDiagnosticHooks(network, enabled=True)
        after = network.pcum(
            {"search": self.search, "template": self.template},
            remote_prompts=remote,
        )["search_tokens"]
        stats = hooks.snapshot()

        self.assertTrue(torch.equal(before, after))
        self.assertTrue(math.isfinite(stats["alignment_gate_mean"]))
        self.assertTrue(math.isfinite(stats["fusion_gate_mean"]))
        hooks.remove()

    def test_formal_diagnostics_do_not_access_gt_visibility(self):
        class NoVisibilityAccess:
            @property
            def target_visible(self):
                raise AssertionError("formal inference accessed GT visibility")

        sequences = [NoVisibilityAccess(), NoVisibilityAccess(), NoVisibilityAccess()]
        selected = visibility_for_remote_selection(True, False, sequences, 3)
        self.assertIsNone(selected)

    def test_diagnostic_hooks_are_removed(self):
        network = nn.Module()
        network.pcum = PCUM(
            token_dim=self.dim,
            prompt_dim=self.dim,
            fusion_mode="gated_add",
            enabled=True,
        )
        hooks = PCUMDiagnosticHooks(network, enabled=True)
        self.assertEqual(hooks.handle_count, 2)
        hooks.remove()
        self.assertEqual(hooks.handle_count, 0)
        self.assertEqual(len(network.pcum.aligner._forward_hooks), 0)
        self.assertEqual(len(network.pcum.fusion.gate._forward_hooks), 0)

    def test_sequence_wrapper_removes_each_tracker_hook(self):
        class SequenceStub:
            def init_info(self):
                return {"init_bbox": [0, 0, 10, 10]}

        class ActiveTrackerStub:
            def __init__(self):
                self.closed = False

            def close_pcum_diagnostics(self):
                self.closed = True

        wrapper = EvaluationTracker.__new__(EvaluationTracker)
        wrapper.get_parameters = lambda: edict({"debug": 0})
        active = [ActiveTrackerStub(), ActiveTrackerStub(), ActiveTrackerStub()]
        wrapper.create_tracker = lambda params: active.pop(0)
        created = []

        def fake_track(tracker_a, tracker_b, tracker_c, *args):
            created.extend([tracker_a, tracker_b, tracker_c])
            return {}, {}, {}

        wrapper.Fuse_three_multi_track = fake_track
        sequence = SequenceStub()
        EvaluationTracker.Fuse_three_multi_run_sequence(
            wrapper, sequence, sequence, sequence, debug=False
        )
        self.assertTrue(all(tracker.closed for tracker in created))

    def test_three_uav_diagnostic_fields_do_not_shift(self):
        rows = []
        for index, uav in enumerate(("A", "B", "C")):
            bbox = [10.0 * index, 0.0, 10.0, 10.0]
            snapshot = {"bbox": bbox, "confidence": 0.5, "score_max": 0.5, "apce": 100.0}
            rows.append(build_frame_diagnostic_row(
                diagnostic_label="formal_no_gt_mask",
                uses_gt_visible_mask=False,
                sequence_name="case-{}".format(index + 1),
                frame_id=1,
                current_uav=uav,
                remote_uav_ids=[peer for peer in ("A", "B", "C") if peer != uav],
                local=snapshot,
                raw_collaborative=snapshot,
                final=snapshot,
                gt_bbox=bbox,
                final_source="raw_collaborative",
            ))

        self.assertEqual([row["current_uav"] for row in rows], ["A", "B", "C"])
        self.assertEqual([row["sequence_name"] for row in rows], ["case-1", "case-2", "case-3"])
        self.assertTrue(all(row["local_iou"] == 1.0 for row in rows))

    def test_multi_remote_json_fields_round_trip(self):
        snapshot = {"bbox": [1, 2, 3, 4], "confidence": 0.5, "score_max": 0.5, "apce": 100.0}
        row = build_frame_diagnostic_row(
            diagnostic_label="formal_no_gt_mask",
            uses_gt_visible_mask=False,
            sequence_name="case-1",
            frame_id=1,
            current_uav="A",
            remote_uav_ids=["B", "C"],
            local=snapshot,
            raw_collaborative=snapshot,
            final=snapshot,
            gt_bbox=[1, 2, 3, 4],
            remote_confidences={"B": 0.76, "C": 0.31},
            prompt_similarities={"B": 0.68, "C": 0.42},
            remote_visibility_gt={"B": 1, "C": 0},
            remote_participated={"B": True, "C": True},
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as results_dir:
            tracker = FakeTrackerInfo(results_dir, save_decision_log=False)
            seq = FakeSequence("case-1")
            _save_tracker_output(seq, tracker, {"pcum_frame_diagnostics": [row]})
            path = os.path.join(
                results_dir,
                diagnostic_filename("entertrack", "fake", 0, "case-1", "A"),
            )
            with open(path, newline="") as fh:
                loaded = next(csv.DictReader(fh))

        self.assertEqual(json.loads(loaded["remote_confidences"]), {"B": 0.76, "C": 0.31})
        self.assertEqual(json.loads(loaded["remote_visibility_gt"]), {"B": 1, "C": 0})

    def test_film_diagnostics_and_matching_configs(self):
        oracle_cfg = copy.deepcopy(cfg)
        formal_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/pcum_diagnostic_reproduction_oracle_mask.yaml",
            base_cfg=oracle_cfg,
        )
        update_config_from_file(
            "experiments/entertrack/pcum_diagnostic_formal_no_gt_mask.yaml",
            base_cfg=formal_cfg,
        )
        self.assertEqual(oracle_cfg.MODEL.PCUM.FUSION, "film")
        self.assertEqual(formal_cfg.MODEL.PCUM.FUSION, "film")
        self.assertEqual(oracle_cfg.TEST.CHECKPOINT_NAME, formal_cfg.TEST.CHECKPOINT_NAME)
        self.assertTrue(oracle_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)
        self.assertFalse(formal_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)

        network = nn.Module()
        network.pcum = PCUM(
            token_dim=self.dim,
            prompt_dim=self.dim,
            fusion_mode="film",
            enabled=True,
        )
        hooks = PCUMDiagnosticHooks(network, enabled=True)
        network.pcum(
            {"search": self.search, "template": self.template},
            remote_prompts=torch.randn(self.batch, 4, self.dim),
        )
        stats = hooks.snapshot()
        self.assertTrue(math.isfinite(stats["alignment_gate_mean"]))
        self.assertTrue(math.isnan(stats["fusion_gate_mean"]))
        hooks.remove()

    def test_fallback_before_and_after_are_recorded(self):
        local = {"bbox": [0, 0, 10, 10], "confidence": 0.8, "score_max": 0.8, "apce": 150.0}
        raw = {"bbox": [20, 20, 10, 10], "confidence": 0.2, "score_max": 0.2, "apce": 40.0}
        row = build_frame_diagnostic_row(
            diagnostic_label="formal_no_gt_mask",
            uses_gt_visible_mask=False,
            sequence_name="case-1",
            frame_id=2,
            current_uav="A",
            remote_uav_ids=["B", "C"],
            local=local,
            raw_collaborative=raw,
            final=local,
            gt_bbox=[0, 0, 10, 10],
            final_source="local",
            fallback_triggered=True,
            fallback_reason="remote_score_drop",
        )
        self.assertLess(row["instant_delta_iou"], 0.0)
        self.assertGreater(row["fallback_delta_iou"], 0.0)
        self.assertEqual(row["final_delta_iou"], 0.0)
        self.assertEqual(row["final_source"], "local")
        self.assertEqual(row["fallback_triggered"], 1)
        self.assertEqual(row["fallback_reason"], "remote_score_drop")

    def test_response_entropy_is_resolution_normalized(self):
        uniform_small = torch.ones(1, 1, 4, 4)
        uniform_large = torch.ones(1, 1, 16, 16)
        self.assertAlmostEqual(normalized_response_entropy(uniform_small), 1.0, places=6)
        self.assertAlmostEqual(normalized_response_entropy(uniform_large), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
