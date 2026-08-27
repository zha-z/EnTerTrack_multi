import copy
import csv
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

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
from lib.test.tracker.entertrack import (  # noqa: E402
    EnTeRTrack as RuntimeEnTeRTrack,
)
from lib.test.evaluation.tracker import Tracker as EvaluationTracker  # noqa: E402
from lib.test.evaluation.running import _save_tracker_output  # noqa: E402


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
        self.assertFalse(cfg.TEST.PLAIN_COLLABORATION.ENABLED)

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

    def test_metadata_accepts_common_visible_shared_frame_slot(self):
        data = {
            "target_id": ["T0", "T1"],
            "view_ids": [["A", "A"], ["B", "B"], ["C", "C"]],
            "search_frame_ids": [[7, 9]],
        }
        validate_synchronized_abc_metadata(data, 3, 2)

    def test_metadata_rejects_invalid_frame_slot_count(self):
        data = {
            "target_id": ["T0"],
            "view_ids": [["A"], ["B"], ["C"]],
            "search_frame_ids": [[7], [7]],
        }
        with self.assertRaisesRegex(ValueError, "one shared frame slot"):
            validate_synchronized_abc_metadata(data, 3, 1)

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
        self.assertTrue(local_cfg.TEST.PLAIN_COLLABORATION.ENABLED)
        self.assertTrue(local_cfg.TEST.PLAIN_COLLABORATION.SAVE_DIAGNOSTICS)
        self.assertTrue(local_cfg.TRAIN.MULTIVIEW.REQUIRE_ALL_VIEWS_VISIBLE)
        self.assertTrue(local_cfg.TRAIN.MULTIVIEW.CANONICAL_VIEW_ORDER)
        self.assertFalse(local_cfg.MODEL.PCUM.ENABLED)
        self.assertFalse(local_cfg.MODEL.C3R.ENABLED)
        self.assertFalse(local_cfg.MODEL.FCVC.ENABLED)
        self.assertEqual(local_cfg.MODEL.BACKBONE.CE_LOC, [])

    def test_d0_diagnostic_configs_only_change_commit_or_local_path(self):
        expected = {
            "plain_collaboration_v1_d0_local": (False, False, False, False),
            "plain_collaboration_v1_d0_closed_loop": (True, True, False, True),
            "plain_collaboration_v1_d0_safe": (True, True, True, True),
        }
        for name, values in expected.items():
            local_cfg = copy.deepcopy(cfg)
            update_config_from_file(
                "experiments/entertrack/{}.yaml".format(name),
                base_cfg=local_cfg,
            )
            plain = local_cfg.TEST.PLAIN_COLLABORATION
            self.assertEqual((
                bool(local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED),
                bool(plain.ENABLED),
                bool(plain.SAFE_COMMIT),
                bool(plain.SAVE_COUNTERFACTUAL_DIAGNOSTICS),
            ), values)
            self.assertFalse(local_cfg.MODEL.PCUM.ENABLED)
            self.assertFalse(local_cfg.MODEL.C3R.ENABLED)
            self.assertFalse(local_cfg.MODEL.FCVC.ENABLED)
            self.assertEqual(local_cfg.MODEL.BACKBONE.CE_LOC, [])

    def test_e15_sender_counterfactual_config_is_safe_and_default_off(self):
        self.assertFalse(
            cfg.TEST.PLAIN_COLLABORATION
            .SAVE_SENDER_COUNTERFACTUAL_DIAGNOSTICS)
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/"
            "plain_collaboration_v1_e15_sender_counterfactual.yaml",
            base_cfg=local_cfg,
        )
        plain = local_cfg.TEST.PLAIN_COLLABORATION
        self.assertTrue(local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED)
        self.assertTrue(plain.ENABLED)
        self.assertTrue(plain.SAFE_COMMIT)
        self.assertTrue(plain.SAVE_SENDER_COUNTERFACTUAL_DIAGNOSTICS)
        self.assertFalse(plain.SAVE_COUNTERFACTUAL_DIAGNOSTICS)
        self.assertFalse(local_cfg.MODEL.PCUM.ENABLED)
        self.assertFalse(local_cfg.MODEL.C3R.ENABLED)
        self.assertFalse(local_cfg.MODEL.FCVC.ENABLED)


class PlainCollaborationInferenceTests(unittest.TestCase):
    @staticmethod
    def _candidate(feature_value):
        feature = torch.full((1, 320, 12), float(feature_value))
        return {
            "out_dict": {"backbone_feat": feature},
            "resize_factor": 1.0,
            "image": torch.zeros(96, 128, 3),
            "crop_bbox": [10.0, 10.0, 20.0, 20.0],
            "output": {"target_bbox": [10.0, 10.0, 20.0, 20.0]},
            "target_bbox": [10.0, 10.0, 20.0, 20.0],
            "prev_bbox": [9.0, 10.0, 20.0, 20.0],
            "max_score": torch.tensor(0.7),
            "apce": torch.tensor(30.0),
            "response": torch.ones(1, 1, 16, 16),
        }

    def _runtime_tracker(self):
        runtime = RuntimeEnTeRTrack.__new__(RuntimeEnTeRTrack)
        runtime.plain_collaboration_enabled = True
        runtime.network = EnTeRTrack(
            transformer=DummyBackbone(),
            box_head=FakeCenterHead(),
            head_type="CENTER",
            plain_collaboration=PlainCollaborationV1(
                token_dim=12, num_heads=3, enabled=True),
            plain_collaboration_freeze_local=True,
        )
        runtime.params = SimpleNamespace(search_size=256, search_factor=4.0)
        runtime.save_all_boxes = False
        runtime.debug = False
        runtime.frame_id = 0
        runtime.state = [10.0, 10.0, 20.0, 20.0]
        runtime.plain_collaboration_safe_commit = False
        runtime.plain_collaboration_counterfactual_diagnostics = False
        runtime.plain_collaboration_sender_counterfactual_diagnostics = False
        runtime._decode_prediction = lambda output, resize_factor, return_score: (
            [0.0, 0.0, 20.0, 20.0],
            torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            torch.tensor([0.8]),
            torch.ones(1, 1, 16, 16),
        )
        runtime.map_box_back = lambda box, resize_factor, reference_bbox=None: (
            [12.0, 13.0, 20.0, 20.0])
        runtime.calAPCE = lambda response: torch.tensor(42.0)
        return runtime

    def test_runtime_fuses_two_canonical_senders_and_commits(self):
        runtime = self._runtime_tracker()
        local = self._candidate(1.0)
        remote_b = self._candidate(2.0)
        remote_c = self._candidate(3.0)
        candidate = runtime.plain_collaboration_candidate(
            local,
            (remote_b, remote_c),
            receiver_view="A",
            sender_views=("B", "C"),
            frame_id=7,
        )
        diagnostics = candidate["plain_collaboration_diagnostics"]
        self.assertTrue(candidate["used_remote"])
        self.assertEqual(diagnostics["valid_remote_count"], 2)
        self.assertEqual(diagnostics["search_token_count"], 256)
        self.assertAlmostEqual(diagnostics["sender_weight_0"], 0.5)
        self.assertAlmostEqual(diagnostics["sender_weight_1"], 0.5)
        output, score, apce = runtime.plain_collaboration_finalize_frame(
            candidate)
        self.assertEqual(runtime.frame_id, 1)
        self.assertEqual(runtime.state, candidate["target_bbox"])
        self.assertEqual(output["plain_collaboration_diagnostics"]["frame_id"], 7)
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.isfinite(apce).all())

    def test_safe_commit_reports_collaboration_but_keeps_local_state(self):
        runtime = self._runtime_tracker()
        runtime.plain_collaboration_safe_commit = True
        runtime.plain_collaboration_counterfactual_diagnostics = True
        local = self._candidate(1.0)
        collaborative = runtime.plain_collaboration_candidate(
            local,
            (self._candidate(2.0), self._candidate(3.0)),
            receiver_view="A",
            sender_views=("B", "C"),
            frame_id=7,
            target_id="T0",
        )
        row = collaborative["plain_collaboration_counterfactual"]
        self.assertEqual(
            row["persistent_state_digest_before"],
            row["persistent_state_digest_after"],
        )
        output, _, _ = runtime.plain_collaboration_finalize_frame(
            local, collaborative)
        self.assertEqual(runtime.state, local["target_bbox"])
        self.assertEqual(output["target_bbox"], collaborative["target_bbox"])
        saved = output["plain_collaboration_counterfactual"]
        self.assertEqual(saved["state_output_bbox_x"], 10.0)
        self.assertEqual(saved["reported_output_bbox_x"], 12.0)
        self.assertFalse(saved["uses_gt"])

    def test_runtime_rejects_noncanonical_sender_order(self):
        runtime = self._runtime_tracker()
        with self.assertRaisesRegex(ValueError, "canonical two-sender order"):
            runtime.plain_collaboration_candidate(
                self._candidate(1.0),
                (self._candidate(2.0), self._candidate(3.0)),
                receiver_view="A",
                sender_views=("C", "B"),
                frame_id=1,
            )

    def test_single_sender_is_diagnostic_only_and_prediction_only(self):
        runtime = self._runtime_tracker()
        local = self._candidate(1.0)
        remote = self._candidate(2.0)
        with self.assertRaisesRegex(ValueError, "canonical two-sender order"):
            runtime.plain_collaboration_candidate(
                local, (remote,), "A", ("B",), frame_id=1)

        runtime.plain_collaboration_safe_commit = True
        runtime.plain_collaboration_sender_counterfactual_diagnostics = True
        digest = runtime.fcvc_persistent_state_digest()
        candidate = runtime.plain_collaboration_candidate(
            local, (remote,), "A", ("B",), frame_id=1, target_id="T0")
        diagnostic = candidate["plain_collaboration_diagnostics"]
        self.assertEqual(diagnostic["valid_remote_count"], 1)
        self.assertAlmostEqual(diagnostic["sender_weight_0"], 1.0)
        self.assertTrue(math.isnan(diagnostic["sender_weight_1"]))
        self.assertEqual(runtime.fcvc_persistent_state_digest(), digest)
        row = runtime.plain_collaboration_sender_counterfactual_row(
            local_candidate=local,
            branch_candidate=candidate,
            remote_candidates=(remote,),
            receiver_view="A",
            sender_views=("B",),
            branch_name="sender0_only",
            frame_id=1,
            target_id="T0",
            state_digest_before=digest,
        )
        self.assertEqual(row["remote_count"], 1)
        self.assertEqual(row["sender_0_view"], "B")
        self.assertEqual(row["sender_1_view"], "")
        self.assertFalse(row["uses_gt"])
        self.assertEqual(
            row["persistent_state_digest_before"],
            row["persistent_state_digest_after"],
        )

    def test_runtime_rejects_checkpoint_without_adapter(self):
        runtime = RuntimeEnTeRTrack.__new__(RuntimeEnTeRTrack)
        network = DummyTrainModel()
        local_only_state = {
            name: value for name, value in network.state_dict().items()
            if not name.startswith("plain_collaboration.")
        }
        with tempfile.NamedTemporaryFile(suffix=".pth.tar") as checkpoint_file:
            torch.save({"net": local_only_state}, checkpoint_file.name)
            with self.assertRaises(RuntimeError):
                runtime._load_network(network, checkpoint_file.name)

    def test_runtime_rejects_single_view_api(self):
        runtime = RuntimeEnTeRTrack.__new__(RuntimeEnTeRTrack)
        runtime.plain_collaboration_enabled = True
        with self.assertRaisesRegex(RuntimeError, "three-view runner"):
            runtime.track(torch.zeros(8, 8, 3), {})
        with self.assertRaisesRegex(RuntimeError, "independent Fusetrack"):
            runtime.Fusetrack(torch.zeros(8, 8, 3), {})

    def test_three_view_runner_dispatches_plain_collaboration(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED = True
        local_cfg.TEST.PLAIN_COLLABORATION.ENABLED = True
        local_cfg.TEST.PLAIN_COLLABORATION.SAVE_DIAGNOSTICS = True

        class SequenceStub:
            def __init__(self, name):
                self.name = name
                self.frames = ["frame0", "frame1"]
                self.ground_truth_rect = [
                    [10.0, 10.0, 20.0, 20.0],
                    [11.0, 10.0, 20.0, 20.0],
                ]

            def frame_info(self, frame_num):
                return {}

        class TrackerStub:
            def __init__(self, view):
                self.view = view
                self.cfg = local_cfg
                self.params = SimpleNamespace(
                    save_all_boxes=False,
                    no_gt_inference=True,
                )
                self.plain_collaboration_enabled = True
                self.plain_collaboration_safe_commit = False
                self.fcvc_enabled = False
                self.c3r_enabled = False
                self.local_calls = 0
                self.sender_orders = []

            def initialize(self, image, info):
                return None

            def plain_collaboration_local_candidate(self, image):
                self.local_calls += 1
                return {"view": self.view}

            def plain_collaboration_candidate(
                    self, local, remotes, receiver_view, sender_views, frame_id,
                    target_id=""):
                self.assert_equal(receiver_view, self.view)
                self.assert_equal(
                    tuple(item["view"] for item in remotes), tuple(sender_views))
                self.sender_orders.append(tuple(sender_views))
                return {"frame_id": frame_id, "receiver_view": receiver_view}

            @staticmethod
            def assert_equal(actual, expected):
                if actual != expected:
                    raise AssertionError("{} != {}".format(actual, expected))

            def plain_collaboration_finalize_frame(
                    self, local, candidate, info=None, debug_name=""):
                diagnostic = {
                    "frame_id": candidate["frame_id"],
                    "receiver_view": self.view,
                    "sender_view_0": self.sender_orders[-1][0],
                    "sender_view_1": self.sender_orders[-1][1],
                    "used_remote": True,
                    "valid_remote_count": 2,
                    "search_token_count": 256,
                    "sender_weight_0": 0.5,
                    "sender_weight_1": 0.5,
                    "residual_norm": 0.1,
                    "relative_residual_norm": 0.01,
                    "residual_scale": 0.01,
                }
                return ({
                    "target_bbox": [11.0, 10.0, 20.0, 20.0],
                    "plain_collaboration_diagnostics": diagnostic,
                }, torch.tensor(0.8), torch.tensor(20.0))

            def Fusetrack(self, image, info):
                raise AssertionError("plain runner fell back to Fusetrack")

        wrapper = EvaluationTracker.__new__(EvaluationTracker)
        wrapper._read_image = lambda path: path
        trackers = [TrackerStub(view) for view in ("A", "B", "C")]
        sequences = [SequenceStub("T0-{}".format(index)) for index in (1, 2, 3)]
        outputs = EvaluationTracker.Fuse_three_multi_track(
            wrapper,
            *trackers,
            *sequences,
            *({"init_bbox": [10.0, 10.0, 20.0, 20.0]},) * 3
        )
        self.assertEqual([tracker.local_calls for tracker in trackers], [1, 1, 1])
        self.assertEqual(trackers[0].sender_orders, [("B", "C")])
        self.assertEqual(trackers[1].sender_orders, [("A", "C")])
        self.assertEqual(trackers[2].sender_orders, [("A", "B")])
        for output in outputs:
            self.assertEqual(len(output["target_bbox"]), 2)
            self.assertTrue(
                output["plain_collaboration_diagnostics"][1]["used_remote"])

    def test_plain_diagnostics_are_saved_as_csv(self):
        row = {
            "frame_id": 1,
            "receiver_view": "A",
            "sender_view_0": "B",
            "sender_view_1": "C",
            "used_remote": True,
            "valid_remote_count": 2,
            "search_token_count": 256,
            "sender_weight_0": 0.5,
            "sender_weight_1": 0.5,
            "residual_norm": 1.0,
            "relative_residual_norm": 0.01,
            "residual_scale": 0.02,
        }
        with tempfile.TemporaryDirectory() as directory:
            tracker = SimpleNamespace(
                results_dir=directory,
                name="entertrack",
                parameter_name="plain_collaboration_v1",
                run_id=1,
            )
            sequence = SimpleNamespace(
                dataset="threemdot_val",
                name="T0-1",
                object_ids=None,
            )
            _save_tracker_output(
                sequence,
                tracker,
                {"plain_collaboration_diagnostics": [row]},
            )
            path = os.path.join(
                directory, "T0-1_plain_collaboration.csv")
            self.assertTrue(os.path.isfile(path))
            with open(path, newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["receiver_view"], "A")
            self.assertEqual(saved[0]["valid_remote_count"], "2")

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
