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
    SaliencyTokenSelector,
    build_pcum,
    build_pseudo_remote_prompts,
)
from lib.config.entertrack.config import cfg  # noqa: E402
from lib.config.entertrack.config import update_config_from_file  # noqa: E402
from lib.test.evaluation.tracker import _pcum_motion_reliability  # noqa: E402
from lib.test.evaluation.tracker import Tracker as EvaluationTracker  # noqa: E402
from lib.test.evaluation.running import run_three_multi_sequence  # noqa: E402
from lib.test.evaluation.running import _save_tracker_output  # noqa: E402
from lib.test.utils.pcum_diagnostics import (  # noqa: E402
    PCUMDiagnosticHooks,
    build_frame_diagnostic_row,
    diagnostic_filename,
    normalized_response_entropy,
    visibility_for_remote_selection,
)
from lib.train.data.sampler_threemdot import TrackingSamplerThreeMDOT  # noqa: E402
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT  # noqa: E402


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

    def test_pcum_motion_reliability_is_conservative(self):
        stable_visible = {
            "prev_bbox": [100.0, 100.0, 40.0, 40.0],
            "target_bbox": [106.0, 104.0, 42.0, 41.0],
            "max_score": 0.8,
            "apce": 180.0,
            "visible": True,
        }
        invisible = dict(stable_visible, visible=False)
        large_jump = dict(stable_visible, target_bbox=[260.0, 260.0, 42.0, 41.0])

        stable_score = _pcum_motion_reliability(stable_visible)
        invisible_score = _pcum_motion_reliability(invisible)
        jump_score = _pcum_motion_reliability(large_jump)

        self.assertGreater(stable_score, 0.1)
        self.assertEqual(invisible_score, 0.0)
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
