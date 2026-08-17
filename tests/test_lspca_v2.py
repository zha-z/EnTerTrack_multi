import copy
import hashlib
import inspect
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lib.config.entertrack.config import get_default_config, update_config_from_file
from lib.models.entertrack.entertrack import EnTeRTrack, build_entertrack
from lib.models.entertrack.pcum import build_pcum
from lib.test.evaluation.running import three_view_triplets
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT
from lib.train.data.sampler_threemdot import TrackingSamplerThreeMDOT
from lib.train.optimizer_groups import build_optimizer_param_groups
from lib.train.pcum_freeze import apply_partial_adaptation_freeze
from lib.train.train_script import use_grouped_multiview_loader
from tracking.audit_j0_j1_partial_adaptation import (
    build_for_audit,
    optimizer_summary,
    run_one_step,
    summarize_trainable,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/entertrack"
J0_BASE = "ostrack_deit_tiny_b0_j0v2_lspca_control_ep15"
J1_BASE = "ostrack_deit_tiny_b0_j1v2_lspca_pcum_ep15"
FROZEN_B0_SHA256 = (
    "88706aa3087d245c22c152d3feb5417e20bd12f06942283cc0c513c53d2c6128"
)


def load_config(name):
    resolved = get_default_config()
    update_config_from_file(str(CONFIG_DIR / (name + ".yaml")), base_cfg=resolved)
    return resolved


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            child = "%s.%s" % (prefix, key) if prefix else str(key)
            result.update(flatten(value[key], child))
        return result
    return {prefix: value}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecordingThreeMDOTDataset:
    def __init__(self):
        self.sequence_list = ["md0001-1", "md0001-2", "md0001-3"]
        self.seq_per_class = {"md0001": [0, 1, 2]}
        self.visible = [torch.ones(8, dtype=torch.bool) for _ in range(3)]
        self.calls = []

    def __len__(self):
        return len(self.sequence_list)

    def get_name(self):
        return "THREEMDOT"

    def is_video_sequence(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        return {
            "visible": self.visible[seq_id],
            "valid": self.visible[seq_id].clone(),
        }

    def get_frames(self, seq_id, frame_ids, seq_info_dict):
        del seq_info_dict
        self.calls.append((self.sequence_list[seq_id], tuple(frame_ids)))
        frames = [
            np.full((8, 8, 3), seq_id * 16 + frame_id, dtype=np.uint8)
            for frame_id in frame_ids
        ]
        anno = {
            "bbox": [torch.tensor([2.0, 2.0, 3.0, 3.0]) for _ in frame_ids]
        }
        return frames, anno, {"object_class_name": "target"}


def stochastic_trace_processing(data):
    data["augmentation_trace"] = torch.tensor([
        random.random(),
        float(np.random.random()),
        float(torch.rand(()).item()),
    ])
    data["valid"] = True
    return data


def sample_trace(seed):
    dataset = RecordingThreeMDOTDataset()
    sampler = TrackingSamplerThreeMDOT(
        datasets=[dataset],
        p_datasets=None,
        samples_per_epoch=1,
        max_gap=3,
        num_search_frames=1,
        num_template_frames=1,
        processing=stochastic_trace_processing,
        frame_sample_mode="causal",
        require_all_views_visible=True,
        canonical_view_order=True,
        max_retry=20,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = sampler[0]
    return dataset.calls, data


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.norm = nn.LayerNorm(4)

    def forward(self, value):
        value = self.blocks[4](value)
        value = self.blocks[5](value)
        return self.norm(value)


class TinyLSPCANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.box_head = nn.Linear(4, 4)
        self.pcum = nn.Linear(4, 4)

    def forward(self, value):
        return self.box_head(self.pcum(self.backbone(value)))


class LSPCAV2ConfigTests(unittest.TestCase):
    def test_three_view_runner_binds_interleaved_manifest_by_target(self):
        sequence = lambda name: type("Sequence", (), {"name": name})()
        interleaved = [
            sequence("md3013-1"), sequence("md3013-2"), sequence("md3013-3"),
            sequence("md3019-1"), sequence("md3019-2"), sequence("md3019-3"),
        ]
        triplets = three_view_triplets(interleaved)
        self.assertEqual(
            [[item.name for item in triplet] for triplet in triplets],
            [
                ["md3013-1", "md3013-2", "md3013-3"],
                ["md3019-1", "md3019-2", "md3019-3"],
            ],
        )
        legacy_order = [
            sequence("md3013-1"), sequence("md3019-1"),
            sequence("md3013-2"), sequence("md3019-2"),
            sequence("md3013-3"), sequence("md3019-3"),
        ]
        self.assertEqual(
            [[item.name for item in triplet]
             for triplet in three_view_triplets(legacy_order)],
            [
                ["md3013-1", "md3013-2", "md3013-3"],
                ["md3019-1", "md3019-2", "md3019-3"],
            ],
        )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            three_view_triplets(interleaved[:-1])

    def test_new_loader_switch_defaults_off_and_preserves_legacy_behavior(self):
        self.assertFalse(use_grouped_multiview_loader(get_default_config()))
        legacy_j0 = load_config("ostrack_deit_tiny_b0_j0_partial_adapt_ep15_fold0")
        legacy_j1 = load_config("ostrack_deit_tiny_b0_j1_partial_adapt_ep15_fold0")
        self.assertFalse(use_grouped_multiview_loader(legacy_j0))
        self.assertTrue(use_grouped_multiview_loader(legacy_j1))

    def test_all_fold_pairs_use_one_grouped_loader_and_identical_manifests(self):
        invariant_paths = [
            "B0_CHECKPOINT",
            "MODEL.PRETRAIN_FILE",
            "TRAIN.SEED",
            "TRAIN.EPOCH",
            "TRAIN.BATCH_SIZE",
            "TRAIN.OPTIMIZER",
            "TRAIN.WEIGHT_DECAY",
            "TRAIN.PARTIAL_ADAPTATION.BACKBONE_BLOCKS",
            "TRAIN.PARTIAL_ADAPTATION.BACKBONE_LR",
            "TRAIN.PARTIAL_ADAPTATION.HEAD_LR",
            "TRAIN.PCUM.REAL_MULTIVIEW_LOSS_WEIGHTS",
            "DATA.TRAIN.SPLIT_FILE",
            "DATA.VAL.SPLIT_FILE",
        ]

        def value_at(config, path):
            value = config
            for part in path.split("."):
                value = getattr(value, part)
            return list(value) if isinstance(value, list) else value

        for fold in range(5):
            j0 = load_config(J0_BASE + "_fold%d" % fold)
            j1 = load_config(J1_BASE + "_fold%d" % fold)
            self.assertTrue(use_grouped_multiview_loader(j0))
            self.assertTrue(use_grouped_multiview_loader(j1))
            self.assertTrue(j0.TRAIN.MULTIVIEW.ENABLED)
            self.assertTrue(j1.TRAIN.MULTIVIEW.ENABLED)
            for path in invariant_paths:
                self.assertEqual(value_at(j0, path), value_at(j1, path), path)
            self.assertEqual(
                sha256_file(j0.DATA.TRAIN.SPLIT_FILE),
                sha256_file(j1.DATA.TRAIN.SPLIT_FILE),
            )
            self.assertEqual(
                sha256_file(j0.DATA.VAL.SPLIT_FILE),
                sha256_file(j1.DATA.VAL.SPLIT_FILE),
            )

    def test_fold0_resolved_config_diff_is_allowlisted(self):
        j0 = flatten(load_config(J0_BASE + "_fold0"))
        j1 = flatten(load_config(J1_BASE + "_fold0"))
        differences = {
            key for key in set(j0) | set(j1) if j0.get(key) != j1.get(key)
        }
        expected = {
            "BASE_CONFIG",
            "MODEL.PCUM.ENABLED",
            "MODEL_ROLE",
            "TEST.CHECKPOINT_NAME",
            "TEST.PCUM.USE_REMOTE",
            "TEST.SAVE_DIR",
            "TRAIN.PCUM.PAIRED_SUPERVISION",
            "TRAIN.PCUM.SAFE_LOSS_WEIGHT",
            "TRAIN.PCUM.USE_REAL_MULTIVIEW",
        }
        self.assertEqual(differences, expected)

    def test_no_gt_inference_boundary(self):
        for name in (J0_BASE + "_fold0", J1_BASE + "_fold0"):
            config = load_config(name)
            self.assertFalse(config.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)
            self.assertIn(config.TEST.PCUM.REMOTE_STATE_SOURCE, ("tracker", "none"))
        forbidden = {
            "gt", "ground_truth", "target_visible", "gt_visibility",
            "oracle_mask", "test_iou",
        }
        self.assertTrue(
            forbidden.isdisjoint(inspect.signature(EnTeRTrack.forward).parameters)
        )

    def test_sampler_and_augmentation_trace_is_seed_identical(self):
        first_calls, first = sample_trace(20260715)
        second_calls, second = sample_trace(20260715)
        self.assertEqual(first_calls, second_calls)
        self.assertEqual([name for name, _ in first_calls], [
            "md0001-1", "md0001-1", "md0001-2",
            "md0001-2", "md0001-3", "md0001-3",
        ])
        self.assertTrue(torch.equal(
            first["augmentation_trace"], second["augmentation_trace"]))
        self.assertTrue(torch.equal(
            first["template_view_valid"], second["template_view_valid"]))
        self.assertTrue(torch.equal(
            first["search_view_valid"], second["search_view_valid"]))


class LSPCAV2ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.j0_cfg, cls.j0_model, cls.j0_groups = build_for_audit(
            J0_BASE + "_fold0")
        cls.j1_cfg, cls.j1_model, cls.j1_groups = build_for_audit(
            J1_BASE + "_fold0")
        cls.j0_smoke = run_one_step(
            "J0v2", cls.j0_cfg, cls.j0_model, cls.j0_groups)
        cls.j1_smoke = run_one_step(
            "J1v2", cls.j1_cfg, cls.j1_model, cls.j1_groups)

    def test_b0_checkpoint_hash_and_strict_initialization(self):
        self.assertEqual(sha256_file(self.j0_cfg.B0_CHECKPOINT), FROZEN_B0_SHA256)
        j0_audit = self.j0_model.initialization_audit
        j1_audit = self.j1_model.initialization_audit
        self.assertEqual(j0_audit["checkpoint_epoch"], 25)
        self.assertTrue(j0_audit["strict_core_load"])
        self.assertEqual(j0_audit["missing_keys"], [])
        self.assertEqual(j0_audit["unexpected_keys"], [])
        self.assertTrue(j1_audit["strict_full_load"])
        self.assertEqual(j1_audit["inherited_a0_pcum_parameters"], 0)
        self.assertGreater(j1_audit["fresh_pcum_key_count"], 0)

    def test_trainable_sets_and_optimizer_groups(self):
        j0_total = sum(parameter.numel() for parameter in self.j0_model.parameters())
        j1_total = sum(parameter.numel() for parameter in self.j1_model.parameters())
        self.assertEqual(j0_total, 5409477)
        self.assertEqual(j1_total, 6092230)
        self.assertAlmostEqual(
            100.0 * (j1_total - j0_total) / j0_total,
            12.621423845,
            places=6,
        )
        self.assertLessEqual((j1_total - j0_total) / j0_total, 0.15)
        j0_totals = summarize_trainable(self.j0_model)[1]
        j1_totals = summarize_trainable(self.j1_model)[1]
        self.assertEqual(set(j0_totals) - {"total"}, {"last_backbone", "head"})
        self.assertEqual(
            set(j1_totals) - {"total"}, {"last_backbone", "head", "pcum"})
        self.assertEqual(j0_totals["last_backbone"], j1_totals["last_backbone"])
        self.assertEqual(j0_totals["head"], j1_totals["head"])
        j0_groups = optimizer_summary(self.j0_groups, self.j0_model)
        j1_groups = optimizer_summary(self.j1_groups, self.j1_model)
        self.assertEqual(
            [(row["group_name"], row["lr"]) for row in j0_groups],
            [("last_backbone", 2.4e-6), ("head", 8e-6)],
        )
        self.assertEqual(
            [(row["group_name"], row["lr"]) for row in j1_groups],
            [("last_backbone", 2.4e-6), ("head", 8e-6), ("pcum", 8e-5)],
        )

    def test_synthetic_one_step_is_finite_and_updates_only_allowed_sets(self):
        for smoke in (self.j0_smoke, self.j1_smoke):
            self.assertTrue(smoke["loss_finite"])
            self.assertEqual(smoke["grad_summary"]["bad_grad_names"], [])
            self.assertGreater(smoke["grad_summary"]["grad_tensor_count"], 0)
            self.assertTrue(smoke["last_backbone_changed"])
            self.assertTrue(smoke["head_changed"])
            self.assertEqual(smoke["unexpected_backbone_changed"], [])
        self.assertFalse(self.j0_smoke["pcum_changed"])
        self.assertTrue(self.j1_smoke["pcum_changed"])

    def test_resume_strictly_preserves_trained_pcum_state(self):
        saved = copy.deepcopy(self.j1_model.state_dict())
        torch.manual_seed(99)
        resumed = build_entertrack(self.j1_cfg, training=True)
        resumed.load_state_dict(saved, strict=True)
        for name, value in saved.items():
            if name.startswith("pcum."):
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)

    def test_no_remote_delay_and_wrong_remote_are_finite(self):
        pcum = build_pcum(self.j1_cfg, token_dim=192)
        features = {
            "search": torch.randn(2, 20, 192),
            "template": torch.randn(2, 8, 192),
        }
        local = pcum(features)
        remote = [torch.randn(2, 4, 192), torch.randn(2, 4, 192)]
        states = [
            {"score": torch.tensor([0.8, 0.7]), "apce": torch.tensor([0.9, 0.8])},
            {"score": torch.tensor([0.6, 0.5]), "apce": torch.tensor([0.7, 0.6])},
        ]
        raw = pcum(features, remote_prompts=remote, remote_states=states)
        actor = EnTeRTrackActorThreeMDOT(
            net=None,
            objective={},
            loss_weight={},
            settings=type("Settings", (), {"batchsize": 2})(),
            cfg=self.j1_cfg,
        )
        delayed = actor._make_delay_remote_bank(remote)
        delay = pcum(features, remote_prompts=delayed, remote_states=states)
        wrong = pcum(
            features,
            remote_prompts=[torch.flip(item, dims=[0]) for item in remote],
            remote_states=states,
        )
        for output in (local, raw, delay, wrong):
            self.assertTrue(torch.isfinite(output["search_tokens"]).all())
            self.assertEqual(output["search_tokens"].shape, features["search"].shape)
        self.assertIsNotNone(raw["remote_weights"])
        self.assertTrue(torch.isfinite(raw["remote_weights"]).all())
        self.assertTrue(torch.allclose(
            raw["remote_weights"].sum(dim=-1),
            torch.ones(raw["remote_weights"].shape[0]),
            atol=1e-6,
        ))
        self.assertIsNotNone(raw["remote_aggregation_diagnostics"])

    def test_partial_adaptation_helpers_accept_real_ddp_prefixes(self):
        if dist.is_initialized():
            self.skipTest("process group already initialized by another test")
        config = copy.deepcopy(self.j1_cfg)
        model = TinyLSPCANet()
        apply_partial_adaptation_freeze(model, config)
        rendezvous = tempfile.NamedTemporaryFile(delete=False)
        rendezvous.close()
        try:
            dist.init_process_group(
                backend="gloo",
                init_method="file://" + rendezvous.name,
                rank=0,
                world_size=1,
            )
            wrapped = DistributedDataParallel(
                model,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            groups = build_optimizer_param_groups(wrapped, config)
            self.assertEqual(
                {group["group_name"] for group in groups},
                {"last_backbone", "head", "pcum"},
            )
            optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
            before = {
                name: value.detach().clone()
                for name, value in wrapped.module.state_dict().items()
            }
            loss = wrapped(torch.randn(2, 4)).square().mean()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            changed = [
                name for name, value in wrapped.module.state_dict().items()
                if not torch.equal(before[name], value)
            ]
            self.assertTrue(any(name.startswith("backbone.blocks.4.") for name in changed))
            self.assertTrue(any(name.startswith("backbone.blocks.5.") for name in changed))
            self.assertTrue(any(name.startswith("backbone.norm.") for name in changed))
            self.assertTrue(any(name.startswith("box_head.") for name in changed))
            self.assertTrue(any(name.startswith("pcum.") for name in changed))
            self.assertFalse(any(
                name.startswith("backbone.blocks.%d." % index)
                for name in changed for index in range(4)
            ))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            if os.path.exists(rendezvous.name):
                os.unlink(rendezvous.name)


if __name__ == "__main__":
    unittest.main()
